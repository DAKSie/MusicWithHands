from __future__ import annotations

import time
from typing import Dict, List, TypedDict
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections


class TrackedHand(TypedDict):
    landmarks: np.ndarray
    handedness: str


class HandTracker:
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        smoothing_alpha: float = 0.55,
        model_path: str | None = None,
    ) -> None:
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self._model_path = self._resolve_model_path(model_path)
        self._ensure_model_exists(self._model_path)

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self.connections = sorted(
            (int(connection.start), int(connection.end))
            for connection in HandLandmarksConnections.HAND_CONNECTIONS
        )
        self._previous_landmarks: Dict[int, np.ndarray] = {}

    def process(self, frame_bgr: np.ndarray) -> List[TrackedHand]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.perf_counter() * 1000)
        results = self._landmarker.detect_for_video(image, timestamp_ms)
        tracked_hands: List[TrackedHand] = []

        if results.hand_landmarks:
            for hand_index, hand_landmarks in enumerate(results.hand_landmarks):
                current = np.array(
                    [[point.x, point.y, point.z] for point in hand_landmarks],
                    dtype=np.float32,
                )
                handedness = "Unknown"
                if results.handedness and hand_index < len(results.handedness):
                    category_list = results.handedness[hand_index]
                    if category_list:
                        handedness = str(category_list[0].category_name or "Unknown")
                tracked_hands.append({
                    "landmarks": self._smooth(hand_index, current),
                    "handedness": handedness,
                })
            self._trim_previous(len(tracked_hands))
        else:
            self._previous_landmarks.clear()

        return tracked_hands

    def normalized_to_pixels(self, landmarks: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
        x_coords = np.clip(landmarks[:, 0] * frame_width, 0, frame_width - 1)
        y_coords = np.clip(landmarks[:, 1] * frame_height, 0, frame_height - 1)
        return np.stack((x_coords, y_coords), axis=1).astype(np.int32)

    def hand_center_pixels(self, landmarks: np.ndarray, frame_width: int, frame_height: int) -> tuple[int, int]:
        center = np.mean(landmarks[:, :2], axis=0)
        x = int(np.clip(center[0] * frame_width, 0, frame_width - 1))
        y = int(np.clip(center[1] * frame_height, 0, frame_height - 1))
        return x, y

    def is_hand_open(self, landmarks: np.ndarray, extended_threshold: int = 4) -> bool:
        return self.extended_finger_count(landmarks) >= extended_threshold

    def extended_finger_count(self, landmarks: np.ndarray) -> int:
        wrist = landmarks[0, :2]
        finger_pairs = [(4, 3), (8, 6), (12, 10), (16, 14), (20, 18)]
        count = 0

        for tip_index, pip_index in finger_pairs:
            tip_distance = float(np.linalg.norm(landmarks[tip_index, :2] - wrist))
            pip_distance = float(np.linalg.norm(landmarks[pip_index, :2] - wrist))
            if tip_distance > (pip_distance + 0.015):
                count += 1

        return count

    def close(self) -> None:
        self._landmarker.close()

    def _smooth(self, hand_index: int, current_landmarks: np.ndarray) -> np.ndarray:
        previous = self._previous_landmarks.get(hand_index)
        if previous is None:
            self._previous_landmarks[hand_index] = current_landmarks
            return current_landmarks

        smoothed = (self.smoothing_alpha * current_landmarks) + ((1.0 - self.smoothing_alpha) * previous)
        self._previous_landmarks[hand_index] = smoothed
        return smoothed

    def _trim_previous(self, hand_count: int) -> None:
        stale_indices = [index for index in self._previous_landmarks if index >= hand_count]
        for index in stale_indices:
            del self._previous_landmarks[index]

    @staticmethod
    def _resolve_model_path(model_path: str | None) -> Path:
        if model_path:
            return Path(model_path)
        return Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"

    @staticmethod
    def _ensure_model_exists(model_path: Path) -> None:
        if model_path.exists():
            return

        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
        try:
            urlretrieve(model_url, model_path)
        except Exception as exc:
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError(f"Failed to download hand model: {exc}") from exc
