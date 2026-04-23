from __future__ import annotations

import math
import time
from typing import List

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

from hand_tracker import HandTracker, TrackedHand
from note_wheel import NoteWheel
from synth8 import Synth8


BACKGROUND_COLOR = (255, 253, 241, 255)
SURFACE_COLOR = (255, 206, 153, 255)
ACCENT_COLOR = (255, 150, 68, 255)
TEXT_COLOR = (86, 47, 0, 255)


def _rgba(value: tuple[int, int, int, int], alpha: int | None = None) -> tuple[int, int, int, int]:
    if alpha is None:
        return value
    return (value[0], value[1], value[2], alpha)

class HandTrackingApp:
    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        target_fps: int = 60,
        debug_print_interval: float = 0.2,
        camera_index: int = 0,
        max_num_hands: int = 2,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.target_fps = target_fps
        self.debug_print_interval = debug_print_interval
        self.camera_index = camera_index
        self.max_num_hands = max_num_hands

        self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            raise RuntimeError("Unable to open camera. Verify webcam access and camera index.")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.capture.set(cv2.CAP_PROP_FPS, self.target_fps)

        self.hand_tracker = HandTracker(
            max_num_hands=self.max_num_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
            smoothing_alpha=0.55,
        )

        self.connections = self.hand_tracker.connections
        self.connection_colors = self._build_connection_colors(len(self.connections))

        self.texture_tag = "camera_texture"
        self.drawlist_tag = "camera_drawlist"
        self.overlay_layer_tag = "overlay_layer"
        self.camera_container_tag = "camera_container"
        self.fps_text_tag = "fps_text"
        self.synth_combo_tag = "synth_combo"
        self.playing_text_tag = "playing_text"
        self.synth_status_tag = "synth_status"
        self.window_tag = "main_window"
        self.theme_tag = "app_theme"

        self.note_wheel = NoteWheel(self.frame_width, self.frame_height)
        self.synth = Synth8()

        self._texture_buffer = np.zeros(self.frame_width * self.frame_height * 4, dtype=np.float32)
        self._rgba_frame = np.zeros((self.frame_height, self.frame_width, 4), dtype=np.uint8)
        self._camera_display_scale = 1.0
        self._camera_display_offset = (0.0, 0.0)
        self._camera_canvas_size = (0, 0)

        self._frame_counter = 0
        self._fps_timer_start = time.perf_counter()
        self._last_debug_print = 0.0
        self._active_section: int | None = None
        self._active_mode_is_chord = False
        self._control_point: tuple[int, int] | None = None
        self._release_fast_min_seconds = 0.08
        self._release_slow_max_seconds = 1.2
        self._free_release_seconds = 0.35
        self._free_release_fastness = 0.5
        self._free_reverb_amount = 0.0
        self._free_finger_count = 0
        self._sustain_latched = False

        self._setup_ui()

    def run(self) -> None:
        try:
            while dpg.is_dearpygui_running():
                has_frame, frame_bgr = self.capture.read()
                self._sync_camera_canvas_layout()
                if has_frame:
                    frame_bgr = cv2.flip(frame_bgr, 1)
                    tracked_hands = self.hand_tracker.process(frame_bgr)
                    landmarks = [tracked_hand["landmarks"] for tracked_hand in tracked_hands]
                    self._update_camera_texture(frame_bgr)
                    self._update_music_control(tracked_hands)
                    self._draw_hand_overlay(landmarks)
                    self._print_landmarks(landmarks)
                    self._update_fps()

                dpg.render_dearpygui_frame()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if hasattr(self, "capture") and self.capture.isOpened():
            self.capture.release()

        if hasattr(self, "hand_tracker"):
            self.hand_tracker.close()

        if hasattr(self, "synth"):
            self.synth.close()

        if dpg.is_dearpygui_running() or dpg.does_item_exist(self.window_tag):
            dpg.destroy_context()

    def _setup_ui(self) -> None:
        dpg.create_context()

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                width=self.frame_width,
                height=self.frame_height,
                default_value=self._texture_buffer,
                format=dpg.mvFormat_Float_rgba,
                tag=self.texture_tag,
            )

        self._setup_theme()

        with dpg.window(
            tag=self.window_tag,
            label="Real-Time Hand Tracking",
            no_resize=False,
            no_move=True,
            no_collapse=True,
            no_scrollbar=True,
        ):
            with dpg.tab_bar():
                with dpg.tab(label="Camera"):
                    with dpg.child_window(
                        tag=self.camera_container_tag,
                        border=False,
                        horizontal_scrollbar=False,
                        no_scrollbar=True,
                        width=-1,
                        height=-1,
                    ):
                        with dpg.drawlist(tag=self.drawlist_tag, width=-1, height=-1):
                            pass

                with dpg.tab(label="Controls"):
                    dpg.add_text("Synth preset")
                    dpg.add_combo(
                        items=self.synth.get_sound_names(),
                        default_value=self.synth.current_sound,
                        tag=self.synth_combo_tag,
                        callback=self._on_synth_changed,
                        width=-1,
                    )

                with dpg.tab(label="Status"):
                    dpg.add_text("FPS: 0.00", tag=self.fps_text_tag)
                    dpg.add_text("Now Playing: --", tag=self.playing_text_tag)
                    synth_state = "Synth8: Ready" if self.synth.enabled else "Synth8: Audio device unavailable"
                    dpg.add_text(synth_state, tag=self.synth_status_tag)

        dpg.create_viewport(
            title="Music with fingers rigamarole",
            width=1280,
            height=800,
            resizable=True,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window(self.window_tag, True)
        dpg.set_viewport_vsync(False)

    def _setup_theme(self) -> None:
        with dpg.theme(tag=self.theme_tag):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, BACKGROUND_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, BACKGROUND_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, BACKGROUND_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_Text, TEXT_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_Border, TEXT_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_Separator, TEXT_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, SURFACE_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, ACCENT_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, ACCENT_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_Button, SURFACE_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, ACCENT_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, ACCENT_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_Header, SURFACE_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, ACCENT_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, ACCENT_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_Tab, SURFACE_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered, ACCENT_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, BACKGROUND_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused, SURFACE_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive, BACKGROUND_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, BACKGROUND_COLOR)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, BACKGROUND_COLOR)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0.0)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 0.0)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 0.0)
                dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 0.0)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 14.0, 14.0)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10.0, 6.0)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10.0, 10.0)
                dpg.add_theme_style(dpg.mvStyleVar_IndentSpacing, 12.0)
                dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 1.0)
                dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 1.0)
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1.0)

        dpg.bind_theme(self.theme_tag)

    def _update_camera_texture(self, frame_bgr: np.ndarray) -> None:
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA, dst=self._rgba_frame)
        np.divide(self._rgba_frame.reshape(-1), 255.0, out=self._texture_buffer, casting="unsafe")
        dpg.set_value(self.texture_tag, self._texture_buffer)

    def _sync_camera_canvas_layout(self) -> None:
        if not dpg.does_item_exist(self.camera_container_tag) or not dpg.does_item_exist(self.drawlist_tag):
            return

        container_width, container_height = dpg.get_item_rect_size(self.camera_container_tag)
        canvas_width = max(1, int(container_width))
        canvas_height = max(1, int(container_height))

        if (canvas_width, canvas_height) != self._camera_canvas_size:
            dpg.configure_item(self.drawlist_tag, width=canvas_width, height=canvas_height)
            self._camera_canvas_size = (canvas_width, canvas_height)

        scale = min(canvas_width / self.frame_width, canvas_height / self.frame_height)
        draw_width = self.frame_width * scale
        draw_height = self.frame_height * scale
        offset_x = (canvas_width - draw_width) * 0.5
        offset_y = (canvas_height - draw_height) * 0.5

        self._camera_display_scale = scale
        self._camera_display_offset = (offset_x, offset_y)

    def _camera_point_to_display(self, point: tuple[int, int]) -> tuple[int, int]:
        offset_x, offset_y = self._camera_display_offset
        scale = self._camera_display_scale
        return (
            int(round(offset_x + (point[0] * scale))),
            int(round(offset_y + (point[1] * scale))),
        )

    def _camera_length_to_display(self, value: float) -> float:
        return max(1.0, value * self._camera_display_scale)

    def _draw_hand_overlay(self, hands: List[np.ndarray]) -> None:
        if not dpg.does_item_exist(self.drawlist_tag):
            return

        dpg.delete_item(self.drawlist_tag, children_only=True)

        offset_x, offset_y = self._camera_display_offset
        draw_width = self.frame_width * self._camera_display_scale
        draw_height = self.frame_height * self._camera_display_scale
        image_min = (int(round(offset_x)), int(round(offset_y)))
        image_max = (int(round(offset_x + draw_width)), int(round(offset_y + draw_height)))

        dpg.draw_image(self.texture_tag, image_min, image_max, parent=self.drawlist_tag)
        with dpg.draw_node(tag=self.overlay_layer_tag, parent=self.drawlist_tag):
            self.note_wheel.draw(
                self.overlay_layer_tag,
                self._active_section,
                scale=self._camera_display_scale,
                offset=self._camera_display_offset,
            )

            if self._control_point is not None:
                dpg.draw_circle(
                    center=self._camera_point_to_display(self._control_point),
                    radius=self._camera_length_to_display(7.5),
                    color=ACCENT_COLOR,
                    fill=_rgba(SURFACE_COLOR, 220),
                    thickness=self._camera_length_to_display(2.0),
                    parent=self.overlay_layer_tag,
                )

            self._draw_control_gauges(self.overlay_layer_tag)

            for landmarks in hands:
                pixels = self.hand_tracker.normalized_to_pixels(landmarks, self.frame_width, self.frame_height)

                for connection_index, (start_index, end_index) in enumerate(self.connections):
                    x1, y1 = pixels[start_index]
                    x2, y2 = pixels[end_index]
                    dpg.draw_line(
                        p1=self._camera_point_to_display((int(x1), int(y1))),
                        p2=self._camera_point_to_display((int(x2), int(y2))),
                        color=self.connection_colors[connection_index],
                        thickness=self._camera_length_to_display(2.0),
                        parent=self.overlay_layer_tag,
                    )

                for x, y in pixels:
                    dpg.draw_circle(
                        center=self._camera_point_to_display((int(x), int(y))),
                        radius=self._camera_length_to_display(2.2),
                        color=_rgba(BACKGROUND_COLOR, 220),
                        fill=_rgba(BACKGROUND_COLOR, 180),
                        parent=self.overlay_layer_tag,
                    )

    def _update_music_control(self, hands: List[TrackedHand]) -> None:
        right_hand_landmarks: np.ndarray | None = None
        left_hand_landmarks: np.ndarray | None = None

        for tracked_hand in hands:
            landmarks = tracked_hand["landmarks"]
            handedness = str(tracked_hand["handedness"]).strip().lower()
            if handedness == "right" and right_hand_landmarks is None:
                right_hand_landmarks = landmarks
            elif handedness == "left" and left_hand_landmarks is None:
                left_hand_landmarks = landmarks

        if right_hand_landmarks is None:
            for tracked_hand in hands:
                landmarks = tracked_hand["landmarks"]
                if not any(landmarks is assigned for assigned in [left_hand_landmarks]):
                    right_hand_landmarks = landmarks
                    break

        if left_hand_landmarks is None:
            for tracked_hand in hands:
                landmarks = tracked_hand["landmarks"]
                if landmarks is not right_hand_landmarks:
                    left_hand_landmarks = landmarks
                    break

        selected_section: int | None = None
        selected_center: tuple[int, int] | None = None

        if left_hand_landmarks is not None:
            left_center = self.hand_tracker.hand_center_pixels(left_hand_landmarks, self.frame_width, self.frame_height)
            selected_section = self.note_wheel.section_at_point(left_center)
            if selected_section is not None:
                selected_center = left_center

        self._update_free_hand_controls(right_hand_landmarks)

        if selected_section is None:
            if not self._sustain_latched:
                self.synth.set_active_frequencies([])
                self._active_section = None
                self._control_point = None
                dpg.set_value(self.playing_text_tag, "Now Playing: --")
            return

        frequencies = self.note_wheel.frequencies_for_section(selected_section, True)
        self.synth.set_active_frequencies(frequencies)
        self._active_section = selected_section
        self._active_mode_is_chord = True
        self._control_point = selected_center

        label = f"Now Playing: {self.note_wheel.major_chord_name_for_section(selected_section)} chord"
        dpg.set_value(self.playing_text_tag, label)

    def _update_free_hand_controls(self, free_hand_landmarks: np.ndarray | None) -> None:
        if free_hand_landmarks is None:
            self._free_release_seconds = 0.35
            self._free_release_fastness = 0.5
            self._free_reverb_amount = 0.0
            self._free_finger_count = 0
            self.synth.set_release_seconds(self._free_release_seconds)
            self.synth.set_reverb_amount(self._free_reverb_amount)
            return

        free_center = self.hand_tracker.hand_center_pixels(free_hand_landmarks, self.frame_width, self.frame_height)
        y_norm = float(np.clip(free_center[1] / max(1, self.frame_height - 1), 0.0, 1.0))
        release_fastness = y_norm
        release_seconds = self._release_slow_max_seconds - (
            (self._release_slow_max_seconds - self._release_fast_min_seconds) * release_fastness
        )

        finger_count = self.hand_tracker.extended_finger_count(free_hand_landmarks)
        if finger_count == 1:
            self._sustain_latched = True
        elif finger_count >= 2:
            self._sustain_latched = False

        reverb_amount = self._rotation_to_reverb_amount(free_hand_landmarks)

        self._free_release_seconds = release_seconds
        self._free_release_fastness = release_fastness
        self._free_reverb_amount = reverb_amount
        self._free_finger_count = finger_count
        self.synth.set_release_seconds(release_seconds)
        self.synth.set_reverb_amount(reverb_amount)

    def _rotation_to_reverb_amount(self, landmarks: np.ndarray) -> float:
        wrist = landmarks[0, :2]
        middle_mcp = landmarks[9, :2]
        delta = middle_mcp - wrist
        length = float(np.linalg.norm(delta))
        if length < 1e-6:
            return 0.0
        angle = abs(math.atan2(float(delta[0]), -float(delta[1])))
        return float(np.clip(angle / (math.pi * 0.5), 0.0, 1.0))

    def _draw_control_gauges(self, parent_tag: str) -> None:
        panel_x = int(round(self._camera_display_offset[0] + (14.0 * self._camera_display_scale)))
        panel_y = int(round(self._camera_display_offset[1] + (14.0 * self._camera_display_scale)))
        panel_width = int(round(max(220.0, 250.0 * self._camera_display_scale)))
        panel_height = int(round(max(120.0, 132.0 * self._camera_display_scale)))
        bar_width = panel_width - 20
        bar_height = int(round(max(10.0, 12.0 * self._camera_display_scale)))
        text_size = max(12, int(round(14.0 * self._camera_display_scale)))

        dpg.draw_rectangle(
            pmin=(panel_x, panel_y),
            pmax=(panel_x + panel_width, panel_y + panel_height),
            color=(86, 47, 0, 230),
            fill=(255, 253, 241, 180),
            thickness=max(1.0, self._camera_length_to_display(1.2)),
            parent=parent_tag,
        )

        y_cursor = panel_y + 8
        self._draw_gauge_row(
            parent_tag,
            label=f"Release speed {self._free_release_seconds:.2f}s",
            value=self._free_release_fastness,
            x=panel_x + 10,
            y=y_cursor,
            width=bar_width,
            height=bar_height,
            text_size=text_size,
        )

        y_cursor += int(round(36.0 * self._camera_display_scale))
        self._draw_gauge_row(
            parent_tag,
            label=f"Reverb {int(round(self._free_reverb_amount * 100.0))}%",
            value=self._free_reverb_amount,
            x=panel_x + 10,
            y=y_cursor,
            width=bar_width,
            height=bar_height,
            text_size=text_size,
        )

        y_cursor += int(round(36.0 * self._camera_display_scale))
        sustain_value = 1.0 if self._sustain_latched else 0.0
        sustain_label = "Sustain ON" if self._sustain_latched else "Sustain OFF"
        self._draw_gauge_row(
            parent_tag,
            label=f"{sustain_label} ({self._free_finger_count} fingers)",
            value=sustain_value,
            x=panel_x + 10,
            y=y_cursor,
            width=bar_width,
            height=bar_height,
            text_size=text_size,
        )

    def _draw_gauge_row(
        self,
        parent_tag: str,
        label: str,
        value: float,
        x: int,
        y: int,
        width: int,
        height: int,
        text_size: int,
    ) -> None:
        clamped = float(np.clip(value, 0.0, 1.0))
        dpg.draw_text(
            pos=(x, y),
            text=label,
            color=(86, 47, 0, 255),
            size=text_size,
            parent=parent_tag,
        )

        bar_top = y + int(round(16.0 * self._camera_display_scale))
        dpg.draw_rectangle(
            pmin=(x, bar_top),
            pmax=(x + width, bar_top + height),
            color=(86, 47, 0, 220),
            fill=(255, 206, 153, 120),
            thickness=max(1.0, self._camera_length_to_display(1.0)),
            parent=parent_tag,
        )

        fill_width = int(round((width - 2) * clamped))
        if fill_width > 0:
            dpg.draw_rectangle(
                pmin=(x + 1, bar_top + 1),
                pmax=(x + 1 + fill_width, bar_top + height - 1),
                color=(255, 150, 68, 230),
                fill=(255, 150, 68, 210),
                thickness=0.0,
                parent=parent_tag,
            )

    def _on_synth_changed(self, _sender: str, app_data: str, _user_data: object) -> None:
        self.synth.set_sound(str(app_data))

    def _print_landmarks(self, hands: List[np.ndarray]) -> None:
        now = time.perf_counter()
        if (now - self._last_debug_print) < self.debug_print_interval:
            return

        self._last_debug_print = now
        for hand_index, landmarks in enumerate(hands):
            chunks = []
            for point_index, point in enumerate(landmarks):
                chunks.append(
                    f"{point_index}:({point[0]:.4f},{point[1]:.4f},{point[2]:.4f})"
                )
            print(f"hand_{hand_index} {' '.join(chunks)}")

    def _update_fps(self) -> None:
        self._frame_counter += 1
        now = time.perf_counter()
        elapsed = now - self._fps_timer_start
        if elapsed >= 0.5:
            fps = self._frame_counter / elapsed
            dpg.set_value(self.fps_text_tag, f"FPS: {fps:.2f}")
            self._frame_counter = 0
            self._fps_timer_start = now

    @staticmethod
    def _build_connection_colors(connection_count: int) -> List[tuple[int, int, int, int]]:
        colors: List[tuple[int, int, int, int]] = []
        if connection_count <= 0:
            return colors

        palette = [TEXT_COLOR, ACCENT_COLOR]
        for index in range(connection_count):
            colors.append(palette[index % len(palette)])
        return colors


def main() -> None:
    app = HandTrackingApp()
    app.run()


if __name__ == "__main__":
    main()
