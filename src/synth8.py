from __future__ import annotations

import threading
import math
from typing import Dict, List

import numpy as np
import sounddevice as sd


class Synth8:
    def __init__(self, sample_rate: int = 44100, block_size: int = 512) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._presets: Dict[str, Dict[str, float | str]] = {
            "Warm Sine": {"wave": "sine", "drive": 1.1, "detune": 0.0015},
            "Bright Square": {"wave": "square", "drive": 0.7, "detune": 0.0008},
            "Soft Saw": {"wave": "saw", "drive": 0.6, "detune": 0.0012},
            "Triangle Pad": {"wave": "triangle", "drive": 0.9, "detune": 0.001},
            "Retro Organ": {"wave": "organ", "drive": 0.8, "detune": 0.0009},
            "Glass Bell": {"wave": "bell", "drive": 1.0, "detune": 0.0013},
            "Pulse Lead": {"wave": "pulse", "drive": 0.85, "detune": 0.0014},
            "Pluck": {"wave": "pluck", "drive": 1.05, "detune": 0.0011},
        }
        self._current_sound = "Warm Sine"
        self._active_frequencies: List[float] = []
        self._phase_map: Dict[float, float] = {}
        self._target_gain = 0.0
        self._current_gain = 0.0
        self._release_seconds = 0.35
        self._reverb_amount = 0.0
        self._reverb_buffer = np.zeros(max(1, int(self.sample_rate * 0.18)), dtype=np.float32)
        self._reverb_index = 0
        self._lock = threading.Lock()
        self.enabled = True
        self._stream: sd.OutputStream | None = None

        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except Exception:
            self.enabled = False
            self._stream = None

    @property
    def current_sound(self) -> str:
        return self._current_sound

    def get_sound_names(self) -> List[str]:
        return list(self._presets.keys())

    def set_sound(self, sound_name: str) -> None:
        if sound_name in self._presets:
            with self._lock:
                self._current_sound = sound_name

    def set_active_frequencies(self, frequencies: List[float]) -> None:
        with self._lock:
            unique = sorted({round(float(frequency), 4) for frequency in frequencies if frequency > 0.0})
            self._active_frequencies = unique
            self._target_gain = 0.24 if unique else 0.0
            if unique:
                active_keys = set(unique)
                stale = [frequency for frequency in self._phase_map if frequency not in active_keys]
                for frequency in stale:
                    del self._phase_map[frequency]

    def set_release_seconds(self, seconds: float) -> None:
        value = float(np.clip(seconds, 0.05, 1.5))
        with self._lock:
            self._release_seconds = value

    def set_reverb_amount(self, amount: float) -> None:
        value = float(np.clip(amount, 0.0, 1.0))
        with self._lock:
            self._reverb_amount = value

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _callback(self, outdata: np.ndarray, frames: int, _time: object, _status: sd.CallbackFlags) -> None:
        with self._lock:
            preset = self._presets[self._current_sound]
            frequencies = list(self._active_frequencies)
            gain_target = self._target_gain
            release_seconds = self._release_seconds
            reverb_amount = self._reverb_amount
            phase_frequencies = list(self._phase_map.keys())

        render_frequencies = list(frequencies)
        if not render_frequencies and self._current_gain > 1e-4 and phase_frequencies:
            render_frequencies = phase_frequencies

        mix = np.zeros(frames, dtype=np.float32)

        for frequency in render_frequencies:
            phase = self._phase_map.get(frequency, 0.0)
            delta = (2.0 * np.pi * frequency) / self.sample_rate
            phase_values = phase + (delta * np.arange(frames, dtype=np.float32))
            signal = self._waveform(str(preset["wave"]), phase_values)
            detune = float(preset["detune"])
            if detune > 0.0:
                detuned_delta = (2.0 * np.pi * (frequency * (1.0 + detune))) / self.sample_rate
                detuned_phase_values = phase + (detuned_delta * np.arange(frames, dtype=np.float32))
                signal = (signal * 0.75) + (self._waveform(str(preset["wave"]), detuned_phase_values) * 0.25)
            mix += signal
            self._phase_map[frequency] = float((phase + (delta * frames)) % (2.0 * np.pi))

        if render_frequencies:
            mix /= max(len(render_frequencies), 1)

        attack_blend = 0.12
        release_blend = 1.0 - math.exp(-frames / max(1.0, self.sample_rate * release_seconds))
        gain_blend = attack_blend if gain_target >= self._current_gain else release_blend
        self._current_gain += (gain_target - self._current_gain) * gain_blend
        driven = np.tanh(mix * float(preset["drive"]))
        output = driven * self._current_gain
        output = self._apply_reverb(output, reverb_amount)
        outdata[:, 0] = output.astype(np.float32)

        if gain_target <= 0.0 and not frequencies and self._current_gain < 1e-4:
            with self._lock:
                if not self._active_frequencies and self._target_gain <= 0.0:
                    self._phase_map.clear()

    def _apply_reverb(self, signal: np.ndarray, amount: float) -> np.ndarray:
        if amount <= 0.001:
            return signal

        wet = float(np.clip(0.05 + (0.45 * amount), 0.0, 0.8))
        feedback = float(np.clip(0.15 + (0.65 * amount), 0.0, 0.92))
        damping = float(np.clip(0.2 + (0.6 * amount), 0.0, 0.98))

        output = np.empty_like(signal)
        buffer = self._reverb_buffer
        index = self._reverb_index
        dry = 1.0 - wet

        for sample_index in range(signal.shape[0]):
            delayed = float(buffer[index])
            input_sample = float(signal[sample_index])
            reverberated = input_sample + (delayed * feedback)
            buffer[index] = (reverberated * (1.0 - damping)) + (delayed * damping)
            output[sample_index] = (input_sample * dry) + (delayed * wet)
            index += 1
            if index >= buffer.shape[0]:
                index = 0

        self._reverb_index = index
        return output

    def _waveform(self, wave: str, phase: np.ndarray) -> np.ndarray:
        if wave == "sine":
            return np.sin(phase)
        if wave == "square":
            return np.where(np.sin(phase) >= 0.0, 1.0, -1.0)
        if wave == "saw":
            return (2.0 * ((phase / (2.0 * np.pi)) % 1.0)) - 1.0
        if wave == "triangle":
            return (2.0 * np.abs((2.0 * ((phase / (2.0 * np.pi)) % 1.0)) - 1.0)) - 1.0
        if wave == "organ":
            return (np.sin(phase) * 0.65) + (np.sin(phase * 2.0) * 0.25) + (np.sin(phase * 3.0) * 0.1)
        if wave == "bell":
            return (np.sin(phase) * 0.7) + (np.sin(phase * 2.6) * 0.2) + (np.sin(phase * 4.1) * 0.1)
        if wave == "pulse":
            return np.where(np.sin(phase) >= 0.35, 1.0, -1.0)
        if wave == "pluck":
            return (np.sin(phase) * 0.5) + (np.sin(phase * 2.0) * 0.35) + (np.sin(phase * 3.0) * 0.15)
        return np.sin(phase)
