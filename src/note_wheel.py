from __future__ import annotations

import math
from typing import List

import dearpygui.dearpygui as dpg


class NoteWheel:
    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.center = (frame_width - 130, frame_height - 130)
        self.outer_radius = min(frame_width, frame_height) * 0.22
        self.inner_radius = self.outer_radius * 0.46
        self.section_count = 12
        self.section_angle = (2.0 * math.pi) / self.section_count
        self.labels = ["C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F"]
        self.pitch_classes = {
            "C": 0,
            "C#": 1,
            "D": 2,
            "D#": 3,
            "E": 4,
            "F": 5,
            "F#": 6,
            "G": 7,
            "G#": 8,
            "A": 9,
            "A#": 10,
            "B": 11,
        }

    def section_at_point(self, point: tuple[int, int]) -> int | None:
        cx, cy = self.center
        px, py = point
        dx = px - cx
        dy = py - cy
        radius = math.hypot(dx, dy)

        if radius < self.inner_radius or radius > self.outer_radius:
            return None

        angle = math.atan2(dx, -dy) % (2.0 * math.pi)
        section = int(angle // self.section_angle)
        return section if 0 <= section < self.section_count else None

    def label_for_section(self, section: int) -> str:
        return self.labels[section % self.section_count]

    def major_chord_name_for_section(self, section: int) -> str:
        return f"{self.label_for_section(section)}maj"

    def frequencies_for_section(self, section: int, play_chord: bool) -> List[float]:
        note_name = self.label_for_section(section)
        root_pitch = self.pitch_classes[note_name]
        root_midi = 60 + root_pitch

        if play_chord:
            midi_notes = [root_midi, root_midi + 4, root_midi + 7]
        else:
            midi_notes = [root_midi]

        return [self._midi_to_frequency(midi_note) for midi_note in midi_notes]

    def draw(
        self,
        parent_tag: str,
        active_section: int | None,
        scale: float = 1.0,
        offset: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        center = self._transform_point(self.center, scale, offset)
        outer_radius = max(1.0, float(self.outer_radius) * scale)
        inner_radius = max(1.0, float(self.inner_radius) * scale)

        dpg.draw_circle(center=center, radius=outer_radius, color=(86, 47, 0, 220), thickness=2.0, parent=parent_tag)
        dpg.draw_circle(center=center, radius=inner_radius, color=(86, 47, 0, 200), thickness=2.0, parent=parent_tag)

        for section in range(self.section_count):
            boundary_angle = section * self.section_angle
            x_inner, y_inner = self._point_on_ring(boundary_angle, self.inner_radius, scale, offset)
            x_outer, y_outer = self._point_on_ring(boundary_angle, self.outer_radius, scale, offset)

            dpg.draw_line(
                p1=(int(x_inner), int(y_inner)),
                p2=(int(x_outer), int(y_outer)),
                color=(86, 47, 0, 170),
                thickness=max(1.0, 1.4 * scale),
                parent=parent_tag,
            )

            mid_angle = (section + 0.5) * self.section_angle
            label_x, label_y = self._point_on_ring(mid_angle, self.outer_radius + 16.0, scale, offset)
            label_color = (86, 47, 0, 255)

            if active_section == section:
                marker_x, marker_y = self._point_on_ring(mid_angle, (self.inner_radius + self.outer_radius) * 0.5, scale, offset)
                dpg.draw_circle(
                    center=(int(marker_x), int(marker_y)),
                    radius=max(1.0, 11.0 * scale),
                    color=(255, 150, 68, 250),
                    fill=(255, 206, 153, 220),
                    thickness=max(1.0, 2.0 * scale),
                    parent=parent_tag,
                )
                label_color = (255, 150, 68, 255)

            dpg.draw_text(
                pos=(int(label_x) - int(round(10.0 * scale)), int(label_y) - int(round(8.0 * scale))),
                text=self.labels[section],
                color=label_color,
                size=max(12, int(round(16.0 * scale))),
                parent=parent_tag,
            )

    def _point_on_ring(
        self,
        angle: float,
        radius: float,
        scale: float = 1.0,
        offset: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[float, float]:
        cx, cy = self.center
        offset_x, offset_y = offset
        x = offset_x + ((cx + (radius * math.sin(angle))) * scale)
        y = offset_y + ((cy - (radius * math.cos(angle))) * scale)
        return x, y

    @staticmethod
    def _transform_point(point: tuple[float, float], scale: float, offset: tuple[float, float]) -> tuple[int, int]:
        offset_x, offset_y = offset
        return (
            int(round(offset_x + (point[0] * scale))),
            int(round(offset_y + (point[1] * scale))),
        )

    @staticmethod
    def _midi_to_frequency(midi_note: int) -> float:
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
