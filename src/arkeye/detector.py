"""Foreground candidates and masked battle-template classification."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

import cv2
import numpy as np
from numpy.typing import NDArray

from arkeye.assets import TemplateLibrary
from arkeye.health_bars import HealthBarDetector
from arkeye.ui_panels import UIPanelDetector

BoundingBox = tuple[int, int, int, int]


class DetectionError(ValueError):
    """Raised when detector inputs have inconsistent dimensions."""


@dataclass(frozen=True, slots=True)
class Candidate:
    bbox: BoundingBox
    area: int
    mask: NDArray[np.uint8]

    @property
    def foot_point(self) -> tuple[float, float]:
        x, y, width, height = self.bbox
        return x + width / 2.0, y + height - 1.0


@dataclass(frozen=True, slots=True)
class TemplateMatch:
    enemy_id: str
    template_name: str
    confidence: float
    color_score: float
    silhouette_score: float
    aspect_score: float


@dataclass(frozen=True, slots=True)
class LocalizedMatch:
    bbox: BoundingBox
    enemy_id: str
    template_name: str
    confidence: float
    alpha: NDArray[np.uint8] | None = None


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: BoundingBox
    foot_point: tuple[float, float]
    enemy_id: str
    confidence: float
    template_name: str | None
    appearance: NDArray[np.float32] | None = None
    possible_friendly: bool = False
    candidate_enemy_id: str | None = None
    faction_evidence: str | None = None
    health_bar_bbox: BoundingBox | None = None
    skill_icon_bbox: BoundingBox | None = None
    template_matches: dict[str, TemplateMatch] | None = None


def appearance_signature(image: NDArray[np.uint8], mask: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Build a masked HSV histogram for track association and reidentification."""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], mask, [12, 8], [0, 180, 0, 256]).astype(np.float32)
    total = float(histogram.sum())
    if total > 0:
        histogram /= total
    return histogram.reshape(-1, 1)


class ForegroundDetector:
    def __init__(
        self,
        background: NDArray[np.uint8],
        *,
        roi_mask: NDArray[np.uint8] | None = None,
        difference_threshold: int = 35,
        min_area: int = 40,
        blur_size: int = 1,
        suppress_red_guides: bool = False,
        ui_mask: NDArray[np.uint8] | None = None,
    ) -> None:
        if background.ndim != 3 or background.shape[2] != 3 or background.dtype != np.uint8:
            raise DetectionError("Background must be a uint8 BGR image")
        if blur_size < 1 or blur_size % 2 == 0:
            raise DetectionError("blur_size must be a positive odd number")
        self.background = background
        self.blur_size = blur_size
        self._background = self._blur(background)
        self.last_mask: NDArray[np.uint8] | None = None
        self.last_guide_mask: NDArray[np.uint8] = np.zeros(background.shape[:2], dtype=np.uint8)
        height, width = background.shape[:2]
        if roi_mask is not None and roi_mask.shape != (height, width):
            raise DetectionError("ROI mask dimensions must match the background")
        self.roi_mask = roi_mask
        self.ui_mask = ui_mask
        self.last_artifact_mask = np.zeros_like(self.last_guide_mask)
        self.difference_threshold = difference_threshold
        self.min_area = min_area
        self.suppress_red_guides = suppress_red_guides
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def _blur(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return image if self.blur_size == 1 else cv2.GaussianBlur(image, (self.blur_size, self.blur_size), 0)

    def _red_guide_mask(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Remove thin red overlays by geometry, not by stage-specific coordinates."""

        blue, green, red = cv2.split(frame)
        red_dominance = red.astype(np.int16) - np.maximum(blue, green).astype(np.int16)
        red_mask = ((red >= 110) & (red_dominance >= 45)).astype(np.uint8) * 255
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, self._open_kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
        guide_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for label in range(1, count):
            _, _, width, height, area = (int(value) for value in stats[label])
            longest = max(width, height)
            if longest < 24:
                continue
            thin = min(width, height) <= 8
            sparse = area / (width * height) <= 0.35 and area <= longest * 12
            if thin or sparse:
                guide_mask[labels == label] = 255
        if np.any(guide_mask):
            guide_mask = cv2.dilate(guide_mask, self._open_kernel)
        return guide_mask

    def _artifact_mask(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Mask small yellow interaction icons without erasing nearby units."""

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, (15, 120, 140), (45, 255, 255))
        contours, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        artifact = np.zeros(frame.shape[:2], dtype=np.uint8)
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, width, height = cv2.boundingRect(contour)
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, perimeter * 0.08, True)
            if area < 20 or area > 2000 or not 0.65 <= width / height <= 1.55 or len(polygon) != 4:
                continue
            if not cv2.isContourConvex(polygon) or area / (width * height) < 0.35:
                continue
            artifact[y : y + height, x : x + width] = 255
        if np.any(artifact):
            artifact = cv2.dilate(artifact, self._open_kernel)
        return artifact

    def foreground_mask(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if frame.shape != self.background.shape or frame.dtype != np.uint8:
            raise DetectionError("Frame dimensions and type must match the background")
        difference = cv2.absdiff(self._blur(frame), self._background)
        channel_difference = np.max(difference, axis=2)
        mask = (channel_difference >= self.difference_threshold).astype(np.uint8) * 255
        if self.roi_mask is not None:
            mask = cv2.bitwise_and(mask, self.roi_mask)
        if self.ui_mask is not None:
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(self.ui_mask))
        self.last_guide_mask = self._red_guide_mask(frame) if self.suppress_red_guides else np.zeros_like(mask)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(self.last_guide_mask))
        self.last_artifact_mask = self._artifact_mask(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel)
        self.last_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)
        return self.last_mask

    def detect(self, frame: NDArray[np.uint8]) -> tuple[Candidate, ...]:
        mask = self.foreground_mask(frame)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates: list[Candidate] = []
        for label in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[label])
            if area < self.min_area:
                continue
            local_mask = np.where(labels[y : y + height, x : x + width] == label, 255, 0).astype(np.uint8)
            candidates.append(Candidate(bbox=(x, y, width, height), area=area, mask=local_mask))
        return tuple(candidates)


class TemplateClassifier:
    def __init__(self, library: TemplateLibrary) -> None:
        self.library = library

    def score_by_enemy(
        self,
        image: NDArray[np.uint8],
        silhouette: NDArray[np.uint8],
        *,
        allowed_enemy_ids: Iterable[str] | None = None,
    ) -> dict[str, TemplateMatch]:
        """Return the best unmodified template score for every candidate class."""

        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise DetectionError("Candidate image must be a uint8 BGR image")
        height, width = image.shape[:2]
        if silhouette.shape != (height, width):
            raise DetectionError("Candidate silhouette must match its image")

        best_by_enemy: dict[str, TemplateMatch] = {}
        observed_mask = silhouette > 0
        for original in self.library.for_enemies(allowed_enemy_ids):
            scale = min(width / original.image.shape[1], height / original.image.shape[0])
            template_width = max(1, round(original.image.shape[1] * scale))
            template_height = max(1, round(original.image.shape[0] * scale))
            template = original.resized(template_width, template_height)
            template_image = np.zeros_like(image)
            template_mask = np.zeros((height, width), dtype=bool)
            offset_x = (width - template_width) // 2
            offset_y = height - template_height
            template_image[offset_y : offset_y + template_height, offset_x : offset_x + template_width] = template.image
            template_mask[offset_y : offset_y + template_height, offset_x : offset_x + template_width] = template.alpha > 127
            if not np.any(template_mask):
                continue
            difference = cv2.absdiff(image, template_image)
            color_error = float(np.mean(difference[template_mask])) / 255.0
            color_score = max(0.0, 1.0 - color_error)
            union = np.logical_or(observed_mask, template_mask)
            silhouette_score = (
                float(np.logical_and(observed_mask, template_mask).sum()) / float(union.sum()) if union.any() else 0.0
            )
            candidate_aspect = width / height
            template_aspect = original.image.shape[1] / original.image.shape[0]
            aspect_score = min(candidate_aspect, template_aspect) / max(candidate_aspect, template_aspect)
            confidence = 0.8 * color_score + 0.1 * silhouette_score + 0.1 * aspect_score
            match = TemplateMatch(
                enemy_id=original.enemy_id,
                template_name=original.name,
                confidence=confidence,
                color_score=color_score,
                silhouette_score=silhouette_score,
                aspect_score=aspect_score,
            )
            previous = best_by_enemy.get(original.enemy_id)
            if previous is None or match.confidence > previous.confidence:
                best_by_enemy[original.enemy_id] = match
        return best_by_enemy

    @staticmethod
    def _best_with_ambiguity(best_by_enemy: dict[str, TemplateMatch]) -> TemplateMatch | None:
        if not best_by_enemy:
            return None
        best = max(best_by_enemy.values(), key=lambda match: match.confidence)
        rivals = [match for enemy_id, match in best_by_enemy.items() if enemy_id != best.enemy_id]
        if rivals:
            rival = max(rivals, key=lambda match: match.confidence)
            if best.confidence - rival.confidence < 0.035:
                best = replace(best, confidence=min(best.confidence, 0.59))
        return best

    def classify_with_scores(
        self,
        image: NDArray[np.uint8],
        silhouette: NDArray[np.uint8],
        *,
        allowed_enemy_ids: Iterable[str] | None = None,
    ) -> tuple[TemplateMatch | None, dict[str, TemplateMatch]]:
        best_by_enemy = self.score_by_enemy(image, silhouette, allowed_enemy_ids=allowed_enemy_ids)
        return self._best_with_ambiguity(best_by_enemy), best_by_enemy

    def classify(
        self,
        image: NDArray[np.uint8],
        silhouette: NDArray[np.uint8],
        *,
        allowed_enemy_ids: Iterable[str] | None = None,
    ) -> TemplateMatch | None:
        best, _ = self.classify_with_scores(image, silhouette, allowed_enemy_ids=allowed_enemy_ids)
        return best

    def locate(
        self,
        image: NDArray[np.uint8],
        silhouette: NDArray[np.uint8],
        *,
        allowed_enemy_ids: Iterable[str] | None = None,
        analysis_scale: float = 1.0,
        confidence_threshold: float = 0.60,
    ) -> tuple[LocalizedMatch, ...]:
        """Locate several templates independently inside one foreground blob."""

        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise DetectionError("Candidate image must be a uint8 BGR image")
        height, width = image.shape[:2]
        if silhouette.shape != (height, width):
            raise DetectionError("Candidate silhouette must match its image")
        candidates: list[LocalizedMatch] = []
        for original in self.library.for_enemies(allowed_enemy_ids):
            for factor in (0.85, 1.0, 1.15):
                scale = analysis_scale * factor
                template_width = max(4, round(original.image.shape[1] * scale))
                template_height = max(4, round(original.image.shape[0] * scale))
                if template_width > width or template_height > height:
                    continue
                template = original.resized(template_width, template_height)
                result = cv2.matchTemplate(image, template.image, cv2.TM_SQDIFF_NORMED, mask=template.alpha)
                finite_result = np.where(np.isfinite(result), result, np.inf)
                locations = []
                for _ in range(4):
                    minimum, _, location, _ = cv2.minMaxLoc(finite_result)
                    if not np.isfinite(minimum) or minimum > 0.55:
                        break
                    locations.append((location[1], location[0]))
                    radius = max(2, round(min(template_width, template_height) * 0.5))
                    cv2.circle(finite_result, location, radius, np.inf, -1)
                template_mask = template.alpha > 127
                for y, x in locations:
                    color_score = max(0.0, 1.0 - float(result[y, x]))
                    observed_mask = silhouette[y : y + template_height, x : x + template_width] > 0
                    union = np.logical_or(observed_mask, template_mask)
                    silhouette_score = (
                        float(np.logical_and(observed_mask, template_mask).sum()) / float(union.sum())
                        if union.any()
                        else 0.0
                    )
                    confidence = 0.7 * color_score + 0.3 * silhouette_score
                    if confidence >= confidence_threshold:
                        candidates.append(
                            LocalizedMatch(
                                bbox=(int(x), int(y), template_width, template_height),
                                enemy_id=original.enemy_id,
                                template_name=original.name,
                                confidence=confidence,
                                alpha=template.alpha.copy(),
                            )
                        )

        filtered: list[LocalizedMatch] = []
        for candidate in sorted(candidates, key=lambda item: (-round(item.confidence, 2), item.bbox[0], item.bbox[1])):
            center_x = candidate.bbox[0] + candidate.bbox[2] / 2
            center_y = candidate.bbox[1] + candidate.bbox[3] / 2
            duplicate = False
            for existing in filtered:
                same_class = existing.enemy_id == candidate.enemy_id
                existing_x = existing.bbox[0] + existing.bbox[2] / 2
                existing_y = existing.bbox[1] + existing.bbox[3] / 2
                distance = ((center_x - existing_x) ** 2 + (center_y - existing_y) ** 2) ** 0.5
                tolerance = 0.60 if same_class else 0.20
                if distance < max(candidate.bbox[2], candidate.bbox[3]) * tolerance:
                    duplicate = True
                    break
            if not duplicate:
                filtered.append(candidate)
        return tuple(sorted(filtered[:8], key=lambda item: item.bbox[0]))


class BattlefieldDetector:
    def __init__(
        self,
        background: NDArray[np.uint8],
        classifier: TemplateClassifier,
        *,
        roi_mask: NDArray[np.uint8] | None = None,
        deployable_mask: NDArray[np.uint8] | None = None,
        ui_mask: NDArray[np.uint8] | None = None,
        allowed_enemy_ids: Iterable[str] | None = None,
        confidence_threshold: float = 0.70,
        class_confidence_thresholds: dict[str, float] | None = None,
        difference_threshold: int = 35,
        min_area: int = 40,
        blur_size: int = 1,
        analysis_scale: float = 1.0,
        health_bars: HealthBarDetector | None = None,
        ui_panels: UIPanelDetector | None = None,
    ) -> None:
        if not 0 < analysis_scale <= 1:
            raise DetectionError("analysis_scale must be in (0, 1]")
        self.ui_panels = ui_panels
        self.last_panel_rejections: list[dict[str, object]] = []
        self._frame_counter = 0
        self.analysis_scale = analysis_scale
        if analysis_scale == 1:
            scaled_background = background
            scaled_roi = roi_mask
            scaled_deployable = deployable_mask
            scaled_ui = ui_mask
        else:
            height, width = background.shape[:2]
            scaled_size = (round(width * analysis_scale), round(height * analysis_scale))
            scaled_background = cv2.resize(background, scaled_size, interpolation=cv2.INTER_AREA)
            scaled_roi = cv2.resize(roi_mask, scaled_size, interpolation=cv2.INTER_NEAREST) if roi_mask is not None else None
            scaled_deployable = cv2.resize(deployable_mask, scaled_size, interpolation=cv2.INTER_NEAREST) if deployable_mask is not None else None
            scaled_ui = cv2.resize(ui_mask, scaled_size, interpolation=cv2.INTER_NEAREST) if ui_mask is not None else None
        self.deployable_mask = scaled_deployable
        self.foreground = ForegroundDetector(
            scaled_background,
            roi_mask=scaled_roi,
            difference_threshold=difference_threshold,
            min_area=max(4, round(min_area * analysis_scale**2)),
            blur_size=blur_size,
            ui_mask=scaled_ui,
        )
        self.classifier = classifier
        self.health_bars = health_bars
        self.allowed_enemy_ids = tuple(allowed_enemy_ids) if allowed_enemy_ids is not None else None
        self.confidence_threshold = confidence_threshold
        self.class_confidence_thresholds = class_confidence_thresholds or {}
        self.debug_images: list[tuple[str, NDArray[np.uint8]]] = []

    @staticmethod
    def _has_overlap(bbox: BoundingBox, tracks) -> bool:
        if not tracks:
            return False
        x, y, width, height = bbox
        overlap_count = 0
        for track in tracks:
            if track.status not in {"confirmed", "occluded"}:
                continue
            track_x, track_y, track_width, track_height = track.bbox
            if x < track_x + track_width and track_x < x + width and y < track_y + track_height and track_y < y + height:
                overlap_count += 1
        return overlap_count >= 2

    def _is_deployable(self, foot_point: tuple[float, float]) -> bool:
        if self.deployable_mask is None:
            return True
        height, width = self.deployable_mask.shape
        x = min(width - 1, max(0, round(foot_point[0] * self.analysis_scale)))
        y = min(height - 1, max(0, round(foot_point[1] * self.analysis_scale)))
        return self.deployable_mask[y, x] > 0

    def _needs_split(self, candidate: Candidate, match: TemplateMatch | None, recognized: bool, tracks) -> bool:
        uncertain = not recognized or (match is not None and match.confidence < 0.85)
        if self._has_overlap(candidate.bbox, tracks) and uncertain:
            return True
        if match is None:
            return False
        for template in self.classifier.library.templates:
            if template.name == match.template_name:
                expected_area = float(np.count_nonzero(template.alpha > 127)) * self.analysis_scale**2
                return candidate.area > max(expected_area * 1.8, expected_area + self.foreground.min_area * 2)
        return False

    @staticmethod
    def _is_prompt_panel(candidate: Candidate, image: NDArray[np.uint8]) -> bool:
        """Reject large, dark, solid tutorial cards without stage-specific coordinates."""

        x, y, width, height = candidate.bbox
        del x, y
        if width < 180 or height < 80 or width * height < 30_000 or not 1.2 <= width / height <= 4.5:
            return False
        if candidate.area / (width * height) < 0.45:
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)) < 135.0

    def _skill_icons(self) -> list[BoundingBox]:
        count, _, stats, _ = cv2.connectedComponentsWithStats(self.foreground.last_artifact_mask, connectivity=8)
        icons = []
        for label in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[label])
            if area >= 4:
                icons.append(tuple(round(value / self.analysis_scale) for value in (x, y, width, height)))
        return icons

    @staticmethod
    def _bind_skill_icon(bbox: BoundingBox, icons: list[BoundingBox]) -> BoundingBox | None:
        x, y, width, _height = bbox
        center_x = x + width / 2
        candidates = []
        for icon in icons:
            ix, iy, iw, ih = icon
            icon_x, icon_y = ix + iw / 2, iy + ih / 2
            if abs(icon_x - center_x) <= max(40.0, width * 0.75) and -20 <= y - icon_y <= 100:
                candidates.append((abs(icon_x - center_x) + abs(y - icon_y) * 0.5, icon))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _detection_from_friendly_bar(self, evidence, frame: NDArray[np.uint8], icons: list[BoundingBox]) -> Detection | None:
        bar_x, bar_y, bar_width, bar_height = evidence.bbox
        del bar_height
        center_x = bar_x + bar_width / 2
        body_width = max(60, round(bar_width * 0.85))
        body_height = max(100, round(bar_width * 1.25))
        foot = (center_x, float(bar_y - 10))
        x, y = round(center_x - body_width / 2), round(foot[1] - body_height)
        bbox = (x, y, body_width, body_height)
        if x < 0 or y < 0 or x + body_width > frame.shape[1] or y + body_height > frame.shape[0]:
            return None
        if not self._is_deployable(foot):
            return None
        mask = self.foreground.last_mask
        if mask is None:
            return None
        left, top = round(x * self.analysis_scale), round(y * self.analysis_scale)
        right, bottom = round((x + body_width) * self.analysis_scale), round((y + body_height) * self.analysis_scale)
        local_mask = mask[top:bottom, left:right]
        if local_mask.size == 0 or cv2.countNonZero(local_mask) < max(20, local_mask.size * 0.02):
            return None
        body_image = frame[y : y + body_height, x : x + body_width]
        body_mask = cv2.resize(local_mask, (body_width, body_height), interpolation=cv2.INTER_NEAREST)
        match = self.classifier.classify(body_image, body_mask, allowed_enemy_ids=("friendly",))
        threshold = self.class_confidence_thresholds.get("friendly", self.confidence_threshold)
        recognized = match is not None and match.confidence >= threshold
        return Detection(
            bbox=bbox,
            foot_point=foot,
            enemy_id="friendly" if recognized and match else "unknown",
            confidence=match.confidence if recognized and match else 0.70,
            template_name=match.template_name if recognized and match else None,
            appearance=appearance_signature(body_image, body_mask),
            possible_friendly=True,
            faction_evidence="friendly",
            health_bar_bbox=evidence.bbox,
            skill_icon_bbox=self._bind_skill_icon(bbox, icons),
        )

    def _apply_health_bars(self, detections: list[Detection], frame: NDArray[np.uint8], evidence=None) -> tuple[Detection, ...]:
        if self.health_bars is None:
            return tuple(detections)
        evidence = self.health_bars.detect(frame) if evidence is None else evidence
        icons = self._skill_icons()
        anchored = []
        for detection in detections:
            matched = self.health_bars.match(detection.foot_point, detection.bbox, evidence)
            if matched is None:
                anchored.append(detection)
                continue
            if matched.kind == "device":
                continue
            enemy_id = detection.enemy_id
            candidate_id = detection.candidate_enemy_id
            possible_friendly = detection.possible_friendly
            template_name = detection.template_name
            confidence = detection.confidence
            if matched.kind == "enemy" and enemy_id == "friendly":
                enemy_id, candidate_id, possible_friendly, template_name = "unknown", None, False, None
            if matched.kind == "friendly":
                enemy_id, candidate_id, possible_friendly = "friendly", "friendly", True
                confidence = max(confidence, matched.confidence)
                if detection.candidate_enemy_id != "friendly":
                    template_name = None
            anchored.append(
                replace(
                    detection,
                    enemy_id=enemy_id,
                    confidence=confidence,
                    template_name=template_name,
                    candidate_enemy_id=candidate_id,
                    possible_friendly=possible_friendly,
                    faction_evidence=matched.kind,
                    health_bar_bbox=matched.bbox,
                    skill_icon_bbox=self._bind_skill_icon(detection.bbox, icons) if matched.kind == "friendly" else None,
                )
            )
        matched_bars = {detection.health_bar_bbox for detection in anchored if detection.health_bar_bbox is not None}
        for item in evidence:
            if item.kind != "friendly" or item.bbox in matched_bars:
                continue
            center_x, center_y = item.center
            if any(np.linalg.norm(np.asarray(detection.foot_point) - np.asarray((center_x, center_y - 10))) < 60 for detection in anchored):
                continue
            created = self._detection_from_friendly_bar(item, frame, icons)
            if created is not None:
                anchored.append(created)
        return tuple(anchored)

    def detect(self, frame: NDArray[np.uint8], *, overlapping_tracks=None) -> tuple[Detection, ...]:
        if self.analysis_scale != 1:
            height, width = frame.shape[:2]
            analysis_frame = cv2.resize(frame, (round(width * self.analysis_scale), round(height * self.analysis_scale)))
        else:
            analysis_frame = frame
        self.debug_images = [("foreground", self.foreground.last_mask)]
        self._frame_counter += 1
        self.last_panel_rejections = []
        detections: list[Detection] = []
        health_evidence = self.health_bars.detect(frame) if self.health_bars is not None else ()
        for candidate in self.foreground.detect(analysis_frame):
            x, y, width, height = candidate.bbox
            if width * height > analysis_frame.shape[0] * analysis_frame.shape[1] * 0.50:
                continue
            # 血条、技能条这类细横条不是作战单位。它们会被当成候选送上分类器，
            # 偶尔匹配到 friendly 模板（分数不高）就带上「可能我方」，随后被融合层
            # 整条剔除，看起来就像敌方的血条把它自己判成了我方。
            if height < 12 or (height < 20 and width / max(height, 1) > 4.0):
                continue
            image = analysis_frame[y : y + height, x : x + width]
            if self._is_prompt_panel(candidate, image):
                continue
            if self.ui_panels is not None:
                original_bbox = tuple(round(value / self.analysis_scale) for value in candidate.bbox)
                # 面板判定（宽度阈值、OCR）都按原分辨率设计，这里把半分辨率裁剪图还原
                panel_crop = image
                if (original_bbox[2], original_bbox[3]) != (width, height):
                    panel_crop = cv2.resize(image, (original_bbox[2], original_bbox[3]), interpolation=cv2.INTER_CUBIC)
                verdict = self.ui_panels.classify(
                    panel_crop,
                    original_bbox,
                    round(analysis_frame.shape[0] / self.analysis_scale),
                    frame_index=self._frame_counter,
                )
                if verdict.is_panel:
                    self.last_panel_rejections.append(
                        {
                            "bbox": list(original_bbox),
                            "reason": f"判定为游戏内 UI 面板：{verdict.reason}",
                        }
                    )
                    continue
            candidate_allowed_ids = self.allowed_enemy_ids
            if self.health_bars is not None:
                original_bbox = tuple(round(value / self.analysis_scale) for value in candidate.bbox)
                original_foot = tuple(value / self.analysis_scale for value in candidate.foot_point)
                matched = self.health_bars.match(original_foot, original_bbox, health_evidence)
                if matched is not None and matched.kind == "friendly":
                    candidate_allowed_ids = ("friendly",)
                elif matched is not None and matched.kind == "enemy":
                    available = self.allowed_enemy_ids or tuple(template.enemy_id for template in self.classifier.library.templates)
                    candidate_allowed_ids = tuple(enemy_id for enemy_id in available if enemy_id != "friendly")
            match, template_matches = self.classifier.classify_with_scores(
                image,
                candidate.mask,
                allowed_enemy_ids=candidate_allowed_ids,
            )
            threshold = self.class_confidence_thresholds.get(match.enemy_id, self.confidence_threshold) if match else self.confidence_threshold
            recognized = match is not None and match.confidence >= threshold
            artifact_pixels = cv2.countNonZero(cv2.bitwise_and(candidate.mask, self.foreground.last_artifact_mask[y : y + height, x : x + width]))
            if not recognized and artifact_pixels >= candidate.area * 0.5:
                continue
            if self._needs_split(candidate, match, recognized, overlapping_tracks):
                localized = self.classifier.locate(
                    image,
                    candidate.mask,
                    allowed_enemy_ids=candidate_allowed_ids,
                    analysis_scale=self.analysis_scale,
                    confidence_threshold=self.confidence_threshold,
                )
                detection_start = len(detections)
                covered = np.zeros_like(candidate.mask)
                localized_count = 0
                for local in localized:
                    if local.alpha is None:
                        continue
                    local_x, local_y, local_width, local_height = local.bbox
                    local_mask = np.zeros_like(candidate.mask)
                    local_mask[local_y : local_y + local_height, local_x : local_x + local_width] = local.alpha
                    local_mask = cv2.bitwise_and(local_mask, candidate.mask)
                    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(local_mask, connectivity=8)
                    if component_count <= 1:
                        continue
                    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    component_x, component_y, component_width, component_height, area = (
                        int(value) for value in stats[component]
                    )
                    if area < max(4, round(40 * self.analysis_scale**2)):
                        continue
                    component_mask = (labels == component).astype(np.uint8) * 255
                    component_mask = component_mask[
                        component_y : component_y + component_height,
                        component_x : component_x + component_width,
                    ]
                    covered[component_y : component_y + component_height, component_x : component_x + component_width] |= component_mask
                    self.debug_images.append((local.enemy_id, component_mask))
                    component_image = image[
                        component_y : component_y + component_height,
                        component_x : component_x + component_width,
                    ]
                    original_bbox = (
                        round((x + component_x) / self.analysis_scale),
                        round((y + component_y) / self.analysis_scale),
                        round(component_width / self.analysis_scale),
                        round(component_height / self.analysis_scale),
                    )
                    original_foot = (
                        original_bbox[0] + original_bbox[2] / 2,
                        original_bbox[1] + original_bbox[3] - 1,
                    )
                    possible_friendly = local.enemy_id == "friendly" and self._is_deployable(original_foot)
                    friendly_threshold = self.class_confidence_thresholds.get("friendly", self.confidence_threshold)
                    enemy_id = local.enemy_id
                    if local.enemy_id == "friendly" and (not possible_friendly or local.confidence < friendly_threshold):
                        enemy_id = "unknown"
                    detections.append(
                        Detection(
                            bbox=original_bbox,
                            foot_point=original_foot,
                            enemy_id=enemy_id,
                            confidence=local.confidence,
                            template_name=local.template_name,
                            appearance=appearance_signature(component_image, component_mask),
                            possible_friendly=possible_friendly,
                            candidate_enemy_id=local.enemy_id,
                        )
                    )
                    localized_count += 1
                if localized_count and np.count_nonzero(covered) < candidate.area * 0.35:
                    del detections[detection_start:]
                    localized_count = 0
                if localized_count:
                    residual = cv2.bitwise_and(candidate.mask, cv2.bitwise_not(covered))
                    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(residual, connectivity=8)
                    for component in range(1, component_count):
                        component_x, component_y, component_width, component_height, area = (
                            int(value) for value in stats[component]
                        )
                        if area < max(40, round(candidate.area * 0.35)):
                            continue
                        component_mask = (labels == component).astype(np.uint8) * 255
                        component_mask = component_mask[
                            component_y : component_y + component_height,
                            component_x : component_x + component_width,
                        ]
                        component_image = image[
                            component_y : component_y + component_height,
                            component_x : component_x + component_width,
                        ]
                        original_bbox = (
                            round((x + component_x) / self.analysis_scale),
                            round((y + component_y) / self.analysis_scale),
                            round(component_width / self.analysis_scale),
                            round(component_height / self.analysis_scale),
                        )
                        detections.append(
                            Detection(
                                bbox=original_bbox,
                                foot_point=(original_bbox[0] + original_bbox[2] / 2, original_bbox[1] + original_bbox[3] - 1),
                                enemy_id="unknown",
                                confidence=0.0,
                                template_name=None,
                                appearance=appearance_signature(component_image, component_mask),
                            )
                        )
                    continue
            original_bbox = tuple(round(value / self.analysis_scale) for value in candidate.bbox)
            original_foot = tuple(value / self.analysis_scale for value in candidate.foot_point)
            possible_friendly = bool(
                match and match.enemy_id == "friendly" and self._is_deployable(original_foot)
            )
            if recognized and match and match.enemy_id == "friendly" and not possible_friendly:
                recognized = False
            detections.append(
                Detection(
                    bbox=original_bbox,
                    foot_point=original_foot,
                    enemy_id=match.enemy_id if recognized and match else "unknown",
                    confidence=match.confidence if match else 0.0,
                    template_name=match.template_name if recognized and match else None,
                    appearance=appearance_signature(image, candidate.mask),
                    possible_friendly=possible_friendly,
                    candidate_enemy_id=match.enemy_id if match else None,
                    template_matches=template_matches,
                )
            )
        return self._apply_health_bars(detections, frame, health_evidence)
