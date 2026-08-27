import ctypes
import sys


def enable_dpi_awareness():
    """Синхронизирует координаты mss и pyautogui при масштабировании Windows."""

    if sys.platform != "win32":
        return

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


# DPI-режим должен быть задан до импорта tkinter и pyautogui.
enable_dpi_awareness()

import queue
import threading
import time
from collections import deque
from pathlib import Path
import tkinter as tk

import cv2
import mss
import numpy as np
import pyautogui
from pynput import keyboard


# ============================================================
# CONFIG
# ============================================================

# Клавиша, на которую в WoW назначена способность «Рыбалка».
CAST_KEY = "1"

# В обычном интерфейсе WoW это чаще right. Оставлено значение исходного скрипта.
CLICK_BUTTON = "left"

BITE_CLICK_DELAY = 1.0
AFTER_LOOT_DELAY = 2.0
BOBBER_FIND_TIMEOUT = 8.0
BITE_TIMEOUT = 23.0

# Итоговая уверенность ансамбля признаков поплавка.
MATCH_THRESHOLD = 0.56
TRACK_MATCH_THRESHOLD = 0.50

# Красное перо — самый устойчивый якорь на всех десяти эталонах.
RED_TEMPLATE_THRESHOLD = 0.61
RED_SHAPE_MAX_DISTANCE = 0.22
RED_ASPECT_MIN = 2.20
RED_ASPECT_MAX = 4.60
RED_FILL_MIN = 0.28
RED_FILL_MAX = 0.78
MIN_RED_AREA = 18
MAX_CANDIDATES = 60

BOBBER_CONFIRM_FRAMES = 2
BITE_MOVE_THRESHOLD = 7
MOTION_THRESHOLD = 10.0
BITE_LOST_FRAMES = 3
TRACK_INTERVAL = 0.05
TRACK_RADIUS = 120

# Номер физического монитора в mss: 1 — первый монитор.
MONITOR_INDEX = 1

# Нижняя часть экрана обычно занята панелями WoW, поэтому она исключена из
# первичного поиска. Локальный трекинг после обнаружения этим не ограничен.
SEARCH_LEFT = 0.01
SEARCH_TOP = 0.01
SEARCH_RIGHT = 0.99
SEARCH_BOTTOM = 0.93

BASE_DIR = Path(__file__).resolve().parent
REFERENCES_DIR = BASE_DIR / "bobbers"

RED_MODEL_SIZE = (192, 64)
BODY_MODEL_SIZE = (96, 128)
RING_MODEL_SIZE = (72, 96)
BLUE_MODEL_SIZE = (112, 112)


# ============================================================
# GLOBAL STATE
# ============================================================

running = False
bot_thread = None
status_updates = queue.SimpleQueue()
pyautogui.FAILSAFE = True


# ============================================================
# BOBBER VISION
# ============================================================


class BobberDetector:
    """
    Ищет конкретный классический поплавок с эталонных PNG.

    Полные картинки не используются как обычные шаблоны: на них слишком много
    разной воды. Сначала находится форма красного пера, затем кандидат
    подтверждается по пробке, металлическому кольцу и синему перу в ожидаемых
    местах. Формы сравниваются со всеми PNG из папки bobbers.
    """

    def __init__(self, references_dir):
        self.references_dir = Path(references_dir)
        self.references = self._load_references()

    @staticmethod
    def _red_mask(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Насыщенное ядро не даёт принять коричневую воду за красное перо.
        seed_low = cv2.inRange(
            hsv,
            np.array((0, 160, 30), dtype=np.uint8),
            np.array((11, 255, 255), dtype=np.uint8),
        )
        seed_wrap = cv2.inRange(
            hsv,
            np.array((170, 160, 30), dtype=np.uint8),
            np.array((179, 255, 255), dtype=np.uint8),
        )
        seed = cv2.bitwise_or(seed_low, seed_wrap)

        relaxed_low = cv2.inRange(
            hsv,
            np.array((0, 90, 25), dtype=np.uint8),
            np.array((14, 255, 255), dtype=np.uint8),
        )
        relaxed_wrap = cv2.inRange(
            hsv,
            np.array((170, 90, 25), dtype=np.uint8),
            np.array((179, 255, 255), dtype=np.uint8),
        )
        relaxed = cv2.bitwise_or(relaxed_low, relaxed_wrap)

        # Сначала соединяем близкие части пера: на маленьком объекте блик или
        # тень могут разорвать красную область на несколько компонентов.
        kernel = np.ones((3, 3), dtype=np.uint8)
        relaxed = cv2.morphologyEx(relaxed, cv2.MORPH_CLOSE, kernel)

        count, labels = cv2.connectedComponents(relaxed)
        if count <= 1:
            return np.zeros(relaxed.shape, dtype=np.uint8)

        seeded_labels = np.unique(labels[seed > 0])
        seeded_labels = seeded_labels[seeded_labels != 0]
        if seeded_labels.size == 0:
            return np.zeros(relaxed.shape, dtype=np.uint8)

        grown = np.isin(labels, seeded_labels).astype(np.uint8) * 255
        return cv2.morphologyEx(grown, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def _contours(mask):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        return contours

    @staticmethod
    def _candidate_from_contour(contour, offset_x=0, offset_y=0):
        if offset_x or offset_y:
            contour = contour.copy()
            contour[:, 0, 0] += offset_x
            contour[:, 0, 1] += offset_y

        x, y, width, height = cv2.boundingRect(contour)
        if width <= 0 or height <= 0:
            return None

        area = float(cv2.contourArea(contour))
        aspect = width / float(height)
        fill = area / float(width * height)

        return {
            "contour": contour,
            "x0": x,
            "y0": y,
            "red_width": width,
            "red_height": height,
            "red_area": area,
            "red_aspect": aspect,
            "red_fill": fill,
        }

    @staticmethod
    def _valid_red_geometry(candidate, min_area):
        if candidate is None or candidate["red_area"] < min_area:
            return False

        width = candidate["red_width"]
        height = candidate["red_height"]
        if width < 6 or height < 3:
            return False

        return (
            RED_ASPECT_MIN <= candidate["red_aspect"] <= RED_ASPECT_MAX
            and RED_FILL_MIN <= candidate["red_fill"] <= RED_FILL_MAX
        )

    @staticmethod
    def _crop_with_padding(image, left, top, right, bottom):
        left = int(round(left))
        top = int(round(top))
        right = int(round(right))
        bottom = int(round(bottom))

        width = max(1, right - left)
        height = max(1, bottom - top)
        patch = np.zeros((height, width, 3), dtype=image.dtype)

        image_h, image_w = image.shape[:2]
        source_left = max(0, left)
        source_top = max(0, top)
        source_right = min(image_w, right)
        source_bottom = min(image_h, bottom)

        if source_right <= source_left or source_bottom <= source_top:
            return patch

        target_left = source_left - left
        target_top = source_top - top
        target_right = target_left + source_right - source_left
        target_bottom = target_top + source_bottom - source_top

        patch[target_top:target_bottom, target_left:target_right] = image[
            source_top:source_bottom,
            source_left:source_right,
        ]
        return patch

    @staticmethod
    def _largest_near_component(
        mask,
        expected_center,
        max_fraction=0.65,
        min_fraction=0.0,
        max_distance=0.40,
    ):
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
        if count <= 1:
            return np.zeros(mask.shape, dtype=np.uint8)

        image_h, image_w = mask.shape
        image_area = float(image_h * image_w)
        expected_x = expected_center[0] * image_w
        expected_y = expected_center[1] * image_h

        best_label = None
        best_score = -1.0

        for label in range(1, count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area < max(4.0, image_area * min_fraction):
                continue
            if area > image_area * max_fraction:
                continue

            center_x, center_y = centers[label]
            distance = np.hypot(
                (center_x - expected_x) / image_w,
                (center_y - expected_y) / image_h,
            )
            if distance > max_distance:
                continue

            proximity = 1.0 - distance / max_distance
            score = area * proximity * proximity

            if score > best_score:
                best_score = score
                best_label = label

        if best_label is None:
            return np.zeros(mask.shape, dtype=np.uint8)

        return (labels == best_label).astype(np.uint8) * 255

    @staticmethod
    def _canonical_red_mask(candidate):
        x = candidate["x0"]
        y = candidate["y0"]
        width = candidate["red_width"]
        height = candidate["red_height"]

        local_contour = candidate["contour"].copy()
        local_contour[:, 0, 0] -= x
        local_contour[:, 0, 1] -= y

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask, [local_contour], -1, 255, thickness=cv2.FILLED)
        return cv2.resize(mask, RED_MODEL_SIZE, interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _mask_contour(mask):
        contours = BobberDetector._contours(mask)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _body_mask(self, image, candidate):
        x = candidate["x0"]
        y = candidate["y0"]
        width = candidate["red_width"]
        height = candidate["red_height"]

        patch = self._crop_with_padding(
            image,
            x + 0.72 * width,
            y + 0.30 * height,
            x + 1.48 * width,
            y + 3.00 * height,
        )
        patch = cv2.resize(patch, BODY_MODEL_SIZE, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array((10, 35, 40), dtype=np.uint8),
            np.array((30, 220, 255), dtype=np.uint8),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8),
        )
        return self._largest_near_component(
            mask,
            (0.48, 0.46),
            max_fraction=0.60,
            min_fraction=0.005,
            max_distance=0.34,
        )

    def _ring_features(self, image, candidate):
        x = candidate["x0"]
        y = candidate["y0"]
        width = candidate["red_width"]
        height = candidate["red_height"]

        patch = self._crop_with_padding(
            image,
            x + 0.86 * width,
            y - 0.70 * height,
            x + 1.25 * width,
            y + 0.85 * height,
        )
        patch = cv2.resize(patch, RING_MODEL_SIZE, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        silver = cv2.inRange(
            hsv,
            np.array((0, 0, 85), dtype=np.uint8),
            np.array((179, 105, 255), dtype=np.uint8),
        )
        silver = cv2.morphologyEx(
            silver,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
        silver = self._largest_near_component(
            silver,
            (0.52, 0.50),
            max_fraction=0.55,
            min_fraction=0.002,
            max_distance=0.32,
        )

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 40, 120)
        support = cv2.dilate(silver, np.ones((7, 7), dtype=np.uint8))
        edges = cv2.bitwise_and(edges, support)
        return silver, edges

    def _blue_mask(self, image, candidate):
        x = candidate["x0"]
        y = candidate["y0"]
        width = candidate["red_width"]
        height = candidate["red_height"]

        patch = self._crop_with_padding(
            image,
            x + 0.05 * width,
            y - 1.70 * height,
            x + 1.10 * width,
            y + 0.80 * height,
        )
        patch = cv2.resize(patch, BLUE_MODEL_SIZE, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(
            hsv,
            np.array((105, 55, 35), dtype=np.uint8),
            np.array((160, 255, 255), dtype=np.uint8),
        )
        blue = cv2.morphologyEx(
            blue,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
        return self._largest_near_component(
            blue,
            (0.43, 0.53),
            max_fraction=0.48,
            min_fraction=0.004,
            max_distance=0.40,
        )

    def _features(self, image, candidate):
        red = self._canonical_red_mask(candidate)
        red_contour = self._mask_contour(red)
        body = self._body_mask(image, candidate)
        ring, ring_edges = self._ring_features(image, candidate)
        blue = self._blue_mask(image, candidate)

        return {
            "red": red,
            "red_contour": red_contour,
            "body": body,
            "ring": ring,
            "ring_edges": ring_edges,
            "blue": blue,
            "aspect": candidate["red_aspect"],
            "fill": candidate["red_fill"],
        }

    def _load_references(self):
        paths = sorted(self.references_dir.glob("*.png"))
        if not paths:
            raise RuntimeError(
                f"В папке {self.references_dir} не найдено ни одного PNG с поплавком."
            )

        references = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"[VISION] Пропущен повреждённый эталон: {path.name}")
                continue

            mask = self._red_mask(image)
            candidates = []
            reference_min_area = max(100.0, image.shape[0] * image.shape[1] * 0.0005)

            for contour in self._contours(mask):
                candidate = self._candidate_from_contour(contour)
                if self._valid_red_geometry(candidate, reference_min_area):
                    candidates.append(candidate)

            if not candidates:
                print(
                    f"[VISION] Пропущен {path.name}: "
                    "не найдено характерное красное перо"
                )
                continue

            candidate = max(candidates, key=lambda item: item["red_area"])
            features = self._features(image, candidate)
            body_fraction = np.count_nonzero(features["body"]) / features["body"].size
            ring_fraction = np.count_nonzero(features["ring"]) / features["ring"].size
            if body_fraction < 0.012 or ring_fraction < 0.004:
                print(
                    f"[VISION] Пропущен {path.name}: "
                    "не выделены корпус или металлическое кольцо"
                )
                continue

            features["name"] = path.name
            references.append(features)

        if len(references) < 2:
            raise RuntimeError(
                "Для надёжного поиска нужны хотя бы два валидных PNG "
                f"в папке {self.references_dir}."
            )

        print(f"[VISION] Загружено эталонов поплавка: {len(references)}")
        return references

    @staticmethod
    def _mask_correlation(first, second):
        value = cv2.matchTemplate(first, second, cv2.TM_CCORR_NORMED)[0, 0]
        if not np.isfinite(value):
            return 0.0
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _tolerant_dice(first, second, tolerance=3):
        first = first > 0
        second = second > 0
        first_count = int(np.count_nonzero(first))
        second_count = int(np.count_nonzero(second))

        if first_count == 0 or second_count == 0:
            return 0.0

        size = tolerance * 2 + 1
        kernel = np.ones((size, size), dtype=np.uint8)
        first_dilated = cv2.dilate(first.astype(np.uint8), kernel) > 0
        second_dilated = cv2.dilate(second.astype(np.uint8), kernel) > 0

        first_matches = np.count_nonzero(first & second_dilated)
        second_matches = np.count_nonzero(second & first_dilated)
        return float((first_matches + second_matches) / (first_count + second_count))

    @staticmethod
    def _ratio_similarity(first, second):
        if first <= 0 or second <= 0:
            return 0.0
        return float(min(first, second) / max(first, second))

    def _compare(self, features, reference):
        red_score = self._mask_correlation(features["red"], reference["red"])
        if red_score < RED_TEMPLATE_THRESHOLD:
            return None

        if features["red_contour"] is None or reference["red_contour"] is None:
            return None

        shape_distance = cv2.matchShapes(
            features["red_contour"],
            reference["red_contour"],
            cv2.CONTOURS_MATCH_I3,
            0.0,
        )
        if shape_distance > RED_SHAPE_MAX_DISTANCE:
            return None

        body_fraction = np.count_nonzero(features["body"]) / features["body"].size
        ring_fraction = np.count_nonzero(features["ring"]) / features["ring"].size
        if body_fraction < 0.012 or ring_fraction < 0.004:
            return None

        shape_score = max(0.0, 1.0 - shape_distance / RED_SHAPE_MAX_DISTANCE)
        body_score = self._tolerant_dice(features["body"], reference["body"], 4)
        ring_mask_score = self._tolerant_dice(features["ring"], reference["ring"], 3)
        ring_edge_score = self._tolerant_dice(
            features["ring_edges"], reference["ring_edges"], 2
        )
        blue_score = self._tolerant_dice(features["blue"], reference["blue"], 4)
        aspect_score = self._ratio_similarity(features["aspect"], reference["aspect"])
        fill_score = max(0.0, 1.0 - abs(features["fill"] - reference["fill"]) / 0.35)
        geometry_score = 0.65 * aspect_score + 0.35 * fill_score
        ring_score = 0.60 * ring_mask_score + 0.40 * ring_edge_score

        # Одной похожей красной полосы недостаточно: рядом обязательно должны
        # находиться похожие пробковый корпус и металлическое кольцо.
        if body_score < 0.14 or ring_score < 0.07:
            return None

        confidence = (
            0.35 * red_score
            + 0.15 * shape_score
            + 0.10 * geometry_score
            + 0.20 * body_score
            + 0.15 * ring_score
            + 0.05 * blue_score
        )
        return float(confidence)

    def find(self, frame, region=None, threshold=MATCH_THRESHOLD):
        frame_h, frame_w = frame.shape[:2]

        if region is None:
            left = int(frame_w * SEARCH_LEFT)
            top = int(frame_h * SEARCH_TOP)
            right = int(frame_w * SEARCH_RIGHT)
            bottom = int(frame_h * SEARCH_BOTTOM)
        else:
            left, top, right, bottom = region
            left = max(0, int(left))
            top = max(0, int(top))
            right = min(frame_w, int(right))
            bottom = min(frame_h, int(bottom))

        if right <= left or bottom <= top:
            return None

        search_image = frame[top:bottom, left:right]
        red_mask = self._red_mask(search_image)
        candidates = []

        for contour in self._contours(red_mask):
            candidate = self._candidate_from_contour(contour, left, top)
            if not self._valid_red_geometry(candidate, MIN_RED_AREA):
                continue

            if candidate["red_area"] > frame_h * frame_w * 0.08:
                continue

            candidates.append(candidate)

        candidates.sort(key=lambda item: item["red_area"], reverse=True)
        candidates = candidates[:MAX_CANDIDATES]

        best = None
        for candidate in candidates:
            features = self._features(frame, candidate)
            matches = []

            for reference in self.references:
                confidence = self._compare(features, reference)
                if confidence is None:
                    continue
                matches.append((confidence, reference["name"]))

            # Случайное совпадение с одним PNG недостаточно. Поплавок должны
            # независимо поддержать хотя бы два ракурса из банка эталонов.
            if len(matches) < 2:
                continue

            matches.sort(reverse=True)
            strongest = matches[: min(3, len(matches))]
            confidence = float(np.mean([item[0] for item in strongest]))

            if best is None or confidence > best["confidence"]:
                red_width = candidate["red_width"]
                red_height = candidate["red_height"]
                click_x = candidate["x0"] + 1.085 * red_width
                click_y = candidate["y0"] + red_height + 0.20 * red_width

                best = {
                    "x": int(np.clip(round(click_x), 0, frame_w - 1)),
                    "y": int(np.clip(round(click_y), 0, frame_h - 1)),
                    "confidence": confidence,
                    "reference": strongest[0][1],
                    "support": len(matches),
                    "red_width": red_width,
                    "red_height": red_height,
                }

        if best is None or best["confidence"] < threshold:
            return None
        return best


detector = BobberDetector(REFERENCES_DIR)


# ============================================================
# SCREEN / TRACKING
# ============================================================


def selected_monitor(capturer):
    if MONITOR_INDEX < 1 or MONITOR_INDEX >= len(capturer.monitors):
        available = len(capturer.monitors) - 1
        raise RuntimeError(
            f"MONITOR_INDEX={MONITOR_INDEX}, но доступно физических мониторов: {available}."
        )
    return capturer.monitors[MONITOR_INDEX]


def screenshot(capturer, monitor):
    frame = np.array(capturer.grab(monitor))
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def track_bobber(frame, previous_position):
    radius = max(TRACK_RADIUS, int(previous_position["red_width"] * 2.2))
    region = (
        previous_position["x"] - radius,
        previous_position["y"] - radius,
        previous_position["x"] + radius,
        previous_position["y"] + radius,
    )
    return detector.find(frame, region=region, threshold=TRACK_MATCH_THRESHOLD)


def get_motion_roi(frame, position, size):
    x = position["x"]
    y = position["y"]
    roi = BobberDetector._crop_with_padding(
        frame,
        x - size,
        y - size,
        x + size,
        y + size,
    )

    if roi.size == 0:
        return None

    roi = cv2.resize(roi, (96, 96), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (7, 7), 0)


def motion_score(previous, current):
    if previous is None or current is None or previous.shape != current.shape:
        return 0.0
    return float(np.mean(cv2.absdiff(previous, current)))


def screen_position(position, monitor):
    return (
        int(monitor["left"] + position["x"]),
        int(monitor["top"] + position["y"]),
    )


# ============================================================
# FISHING
# ============================================================


def cast():
    print("[CAST] Забрасываем удочку")
    pyautogui.press(CAST_KEY)


def wait_for_bobber(capturer, monitor):
    print("[VISION] Ищу поплавок по 10 эталонам...")
    started = time.time()
    previous_candidate = None
    confirmations = 0

    while running:
        if time.time() - started > BOBBER_FIND_TIMEOUT:
            print("[VISION] Поплавок не найден")
            return None

        frame = screenshot(capturer, monitor)
        candidate = detector.find(frame)

        if candidate is None:
            previous_candidate = None
            confirmations = 0
            time.sleep(0.1)
            continue

        if previous_candidate is not None:
            distance = np.hypot(
                candidate["x"] - previous_candidate["x"],
                candidate["y"] - previous_candidate["y"],
            )
            allowed_distance = max(20, candidate["red_width"] * 0.70)
            width_ratio = min(
                candidate["red_width"], previous_candidate["red_width"]
            ) / max(candidate["red_width"], previous_candidate["red_width"])
            stable_candidate = distance <= allowed_distance and width_ratio >= 0.75
            confirmations = confirmations + 1 if stable_candidate else 1
        else:
            confirmations = 1

        previous_candidate = candidate

        if confirmations >= BOBBER_CONFIRM_FRAMES:
            print(
                "[VISION] Поплавок найден:",
                candidate["x"],
                candidate["y"],
                "confidence:",
                round(candidate["confidence"], 2),
                "reference:",
                candidate["reference"],
                "support:",
                candidate["support"],
            )
            return candidate

        time.sleep(0.1)

    return None


def wait_for_bite(capturer, monitor, initial_position):
    print("[WAIT] Жду поклёвку...")
    position = initial_position
    baseline_positions = deque([initial_position["y"]], maxlen=12)
    recent_motion = deque(maxlen=6)
    motion_baseline = deque(maxlen=30)
    previous_roi = None
    lost_frames = 0
    stable_track_frames = 0
    motion_roi_size = max(30, int(initial_position["red_width"] * 0.60))
    started = time.time()

    # Вода и поплавок стабилизируются после заброса.
    time.sleep(1.5)

    while running:
        if time.time() - started > BITE_TIMEOUT:
            print("\n[WAIT] Таймаут поклёвки")
            return False

        frame = screenshot(capturer, monitor)
        new_position = track_bobber(frame, position)
        detected = new_position is not None

        if not detected:
            lost_frames += 1
        else:
            position = new_position
            lost_frames = 0
            stable_track_frames += 1

        current_roi = get_motion_roi(frame, position, motion_roi_size)
        motion = motion_score(previous_roi, current_roi)
        previous_roi = current_roi
        recent_motion.append(motion)

        movement_threshold = max(
            BITE_MOVE_THRESHOLD,
            min(14, int(position["red_width"] * 0.04)),
        )
        baseline_y = float(np.median(baseline_positions))
        downward_movement = max(0.0, position["y"] - baseline_y) if detected else 0.0

        if detected and downward_movement < movement_threshold * 0.50:
            baseline_positions.append(position["y"])

        if motion > 0 and detected and downward_movement < movement_threshold * 0.50:
            motion_baseline.append(motion)

        adaptive_motion_threshold = MOTION_THRESHOLD
        if len(motion_baseline) >= 6:
            motion_median = float(np.median(motion_baseline))
            motion_mad = float(
                np.median(np.abs(np.asarray(motion_baseline) - motion_median))
            )
            adaptive_motion_threshold = max(
                MOTION_THRESHOLD,
                motion_median + max(2.5, 4.0 * motion_mad),
            )

        print(
            "\r"
            f"down={downward_movement:04.1f}px "
            f"motion={motion:05.2f} "
            f"limit={adaptive_motion_threshold:05.2f} "
            f"conf={position['confidence']:.2f} "
            f"lost={lost_frames}",
            end="",
        )

        strong_position_change = downward_movement >= movement_threshold
        strong_visual_change = motion >= adaptive_motion_threshold
        disappeared_with_motion = (
            stable_track_frames >= 6
            and lost_frames >= BITE_LOST_FRAMES
            and recent_motion
            and max(recent_motion) >= adaptive_motion_threshold
        )

        if (strong_position_change and strong_visual_change) or disappeared_with_motion:
            print("\n[BITE] ПОКЛЁВКА!")
            return position

        time.sleep(TRACK_INTERVAL)

    return False


def follow_bobber_during_delay(capturer, monitor, initial_position, delay):
    """Обновляет точку клика, пока выдерживается настроенная задержка."""

    position = initial_position
    deadline = time.time() + delay

    while running and time.time() < deadline:
        frame = screenshot(capturer, monitor)
        updated = track_bobber(frame, position)
        if updated is not None:
            position = updated

        remaining = deadline - time.time()
        if remaining > 0:
            time.sleep(min(TRACK_INTERVAL, remaining))

    return position


def set_status(text):
    status_updates.put(text)


def fishing_loop():
    global running

    print("\n=========================")
    print(" WoW Fishing started")
    print("=========================")
    print("Переключись в WoW. Запуск через 3 секунды...")

    try:
        with mss.mss() as capturer:
            monitor = selected_monitor(capturer)
            time.sleep(3)

            while running:
                cast()
                time.sleep(1)

                if not running:
                    break

                bobber = wait_for_bobber(capturer, monitor)
                if not bobber:
                    print("[RETRY] Новый заброс")
                    time.sleep(1)
                    continue

                pyautogui.moveTo(*screen_position(bobber, monitor), duration=0.2)
                bite_position = wait_for_bite(capturer, monitor, bobber)

                if not bite_position:
                    print("[RETRY] Поклёвка не обнаружена")
                    continue

                print(f"[BITE] Жду {BITE_CLICK_DELAY} сек...")
                bite_position = follow_bobber_during_delay(
                    capturer,
                    monitor,
                    bite_position,
                    BITE_CLICK_DELAY,
                )

                if not running:
                    break

                print("[CLICK] Кликаю поплавок")
                click_x, click_y = screen_position(bite_position, monitor)
                pyautogui.moveTo(click_x, click_y, duration=0.1)
                pyautogui.click(x=click_x, y=click_y, button=CLICK_BUTTON)
                time.sleep(AFTER_LOOT_DELAY)

    except Exception as error:
        print(f"\n[ERROR] {error}")
    finally:
        running = False
        set_status("Остановлен")
        print("[STOP] Рыбалка остановлена")


# ============================================================
# GUI
# ============================================================


def start_bot():
    global running, bot_thread

    if running or (bot_thread is not None and bot_thread.is_alive()):
        return

    while True:
        try:
            status_updates.get_nowait()
        except queue.Empty:
            break

    running = True
    status_label.config(text="Работает")
    bot_thread = threading.Thread(target=fishing_loop, daemon=True)
    bot_thread.start()


def stop_bot():
    global running
    running = False
    status_label.config(text="Останавливается...")


root = tk.Tk()
root.title("WoW Fishing")
root.geometry("300x180")
root.resizable(False, False)

title = tk.Label(root, text="WoW Fishing", font=("Arial", 18, "bold"))
title.pack(pady=(20, 10))

start_button = tk.Button(
    root,
    text="НАЧАТЬ РЫБАЛКУ",
    command=start_bot,
    width=25,
    height=2,
)
start_button.pack()

stop_button = tk.Button(
    root,
    text="ОСТАНОВИТЬ",
    command=stop_bot,
    width=25,
)
stop_button.pack(pady=10)

status_label = tk.Label(root, text="Остановлен")
status_label.pack()


def process_status_updates():
    while True:
        try:
            text = status_updates.get_nowait()
        except queue.Empty:
            break
        status_label.config(text=text)

    root.after(100, process_status_updates)


root.after(100, process_status_updates)


# ============================================================
# F8 = emergency stop
# ============================================================


def on_key_press(key):
    global running

    if key == keyboard.Key.f8:
        running = False
        set_status("Остановлен")
        print("\n[F8] STOP")


listener = keyboard.Listener(on_press=on_key_press)
listener.daemon = True
listener.start()

root.mainloop()
running = False
