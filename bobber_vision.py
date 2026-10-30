"""Поиск маленького тёмного поплавка и брызг, без доступа к экрану/мыши."""

from collections import deque
from pathlib import Path
import time

import cv2
import numpy as np


class BobberDetector:
    # Координаты центра пробки на сохранённых кадрах калибровки.
    REFERENCE_CENTERS = {
        "bobber_idle.png": (49, 55),
        "bobber_bite.png": (179, 101),
    }

    def __init__(self, references_dir):
        self.references = []
        for name, center in self.REFERENCE_CENTERS.items():
            path = Path(references_dir) / name
            try:
                # imdecode + fromfile поддерживает кириллицу в пути Windows.
                image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            except (OSError, cv2.error):
                image = None
            if image is None:
                raise RuntimeError(f"Не удалось прочитать эталон поплавка: {path}")
            candidates = self._candidates(image)
            if not candidates:
                raise RuntimeError(f"Не найдено синее перо на эталоне: {path}")
            candidate = max(candidates, key=lambda item: item[4])
            x, y, w, h, _ = candidate
            self.references.append({
                "name": name,
                "features": self._features(image, candidate),
                "aspect": w / h,
                "click": ((center[0] - x) / w, (center[1] - y) / h),
            })

    @staticmethod
    def _crop_with_padding(image, left, top, right, bottom):
        left, top, right, bottom = map(lambda value: int(round(value)),
                                       (left, top, right, bottom))
        patch = np.zeros((max(1, bottom - top), max(1, right - left), 3), image.dtype)
        height, width = image.shape[:2]
        sx, sy = max(0, left), max(0, top)
        ex, ey = min(width, right), min(height, bottom)
        if ex > sx and ey > sy:
            patch[sy - top:ey - top, sx - left:ex - left] = image[sy:ey, sx:ex]
        return patch

    @staticmethod
    def _blue_mask(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # На ночном кадре перо почти чёрное; высокий порог яркости его теряет.
        return cv2.inRange(hsv, (100, 45, 15), (140, 255, 255))

    def _candidates(self, image):
        mask = self._blue_mask(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if (w >= 6 and h >= 12 and 0.25 <= w / h <= 2.4
                    and area >= 35 and 0.18 <= area / (w * h) <= 0.85):
                candidates.append((x, y, w, h, area))
        return candidates

    def _features(self, image, candidate):
        x, y, w, h, _ = candidate
        features = {}
        regions = {
            "blue": (0, 0, 1, 1),
            "red": (-0.3, 0.35, 1.7, 1.35),
            "body": (-0.8, 0.85, 0.9, 1.9),
            "ring": (-0.6, 0.15, 0.45, 1.15),
        }
        for name, (left, top, right, bottom) in regions.items():
            patch = self._crop_with_padding(image, x + left*w, y + top*h,
                                             x + right*w, y + bottom*h)
            patch = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            if name == "blue":
                mask = self._blue_mask(patch)
            elif name == "red":
                mask = cv2.inRange(hsv, (0, 100, 20), (14, 255, 255))
                mask |= cv2.inRange(hsv, (170, 100, 20), (179, 255, 255))
            elif name == "body":
                mask = cv2.inRange(hsv, (0, 30, 45), (30, 190, 255))
            else:
                mask = cv2.inRange(hsv, (0, 0, 85), (179, 105, 255))
            features[name] = mask
        return features

    @staticmethod
    def _similarity(first, second):
        if not np.any(first) or not np.any(second):
            return 0.0
        # Допуск в несколько пикселей учитывает растрирование маленького пера.
        kernel = np.ones((5, 5), np.uint8)
        matches = (np.count_nonzero((first > 0) & (cv2.dilate(second, kernel) > 0))
                   + np.count_nonzero((second > 0) & (cv2.dilate(first, kernel) > 0)))
        return matches / (np.count_nonzero(first) + np.count_nonzero(second))

    def find(self, frame, region=None, threshold=0.62):
        height, width = frame.shape[:2]
        if region is None:
            region = (width*.01, height*.01, width*.99, height*.93)
        left, top, right, bottom = map(int, region)
        left, top = max(0, left), max(0, top)
        right, bottom = min(width, right), min(height, bottom)
        if right <= left or bottom <= top:
            return None
        best = None
        for candidate in self._candidates(frame[top:bottom, left:right]):
            x, y, w, h, area = candidate
            x, y = x + left, y + top
            features = self._features(frame, (x, y, w, h, area))
            for reference in self.references:
                aspect = min(w/h, reference["aspect"]) / max(w/h, reference["aspect"])
                if aspect < 0.50:
                    continue
                scores = {name: self._similarity(mask, reference["features"][name])
                          for name, mask in features.items()}
                # Перо само по себе может оказаться растением или элементом UI.
                if (scores["blue"] < 0.65 or scores["red"] < 0.28
                        or scores["body"] < 0.28 or scores["ring"] < 0.10):
                    continue
                confidence = (0.35*scores["blue"] + 0.25*scores["red"]
                              + 0.20*scores["body"] + 0.15*scores["ring"] + 0.05*aspect)
                if confidence < threshold or (best and confidence <= best["confidence"]):
                    continue
                cx, cy = reference["click"]
                best = {
                    "x": int(np.clip(round(x + cx*w), 0, width - 1)),
                    "y": int(np.clip(round(y + cy*h), 0, height - 1)),
                    "confidence": confidence, "reference": reference["name"],
                    "support": 1, "feather_width": w, "feather_height": h,
                    "bobber_height": h,
                }
        return best


class BobberTracker:
    """Сопровождает уже выбранную пробку по её локальному изображению.

    После захвата не пересчитывает центр из размеров пера на каждом кадре.
    Нормированная корреляция серого изображения и границ переносит небольшие
    изменения яркости. Цветовой поиск используется для восстановления трека.
    """

    def __init__(self, detector):
        self.detector = detector
        self.position = None
        self.template = None
        self.edges = None
        self.half_size = None
        self.lost_frames = 0

    @staticmethod
    def _gray(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _edges(gray):
        return cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0),
                             cv2.Sobel(gray, cv2.CV_32F, 0, 1))

    def start(self, frame, position):
        self.position = dict(position)
        self.half_size = max(8, int(position.get("bobber_height", 32) * 0.45))
        self._set_template(frame)
        self.lost_frames = 0

    def _set_template(self, frame):
        p, size = self.position, self.half_size
        patch = BobberDetector._crop_with_padding(frame, p["x"]-size, p["y"]-size,
                                                 p["x"]+size, p["y"]+size)
        self.template = self._gray(patch)
        self.edges = self._edges(self.template)

    def update(self, frame):
        if self.position is None:
            raise RuntimeError("Сначала нужно захватить поплавок через start()")
        p, size = self.position, self.half_size
        radius = max(20, min(64, int(p.get("bobber_height", 32) * 1.3)))
        left, top = p["x"]-radius-size, p["y"]-radius-size
        patch = BobberDetector._crop_with_padding(frame, left, top,
                                                 p["x"]+radius+size, p["y"]+radius+size)
        gray = self._gray(patch)
        result = None
        if float(np.std(self.template)) >= 3:
            gray_scores = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)
            edge_scores = cv2.matchTemplate(self._edges(gray), self.edges, cv2.TM_CCOEFF_NORMED)
            scores = 0.7*gray_scores + 0.3*edge_scores
            scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
            ys, xs = np.indices(scores.shape)
            # При похожих объектах предпочтение получает продолжение того же трека.
            ranked = scores - 0.06*np.hypot(xs-radius, ys-radius)/radius
            _, _, _, point = cv2.minMaxLoc(ranked)
            confidence = float(scores[point[1], point[0]])
            step = np.hypot(point[0]-radius, point[1]-radius)
            if confidence >= 0.63 and step <= max(12, p.get("bobber_height", 32)*0.65):
                result = dict(p, x=left+point[0]+size, y=top+point[1]+size,
                              confidence=confidence, source="tracking")
        if result is None:
            recovered = self.detector.find(frame, (p["x"]-radius, p["y"]-radius,
                                                   p["x"]+radius, p["y"]+radius), threshold=0.62)
            if recovered is not None:
                distance = np.hypot(recovered["x"]-p["x"], recovered["y"]-p["y"])
                if distance <= max(12, p.get("bobber_height", 32)*0.8):
                    result = dict(recovered, source="recovery")
        if result is None:
            self.lost_frames += 1
            return None
        h, w = frame.shape[:2]
        if not (size <= result["x"] < w-size and size <= result["y"] < h-size):
            self.lost_frames += 1
            return None
        self.position = result
        self.lost_frames = 0
        # При восстановлении запоминаем новый вид той же пробки.
        if result["source"] == "recovery":
            self._set_template(frame)
        return dict(result)


    def recover(self, frame):
        """Расширенный поиск рядом с последней целью после краткой потери."""
        if self.position is None:
            return None
        p = self.position
        radius = max(64, int(p.get("bobber_height", 32)*2.5))
        candidate = self.detector.find(frame, (p["x"]-radius, p["y"]-radius,
                                               p["x"]+radius, p["y"]+radius), threshold=0.65)
        if candidate is None:
            return None
        distance = np.hypot(candidate["x"]-p["x"], candidate["y"]-p["y"])
        scale_ratio = candidate.get("bobber_height", 32) / p.get("bobber_height", 32)
        if distance > max(32, p.get("bobber_height", 32)*1.5) or not 0.6 <= scale_ratio <= 1.6:
            return None
        self.start(frame, candidate)
        return dict(candidate, source="recovery")


def _splash_patch(frame, position):
    size = max(24, int(position.get("bobber_height", 32) * 1.25))
    x, y = position["x"], position["y"]
    patch = BobberDetector._crop_with_padding(frame, x-size, y-size, x+size, y+size)
    return cv2.resize(patch, (96, 96), interpolation=cv2.INTER_AREA)


def _splash_mask(patch):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    background = cv2.GaussianBlur(value, (25, 25), 0)
    bright = (value >= 145) & (hsv[:, :, 1] <= 115) & (value - background >= 18)
    # Пробка и кольцо могут блестеть сами по себе, без поклёвки.
    bright[36:62, 36:60] = False
    bright[:36, 38:58] = False
    return bright


def splash_score(frame, position):
    """Статичная оценка для диагностики; сама по себе не означает поклёвку."""
    return float(np.mean(_splash_mask(_splash_patch(frame, position))))


class SplashDetector:
    WARMUP_SECONDS = 0.6
    WARMUP_FRAMES = 10
    CONFIRM_SECONDS = 0.12
    CONFIRM_FRAMES = 3

    def __init__(self):
        self.patches = deque(maxlen=20)
        self.baseline = deque(maxlen=20)
        self.confirmations = 0
        self.score = 0.0
        self.new_score = 0.0
        self.threshold = 0.014
        self.particles = 0
        self.armed = False
        self.state = "warmup"
        self.started = None
        self.last_seen = None
        self.candidate_started = None
        self.anchor = None
        self.current_patch = None
        self.background_patch = None

    def _cancel_candidate(self):
        self.confirmations = 0
        self.candidate_started = None

    def update(self, frame, position, now=None):
        now = time.monotonic() if now is None else now
        if position is None:
            self._cancel_candidate()
            self.state = "lost"
            return False

        if self.last_seen is not None and now - self.last_seen > 0.4:
            # После долгой потери нельзя сравнивать с устаревшим фоном.
            self.__init__()
        self.last_seen = now
        if self.anchor is not None:
            distance = np.hypot(position["x"] - self.anchor["x"],
                                position["y"] - self.anchor["y"])
            old_height = self.anchor.get("bobber_height", 32)
            height_ratio = position.get("bobber_height", 32) / old_height
            if distance > max(24, old_height * 0.85) or not 0.5 <= height_ratio <= 2.0:
                # Смена объекта/ракурса не должна превращаться в всплеск.
                self.__init__()
        if self.anchor is None:
            self.anchor = dict(position)
        # Сравниваем одну область экрана: дрожание координат и наклон пера
        # не должны сдвигать/масштабировать неподвижные блики относительно фона.
        patch = _splash_patch(frame, self.anchor)
        self.current_patch = patch
        bright = _splash_mask(patch)
        self.score = float(np.mean(bright))
        self.new_score = 0.0
        self.particles = 0

        if not self.armed:
            if self.started is None:
                self.started = now
            self.patches.append(patch)
            self.baseline.append(self.score)
            if (len(self.patches) >= self.WARMUP_FRAMES
                    and now - self.started >= self.WARMUP_SECONDS):
                self.armed = True
            self.state = "warmup"
            return False

        self.background_patch = np.median(np.asarray(self.patches), axis=0).astype(np.uint8)
        # Запоминаем и редкие спокойные блики; небольшой сдвиг кольца не является
        # появлением новых частиц. Яркость сравниваем с медианой спокойных кадров.
        known = np.any([_splash_mask(item) for item in self.patches], axis=0).astype(np.uint8)
        known = cv2.dilate(known, np.ones((7, 7), np.uint8)) > 0
        value = np.max(patch, axis=2).astype(np.int16)
        background_value = np.max(self.background_patch, axis=2).astype(np.int16)
        new = bright & ~known & (value - background_value >= 22)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(new.astype(np.uint8))
        # Отдельные точки дождя и единичный блик не подтверждают поклёвку.
        kept = np.zeros(new.shape, dtype=bool)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= 3:
                kept |= labels == label
                self.particles += 1
        self.new_score = float(np.mean(kept))
        ys, xs = np.nonzero(kept)
        spread = bool(xs.size and np.ptp(xs) >= 12 and np.ptp(ys) >= 18)
        median = float(np.median(self.baseline))
        mad = float(np.median(np.abs(np.asarray(self.baseline) - median)))
        self.threshold = max(0.014, median + max(0.006, 4 * mad))
        candidate = (self.score >= self.threshold and self.new_score >= 0.008
                     and self.particles >= 3 and spread)
        if candidate:
            if self.candidate_started is None:
                self.candidate_started = now
            self.confirmations += 1
            self.state = "confirm"
        else:
            self._cancel_candidate()
            self.state = "idle"
            # Не включаем начало всплеска в фон до его подтверждения.
            if self.new_score < 0.004:
                self.patches.append(patch)
                self.baseline.append(self.score)
        if (self.confirmations >= self.CONFIRM_FRAMES
                and now - self.candidate_started >= self.CONFIRM_SECONDS):
            self.state = "bite"
            return True
        return False

    def debug_image(self):
        if self.current_patch is None or self.background_patch is None:
            return None
        image = np.hstack((self.background_patch, self.current_patch))
        image = cv2.resize(image, (576, 288), interpolation=cv2.INTER_NEAREST)
        cv2.putText(image, "IDLE", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(image, "BITE", (298, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        text = f"new={self.new_score:.3f} particles={self.particles} frames={self.confirmations}"
        cv2.putText(image, text, (10, 277), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return image
