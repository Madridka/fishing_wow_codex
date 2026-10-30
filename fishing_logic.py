"""Настройки, анализ движения и управление рыбалкой без зависимости от GUI."""

from collections import deque
from dataclasses import dataclass, replace
import logging
import time

import cv2
import numpy as np

from bobber_vision import BobberTracker, SplashDetector


@dataclass(frozen=True)
class BotConfig:
    cast_key: str = "1"
    click_button: str = "right"
    click_delay: float = 0.75
    after_loot_delay: float = 3.0
    find_timeout: float = 8.0
    bite_timeout: float = 23.0
    monitor_index: int = 1
    track_interval: float = 0.05
    match_threshold: float = 0.62
    motion_threshold: float = 6.0
    cursor_deadband: float = 5.0
    track_lost_timeout: float = 2.0
    old_bobber_timeout: float = 6.0
    bite_mode: str = "motion"
    search_bounds: tuple = (0.01, 0.01, 0.99, 0.93)

    def __post_init__(self):
        if self.bite_mode not in {"motion", "splash", "both"}:
            raise ValueError("Неизвестный режим поклёвки")
        if not 0.5 <= self.click_delay <= 1.0:
            raise ValueError("Задержка клика должна быть от 0,5 до 1 секунды")
        if not 4 <= self.motion_threshold <= 50:
            raise ValueError("Порог движения должен быть от 4 до 50 пикселей")
        if not 1 <= self.after_loot_delay <= 10:
            raise ValueError("Пауза до нового заброса должна быть от 1 до 10 секунд")
        if self.track_lost_timeout <= 0 or self.old_bobber_timeout <= 0:
            raise ValueError("Таймауты восстановления должны быть положительными")


@dataclass
class SessionStats:
    casts: int = 0
    hooks: int = 0
    confirmed_hooks: int = 0
    lost_tracks: int = 0
    unconfirmed_hooks: int = 0


class MotionBiteDetector:
    """Резкое отклонение центра от предсказанного спокойного движения.

    Порог учитывает шум и размер поплавка. Единичный выброс, потеря трека,
    равномерный дрейф и движение мыши сами по себе не считаются поклёвкой.
    """

    def __init__(self, min_displacement=6.0):
        self.minimum = float(min_displacement)
        self.baseline = deque(maxlen=32)
        self.history = deque(maxlen=30)
        self.reset()

    def reset(self):
        self.baseline.clear()
        self.history.clear()
        self.started = None
        self.last_seen = None
        self.pending_since = None
        self.confirmations = 0
        self.armed = False
        self.state = "warmup"
        self.displacement = 0.0
        self.noise = 0.0
        self.threshold = self.minimum
        self.spread = 0.0

    def _cancel(self):
        self.confirmations = 0
        self.pending_since = None

    def update(self, position, now=None):
        now = time.monotonic() if now is None else now
        if position is None:
            self._cancel()
            self.state = "lost"
            return False
        if self.last_seen is not None and now - self.last_seen > 0.35:
            self.reset()
        self.last_seen = now
        point = np.array((position["x"], position["y"]), dtype=float)
        scale = position.get("bobber_height", 32)
        self.history.append((now, point))
        if self.started is None:
            self.started = now
        if not self.armed:
            self.baseline.append((now, point))
            self.armed = len(self.baseline) >= 12 and now - self.started >= 0.8
            self.state = "warmup"
            return False

        times = np.array([item[0] - now for item in self.baseline])
        points = np.array([item[1] for item in self.baseline])
        # Линейная модель отличает спокойный дрейф от внезапной тряски.
        design = np.column_stack((times, np.ones(times.size)))
        fit, _, _, _ = np.linalg.lstsq(design, points, rcond=None)
        predicted = fit[1]
        residuals = np.linalg.norm(points - design @ fit, axis=1)
        self.noise = float(np.percentile(residuals, 90))
        self.threshold = max(self.minimum, scale * 0.14, self.noise * 2.5 + 1.0)
        self.displacement = float(np.linalg.norm(point - predicted))
        recent = [p for t, p in self.history if now - t <= 0.4]
        self.spread = float(np.linalg.norm(np.ptp(recent, axis=0))) if recent else 0.0

        if self.displacement > max(40, scale * 1.8):
            # Большой скачок к другому объекту требует нового спокойного фона.
            self.reset()
            self.state = "reacquire"
            return False
        if self.displacement >= self.threshold:
            if self.pending_since is None:
                self.pending_since = now
            self.confirmations += 1
            self.state = "confirm"
        else:
            self._cancel()
            self.state = "idle"
            self.baseline.append((now, point))
        if (self.confirmations >= 3 and now - self.pending_since >= 0.10):
            self.state = "bite"
            return True
        return False

    def debug_image(self, frame, position):
        radius = max(40, int(position.get("bobber_height", 32) * 2))
        x, y = position["x"], position["y"]
        from bobber_vision import BobberDetector
        patch = BobberDetector._crop_with_padding(frame, x-radius, y-radius, x+radius, y+radius)
        for _, point in self.history:
            px, py = np.rint(point - (x-radius, y-radius)).astype(int)
            cv2.circle(patch, (px, py), 1, (0, 255, 255), -1)
        cv2.drawMarker(patch, (radius, radius), (0, 255, 0), cv2.MARKER_CROSS, 12)
        patch = cv2.resize(patch, (384, 384), interpolation=cv2.INTER_NEAREST)
        text = f"move={self.displacement:.1f} limit={self.threshold:.1f} noise={self.noise:.1f}"
        cv2.putText(patch, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1)
        return patch


class CursorFollower:
    """Медиана нужна только мыши; детектор получает исходные координаты."""

    def __init__(self, position, deadband=5.0):
        self.points = deque(maxlen=3)
        self.last = (position["x"], position["y"])
        self.deadband = deadband

    def target(self, position):
        self.points.append((position["x"], position["y"]))
        target = tuple(np.rint(np.median(self.points, axis=0)).astype(int))
        if np.hypot(target[0] - self.last[0], target[1] - self.last[1]) <= self.deadband:
            return None
        self.last = target
        return target


class FishingBot:
    def __init__(self, detector, config, mouse, capturer_factory, stop_event,
                 output_dir, status_callback=lambda text: None, logger=None,
                 stats=None, stats_callback=lambda snapshot: None):
        self.detector = detector
        self.config = config
        self.mouse = mouse
        self.capturer_factory = capturer_factory
        self.stop = stop_event
        self.output_dir = output_dir
        self.status = status_callback
        self.log = logger or logging.getLogger("fishing")
        self.tracker = None
        self.last_position = None
        self.pending_hook = False
        self.stats = stats if stats is not None else SessionStats()
        self.stats_callback = stats_callback

    def _publish_stats(self):
        # GUI получает отдельный снимок, а не изменяемый объект рабочего потока.
        self.stats_callback(replace(self.stats))

    def _log_stats(self):
        self.log.info("СЧЁТЧИК: успешных подсечек %d | правых кликов %d | забросов %d | потерь центра %d | без подтверждения %d",
                      self.stats.confirmed_hooks, self.stats.hooks, self.stats.casts,
                      self.stats.lost_tracks, self.stats.unconfirmed_hooks)

    @staticmethod
    def screenshot(capturer, monitor):
        return cv2.cvtColor(np.asarray(capturer.grab(monitor)), cv2.COLOR_BGRA2BGR)

    @staticmethod
    def screen_position(position, monitor):
        return int(monitor["left"] + position["x"]), int(monitor["top"] + position["y"])

    def _move(self, position, monitor):
        if not self.stop.is_set():
            self.mouse.moveTo(*self.screen_position(position, monitor), duration=0)

    def wait_for_bobber(self, capturer, monitor):
        self.status("Поиск поплавка")
        self.log.info("Ищу поплавок")
        deadline = time.monotonic() + self.config.find_timeout
        previous = None
        confirmations = 0
        while not self.stop.is_set() and time.monotonic() < deadline:
            frame = self.screenshot(capturer, monitor)
            h, w = frame.shape[:2]
            a, b, c, d = self.config.search_bounds
            pos = self.detector.find(frame, (w*a, h*b, w*c, h*d), self.config.match_threshold)
            if pos is None:
                previous, confirmations = None, 0
            else:
                same = previous is not None and np.hypot(pos["x"]-previous["x"], pos["y"]-previous["y"]) < 15
                confirmations = confirmations + 1 if same else 1
                previous = pos
                if confirmations >= 2:
                    self.tracker = BobberTracker(self.detector)
                    self.tracker.start(frame, pos)
                    self.last_position = dict(pos)
                    self.log.info("Центр найден: x=%d y=%d, уверенность %.2f", pos["x"], pos["y"], pos["confidence"])
                    return pos
            self.stop.wait(0.1)
        self.log.info("Поиск завершён без поплавка")
        return None

    def wait_for_bite(self, capturer, monitor, initial_position):
        self.status("Ожидание поклёвки")
        self._move(initial_position, monitor)
        if self.stop.wait(1.0):
            return None
        movement = MotionBiteDetector(self.config.motion_threshold)
        splashes = SplashDetector() if self.config.bite_mode != "motion" else None
        follower = CursorFollower(initial_position, self.config.cursor_deadband)
        position = initial_position
        deadline = time.monotonic() + self.config.bite_timeout
        next_log = 0.0
        lost_since = None
        next_recovery = 0.0
        while not self.stop.is_set() and time.monotonic() < deadline:
            frame = self.screenshot(capturer, monitor)
            position = self.tracker.update(frame)
            now = time.monotonic()
            if position is None:
                if lost_since is None:
                    lost_since = now
                    next_recovery = now + 0.25
                    self.stats.lost_tracks += 1
                    self._publish_stats()
                    self.status("Центр потерян — восстанавливаю")
                    self.log.warning("Центр потерян. Восстанавливаю тот же поплавок до %.1f с", self.config.track_lost_timeout)
                if now >= next_recovery:
                    position = self.tracker.recover(frame)
                    next_recovery = now + 0.25
                if position is None and now - lost_since >= self.config.track_lost_timeout:
                    self.log.warning("Не удалось восстановить центр; подсечка отменена, завершаю старый заброс")
                    return None
            if position is not None:
                self.last_position = dict(position)
                if lost_since is not None:
                    self.log.info("Центр восстановлен через %.2f с: (%d, %d)", now-lost_since, position["x"], position["y"])
                    self.status("Ожидание поклёвки")
                    movement.reset()
                    if splashes is not None:
                        splashes = SplashDetector()
                    follower = CursorFollower(position, self.config.cursor_deadband)
                    self._move(position, monitor)
                    lost_since = None
            motion_bite = movement.update(position, now)
            splash_bite = splashes.update(frame, position, now) if splashes else False
            if position is not None:
                target = follower.target(position)
                if target is not None:
                    self._move({"x": target[0], "y": target[1]}, monitor)
            if now >= next_log:
                center = f"{position['x']}, {position['y']}" if position else "потерян"
                self.log.info("Центр %s | %s | отклонение %.1f / %.1f px | шум %.1f | разброс %.1f | подтверждение %d",
                              center, movement.state, movement.displacement, movement.threshold,
                              movement.noise, movement.spread, movement.confirmations)
                next_log = now + 1.0
            mode = self.config.bite_mode
            bitten = motion_bite if mode == "motion" else splash_bite if mode == "splash" else motion_bite and splash_bite
            if bitten and position is not None:
                self.log.info("ПОКЛЁВКА: режим %s, отклонение %.1f px, порог %.1f px", mode, movement.displacement, movement.threshold)
                debug = movement.debug_image(frame, position) if mode == "motion" else splashes.debug_image()
                if debug is not None:
                    try:
                        encoded, buffer = cv2.imencode(".png", debug)
                        if encoded:
                            buffer.tofile(self.output_dir / "last_bite_debug.png")
                        else:
                            self.log.warning("Не удалось сохранить кадр поклёвки")
                    except (OSError, cv2.error):
                        self.log.warning("Не удалось сохранить кадр поклёвки", exc_info=True)
                return position
            self.stop.wait(self.config.track_interval)
        self.log.info("Ожидание завершено без подтверждённой поклёвки")
        return None

    def land_fish(self, capturer, monitor, position):
        self.status("Подсечка")
        self.log.info("Правый клик через %.2f с", self.config.click_delay)
        deadline = time.monotonic() + self.config.click_delay
        last_seen = time.monotonic()
        while not self.stop.is_set() and time.monotonic() < deadline:
            updated = self.tracker.update(self.screenshot(capturer, monitor))
            if updated is not None:
                position = updated
                self.last_position = dict(position)
                last_seen = time.monotonic()
                self._move(position, monitor)
            self.stop.wait(min(self.config.track_interval, max(0, deadline-time.monotonic())))
        if self.stop.is_set():
            return False
        if time.monotonic() - last_seen > 0.35:
            self.log.warning("Клик отменён: центр потерян перед подсечкой")
            return False
        self._move(position, monitor)
        if self.stop.is_set():
            return False
        self.mouse.click(*self.screen_position(position, monitor), button=self.config.click_button)
        self.last_position = dict(position)
        self.stats.hooks += 1
        self.pending_hook = True
        self._publish_stats()
        self.log.info("Клик %s по центру (%d, %d)", self.config.click_button, position["x"], position["y"])
        return True

    def wait_for_old_bobber_clear(self, capturer, monitor):
        """Исчезновение подтверждается серией кадров, а не одним пропуском."""
        self.status("Ожидаю исчезновения старого поплавка")
        deadline = time.monotonic() + self.config.old_bobber_timeout
        absent_since = None
        absent_frames = 0
        while not self.stop.is_set() and time.monotonic() < deadline:
            frame = self.screenshot(capturer, monitor)
            tracked = self.tracker.update(frame) if self.tracker is not None else None
            anchor = self.last_position
            if anchor is not None:
                radius = max(48, int(anchor.get("bobber_height", 32)*2))
                region = (anchor["x"]-radius, anchor["y"]-radius,
                          anchor["x"]+radius, anchor["y"]+radius)
            else:
                region = None
            detected = self.detector.find(frame, region=region, threshold=self.config.match_threshold)
            present = tracked is not None or detected is not None
            now = time.monotonic()
            if present:
                absent_since, absent_frames = None, 0
            else:
                if absent_since is None:
                    absent_since = now
                absent_frames += 1
                if absent_frames >= 4 and now - absent_since >= 0.4:
                    self.log.info("Старый поплавок исчез; можно готовить следующий заброс")
                    return True
            self.stop.wait(0.1)
        return False

    def prepare_next_cast(self, capturer, monitor, after_click=False):
        """Не допускает нового заброса поверх старого и подтверждает подсечку."""
        if self.stop.is_set():
            return False
        cooldown_until = time.monotonic() + self.config.after_loot_delay
        if not after_click:
            self.log.info("Завершаю неудавшийся заброс клавишей Esc")
            self.mouse.press("esc")
        cleared = self.wait_for_old_bobber_clear(capturer, monitor)
        if self.stop.is_set():
            return False
        if after_click and self.pending_hook:
            if cleared:
                self.stats.confirmed_hooks += 1
                self.log.info("Успешная подсечка №%d: после правого клика поплавок исчез", self.stats.confirmed_hooks)
            else:
                self.stats.unconfirmed_hooks += 1
                self.log.warning("Подсечка без подтверждения: старый поплавок остаётся видимым")
            self._publish_stats()
            self._log_stats()
            self.pending_hook = False
        if not cleared and after_click:
            self.log.warning("Поплавок не исчез за %.1f с. Отменяю старый заброс клавишей Esc", self.config.old_bobber_timeout)
            self.mouse.press("esc")
            cleared = self.wait_for_old_bobber_clear(capturer, monitor)
        if not cleared:
            if not self.stop.is_set():
                self.log.error("Старый поплавок не исчез. Автоматическая остановка, чтобы не накладывать забросы")
                self.stop.set()
            return False
        self.tracker = None
        self.last_position = None
        remaining = max(0, cooldown_until - time.monotonic())
        if remaining:
            self.status("Пауза перед новым забросом")
            self.log.info("Пауза до следующего заброса: ещё %.1f с", remaining)
            if self.stop.wait(remaining):
                return False
        return not self.stop.is_set()

    def run(self):
        try:
            self._publish_stats()
            self._log_stats()
            self.log.info("Запуск через 3 секунды. Переключитесь в WoW. F8 — остановка")
            self.status("Запуск через 3 секунды")
            if self.stop.wait(3.0):
                return
            with self.capturer_factory() as capturer:
                index = self.config.monitor_index
                if not 1 <= index < len(capturer.monitors):
                    raise RuntimeError(f"Монитор {index} недоступен")
                monitor = capturer.monitors[index]
                # При повторном запуске в игре ещё может идти предыдущий заброс.
                frame = self.screenshot(capturer, monitor)
                existing = self.detector.find(frame, threshold=self.config.match_threshold)
                if existing is not None:
                    self.log.info("Перед запуском найден старый поплавок; сначала завершаю предыдущий заброс")
                    self.last_position = dict(existing)
                    self.tracker = BobberTracker(self.detector)
                    self.tracker.start(frame, existing)
                    if not self.prepare_next_cast(capturer, monitor):
                        return
                while not self.stop.is_set():
                    self.status("Заброс")
                    self.log.info("Заброс: клавиша %s", self.config.cast_key)
                    if self.stop.is_set():
                        break
                    self.mouse.press(self.config.cast_key)
                    self.stats.casts += 1
                    self._publish_stats()
                    if self.stop.wait(1.0):
                        break
                    bobber = self.wait_for_bobber(capturer, monitor)
                    if bobber is None:
                        if not self.prepare_next_cast(capturer, monitor):
                            break
                        continue
                    bite = self.wait_for_bite(capturer, monitor, bobber)
                    clicked = bite is not None and self.land_fish(capturer, monitor, bite)
                    if not self.prepare_next_cast(capturer, monitor, after_click=clicked):
                        break
        except Exception:
            self.log.exception("Ошибка рыбалки")
        finally:
            self.stop.set()
            self.status("Остановлен")
            self._log_stats()
            self.log.info("Рыбалка остановлена")
