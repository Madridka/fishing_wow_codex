"""Регрессии трекинга, движения и подсечки; настоящая мышь не используется."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from bobber_vision import BobberDetector, BobberTracker
from fishing_logic import BotConfig, CursorFollower, FishingBot, MotionBiteDetector, SessionStats


BASE = Path(__file__).resolve().parent


def position(x=300, y=250):
    return dict(x=x, y=y, bobber_height=32, feather_width=39, confidence=1.0)


class MotionTests(unittest.TestCase):
    def warm(self, dx=0, dy=0):
        detector = MotionBiteDetector()
        for i in range(22):
            self.assertFalse(detector.update(position(300+dx*i, 250+dy*i), now=i*0.05))
        self.assertTrue(detector.armed)
        return detector

    def test_small_jitter_never_clicks(self):
        detector = self.warm()
        for i in range(300):
            self.assertFalse(detector.update(position(300+(i%5)-2, 250+(i%3)-1), now=1.1+i*0.05))

    def test_three_axis_independent_shakes(self):
        for dx, dy in [(12, 0), (0, 12), (10, 8)]:
            with self.subTest(dx=dx, dy=dy):
                detector = self.warm()
                results = [detector.update(position(300+dx, 250+dy), now=1.1+i*0.05) for i in range(4)]
                self.assertFalse(results[0])
                self.assertTrue(results[-1])

    def test_oscillating_shake(self):
        detector = self.warm()
        results = [detector.update(position(300+(12 if i%2 else -12), 250), now=1.1+i*0.05) for i in range(4)]
        self.assertTrue(results[-1])

    def test_single_outlier_is_not_bite(self):
        detector = self.warm()
        self.assertFalse(detector.update(position(320, 260), now=1.1))
        for i in range(20):
            self.assertFalse(detector.update(position(), now=1.15+i*0.05))

    def test_smooth_drift_is_not_bite(self):
        detector = self.warm(dx=0.4, dy=0.2)
        for i in range(22, 150):
            self.assertFalse(detector.update(position(300+0.4*i, 250+0.2*i), now=i*0.05))

    def test_giant_jump_relearns(self):
        detector = self.warm()
        for i in range(40):
            self.assertFalse(detector.update(position(430, 350), now=1.1+i*0.05))

    def test_track_loss_cancels_confirmation(self):
        detector = self.warm()
        self.assertFalse(detector.update(position(312), now=1.1))
        self.assertFalse(detector.update(None, now=1.15))
        self.assertEqual(detector.confirmations, 0)
        for i in range(30):
            self.assertFalse(detector.update(position(320), now=1.7+i*0.05))

    def test_fast_duplicate_samples_do_not_confirm(self):
        detector = self.warm()
        for i in range(10):
            self.assertFalse(detector.update(position(312), now=1.1+i*0.001))

    def test_mouse_deadband_and_large_move(self):
        follower = CursorFollower(position())
        for i in range(30):
            self.assertIsNone(follower.target(position(300+(i%5)-2, 250+(i%3)-1)))
        targets = [follower.target(position(314, 258)) for _ in range(3)]
        self.assertIn((314, 258), targets)

    def test_configuration_validation(self):
        for kwargs in [{"click_delay": 0.1}, {"click_delay": 1.1},
                       {"motion_threshold": 0}, {"bite_mode": "unknown"},
                       {"after_loot_delay": 0}, {"after_loot_delay": 11},
                       {"track_lost_timeout": 0}, {"old_bobber_timeout": 0}]:
            with self.assertRaises(ValueError):
                BotConfig(**kwargs)


class TrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = BobberDetector(BASE / "calibration")
        cls.crop = cv2.imread(str(BASE / "calibration/bobber_idle.png"))

    def scene(self, dx=0, dy=0, gain=1.0, offset=0):
        canvas = np.full((500, 700, 3), (55, 65, 70), np.uint8)
        h, w = self.crop.shape[:2]
        canvas[195+dy:195+dy+h, 251+dx:251+dx+w] = self.crop
        return cv2.convertScaleAbs(canvas, alpha=gain, beta=offset)

    def tracker(self):
        frame = self.scene()
        tracker = BobberTracker(self.detector)
        pos = self.detector.find(frame)
        self.assertIsNotNone(pos)
        tracker.start(frame, pos)
        return tracker

    def test_center_translations(self):
        tracker = self.tracker()
        for dx, dy in [(0, 0), (1, -2), (-2, 1), (8, 6), (-5, -4), (12, 9)]:
            with self.subTest(dx=dx, dy=dy):
                pos = tracker.update(self.scene(dx, dy))
                self.assertIsNotNone(pos)
                self.assertLessEqual(abs(pos["x"]-(300+dx)), 1)
                self.assertLessEqual(abs(pos["y"]-(250+dy)), 1)

    def test_tracking_survives_brightness_changes(self):
        tracker = self.tracker()
        for gain, offset in [(0.6, 0), (1.3, 10), (1.8, 20), (0.8, 15)]:
            with self.subTest(gain=gain, offset=offset):
                pos = tracker.update(self.scene(gain=gain, offset=offset))
                self.assertIsNotNone(pos)
                self.assertLessEqual(abs(pos["x"]-300), 1)
                self.assertLessEqual(abs(pos["y"]-250), 1)

    def test_loss_is_not_a_coordinate(self):
        tracker = self.tracker()
        frame = np.full((500, 700, 3), (55, 65, 70), np.uint8)
        self.assertIsNone(tracker.update(frame))
        self.assertIsNotNone(tracker.update(self.scene()))

    def test_outside_search_target_cannot_steal_track(self):
        tracker = self.tracker()
        self.assertIsNone(tracker.update(self.scene(dx=100)))

    def test_nearby_target_jump_is_rejected(self):
        tracker = self.tracker()
        self.assertIsNone(tracker.update(self.scene(dx=30)))

    def test_reference_loading_from_unicode_path(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "эталоны"
            shutil.copytree(BASE / "calibration", target)
            self.assertEqual(len(BobberDetector(target).references), 2)

    def test_expanded_recovery_remains_near_same_target(self):
        tracker = self.tracker()
        self.assertIsNone(tracker.update(self.scene(dx=35)))
        recovered = tracker.recover(self.scene(dx=35))
        self.assertIsNotNone(recovered)
        self.assertLessEqual(abs(recovered["x"]-335), 1)
        self.assertIsNotNone(tracker.update(self.scene(dx=35)))
        self.assertIsNone(tracker.recover(self.scene(dx=120)))


class FakeStop:
    def __init__(self, timeout=30):
        self.now = 0.0
        self.stopped = False
        self.timeout = timeout

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def wait(self, duration):
        self.now += duration
        if self.now >= self.timeout:
            self.set()
        return self.stopped


class RuntimeTests(unittest.TestCase):
    def bot(self, stop, directory):
        bot = FishingBot(Mock(), BotConfig(), Mock(), Mock(), stop, Path(directory))
        bot.tracker = Mock()
        bot.tracker.update.return_value = position()
        bot.screenshot = Mock(return_value=np.zeros((500, 700, 3), np.uint8))
        return bot

    def test_right_click_after_delay_with_monitor_offset(self):
        stop = FakeStop()
        monitor = dict(left=-1920, top=100)
        with tempfile.TemporaryDirectory() as directory:
            bot = self.bot(stop, directory)
            with patch("fishing_logic.time.monotonic", side_effect=lambda: stop.now):
                self.assertTrue(bot.land_fish(Mock(), monitor, position()))
            self.assertGreaterEqual(stop.now, 0.5)
            self.assertLessEqual(stop.now, 1.0)
            bot.mouse.click.assert_called_once_with(-1620, 350, button="right")

    def test_stop_during_delay_prevents_click(self):
        stop = FakeStop(timeout=0.3)
        with tempfile.TemporaryDirectory() as directory:
            bot = self.bot(stop, directory)
            with patch("fishing_logic.time.monotonic", side_effect=lambda: stop.now):
                self.assertFalse(bot.land_fish(Mock(), dict(left=0, top=0), position()))
            bot.mouse.click.assert_not_called()

    def test_lost_track_before_click_cancels_click(self):
        stop = FakeStop()
        with tempfile.TemporaryDirectory() as directory:
            bot = self.bot(stop, directory)
            bot.tracker.update.return_value = None
            with patch("fishing_logic.time.monotonic", side_effect=lambda: stop.now):
                with self.assertLogs("fishing", level="WARNING"):
                    self.assertFalse(bot.land_fish(Mock(), dict(left=0, top=0), position()))
            bot.mouse.click.assert_not_called()

    def test_stop_before_run_does_not_press_cast(self):
        stop = FakeStop()
        stop.set()
        with tempfile.TemporaryDirectory() as directory:
            bot = self.bot(stop, directory)
            bot.run()
            bot.mouse.press.assert_not_called()

    def test_complete_cycle_on_frames_cast_track_motion_and_right_click(self):
        detector = BobberDetector(BASE / "calibration")
        crop = cv2.imread(str(BASE / "calibration/bobber_idle.png"))
        base = np.full((500, 700, 3), (55, 65, 70), np.uint8)
        base[195:275, 251:371] = crop
        shifted = np.roll(base, (8, 12), axis=(0, 1))
        blank = np.full_like(base, (55, 65, 70))
        frames = iter([blank] + [base]*25 + [shifted]*50)
        stop = FakeStop(timeout=20)
        capturer = Mock()
        capturer.monitors = [{}, dict(left=-700, top=10, width=700, height=500)]
        capturer.__enter__ = Mock(return_value=capturer)
        capturer.__exit__ = Mock(return_value=False)
        def grab(_):
            try:
                frame = next(frames)
            except StopIteration:
                stop.set()
                frame = shifted
            return cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        capturer.grab.side_effect = grab
        mouse = Mock()
        clicks = []
        def click(*args, **kwargs):
            clicks.append((stop.now, args, kwargs))
            stop.set()
        mouse.click.side_effect = click
        with tempfile.TemporaryDirectory() as directory:
            bot = FishingBot(detector, BotConfig(), mouse, lambda: capturer, stop, Path(directory))
            with patch("fishing_logic.time.monotonic", side_effect=lambda: stop.now):
                bot.run()
            mouse.press.assert_called_once_with("1")
            self.assertEqual(len(clicks), 1)
            self.assertEqual(clicks[0][1], (-388, 268))
            self.assertEqual(clicks[0][2], dict(button="right"))
            self.assertTrue((Path(directory) / "last_bite_debug.png").exists())

    def test_capture_failure_stops_bot_and_logs_exception(self):
        stop = FakeStop()
        with tempfile.TemporaryDirectory() as directory:
            bot = self.bot(stop, directory)
            bot.capturer_factory.side_effect = RuntimeError("capture failed")
            with self.assertLogs("fishing", level="ERROR"):
                bot.run()
            self.assertTrue(stop.is_set())
            bot.mouse.click.assert_not_called()

    def test_two_casts_wait_for_old_bobber_and_cooldown_and_count_once(self):
        detector = BobberDetector(BASE / "calibration")
        crop = cv2.imread(str(BASE / "calibration/bobber_idle.png"))
        base = np.full((500, 700, 3), (55, 65, 70), np.uint8)
        base[195:275, 251:371] = crop
        shifted = np.roll(base, (8, 12), axis=(0, 1))
        blank = np.full_like(base, (55, 65, 70))
        stop = FakeStop(timeout=30)
        state = dict(phase="idle", casts=0, frames=0, clicked_at=None)
        capturer = Mock()
        capturer.monitors = [{}, dict(left=0, top=0, width=700, height=500)]
        capturer.__enter__ = Mock(return_value=capturer)
        capturer.__exit__ = Mock(return_value=False)
        def grab(_):
            if state["phase"] == "cast":
                state["frames"] += 1
                frame = base if state["frames"] <= 25 else shifted
            elif state["phase"] == "old" and stop.now-state["clicked_at"] < 1.5:
                frame = shifted
            else:
                frame = blank
            return cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        capturer.grab.side_effect = grab
        mouse = Mock()
        def press(key):
            self.assertEqual(key, "1")
            state["casts"] += 1
            if state["casts"] == 2:
                self.assertGreaterEqual(stop.now-state["clicked_at"], 3)
                stop.set()
            state.update(phase="cast", frames=0)
        def click(*_, **kwargs):
            self.assertEqual(kwargs["button"], "right")
            state.update(phase="old", clicked_at=stop.now)
        mouse.press.side_effect = press
        mouse.click.side_effect = click
        with tempfile.TemporaryDirectory() as directory:
            bot = FishingBot(detector, BotConfig(), mouse, lambda: capturer, stop, Path(directory))
            with patch("fishing_logic.time.monotonic", side_effect=lambda: stop.now):
                bot.run()
            self.assertEqual(state["casts"], 2)
            self.assertEqual(bot.stats.casts, 2)
            self.assertEqual(bot.stats.hooks, 1)
            self.assertEqual(bot.stats.confirmed_hooks, 1)
            mouse.click.assert_called_once()

    def test_existing_float_on_restart_is_cleared_before_first_cast(self):
        detector = BobberDetector(BASE / "calibration")
        crop = cv2.imread(str(BASE / "calibration/bobber_idle.png"))
        base = np.full((500, 700, 3), (55, 65, 70), np.uint8)
        base[195:275, 251:371] = crop
        blank = np.full_like(base, (55, 65, 70))
        stop = FakeStop(timeout=20)
        visible = [True]
        capturer = Mock()
        capturer.monitors = [{}, dict(left=0, top=0, width=700, height=500)]
        capturer.__enter__ = Mock(return_value=capturer)
        capturer.__exit__ = Mock(return_value=False)
        capturer.grab.side_effect = lambda _: cv2.cvtColor(base if visible[0] else blank, cv2.COLOR_BGR2BGRA)
        mouse = Mock()
        def press(key):
            if key == "esc":
                visible[0] = False
            elif key == "1":
                self.assertFalse(visible[0])
                self.assertGreaterEqual(stop.now, 6)
                stop.set()
        mouse.press.side_effect = press
        with tempfile.TemporaryDirectory() as directory:
            bot = FishingBot(detector, BotConfig(), mouse, lambda: capturer, stop, Path(directory))
            with patch("fishing_logic.time.monotonic", side_effect=lambda: stop.now):
                bot.run()
        self.assertEqual([args.args[0] for args in mouse.press.call_args_list], ["esc", "1"])
        self.assertEqual(bot.stats.confirmed_hooks, 0)


class CleanupTests(unittest.TestCase):
    def make_bot(self, stop, config=None):
        bot = FishingBot(Mock(), config or BotConfig(), Mock(), Mock(), stop, BASE)
        bot.tracker = Mock()
        bot.tracker.update.return_value = None
        bot.tracker.recover.return_value = None
        bot.detector.find.return_value = None
        bot.screenshot = Mock(return_value=np.zeros((500, 700, 3), np.uint8))
        bot.last_position = position()
        return bot

    def pending(self, bot):
        bot.pending_hook = True
        bot.stats.hooks = 1

    def test_lingering_old_float_and_short_detection_gap_delay_confirmation(self):
        stop = FakeStop()
        bot = self.make_bot(stop)
        self.pending(bot)
        samples = iter([position()]*15 + [None]*2 + [position()]*3)
        bot.tracker.update.side_effect = lambda _: next(samples, None)
        with patch("fishing_logic.time.monotonic", side_effect=lambda: stop.now):
            self.assertTrue(bot.prepare_next_cast(Mock(), {}, after_click=True))
        self.assertGreaterEqual(stop.now, 3)
        self.assertEqual(bot.stats.confirmed_hooks, 1)
        self.assertIsNone(bot.tracker)
        bot.mouse.press.assert_not_called()

    def test_single_missing_frame_cannot_confirm_clear(self):
        stop = FakeStop()
        bot = self.make_bot(stop, BotConfig(old_bobber_timeout=0.8))
        samples = iter([None, None] + [position()]*20)
        bot.tracker.update.side_effect = lambda _: next(samples, position())
        with patch("fishing_logic.time.monotonic", side_effect=lambda: stop.now):
            self.assertFalse(bot.wait_for_old_bobber_clear(Mock(), {}))
        self.assertEqual(bot.stats.confirmed_hooks, 0)

    def test_color_detection_prevents_clearing_when_tracker_is_lost(self):
        stop = FakeStop()
        bot = self.make_bot(stop, BotConfig(old_bobber_timeout=0.7))
        bot.detector.find.return_value = position()
        with patch("fishing_logic.time.monotonic", side_effect=lambda: stop.now):
            self.assertFalse(bot.wait_for_old_bobber_clear(Mock(), {}))

    def test_persistent_float_is_cancelled_without_counting_fish(self):
        stop = FakeStop()
        bot = self.make_bot(stop, BotConfig(old_bobber_timeout=0.7))
        self.pending(bot)
        visible = [True]
        bot.tracker.update.side_effect = lambda _: position() if visible[0] else None
        bot.mouse.press.side_effect = lambda key: visible.__setitem__(0, False)
        with patch("fishing_logic.time.monotonic", side_effect=lambda: stop.now):
            with self.assertLogs("fishing", level="WARNING"):
                self.assertTrue(bot.prepare_next_cast(Mock(), {}, after_click=True))
        self.assertEqual(bot.stats.confirmed_hooks, 0)
        self.assertEqual(bot.stats.unconfirmed_hooks, 1)
        bot.mouse.press.assert_called_once_with("esc")

    def test_never_disappearing_old_float_stops_new_casts(self):
        stop = FakeStop()
        bot = self.make_bot(stop, BotConfig(old_bobber_timeout=0.7))
        self.pending(bot)
        bot.tracker.update.return_value = position()
        with patch("fishing_logic.time.monotonic", side_effect=lambda: stop.now):
            with self.assertLogs("fishing", level="ERROR"):
                self.assertFalse(bot.prepare_next_cast(Mock(), {}, after_click=True))
        self.assertTrue(stop.is_set())
        self.assertEqual(bot.stats.confirmed_hooks, 0)
        bot.mouse.press.assert_called_once_with("esc")

    def test_cancelled_cast_is_not_a_successful_hook(self):
        stop = FakeStop()
        bot = self.make_bot(stop)
        with patch("fishing_logic.time.monotonic", side_effect=lambda: stop.now):
            self.assertTrue(bot.prepare_next_cast(Mock(), {}))
        self.assertEqual(bot.stats.confirmed_hooks, 0)
        bot.mouse.press.assert_called_once_with("esc")

    def test_hook_is_counted_only_once(self):
        stop = FakeStop()
        bot = self.make_bot(stop)
        self.pending(bot)
        with patch("fishing_logic.time.monotonic", side_effect=lambda: stop.now):
            self.assertTrue(bot.prepare_next_cast(Mock(), {}, after_click=True))
            self.assertTrue(bot.prepare_next_cast(Mock(), {}, after_click=True))
        self.assertEqual(bot.stats.confirmed_hooks, 1)

    def test_stop_during_clear_does_not_count_hook(self):
        stop = FakeStop(timeout=0.2)
        bot = self.make_bot(stop)
        self.pending(bot)
        with patch("fishing_logic.time.monotonic", side_effect=lambda: stop.now):
            self.assertFalse(bot.prepare_next_cast(Mock(), {}, after_click=True))
        self.assertEqual(bot.stats.confirmed_hooks, 0)
        bot.mouse.press.assert_not_called()

    def test_stop_during_cooldown_prevents_next_cast(self):
        stop = FakeStop(timeout=1.0)
        bot = self.make_bot(stop)
        self.pending(bot)
        with patch("fishing_logic.time.monotonic", side_effect=lambda: stop.now):
            self.assertFalse(bot.prepare_next_cast(Mock(), {}, after_click=True))
        self.assertEqual(bot.stats.confirmed_hooks, 1)
        bot.mouse.press.assert_not_called()

    def test_lost_center_recovers_for_bounded_time_then_aborts_without_click(self):
        stop = FakeStop()
        bot = self.make_bot(stop, BotConfig(track_lost_timeout=0.5, bite_timeout=10))
        with patch("fishing_logic.time.monotonic", side_effect=lambda: stop.now):
            with self.assertLogs("fishing", level="WARNING"):
                self.assertIsNone(bot.wait_for_bite(Mock(), dict(left=0, top=0), position()))
        self.assertLess(stop.now, 3)
        self.assertEqual(bot.stats.lost_tracks, 1)
        bot.mouse.click.assert_not_called()
        self.assertTrue(bot.tracker.recover.called)

    def test_reacquired_center_relearns_without_false_bite(self):
        stop = FakeStop(timeout=4)
        bot = self.make_bot(stop)
        def tracking(_):
            if 2.0 <= stop.now < 2.35:
                return None
            return position(300 if stop.now < 2 else 312)
        bot.tracker.update.side_effect = tracking
        with patch("fishing_logic.time.monotonic", side_effect=lambda: stop.now):
            with self.assertLogs("fishing", level="INFO") as logs:
                self.assertIsNone(bot.wait_for_bite(Mock(), dict(left=0, top=0), position()))
        self.assertEqual(bot.stats.lost_tracks, 1)
        self.assertTrue(any("восстановлен" in entry for entry in logs.output))
        bot.mouse.click.assert_not_called()

    def test_session_stats_and_snapshots_survive_new_worker(self):
        shared = SessionStats(confirmed_hooks=34)
        snapshots = []
        bot = FishingBot(Mock(), BotConfig(), Mock(), Mock(), FakeStop(), BASE,
                         stats=shared, stats_callback=snapshots.append)
        bot._publish_stats()
        shared.confirmed_hooks += 1
        self.assertEqual(snapshots[0].confirmed_hooks, 34)
        next_bot = FishingBot(Mock(), BotConfig(), Mock(), Mock(), FakeStop(), BASE, stats=shared)
        self.assertEqual(next_bot.stats.confirmed_hooks, 35)


if __name__ == "__main__":
    unittest.main()
