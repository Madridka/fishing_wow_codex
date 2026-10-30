"""Офлайн-проверки на кадрах калибровки; игру и мышь не запускают."""

import unittest
from pathlib import Path

import cv2
import numpy as np

from bobber_vision import BobberDetector, SplashDetector, splash_score


BASE = Path(__file__).resolve().parent


class VisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = BobberDetector(BASE / "calibration")
        cls.idle = cv2.imread(str(BASE / "calibration/bobber_idle.png"))
        cls.bite = cv2.imread(str(BASE / "calibration/bobber_bite.png"))
        cls.idle_scene = cls.scene(cls.idle, (49, 55))
        cls.bite_scene = cls.scene(cls.bite, (179, 101))

    @staticmethod
    def scene(image, center):
        canvas = np.full((500, 700, 3), (55, 65, 70), np.uint8)
        h, w = image.shape[:2]
        x, y = 300 - center[0], 250 - center[1]
        canvas[y:y+h, x:x+w] = image
        return canvas

    def warmed_tracker(self, image=None):
        image = self.idle_scene if image is None else image
        tracker = SplashDetector()
        pos = self.detector.find(image)
        for i in range(15):
            self.assertFalse(tracker.update(image, pos, now=i * 0.05))
        self.assertTrue(tracker.armed)
        return tracker, pos

    def test_scale_and_screen_coordinates(self):
        for image, center in [(self.idle, (49, 55)), (self.bite, (179, 101))]:
            for scale in (0.6, 0.8, 1.0, 1.3, 2.0):
                with self.subTest(center=center, scale=scale):
                    sample = cv2.resize(image, None, fx=scale, fy=scale)
                    canvas = np.full((800, 1200, 3), (55, 65, 70), np.uint8)
                    h, w = sample.shape[:2]
                    canvas[180:180+h, 250:250+w] = sample
                    pos = self.detector.find(canvas)
                    self.assertIsNotNone(pos)
                    self.assertLessEqual(abs(pos["x"] - (250 + center[0]*scale)), 3)
                    self.assertLessEqual(abs(pos["y"] - (180 + center[1]*scale)), 3)
                    local = self.detector.find(canvas, (pos["x"]-120, pos["y"]-120,
                                                        pos["x"]+120, pos["y"]+120))
                    self.assertEqual((pos["x"], pos["y"]), (local["x"], local["y"]))

    def test_empty_scene_and_blue_feather_alone_are_rejected(self):
        canvas = np.full((600, 800, 3), (55, 65, 70), np.uint8)
        self.assertIsNone(self.detector.find(canvas))
        feather = self.detector._blue_mask(self.idle)
        patch = canvas[200:280, 300:420]
        patch[feather > 0] = self.idle[feather > 0]
        self.assertIsNone(self.detector.find(canvas))
        self.assertIsNone(self.detector.find(canvas, (-100, -100, -1, -1)))

    def test_splash_requires_appearance_and_duration(self):
        tracker, idle_pos = self.warmed_tracker()
        bite_pos = self.detector.find(self.bite_scene)
        self.assertFalse(tracker.update(self.bite_scene, bite_pos, now=0.75))
        self.assertFalse(tracker.update(self.idle_scene, idle_pos, now=0.8))
        for now in (0.85, 0.9, 0.95):
            self.assertFalse(tracker.update(self.bite_scene, bite_pos, now=now))
        self.assertTrue(tracker.update(self.bite_scene, bite_pos, now=1.0))

    def test_static_bright_details_do_not_trigger_even_above_old_threshold(self):
        tracker, pos = self.warmed_tracker(self.bite_scene)
        self.assertGreater(tracker.score, 0.012)
        for i in range(100):
            self.assertFalse(tracker.update(self.bite_scene, pos, now=0.75+i*0.05))
        self.assertEqual(tracker.new_score, 0)

    def test_tracking_jitter_and_illumination_do_not_trigger(self):
        tracker, pos = self.warmed_tracker(self.bite_scene)
        for i in range(30):
            jitter = dict(pos, x=pos["x"]+(i%3)-1, y=pos["y"]+(i%3)-1)
            frame = cv2.convertScaleAbs(self.bite_scene, alpha=1.03, beta=i%4)
            self.assertFalse(tracker.update(frame, jitter, now=0.75+i*0.05))

    def test_track_loss_and_reacquisition_do_not_trigger(self):
        tracker, pos = self.warmed_tracker()
        self.assertFalse(tracker.update(self.bite_scene, pos, now=0.75))
        self.assertFalse(tracker.update(self.bite_scene, None, now=0.8))
        self.assertEqual(tracker.confirmations, 0)
        # После длинной потери новое статичное изображение изучается заново.
        for i in range(20):
            self.assertFalse(tracker.update(self.bite_scene, pos, now=1.5+i*0.05))

    def test_fast_duplicate_frames_do_not_confirm_bite(self):
        tracker, pos = self.warmed_tracker()
        for i in range(10):
            self.assertFalse(tracker.update(self.bite_scene, pos, now=0.75+i*0.001))

    def test_bite_can_tilt_and_move_without_relearning_the_splash_as_background(self):
        tracker, pos = self.warmed_tracker()
        moved = dict(pos, x=pos["x"]+10, y=pos["y"]+7, bobber_height=24)
        frame = np.roll(self.bite_scene, (7, 10), axis=(0, 1))
        for now in (0.75, 0.8, 0.85):
            self.assertFalse(tracker.update(frame, moved, now=now))
        self.assertTrue(tracker.update(frame, moved, now=0.9))

    def test_large_target_jump_restarts_warmup(self):
        tracker, pos = self.warmed_tracker()
        moved = dict(pos, x=pos["x"]+80)
        frame = np.roll(self.bite_scene, 80, axis=1)
        self.assertFalse(tracker.update(frame, moved, now=0.75))
        self.assertFalse(tracker.armed)
        for i in range(20):
            self.assertFalse(tracker.update(frame, moved, now=0.8+i*0.05))

    def test_one_bright_object_is_not_a_cloud_of_particles(self):
        tracker, pos = self.warmed_tracker()
        frame = self.idle_scene.copy()
        cv2.rectangle(frame, (270, 230), (278, 260), (255, 245, 255), -1)
        for i in range(10):
            self.assertFalse(tracker.update(frame, pos, now=0.75+i*0.05))

    def test_splash_at_different_scales(self):
        for scale in (0.6, 0.8, 1.0, 1.3, 2.0):
            for image, biting in [(self.idle, False), (self.bite, True)]:
                with self.subTest(scale=scale, biting=biting):
                    sample = cv2.resize(image, None, fx=scale, fy=scale)
                    pos = self.detector.find(sample)
                    self.assertEqual(splash_score(sample, pos) >= 0.012, biting)

    def test_splash_sequence_at_different_scales(self):
        for scale in (0.6, 0.8, 1.0, 1.3, 2.0):
            with self.subTest(scale=scale):
                idle = cv2.resize(self.idle_scene, None, fx=scale, fy=scale)
                bite = cv2.resize(self.bite_scene, None, fx=scale, fy=scale)
                tracker, _ = self.warmed_tracker(idle)
                pos = self.detector.find(bite)
                for now in (0.75, 0.8, 0.85):
                    self.assertFalse(tracker.update(bite, pos, now=now))
                self.assertTrue(tracker.update(bite, pos, now=0.9))

    def test_uniform_brightness_change_is_not_splash(self):
        image = np.full((200, 200, 3), 220, np.uint8)
        self.assertEqual(splash_score(image, {"x": 100, "y": 100}), 0.0)

    @unittest.skipUnless((BASE / "bobber_not_found.png").exists(), "Нет локального скриншота")
    def test_original_missed_screen_and_background(self):
        image = cv2.imread(str(BASE / "bobber_not_found.png"))
        position = self.detector.find(image)
        self.assertIsNotNone(position)
        self.assertLess(abs(position["x"] - 1109), 5)
        self.assertLess(abs(position["y"] - 635), 5)
        self.assertLess(splash_score(image, position), 0.012)
        # Тот же дождь, растительность и интерфейс, но без самого поплавка.
        image[580:660, 1060:1180] = image[580:660, 940:1060]
        self.assertIsNone(self.detector.find(image))


if __name__ == "__main__":
    unittest.main()
