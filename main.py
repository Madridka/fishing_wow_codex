"""Запуск оконного приложения. Импорт модуля не запускает GUI и управление мышью."""

import ctypes
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys


def enable_dpi_awareness():
    """Координаты mss и мыши должны совпадать при масштабировании Windows."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def configure_logging():
    directory = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    try:
        handler = RotatingFileHandler(directory / "fishing.log", maxBytes=1_000_000,
                                      backupCount=3, encoding="utf-8")
    except OSError:
        directory = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "WoWFishing"
        directory.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(directory / "fishing.log", maxBytes=1_000_000,
                                      backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger = logging.getLogger("fishing")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    return directory, directory / "fishing.log"


def main():
    # Вызвать до импорта библиотек, создающих окна или использующих экран.
    enable_dpi_awareness()
    import tkinter as tk
    import mss
    import pyautogui
    from pynput import keyboard
    from app_ui import FishingApp
    from bobber_vision import BobberDetector

    pyautogui.FAILSAFE = True
    # Все паузы контролируются stop_event; встроенная пауза иначе тормозит трекинг.
    pyautogui.PAUSE = 0
    output_dir, log_path = configure_logging()
    references = Path(__file__).resolve().parent / "calibration"
    root = tk.Tk()
    check_only = "--self-test" in sys.argv
    if check_only:
        root.withdraw()
    app = FishingApp(root, lambda: BobberDetector(references), pyautogui,
                     mss.mss, output_dir, log_path)

    if check_only:
        # Проверка собранного .exe без захвата экрана и управления мышью.
        try:
            model = BobberDetector(references)
            app.poll()
            root.update_idletasks()
            if len(model.references) < 2 or not app.log_text.winfo_exists():
                raise RuntimeError("Не загрузились эталоны или окно журнала")
            logging.getLogger("fishing").info("SELF-TEST OK: Tk, журнал, OpenCV, эталоны и зависимости загружены")
            app.close()
            app.poll()
            return 0
        except Exception:
            logging.getLogger("fishing").exception("SELF-TEST FAILED")
            root.destroy()
            return 1
        finally:
            logging.shutdown()

    def on_key(key):
        if key == keyboard.Key.f8:
            logging.getLogger("fishing").info("F8: аварийная остановка")
            app.request_stop()

    listener = keyboard.Listener(on_press=on_key)
    listener.daemon = True
    listener.start()
    try:
        root.mainloop()
    finally:
        app.stop_event.set()
        listener.stop()
        if app.worker is not None:
            app.worker.join(timeout=2)
        logging.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
