"""python build_exe.py — сборка Windows-приложения вместе с эталонами."""

import importlib.util
import os
from pathlib import Path
import sys


def build():
    base = Path(__file__).resolve().parent
    # Не выпускать .exe без Tk: PyInstaller иначе может лишь предупредить об этом.
    import tkinter
    tkinter.Tcl().eval("info patchlevel")
    if importlib.util.find_spec("PyInstaller") is None:
        sys.path.insert(0, str(base / ".build-tools"))
    from PyInstaller.__main__ import run
    os.environ["PYINSTALLER_CONFIG_DIR"] = str(base / ".build-cache")
    run(["--noconfirm", "--clean", "--onefile", "--windowed", "--noupx", "--name", "WoWFishing",
         "--distpath", str(base / "dist"), "--workpath", str(base / "build"),
         "--specpath", str(base / "build"),
         "--add-data", f"{base / 'calibration'}:calibration",
         "--hidden-import", "pynput.keyboard._win32",
         "--hidden-import", "pynput.mouse._win32", str(base / "main.py")])
    print(base / "dist" / "WoWFishing.exe")


if __name__ == "__main__":
    build()
