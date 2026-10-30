"""Окно управления и потокобезопасный журнал."""

import logging
import queue
import threading
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from fishing_logic import BotConfig, FishingBot, SessionStats


class QueueLogHandler(logging.Handler):
    def __init__(self, messages):
        super().__init__()
        self.messages = messages
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s  %(message)s", "%H:%M:%S"))

    def emit(self, record):
        message = self.format(record)
        try:
            self.messages.put_nowait(message)
        except queue.Full:
            try:
                self.messages.get_nowait()
            except queue.Empty:
                pass
            try:
                self.messages.put_nowait(message)
            except queue.Full:
                pass


class FishingApp:
    MODES = {"Движение центра": "motion", "Брызги (прежний режим)": "splash",
             "Движение + брызги": "both"}

    def __init__(self, root, detector_factory, mouse, capturer_factory, output_dir, log_path):
        self.root = root
        self.detector_factory = detector_factory
        self.mouse = mouse
        self.capturer_factory = capturer_factory
        self.output_dir = output_dir
        self.logger = logging.getLogger("fishing")
        self.log_queue = queue.Queue(maxsize=2000)
        self.status_queue = queue.SimpleQueue()
        self.stats_queue = queue.SimpleQueue()
        self.session_stats = SessionStats()
        self.log_handler = QueueLogHandler(self.log_queue)
        self.logger.addHandler(self.log_handler)
        self.worker = None
        self.stop_event = threading.Event()
        self.closing = False
        root.title("WoW Fishing — центр поплавка")
        root.geometry("980x650")
        root.minsize(860, 540)
        root.protocol("WM_DELETE_WINDOW", self.close)

        panel = ttk.Frame(root, padding=14)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="WoW Fishing", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        self.status = tk.StringVar(value="Остановлен")
        ttk.Label(panel, textvariable=self.status).pack(anchor="w", pady=(4, 12))
        self.counter = tk.StringVar(value="Успешных подсечек: 0  |  Забросов: 0")
        ttk.Label(panel, textvariable=self.counter, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(panel, text="Подтверждение — исчезновение поплавка после правого клика. Счётчик за время работы окна.",
                  wraplength=900).pack(anchor="w", pady=(2, 10))
        controls = ttk.Frame(panel)
        controls.pack(fill="x")
        self.start_button = ttk.Button(controls, text="НАЧАТЬ РЫБАЛКУ", command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(controls, text="ОСТАНОВИТЬ · F8", command=self.request_stop)
        self.stop_button.pack(side="left", padx=8)

        options = ttk.Frame(panel)
        options.pack(fill="x", pady=12)
        self.mode = tk.StringVar(value="Движение центра")
        self.mode_box = ttk.Combobox(options, textvariable=self.mode, values=list(self.MODES), state="readonly", width=25)
        self.mode_box.grid(row=0, column=0, rowspan=2, padx=(0, 18))
        ttk.Label(options, text="Порог движения, px").grid(row=0, column=1, sticky="w")
        self.threshold = tk.StringVar(value="6")
        self.threshold_box = ttk.Spinbox(options, textvariable=self.threshold, from_=4, to=50, width=7)
        self.threshold_box.grid(row=1, column=1, sticky="w", padx=(0, 18))
        ttk.Label(options, text="Задержка правого клика, с").grid(row=0, column=2, sticky="w")
        self.delay = tk.StringVar(value="0.75")
        self.delay_box = ttk.Spinbox(options, textvariable=self.delay, from_=0.5, to=1.0, increment=0.05, width=7)
        self.delay_box.grid(row=1, column=2, sticky="w")
        ttk.Label(options, text="Пауза до заброса, с").grid(row=0, column=3, sticky="w", padx=(18, 0))
        self.cooldown = tk.StringVar(value="3")
        self.cooldown_box = ttk.Spinbox(options, textvariable=self.cooldown, from_=1, to=10, increment=0.5, width=7)
        self.cooldown_box.grid(row=1, column=3, sticky="w", padx=(18, 0))
        ttk.Label(panel, text="Дрожание 1–2 px игнорируется. Поклёвка — подтверждённое резкое движение центра.",
                  wraplength=790).pack(anchor="w", pady=(0, 10))
        log_header = ttk.Frame(panel)
        log_header.pack(fill="x")
        ttk.Label(log_header, text="Журнал", font=("Segoe UI", 11, "bold")).pack(side="left")
        ttk.Button(log_header, text="Очистить окно журнала", command=self.clear_log).pack(side="right")
        self.log_text = ScrolledText(panel, height=15, state="disabled", font=("Consolas", 10), wrap="word")
        self.log_text.pack(fill="both", expand=True, pady=(6, 8))
        ttk.Label(panel, text=f"Файл журнала: {log_path}", wraplength=800).pack(anchor="w")
        root.after(100, self.poll)
        self.logger.info("Готово. Правый клик; клавиша заброса 1. По умолчанию анализируется движение центра")

    def start(self):
        if self.closing or (self.worker is not None and self.worker.is_alive()):
            return
        try:
            config = BotConfig(motion_threshold=float(self.threshold.get().replace(",", ".")),
                               click_delay=float(self.delay.get().replace(",", ".")),
                               after_loot_delay=float(self.cooldown.get().replace(",", ".")),
                               bite_mode=self.MODES[self.mode.get()])
        except (ValueError, KeyError) as error:
            self.logger.error("Проверьте настройки: %s", error)
            return
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self.run_bot, args=(config, self.stop_event), daemon=True)
        self.worker.start()
        self.set_controls(True)

    def run_bot(self, config, stop_event):
        try:
            bot = FishingBot(self.detector_factory(), config, self.mouse, self.capturer_factory,
                             stop_event, self.output_dir, self.status_queue.put, self.logger,
                             stats=self.session_stats, stats_callback=self.stats_queue.put)
            bot.run()
        except Exception:
            self.logger.exception("Не удалось запустить рыбалку")
            stop_event.set()
            self.status_queue.put("Остановлен: ошибка запуска")

    def request_stop(self):
        # Этот метод также вызывается из потока клавиатуры: здесь нет вызовов Tk.
        self.stop_event.set()
        if self.worker is not None and self.worker.is_alive():
            self.status_queue.put("Останавливается…")
            self.logger.info("Запрошена остановка")

    def set_controls(self, active):
        self.start_button.configure(state="disabled" if active else "normal")
        self.mode_box.configure(state="disabled" if active else "readonly")
        self.threshold_box.configure(state="disabled" if active else "normal")
        self.delay_box.configure(state="disabled" if active else "normal")
        self.cooldown_box.configure(state="disabled" if active else "normal")

    def clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def poll(self):
        while not self.status_queue.empty():
            self.status.set(self.status_queue.get())
        while not self.stats_queue.empty():
            stats = self.stats_queue.get()
            self.counter.set(f"Успешных подсечек: {stats.confirmed_hooks}  |  Забросов: {stats.casts}  |  Потерь центра: {stats.lost_tracks}")
        entries = []
        for _ in range(200):
            try:
                entries.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        if entries:
            self.log_text.configure(state="normal")
            self.log_text.insert("end", "\n".join(entries) + "\n")
            lines = int(self.log_text.index("end-1c").split(".")[0])
            if lines > 1000:
                self.log_text.delete("1.0", f"{lines-1000}.0")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        active = self.worker is not None and self.worker.is_alive()
        self.set_controls(active)
        if self.closing and not active:
            self.logger.removeHandler(self.log_handler)
            self.root.destroy()
            return
        self.root.after(100, self.poll)

    def close(self):
        self.closing = True
        self.request_stop()
        self.start_button.configure(state="disabled")
