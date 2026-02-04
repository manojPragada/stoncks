from __future__ import annotations

import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

import pandas as pd

from stoncks_app.config import AppConfig
from stoncks_app.data import build_features, load_price_data
from stoncks_app.model import load_model, predict_proba
from stoncks_app.stream import replay_stream
from stoncks_app.trading import evaluate_signal, Signal


@dataclass
class AppState:
    running: bool = False
    latest_signal: Optional[Signal] = None


class StoncksApp:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.root = tk.Tk()
        self.root.title("NQ! Alert Dashboard")
        self.state = AppState()
        self.queue: Queue[Signal] = Queue()
        self._build_ui()

    def _build_ui(self) -> None:
        self.status_label = tk.Label(self.root, text="Status: Idle", font=("Arial", 14))
        self.status_label.pack(pady=10)

        self.signal_label = tk.Label(self.root, text="Signal: -", font=("Arial", 12))
        self.signal_label.pack(pady=5)

        self.edge_label = tk.Label(self.root, text="Expected Edge: -", font=("Arial", 12))
        self.edge_label.pack(pady=5)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        self.start_button = tk.Button(button_frame, text="Start", command=self.start)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(button_frame, text="Stop", command=self.stop)
        self.stop_button.pack(side=tk.LEFT, padx=5)

    def start(self) -> None:
        if self.state.running:
            return
        self.state.running = True
        self.status_label.configure(text="Status: Running")
        threading.Thread(target=self._run_loop, daemon=True).start()
        self.root.after(500, self._poll_queue)

    def stop(self) -> None:
        self.state.running = False
        self.status_label.configure(text="Status: Stopped")

    def _poll_queue(self) -> None:
        try:
            signal = self.queue.get_nowait()
        except Empty:
            pass
        else:
            self.state.latest_signal = signal
            self.signal_label.configure(
                text=f"Signal: {signal.action} (p={signal.probability:.2f})"
            )
            self.edge_label.configure(text=f"Expected Edge: {signal.expected_edge:.5f}")
        if self.state.running:
            self.root.after(500, self._poll_queue)

    def _run_loop(self) -> None:
        model = load_model(self.config.model_path)
        data = load_price_data(self.config.data_path)
        for row in replay_stream(self.config.data_path, self.config.refresh_seconds):
            if not self.state.running:
                break
            row_df = pd.DataFrame([row])
            features = build_features(pd.concat([data, row_df]).tail(100)).tail(1)
            probability = predict_proba(model, features)
            signal = evaluate_signal(probability, self.config.commission_per_contract)
            self.queue.put(signal)

    def run(self) -> None:
        self.root.mainloop()


def launch_app(model_path: Path, data_path: Path) -> None:
    config = AppConfig(model_path=model_path, data_path=data_path)
    app = StoncksApp(config)
    app.run()
