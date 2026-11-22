"""DESTROYER desktop lab: manual play, DQN training and solver evaluation."""

from __future__ import annotations

import json
import os
import queue
import re
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np

from destroyer import (
    ACTIONS,
    DQNAgent,
    Expectimax,
    Game2048,
    NtupleValue,
    PositionHeuristic,
    apply_move,
    encode,
)
from destroyer.arena import (
    AGENT_DQN,
    AGENT_EXPECTIMAX,
    AGENT_MODES,
    AGENT_NTUPLE,
    evaluate,
)
from destroyer.records import RecordBook
from destroyer.ui import theme
from destroyer.ui.widgets import BoardView, MetricCard, Sparkline, Tooltip

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
STATE_PATH = MODELS_DIR / "ui_state.json"
RECORDS_PATH = MODELS_DIR / "records.json"

ACTION_LABELS = {"UP": "↑ UP", "DOWN": "↓ DOWN", "LEFT": "← LEFT", "RIGHT": "→ RIGHT"}
TRAIN_STAT_KEYS = (
    "Episode",
    "Train steps",
    "Env steps",
    "Epsilon",
    "Loss",
    "TD |mean|",
    "TD |max|",
    "Grad norm",
    "Replay",
    "LR",
    "Device",
    "Architecture",
    "Best train",
    "Stored eps",
)

SETTINGS_SPEC = {
    "search_depth": (int, 1, 8, 3),
    "node_budget": (int, 1_000, 10_000_000, 500_000),
    "time_budget_ms": (float, 1.0, 2_000.0, 35.0),
    "autoplay_ms": (int, 10, 2_000, 120),
    "eval_games": (int, 1, 1_000, 100),
    "eval_seed": (int, 0, 2_147_483_647, 1_000_000),
    "auto_restart": (bool, False, False, False),
    "animations": (bool, True, True, True),
    "max_log_lines": (int, 100, 5_000, 500),
}


def _coerce_setting(key: str, value):
    kind, low, high, _default = SETTINGS_SPEC[key]
    if kind is bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if kind is int:
        return max(int(low), min(int(high), int(round(float(value)))))
    return max(float(low), min(float(high), float(value)))


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self._closing = False
        self._saving_state = False
        self._logged_errors: set[str] = set()
        self.state = self._load_state_file()

        self.settings = {key: spec[3] for key, spec in SETTINGS_SPEC.items()}
        for key, value in (self.state.get("settings") or {}).items():
            if key in SETTINGS_SPEC:
                try:
                    self.settings[key] = _coerce_setting(key, value)
                except Exception:
                    pass

        self._record_load_error: str | None = None
        legacy_best_score = max(0, int(self.state.get("high_score", 0)))
        legacy_best_tile = max(0, int(self.state.get("high_tile", 0)))
        try:
            loaded_records = RecordBook.load(RECORDS_PATH)
        except Exception as error:
            self._record_load_error = str(error)
            loaded_records = RecordBook()
        self.records = RecordBook(
            best_score=max(legacy_best_score, loaded_records.best_score),
            best_tile=max(legacy_best_tile, loaded_records.best_tile),
            games_played=loaded_records.games_played,
            last_score=loaded_records.last_score,
            last_tile=loaded_records.last_tile,
        )
        self.high_score = self.records.best_score
        self.high_tile = self.records.best_tile
        self.best_train_score = int(self.state.get("best_train_score", -1))
        self.current_model_name = str(self.state.get("last_model") or "—")
        self.last_ntuple_path = self.state.get("last_ntuple")
        last_mode = self.state.get("last_mode", AGENT_DQN)
        if last_mode not in AGENT_MODES:
            last_mode = AGENT_DQN

        root.title("2048 DESTROYER | Neural Agent Lab")
        root.configure(bg=theme.BG)
        root.minsize(1180, 780)
        geometry = self.state.get("geometry")
        saved = re.match(r"^(\d+)x(\d+)([+-]\d+){2}$", geometry) if isinstance(geometry, str) else None
        if saved and int(saved.group(1)) >= 1180 and int(saved.group(2)) >= 780:
            root.geometry(geometry)
        else:
            root.geometry("1500x940")
        theme.configure_styles(root)

        self.game = Game2048()
        self.game_best_score = self.game.score
        self.game_best_tile = self.game.max_tile
        self.agent = DQNAgent()
        self.agent_mode = tk.StringVar(value=last_mode)
        self.current_mode = last_mode
        self.ntuple: NtupleValue | None = None
        self.ntuple_solver: Expectimax | None = None
        self.classic_solver: Expectimax | None = None

        self.training = False
        self.stop_requested = False
        self.evaluating = False
        self.autoplay = False
        self.auto_job = None
        self.stop_flag = threading.Event()
        self.eval_cancel = threading.Event()
        self.uiq: queue.Queue = queue.Queue()
        self.train_thread: threading.Thread | None = None

        self.episode = 0
        self.episodes: list[dict] = []
        self.eval_records: list = []
        self.eval_report = None
        self.eval_total = 0
        self.last_eval_path: Path | None = None
        self.ntuple_dialog_refresh = None

        self.overlay: str | None = None
        self.win_shown = False
        self._game_record_saved = False
        self.last_spawn: tuple[int, int, int] | None = None
        self.last_action = "—"
        self.last_reward = 0
        self.last_search_result = None
        self._action_values = None
        self._legal_actions: list[str] = []
        self._best_action: str | None = None

        self.buttons: dict[str, ttk.Button] = {}
        self.action_bars: dict[str, tk.Canvas] = {}
        self.action_value_labels: dict[str, tk.Label] = {}
        self.action_name_labels: dict[str, tk.Label] = {}
        self.search_labels: dict[str, tk.Label] = {}
        self.stat_labels: dict[str, tk.Label] = {}
        self.training_summary: dict[str, MetricCard] = {}
        self.eval_labels: dict[str, tk.Label] = {}

        self._build_header()
        self._build_toolbar()
        self._build_status()
        self._build_body()
        self._bind_keys()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.render(full=True)
        self.refresh_stats()
        self._update_status()
        self.log(
            "Application ready. "
            f"DQN architecture: {self.agent.model_architecture} | "
            f"parameters: {sum(p.numel() for p in self.agent.online.parameters()):,} | "
            f"batch: {self.agent.batch} | device: {self.agent.device}"
        )
        self.log("Arrows move · SPACE autoplay · U undo · R new game · RETURN AI step · ESC stop")
        if self._record_load_error:
            self.log(f"Could not load personal records: {self._record_load_error}", "error")
        self.poll_job = self.root.after(80, self.poll)

    # --------------------------------------------------------------- persistence
    @staticmethod
    def _load_state_file() -> dict:
        try:
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_state(self) -> None:
        if self._saving_state:
            return
        self._saving_state = True
        try:
            geometry = None
            try:
                geometry = self.root.geometry()
            except tk.TclError:
                pass
            payload = {
                "settings": self.settings,
                "high_score": self.high_score,
                "high_tile": self.high_tile,
                "best_train_score": self.best_train_score,
                "last_model": self.current_model_name,
                "last_ntuple": self.last_ntuple_path,
                "last_mode": self.current_mode
                if self.current_mode in AGENT_MODES
                else AGENT_DQN,
                "geometry": geometry,
            }
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            temporary = STATE_PATH.with_name(STATE_PATH.name + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temporary, STATE_PATH)
        except Exception as error:
            self._log_error_once("state-save", f"Could not save UI state: {error}")
        finally:
            self._saving_state = False

    # --------------------------------------------------------------- layout
    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg=theme.BG)
        header.pack(fill="x", padx=18, pady=(14, 4))
        title = tk.Frame(header, bg=theme.BG)
        title.pack(side="left")
        tk.Label(title, text="2048", bg=theme.BG, fg=theme.FG,
                 font=("Segoe UI", 29, "bold")).pack(side="left")
        tk.Label(title, text="DESTROYER", bg=theme.BG, fg=theme.ACC,
                 font=("Segoe UI", 29, "bold")).pack(side="left", padx=(8, 0))
        tk.Label(title, text="NEURAL AGENT LAB", bg=theme.BG, fg=theme.MUTED,
                 font=("Consolas", 9)).pack(side="left", padx=(14, 0), pady=(15, 0))

        agent = tk.Frame(header, bg=theme.BG)
        agent.pack(side="right", padx=(0, 12), pady=(8, 0))
        tk.Label(agent, text="AGENT", bg=theme.BG, fg=theme.MUTED,
                 font=("Consolas", 9)).pack(side="left", padx=(0, 6))
        self.mode_box = ttk.Combobox(
            agent,
            textvariable=self.agent_mode,
            values=AGENT_MODES,
            state="readonly",
            width=22,
            takefocus=False,
        )
        self.mode_box.pack(side="left")
        self.mode_box.bind("<<ComboboxSelected>>", self.on_mode_changed)
        Tooltip(self.mode_box, "DQN · classic Expectimax · N-tuple + Expectimax")

        self.status_pill = tk.Label(
            header,
            text="IDLE",
            bg=theme.PANEL_ALT,
            fg=theme.MUTED,
            font=("Consolas", 9, "bold"),
            padx=12,
            pady=5,
        )
        self.status_pill.pack(side="right", pady=(8, 0))

    def _build_toolbar(self) -> None:
        toolbar = tk.Frame(self.root, bg=theme.BG)
        toolbar.pack(fill="x", padx=18, pady=(2, 8))
        left = tk.Frame(toolbar, bg=theme.BG)
        left.pack(side="left")

        self._toolbar_button(left, "new", "NEW GAME", self.new_game, "Accent.TButton",
                             "Start a fresh episode (R)")
        self._toolbar_button(left, "undo", "UNDO", self.undo, "TButton",
                             "Undo the last successful move (U / Ctrl+Z)")
        self._toolbar_button(left, "step", "STEP AI", self.ai_step, "TButton",
                             "Ask the active agent for one move (Return)")
        self.auto_button = self._toolbar_button(left, "auto", "AUTO PLAY", self.toggle_autoplay,
                                                "TButton", "Continuous play; press again to pause (Space)")
        self._toolbar_button(left, "stop", "STOP", self.stop_all, "Danger.TButton",
                             "Stop autoplay, training and evaluation (Esc)")
        self._toolbar_button(left, "train", "TRAIN DQN", self.start_training, "TButton",
                             "Train the DQN baseline in the background")
        self.eval_button = self._toolbar_button(left, "eval", "EVAL ×100", self.evaluate,
                                                "TButton", "Evaluate the active agent on fixed seeds")
        self._toolbar_button(left, "models", "MODELS", self.open_models, "TButton",
                             "Save, load, delete or export DQN checkpoints")
        self._toolbar_button(left, "ntuple", "N-TUPLE", self.open_ntuple, "TButton",
                             "Load or unload an N-tuple evaluator")
        self._toolbar_button(left, "settings", "SETTINGS", self.open_settings, "TButton",
                             "Search budgets, autoplay speed, evaluation size and animation")

        self.eval_button.configure(text=f"EVAL ×{self.settings['eval_games']}")

    def _toolbar_button(self, parent, key, text, command, style="TButton", tip=None):
        button = ttk.Button(parent, text=text, command=command, style=style, takefocus=False)
        button.pack(side="left", padx=(0, 6))
        self.buttons[key] = button
        if tip:
            Tooltip(button, tip)
        return button

    def _build_status(self) -> None:
        bar = tk.Frame(self.root, bg=theme.PANEL_ALT, height=28)
        bar.pack(side="bottom", fill="x")
        bar.pack_propagate(False)
        self.status_left = tk.Label(bar, text="", bg=theme.PANEL_ALT, fg=theme.MUTED,
                                    font=("Consolas", 8), anchor="w")
        self.status_left.pack(side="left", padx=12)
        self.status_right = tk.Label(bar, text="", bg=theme.PANEL_ALT, fg=theme.MUTED,
                                     font=("Consolas", 8), anchor="e")
        self.status_right.pack(side="right", padx=12)

    def _build_body(self) -> None:
        body = tk.Frame(self.root, bg=theme.BG)
        body.pack(fill="both", expand=True, padx=18, pady=(0, 8))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=3, minsize=330)
        body.grid_columnconfigure(1, weight=4, minsize=340)
        body.grid_columnconfigure(2, weight=4, minsize=340)

        left = tk.Frame(body, bg=theme.BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        middle = tk.Frame(body, bg=theme.BG)
        middle.grid(row=0, column=1, sticky="nsew", padx=10)
        right = tk.Frame(body, bg=theme.BG)
        right.grid(row=0, column=2, sticky="nsew", padx=(10, 0))

        self._build_left(left)
        self._build_middle(middle)
        self._build_right(right)

    def _build_left(self, parent) -> None:
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        self.board_view = BoardView(parent, width=300, height=300,
                                    on_new_game=self.new_game,
                                    on_continue=self.continue_after_win)
        self.board_view.grid(row=0, column=0, sticky="nsew")

        metrics = tk.Frame(parent, bg=theme.BG)
        metrics.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for column in range(3):
            metrics.grid_columnconfigure(column, weight=1)
        self.score_card = MetricCard(metrics, "Score", "0")
        self.best_card = MetricCard(metrics, "Best score", "—", accent=theme.GOLD)
        self.tile_card = MetricCard(metrics, "Max tile", "2")
        self.best_tile_card = MetricCard(metrics, "Best tile", "—", accent=theme.GOLD)
        self.moves_card = MetricCard(metrics, "Moves", "0")
        cards = (self.score_card, self.best_card, self.tile_card,
                 self.best_tile_card, self.moves_card)
        for index, card in enumerate(cards):
            row, column = divmod(index, 3)
            card.grid(
                row=row,
                column=column,
                sticky="ew",
                padx=(0, 6 if column < 2 else 0),
                pady=(0, 6 if row == 0 else 0),
            )

        hint = tk.Label(
            parent,
            text="ARROWS move   ·   SPACE auto   ·   U undo   ·   R new   ·   ESC stop",
            bg=theme.BG,
            fg=theme.MUTED,
            font=("Consolas", 8),
            wraplength=330,
            justify="left",
        )
        hint.grid(row=2, column=0, sticky="w", pady=(7, 0))

    def _build_middle(self, parent) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(3, weight=1)

        values_frame = tk.LabelFrame(parent, text=" ACTION VALUE ANALYSIS ", bg=theme.BG,
                                     fg=theme.FG, padx=9, pady=7)
        values_frame.grid(row=0, column=0, sticky="ew")
        for action in ACTIONS:
            row = tk.Frame(values_frame, bg=theme.BG)
            row.pack(fill="x", pady=2)
            name = tk.Label(row, text=ACTION_LABELS[action], width=8, anchor="w",
                            bg=theme.BG, fg=theme.FG, font=("Consolas", 9))
            name.pack(side="left")
            bar = tk.Canvas(row, width=1, height=18, bg=theme.PANEL,
                            highlightthickness=0, bd=0)
            bar.pack(side="left", fill="x", expand=True, padx=(4, 8))
            value = tk.Label(row, text="—", width=11, anchor="e", bg=theme.BG,
                             fg=theme.FG, font=("Consolas", 9))
            value.pack(side="right")
            self.action_bars[action] = bar
            self.action_name_labels[action] = name
            self.action_value_labels[action] = value
            bar.bind("<Configure>", lambda _event, a=action: self._draw_action_bar(a))

        search_frame = tk.LabelFrame(parent, text=" SEARCH / POLICY ", bg=theme.BG,
                                     fg=theme.FG, padx=9, pady=6)
        search_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        for column in (1, 3):
            search_frame.grid_columnconfigure(column, weight=1)
        for index, key in enumerate(("Backend", "Depth", "Nodes", "Cache", "Time", "Budget")):
            row = index // 2
            column = (index % 2) * 2
            tk.Label(search_frame, text=key, bg=theme.BG, fg=theme.MUTED,
                     font=("Consolas", 8), anchor="w").grid(
                row=row, column=column, sticky="w", padx=(0, 5), pady=1)
            value = tk.Label(search_frame, text="—", bg=theme.BG, fg=theme.FG,
                             font=("Consolas", 8), anchor="w")
            value.grid(row=row, column=column + 1, sticky="w", padx=(0, 14), pady=1)
            self.search_labels[key] = value

        stats_frame = tk.LabelFrame(parent, text=" NETWORK / TRAINING ", bg=theme.BG,
                                    fg=theme.FG, padx=9, pady=6)
        stats_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        for index, key in enumerate(TRAIN_STAT_KEYS):
            row = index // 2
            pair = index % 2
            base = pair * 3
            stats_frame.grid_columnconfigure(base + 1, weight=1)
            tk.Label(stats_frame, text=key, bg=theme.BG, fg=theme.MUTED,
                     font=("Consolas", 8), anchor="w").grid(
                row=row, column=base, sticky="w", padx=(0, 4), pady=1)
            value = tk.Label(stats_frame, text="—", bg=theme.BG, fg=theme.FG,
                             font=("Consolas", 8), anchor="e")
            value.grid(row=row, column=base + 1, sticky="e", padx=(0, 14), pady=1)
            self.stat_labels[key] = value
        rows = (len(TRAIN_STAT_KEYS) + 1) // 2
        epsilon_row = tk.Frame(stats_frame, bg=theme.BG)
        epsilon_row.grid(row=rows, column=0, columnspan=6, sticky="ew", pady=(5, 0))
        tk.Label(epsilon_row, text="EPSILON", bg=theme.BG, fg=theme.MUTED,
                 font=("Consolas", 8)).pack(side="left", padx=(0, 6))
        self.epsilon_bar = ttk.Progressbar(epsilon_row, style="Accent.Horizontal.TProgressbar",
                                           maximum=100, value=0)
        self.epsilon_bar.pack(side="left", fill="x", expand=True)

        debug_frame = tk.LabelFrame(parent, text=" LIVE DEBUG ", bg=theme.BG,
                                    fg=theme.FG, padx=6, pady=4)
        debug_frame.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        debug_frame.grid_rowconfigure(1, weight=1)
        debug_frame.grid_columnconfigure(0, weight=1)
        debug_header = tk.Frame(debug_frame, bg=theme.BG)
        debug_header.grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Label(debug_header, text="latest events", bg=theme.BG, fg=theme.MUTED,
                 font=("Consolas", 7)).pack(side="left")
        clear = ttk.Button(debug_header, text="CLEAR", style="Ghost.TButton",
                           command=self.clear_log, takefocus=False)
        clear.pack(side="right")
        self.debug_text = tk.Text(
            debug_frame,
            width=24,
            height=7,
            bg="#091016",
            fg="#a7e4c6",
            insertbackground=theme.FG,
            font=("Consolas", 8),
            wrap="word",
            relief="flat",
            bd=0,
            state="disabled",
        )
        self.debug_text.grid(row=1, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(debug_frame, orient="vertical",
                                  command=self.debug_text.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.debug_text.configure(yscrollcommand=scrollbar.set)
        self.debug_text.tag_configure("time", foreground=theme.DIM)
        self.debug_text.tag_configure("info", foreground="#a7e4c6")
        self.debug_text.tag_configure("warn", foreground=theme.WARN)
        self.debug_text.tag_configure("error", foreground=theme.DANGER)

    def _build_right(self, parent) -> None:
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)

        training = tk.Frame(notebook, bg=theme.BG)
        notebook.add(training, text="TRAINING")
        training.grid_rowconfigure(2, weight=1)
        training.grid_columnconfigure(0, weight=1)

        summary = tk.Frame(training, bg=theme.BG)
        summary.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        for column in range(4):
            summary.grid_columnconfigure(column, weight=1)
        cards = (
            ("episodes", MetricCard(summary, "Episodes", "0")),
            ("average", MetricCard(summary, "Avg 50", "—")),
            ("best", MetricCard(summary, "Best", "—", accent=theme.GOLD)),
            ("rate", MetricCard(summary, "P(2048)", "—", accent=theme.ACC)),
        )
        for index, (key, card) in enumerate(cards):
            card.grid(row=0, column=index, sticky="ew", padx=(0, 6 if index < 3 else 0))
            self.training_summary[key] = card

        top = tk.Frame(training, bg=theme.BG)
        top.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.chart = Sparkline(top, width=1, height=170)
        self.chart.pack(side="left", fill="both", expand=True)
        self.train_preview = BoardView(top, width=184, height=184)
        self.train_preview.pack(side="right", padx=(8, 0))

        tree_frame = tk.Frame(training, bg=theme.BG)
        tree_frame.grid(row=2, column=0, sticky="nsew")
        self.training_tree = ttk.Treeview(
            tree_frame,
            columns=("ep", "score", "tile", "moves", "loss", "eps"),
            show="headings",
            height=9,
            selectmode="browse",
        )
        for column, title, width in (
            ("ep", "EP", 46),
            ("score", "SCORE", 82),
            ("tile", "MAX", 58),
            ("moves", "MOVES", 62),
            ("loss", "LOSS", 72),
            ("eps", "EPS", 58),
        ):
            self.training_tree.heading(column, text=title)
            self.training_tree.column(column, width=width, anchor="center", stretch=True)
        self.training_tree.tag_configure("odd", background=theme.PANEL_ALT)
        training_scroll = ttk.Scrollbar(tree_frame, orient="vertical",
                                        command=self.training_tree.yview)
        self.training_tree.configure(yscrollcommand=training_scroll.set)
        self.training_tree.pack(side="left", fill="both", expand=True)
        training_scroll.pack(side="right", fill="y")

        tk.Label(training, text="episode log · latest 100 records", bg=theme.BG,
                 fg=theme.MUTED, font=("Consolas", 7)).grid(
            row=3, column=0, sticky="w", pady=(4, 0))

        evaluation = tk.Frame(notebook, bg=theme.BG)
        notebook.add(evaluation, text="EVALUATION")
        evaluation.grid_rowconfigure(2, weight=1)
        evaluation.grid_columnconfigure(0, weight=1)

        eval_grid = tk.Frame(evaluation, bg=theme.BG)
        eval_grid.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        for column in (1, 3):
            eval_grid.grid_columnconfigure(column, weight=1)
        for index, key in enumerate((
            "Agent", "Games", "Avg score", "Median", "P(2048)",
            "P(4096)", "P(8192)", "Nodes/game", "Elapsed",
        )):
            row = index // 2
            column = (index % 2) * 2
            tk.Label(eval_grid, text=key, bg=theme.BG, fg=theme.MUTED,
                     font=("Consolas", 8), anchor="w").grid(
                row=row, column=column, sticky="w", padx=(0, 5), pady=1)
            value = tk.Label(eval_grid, text="—", bg=theme.BG, fg=theme.FG,
                             font=("Consolas", 8), anchor="w")
            value.grid(row=row, column=column + 1, sticky="w", padx=(0, 14), pady=1)
            self.eval_labels[key] = value

        progress_frame = tk.Frame(evaluation, bg=theme.BG)
        progress_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.eval_progress = ttk.Progressbar(
            progress_frame, style="Blue.Horizontal.TProgressbar", maximum=100, value=0)
        self.eval_progress.pack(side="left", fill="x", expand=True)
        self.eval_progress_label = tk.Label(progress_frame, text="idle", bg=theme.BG,
                                            fg=theme.MUTED, font=("Consolas", 8))
        self.eval_progress_label.pack(side="right", padx=(8, 0))

        eval_tree_frame = tk.Frame(evaluation, bg=theme.BG)
        eval_tree_frame.grid(row=2, column=0, sticky="nsew")
        self.eval_tree = ttk.Treeview(
            eval_tree_frame,
            columns=("seed", "score", "tile", "moves", "nodes"),
            show="headings",
            height=9,
            selectmode="browse",
        )
        for column, title, width in (
            ("seed", "SEED", 84),
            ("score", "SCORE", 82),
            ("tile", "MAX", 58),
            ("moves", "MOVES", 62),
            ("nodes", "NODES", 88),
        ):
            self.eval_tree.heading(column, text=title)
            self.eval_tree.column(column, width=width, anchor="center", stretch=True)
        self.eval_tree.tag_configure("odd", background=theme.PANEL_ALT)
        eval_scroll = ttk.Scrollbar(eval_tree_frame, orient="vertical",
                                    command=self.eval_tree.yview)
        self.eval_tree.configure(yscrollcommand=eval_scroll.set)
        self.eval_tree.pack(side="left", fill="both", expand=True)
        eval_scroll.pack(side="right", fill="y")

        eval_buttons = tk.Frame(evaluation, bg=theme.BG)
        eval_buttons.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        self.save_report_button = ttk.Button(
            eval_buttons, text="SAVE REPORT", command=self.save_eval_report,
            state="disabled", takefocus=False)
        self.save_report_button.pack(side="left", padx=(0, 6))
        ttk.Button(eval_buttons, text="OPEN REPORTS", command=self.open_reports,
                   takefocus=False).pack(side="left", padx=(0, 6))
        self.eval_cancel_button = ttk.Button(
            eval_buttons, text="CANCEL", style="Danger.TButton",
            command=self.cancel_evaluation, state="disabled", takefocus=False)
        self.eval_cancel_button.pack(side="right")

    # --------------------------------------------------------------- logging
    def log(self, message: str, level: str = "info") -> None:
        if level not in ("info", "warn", "error"):
            level = "info"
        try:
            self.debug_text.configure(state="normal")
            self.debug_text.insert("end", f"[{time.strftime('%H:%M:%S')}] ", "time")
            self.debug_text.insert("end", f"{message}\n", level)
            lines = int(self.debug_text.index("end-1c").split(".")[0])
            limit = int(self.settings.get("max_log_lines", 500))
            if lines > limit:
                self.debug_text.delete("1.0", f"{lines - limit + 1}.0")
            self.debug_text.see("end")
            self.debug_text.configure(state="disabled")
        except tk.TclError:
            pass

    def clear_log(self) -> None:
        try:
            self.debug_text.configure(state="normal")
            self.debug_text.delete("1.0", "end")
            self.debug_text.configure(state="disabled")
        except tk.TclError:
            pass

    def _log_error_once(self, key: str, message: str) -> None:
        if key in self._logged_errors:
            return
        self._logged_errors.add(key)
        self.log(message, "error")

    # --------------------------------------------------------------- rendering
    def render(self, full: bool = True) -> None:
        spawn, self.last_spawn = self.last_spawn, None
        self.board_view.set_state(
            self.game.board,
            spawn=spawn,
            overlay=self.overlay,
            animations=self.settings["animations"],
        )
        self.score_card.set(f"{self.game.score:,}")
        self.best_card.set(f"{self.high_score:,}" if self.high_score >= 0 else "—")
        self.tile_card.set(str(self.game.max_tile))
        self.best_tile_card.set(f"{self.high_tile:,}" if self.high_tile > 0 else "—")
        self.moves_card.set(str(self.game.moves))
        if full:
            self._render_action_values()
            self._update_search_panel()
        self._sync_controls()
        self._update_status()

    def _render_action_values(self) -> None:
        legal = self.game.legal_actions()
        self._legal_actions = legal
        mode = self.agent_mode.get()
        values = None
        try:
            if mode == AGENT_DQN:
                values = self.agent.q_values(self.game.board)
            elif mode == AGENT_EXPECTIMAX:
                solver = self._get_solver()
                computed = []
                for action in ACTIONS:
                    result = apply_move(self.game.board, action)
                    computed.append(float(solver.evaluator(result.board)) if result.moved else 0.0)
                values = np.asarray(computed, dtype=np.float64)
            elif mode == AGENT_NTUPLE:
                if self.ntuple is not None:
                    computed = []
                    for action in ACTIONS:
                        result = apply_move(self.game.board, action)
                        if result.moved:
                            computed.append(float(result.score_delta) + self.ntuple.value(result.board))
                        else:
                            computed.append(0.0)
                    values = np.asarray(computed, dtype=np.float64)
        except Exception as error:
            self._log_error_once("action-values", f"Action analysis failed: {error}")
            values = None

        self._action_values = values
        self._best_action = None
        if values is not None and legal:
            best_index = max(
                (ACTIONS.index(action) for action in legal),
                key=lambda index: float(values[index]),
            )
            self._best_action = ACTIONS[best_index]

        for index, action in enumerate(ACTIONS):
            is_legal = action in legal
            if values is None:
                text = "—"
            else:
                value = float(values[index])
                text = f"{value:+,.2f}" if abs(value) < 1000 else f"{value:+,.0f}"
            label = self.action_value_labels[action]
            label.config(
                text=text,
                fg=theme.ACC if action == self._best_action
                else (theme.FG if is_legal else theme.DIM),
            )
            name = self.action_name_labels[action]
            name.config(
                fg=theme.ACC if action == self._best_action
                else (theme.FG if is_legal else theme.DIM),
                font=("Consolas", 9, "bold") if action == self._best_action
                else ("Consolas", 9),
            )
            self._draw_action_bar(action)

    def _draw_action_bar(self, action: str) -> None:
        bar = self.action_bars.get(action)
        if bar is None:
            return
        bar.delete("all")
        width = max(60, bar.winfo_width())
        values = self._action_values
        if values is None:
            bar.create_text(width / 2, 9, text="no evaluator", fill=theme.DIM,
                            font=("Consolas", 7))
            return
        index = ACTIONS.index(action)
        value = float(values[index])
        legal = action in self._legal_actions
        scale = max(1e-9, max(abs(float(item)) for item in values))
        zero = width * 0.5
        target = zero + (value / scale) * (width * 0.47)
        color = (
            theme.ACC if action == self._best_action
            else (theme.BLUE if legal else "#33414f")
        )
        if abs(target - zero) < 1:
            bar.create_line(zero, 4, zero, 14, fill=color, width=2)
        else:
            bar.create_rectangle(min(zero, target), 3, max(zero, target), 15,
                                 fill=color, outline="")
        bar.create_line(zero, 1, zero, 17, fill="#5d7186")

    def _record_search_result(self, result) -> None:
        self.last_search_result = result
        self._update_search_panel()

    def _update_search_panel(self) -> None:
        mode = self.agent_mode.get()
        if mode == AGENT_DQN:
            self.search_labels["Backend"].config(text="torch-mlp")
            self.search_labels["Depth"].config(text="greedy")
            self.search_labels["Nodes"].config(text="—")
            self.search_labels["Cache"].config(text="—")
            self.search_labels["Time"].config(text="—")
            self.search_labels["Budget"].config(text="argmax Q")
            return
        try:
            solver = self._get_solver()
        except Exception:
            solver = None
        backend = solver.backend if solver is not None else "not loaded"
        result = self.last_search_result
        self.search_labels["Backend"].config(text=backend)
        if result is not None:
            depth = str(result.completed_depth)
            if not result.complete:
                depth += "*"
            self.search_labels["Depth"].config(text=depth)
            self.search_labels["Nodes"].config(text=f"{result.nodes:,}")
            self.search_labels["Cache"].config(text=f"{result.cache_hits:,}")
            self.search_labels["Time"].config(text=f"{result.elapsed_ms:.1f} ms")
        else:
            self.search_labels["Depth"].config(text="—")
            self.search_labels["Nodes"].config(text="—")
            self.search_labels["Cache"].config(text="—")
            self.search_labels["Time"].config(text="—")
        self.search_labels["Budget"].config(
            text=f"{self.settings['search_depth']}d · "
                 f"{self.settings['node_budget'] // 1000}k · "
                 f"{self.settings['time_budget_ms']:.0f}ms"
        )

    def refresh_stats(self) -> None:
        recent = self.episodes[-50:]
        count = max(1, len(recent))
        average = sum(item["score"] for item in recent) / count
        average_tile = sum(item["tile"] for item in recent) / count
        rate2048 = sum(item["tile"] >= 2048 for item in recent) / count
        agent = self.agent
        values = {
            "Episode": f"{self.episode:,}",
            "Train steps": f"{agent.steps:,}",
            "Env steps": f"{agent.environment_steps:,}",
            "Epsilon": f"{agent.epsilon:.4f}",
            "Loss": f"{agent.loss_avg:.3e}",
            "TD |mean|": f"{agent.td_error_mean:.3e}",
            "TD |max|": f"{agent.td_error_max:.3e}",
            "Grad norm": f"{agent.grad_norm:.3e}",
            "Replay": f"{len(agent.replay):,}",
            "LR": f"{agent.learning_rate:.1e}",
            "Device": str(agent.device),
            "Architecture": agent.model_architecture,
            "Best train": f"{self.best_train_score:,}" if self.best_train_score >= 0 else "—",
            "Stored eps": f"{len(self.episodes):,}",
        }
        for key, value in values.items():
            label = self.stat_labels.get(key)
            if label is not None:
                label.config(text=str(value))
        try:
            self.epsilon_bar["value"] = max(0.0, min(100.0, agent.epsilon * 100.0))
        except tk.TclError:
            pass
        self.training_summary["episodes"].set(f"{self.episode:,}")
        self.training_summary["average"].set(f"{average:,.0f}" if recent else "—")
        self.training_summary["best"].set(
            f"{self.best_train_score:,}" if self.best_train_score >= 0 else "—")
        self.training_summary["rate"].set(f"{rate2048 * 100:.1f}%" if recent else "—")

    def _solver_backend(self) -> str:
        mode = self.agent_mode.get()
        if mode == AGENT_DQN:
            return "torch-mlp"
        solver = self.ntuple_solver if mode == AGENT_NTUPLE else self.classic_solver
        return solver.backend if solver is not None else "not loaded"

    def _update_status(self) -> None:
        if self.training:
            state = "STOPPING" if self.stop_requested else "TRAINING"
            background, foreground = theme.WARN, "#1a1205"
        elif self.evaluating:
            state = f"EVAL {len(self.eval_records)}/{self.eval_total}"
            background, foreground = theme.BLUE, "#061421"
        elif self.autoplay:
            state = "AUTOPLAY"
            background, foreground = theme.ACC_DARK, "#eafff4"
        else:
            state = "IDLE"
            background, foreground = theme.PANEL_ALT, theme.MUTED
        try:
            self.status_pill.config(text=state, bg=background, fg=foreground)
            self.status_left.config(
                text=f"MODE {self.agent_mode.get()}   ·   SEARCH {self._solver_backend()}   ·   "
                     f"MODEL {self.current_model_name}   ·   DEVICE {self.agent.device}"
            )
            self.status_right.config(
                text=f"AUTO {'ON' if self.autoplay else 'OFF'}   ·   "
                     f"TRAIN {'ON' if self.training else 'OFF'}   ·   "
                     f"EVAL {'ON' if self.evaluating else 'OFF'}   ·   "
                     f"HIGH {self.high_score if self.high_score >= 0 else '—'}"
            )
        except tk.TclError:
            pass

    # --------------------------------------------------------------- controls
    def _sync_controls(self) -> None:
        mode = self.agent_mode.get()
        training = self.training
        evaluating = self.evaluating

        def enable(key, enabled):
            button = self.buttons.get(key)
            if button is not None:
                button.state(["!disabled"] if enabled else ["disabled"])

        enable("new", not training)
        enable("undo", not training and self.game.can_undo)
        enable("step", not training and not evaluating and not self.autoplay
               and self.overlay != "gameover")
        enable("auto", not training and not evaluating)
        enable("stop", training or evaluating or self.autoplay)
        enable("train", not training and not evaluating and mode == AGENT_DQN)
        enable("eval", not training and not evaluating
               and not (mode == AGENT_NTUPLE and self.ntuple is None))
        enable("models", not training and not evaluating)
        enable("ntuple", not training and not evaluating)
        enable("settings", not training and not evaluating)

        if self.auto_button is not None:
            self.auto_button.configure(
                text="PAUSE" if self.autoplay else "AUTO PLAY",
                style="Accent.TButton" if self.autoplay else "TButton",
            )
        if self.buttons.get("stop") is not None:
            self.buttons["stop"].configure(
                text="STOPPING…" if (self.training and self.stop_requested) else "STOP")
        try:
            self.mode_box.configure(
                state="disabled" if (training or evaluating) else "readonly")
        except tk.TclError:
            pass
        if hasattr(self, "eval_cancel_button"):
            self.eval_cancel_button.state(
                ["!disabled"] if evaluating else ["disabled"])
        if hasattr(self, "save_report_button"):
            self.save_report_button.state(
                ["!disabled"] if self.eval_report is not None else ["disabled"])

    # --------------------------------------------------------------- keyboard
    def _typing(self) -> bool:
        try:
            widget = self.root.focus_get()
        except Exception:
            widget = None
        return isinstance(widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Combobox, ttk.Spinbox))

    def _bind_keys(self) -> None:
        for key, action in (
            ("<Up>", "UP"), ("<Down>", "DOWN"),
            ("<Left>", "LEFT"), ("<Right>", "RIGHT"),
        ):
            self.root.bind(key, lambda _event, a=action: self._key_manual(a))
        self.root.bind("<space>", self._key_space)
        self.root.bind("<Key-r>", self._key_new)
        self.root.bind("<Key-R>", self._key_new)
        self.root.bind("<Key-u>", self._key_undo)
        self.root.bind("<Key-U>", self._key_undo)
        self.root.bind("<Control-z>", self._key_undo)
        self.root.bind("<Return>", self._key_step)
        self.root.bind("<Escape>", self._key_stop)
        self.root.bind("<Key-1>", lambda _event: self._key_mode(0))
        self.root.bind("<Key-2>", lambda _event: self._key_mode(1))
        self.root.bind("<Key-3>", lambda _event: self._key_mode(2))

    def _key_manual(self, action: str):
        if self._typing():
            return None
        self.manual(action)
        return "break"

    def _key_space(self, _event=None):
        if self._typing():
            return None
        self.toggle_autoplay()
        return "break"

    def _key_new(self, _event=None):
        if self._typing():
            return None
        self.new_game()
        return "break"

    def _key_undo(self, _event=None):
        if self._typing():
            return None
        self.undo()
        return "break"

    def _key_step(self, _event=None):
        if self._typing():
            return None
        if not self.autoplay:
            self.ai_step()
        return "break"

    def _key_stop(self, _event=None):
        if self._typing():
            return None
        self.stop_all()
        return "break"

    def _key_mode(self, index: int):
        if self._typing():
            return None
        if index < len(AGENT_MODES):
            self.agent_mode.set(AGENT_MODES[index])
            self.on_mode_changed()
        return "break"

    # --------------------------------------------------------------- solvers
    def _make_solver(self, mode: str, ntuple: NtupleValue | None = None) -> Expectimax:
        depth = int(self.settings["search_depth"])
        budget = int(self.settings["node_budget"])
        time_budget = float(self.settings["time_budget_ms"])
        if mode == AGENT_EXPECTIMAX:
            return Expectimax(
                PositionHeuristic(),
                max_depth=depth,
                node_budget=budget,
                time_budget_ms=time_budget,
            )
        if mode == AGENT_NTUPLE:
            if ntuple is None:
                raise RuntimeError("Load an N-tuple checkpoint first.")
            return Expectimax(
                ntuple.state_value,
                max_depth=depth,
                node_budget=budget,
                time_budget_ms=time_budget,
                fallback_evaluator=ntuple.value,
            )
        raise ValueError(f"Search is not used by {mode}")

    def _get_solver(self) -> Expectimax | None:
        mode = self.agent_mode.get()
        if mode == AGENT_EXPECTIMAX:
            if self.classic_solver is None:
                self.classic_solver = self._make_solver(mode)
            return self.classic_solver
        if mode == AGENT_NTUPLE:
            if self.ntuple is None:
                raise RuntimeError("Load an N-tuple checkpoint first.")
            if self.ntuple_solver is None:
                self.ntuple_solver = self._make_solver(mode, self.ntuple)
            return self.ntuple_solver
        return None

    def _apply_solver_settings(self) -> None:
        for solver in (self.classic_solver, self.ntuple_solver):
            if solver is None:
                continue
            try:
                solver.configure(
                    max_depth=int(self.settings["search_depth"]),
                    node_budget=int(self.settings["node_budget"]),
                    time_budget_ms=float(self.settings["time_budget_ms"]),
                )
            except Exception as error:
                self.log(f"Could not update search settings: {error}", "warn")

    def on_mode_changed(self, _event=None) -> None:
        mode = self.agent_mode.get()
        if self.training or self.evaluating:
            self.agent_mode.set(self.current_mode)
            messagebox.showinfo(
                "Busy",
                "Agent mode is locked while training or evaluation is active.",
                parent=self.root,
            )
            return
        if mode == AGENT_NTUPLE and self.ntuple is None:
            self.agent_mode.set(self.current_mode)
            if messagebox.askyesno(
                "N-tuple required",
                "Load a v4 N-tuple checkpoint now?",
                parent=self.root,
            ):
                self.open_ntuple()
            return
        self.stop_autoplay("mode change")
        self.current_mode = mode
        if mode == AGENT_EXPECTIMAX:
            try:
                self._get_solver()
            except Exception as error:
                self.log(f"Could not create Expectimax solver: {error}", "error")
        self.last_search_result = None
        self.log(f"Active agent: {mode}")
        self._save_state()
        self.render(full=True)

    # --------------------------------------------------------------- game actions
    def new_game(self, from_auto: bool = False) -> None:
        if self.training:
            self.log("New game is disabled while DQN training is active.", "warn")
            return
        self.game.reset()
        self.game_best_score = self.game.score
        self.game_best_tile = self.game.max_tile
        self._game_record_saved = False
        self.overlay = None
        self.win_shown = False
        self.last_spawn = None
        self.last_search_result = None
        self.log("New episode started.")
        self._save_state()
        self.render(full=True)
        if self.autoplay and self.auto_job is None and not from_auto:
            self._autoplay_tick()

    def continue_after_win(self) -> None:
        if self.overlay == "win":
            self.overlay = None
            self.log("Continuing past 2048.")
            self.render(full=False)

    def manual(self, action: str) -> None:
        if self.training:
            self.log("Manual play is disabled while training is active.", "warn")
            return
        if self.game.game_over():
            self._finish_game_record()
            self.overlay = "gameover"
            self.render(full=True)
            return
        self.stop_autoplay("manual move")
        self.overlay = None
        step = self.game.move(action)
        if not step.moved:
            self.board_view.flash(theme.DANGER)
            self.log(f"MANUAL {action}: blocked", "warn")
            self._sync_controls()
            return
        self._after_move(step, action, source="MANUAL")

    def undo(self) -> None:
        if self.training:
            self.log("Undo is disabled while training is active.", "warn")
            return
        if not self.game.undo():
            self.log("Nothing to undo.", "warn")
            return
        if self._game_record_saved:
            self._game_record_saved = False
            self.game_best_score = self.game.score
            self.game_best_tile = self.game.max_tile
        self.stop_autoplay("undo")
        self.overlay = None
        if self.game.max_tile < 2048:
            self.win_shown = False
        self.last_spawn = None
        self.last_search_result = None
        self.log(f"Undo: score {self.game.score:,} · max {self.game.max_tile}.")
        self.render(full=True)

    def ai_step(self, auto: bool = False) -> bool:
        if self.training:
            if not auto:
                self.log("AI step is disabled while training is active.", "warn")
            return False
        if self.evaluating and not auto:
            self.log("AI step is paused during evaluation; manual play still works.", "warn")
            return False
        if self.game.game_over():
            self._finish_game_record()
            self.overlay = "gameover"
            self.render(full=True)
            return False
        if self.overlay == "win":
            self.overlay = None
        legal = self.game.legal_actions()
        try:
            action, result, elapsed = self._choose_action(self.game.board, legal)
        except Exception as error:
            self.log(f"AGENT ERROR: {error}", "error")
            self.stop_autoplay("agent error")
            return False
        step = self.game.move(action)
        if result is not None:
            self._record_search_result(result)
        self._after_move(
            step,
            action,
            source="AUTO" if auto else "AI",
            elapsed_ms=elapsed,
        )
        return True

    def _choose_action(self, board, legal):
        mode = self.agent_mode.get()
        started = time.perf_counter()
        if mode == AGENT_DQN:
            action = ACTIONS[self.agent.act(board, legal, explore=False)]
            return action, None, (time.perf_counter() - started) * 1000.0
        solver = self._get_solver()
        if solver is None:
            raise RuntimeError(f"No solver is available for {mode}.")
        result = solver.choose(board)
        if result is None:
            raise RuntimeError("Search returned no action for a non-terminal board.")
        if result.action not in legal:
            raise RuntimeError(f"Search returned an illegal action: {result.action}")
        elapsed = result.elapsed_ms or (time.perf_counter() - started) * 1000.0
        return result.action, result, elapsed

    def _after_move(self, step, action: str, *, source: str, elapsed_ms: float = 0.0) -> None:
        self.last_action = action
        self.last_reward = step.score_delta
        self.last_spawn = step.spawned
        self.game_best_score = max(self.game_best_score, self.game.score)
        self.game_best_tile = max(self.game_best_tile, self.game.max_tile)
        if self.game.score > self.high_score:
            self.high_score = self.game.score
        if self.game.max_tile > self.high_tile:
            self.high_tile = self.game.max_tile
        if step.game_over:
            self._finish_game_record()
            self.overlay = "gameover"
            self.log(
                f"GAME OVER — score {self.game.score:,}, max tile {self.game.max_tile:,}, "
                f"moves {self.game.moves:,}.",
                "warn",
            )
        elif self.game.max_tile >= 2048 and not self.win_shown:
            self.win_shown = True
            if self.autoplay:
                self.overlay = None
                self.log("2048 reached; autoplay continues.", "info")
            else:
                self.overlay = "win"
                self.log("2048 reached. Keep playing or start a new game.", "info")
        else:
            self.overlay = None
        suffix = f" · {elapsed_ms:.1f} ms" if elapsed_ms else ""
        self.log(
            f"{source} {action}: +{step.score_delta} · score {self.game.score:,} · "
            f"max {self.game.max_tile:,}{suffix}"
        )
        self.render(full=True)

    def _finish_game_record(self) -> None:
        """Persist this completed game once, keeping score/tile records monotonic."""

        if self._game_record_saved:
            return
        previous_score = self.records.best_score
        previous_tile = self.records.best_tile
        try:
            self.records = self.records.with_completed_game(
                RECORDS_PATH,
                score=self.game_best_score,
                max_tile=self.game_best_tile,
            )
        except Exception as error:
            self.log(f"PERSONAL RECORD SAVE FAILED: {error}", "error")
            return
        self.high_score = self.records.best_score
        self.high_tile = self.records.best_tile
        self._game_record_saved = True
        self._save_state()
        if self.high_score > previous_score or self.high_tile > previous_tile:
            self.log(
                f"NEW RECORD — best score {self.high_score:,} · best tile {self.high_tile:,}.",
                "info",
            )

    # --------------------------------------------------------------- autoplay
    def toggle_autoplay(self) -> None:
        if self.autoplay:
            self.stop_autoplay("paused")
            return
        if self.training or self.evaluating:
            self.log("Autoplay is unavailable while training or evaluation is active.", "warn")
            return
        if self.game.game_over():
            self.new_game(from_auto=True)
        if self.overlay == "win":
            self.overlay = None
        self.autoplay = True
        self.log("Autoplay started.")
        self._sync_controls()
        self._update_status()
        self._autoplay_tick()

    def stop_autoplay(self, reason: str | None = None) -> None:
        if self.auto_job is not None:
            try:
                self.root.after_cancel(self.auto_job)
            except tk.TclError:
                pass
            self.auto_job = None
        was_running = self.autoplay
        self.autoplay = False
        if was_running:
            self.log(f"Autoplay stopped ({reason})." if reason else "Autoplay stopped.")
            self._sync_controls()
            self._update_status()

    def _autoplay_tick(self) -> None:
        self.auto_job = None
        if not self.autoplay or self._closing:
            return
        if self.training or self.evaluating:
            self.stop_autoplay("busy")
            return
        if self.game.game_over():
            self._autoplay_finish()
            return
        self.ai_step(auto=True)
        if not self.autoplay or self._closing:
            return
        if self.game.game_over():
            self._autoplay_finish()
            return
        try:
            self.auto_job = self.root.after(int(self.settings["autoplay_ms"]), self._autoplay_tick)
        except tk.TclError:
            self.auto_job = None

    def _autoplay_finish(self) -> None:
        if self.settings["auto_restart"]:
            self.log("Autoplay: restarting after game over.")
            try:
                self.auto_job = self.root.after(900, self._auto_restart)
            except tk.TclError:
                self.auto_job = None
        else:
            self.stop_autoplay("game over")

    def _auto_restart(self) -> None:
        self.auto_job = None
        if not self.autoplay or self._closing:
            return
        self.new_game(from_auto=True)
        try:
            self.auto_job = self.root.after(200, self._autoplay_tick)
        except tk.TclError:
            self.auto_job = None

    def stop_all(self) -> None:
        acted = False
        if self.autoplay:
            self.stop_autoplay("stop requested")
            acted = True
        if self.evaluating:
            self.cancel_evaluation()
            acted = True
        if self.training and not self.stop_requested:
            self.stop_requested = True
            self.stop_flag.set()
            self.log("Training stop requested.", "warn")
            acted = True
        if not acted:
            self.log("Nothing is running.")
        self._sync_controls()
        self._update_status()

    # --------------------------------------------------------------- training
    def start_training(self) -> None:
        if self.training:
            return
        if self.evaluating:
            messagebox.showinfo("Busy", "Wait for the current evaluation to finish.",
                                parent=self.root)
            return
        if self.agent_mode.get() != AGENT_DQN:
            messagebox.showinfo(
                "DQN training",
                "The GUI trainer trains the DQN baseline. "
                "Use scripts/train_ntuple.py for N-tuple training.",
                parent=self.root,
            )
            return
        self.stop_autoplay("training")
        self.training = True
        self.stop_requested = False
        self.stop_flag.clear()
        self.train_thread = threading.Thread(target=self.train_loop, daemon=True)
        self.train_thread.start()
        self.log("DQN training started (8 environments, 4 updates per collection round).")
        self._sync_controls()
        self._update_status()

    def train_loop(self) -> None:
        env_count = 8
        update_ratio = 4
        episode_counter = self.episode
        reason = "finished"
        try:
            envs = [Game2048() for _ in range(env_count)]
            episode_scores = [0] * env_count
            episode_steps = [0] * env_count
            while not self.stop_flag.is_set():
                for index, game in enumerate(envs):
                    if self.stop_flag.is_set():
                        break
                    legal = game.legal_actions()
                    action_index = self.agent.act(game.board, legal, True)
                    action = ACTIONS[action_index]
                    state = encode(game.board)
                    step = game.move(action)
                    done = step.game_over
                    next_legal = [] if done else game.legal_actions()
                    next_mask = [candidate in next_legal for candidate in ACTIONS]
                    reward = float(step.score_delta)
                    self.agent.replay.add(
                        state, action_index, reward,
                        encode(game.board), done, next_mask,
                    )
                    self.agent.environment_steps += 1
                    self.agent.update_epsilon()
                    episode_scores[index] += step.score_delta
                    episode_steps[index] += 1
                    if done:
                        episode_counter += 1
                        record = {
                            "episode": episode_counter,
                            "score": episode_scores[index],
                            "tile": game.max_tile,
                            "moves": episode_steps[index],
                            "loss": self.agent.loss_avg,
                            "epsilon": self.agent.epsilon,
                        }
                        self.uiq.put(("episode", record))
                        if episode_counter % 10 == 0:
                            try:
                                self.agent.save(
                                    MODELS_DIR / "latest.pt",
                                    episode_counter,
                                    include_replay=True,
                                )
                            except Exception as error:
                                self.uiq.put(("log", f"AUTOSAVE FAILED: {error!r}", "error"))
                        if record["score"] > self.best_train_score:
                            self.best_train_score = record["score"]
                            try:
                                self.agent.save(
                                    MODELS_DIR / "best.pt",
                                    episode_counter,
                                    include_replay=False,
                                )
                            except Exception as error:
                                self.uiq.put(("log", f"BEST SAVE FAILED: {error!r}", "error"))
                        envs[index] = Game2048()
                        episode_scores[index] = 0
                        episode_steps[index] = 0
                    if self.agent.environment_steps % 500 == 0:
                        self.uiq.put((
                            "progress",
                            [row[:] for row in game.board],
                            episode_scores[index],
                            episode_steps[index],
                            action,
                            reward,
                            done,
                        ))
                if len(self.agent.replay) >= self.agent.warmup:
                    for _ in range(update_ratio):
                        if self.stop_flag.is_set():
                            break
                        loss = self.agent.learn()
                        if loss is None:
                            break
            reason = "stopped"
        except Exception:
            reason = "error"
            self.uiq.put(("train_error", traceback.format_exc()))
        finally:
            try:
                self.agent.save(MODELS_DIR / "latest.pt", episode_counter, include_replay=True)
            except Exception as error:
                self.uiq.put(("log", f"FINAL SAVE FAILED: {error!r}", "error"))
            self.uiq.put(("train_stopped", reason))

    def _on_training_episode(self, record: dict) -> None:
        self.episodes.append(record)
        if len(self.episodes) > 2000:
            del self.episodes[:-2000]
        self.episode = max(self.episode, int(record["episode"]))
        tag = "odd" if int(record["episode"]) % 2 else ""
        self.training_tree.insert(
            "", 0,
            values=(
                record["episode"],
                f"{record['score']:,}",
                record["tile"],
                record["moves"],
                f"{record['loss']:.2e}",
                f"{record.get('epsilon', 0.0):.3f}",
            ),
            tags=(tag,) if tag else (),
        )
        children = self.training_tree.get_children()
        for item in children[100:]:
            self.training_tree.delete(item)
        self.chart.set_episodes(self.episodes)
        self.log(
            f"EP {record['episode']} score={record['score']:,} max={record['tile']:,} "
            f"moves={record['moves']} loss={record['loss']:.3e}"
        )
        self.refresh_stats()
        self._sync_controls()

    def _on_training_progress(self, board, score, steps, action, reward, done) -> None:
        self.train_preview.set_state(board, animations=False)
        self.log(
            f"train step={steps} action={action} reward={reward:+.0f} terminal={done}"
        )

    def _on_train_stopped(self, reason: str) -> None:
        self.training = False
        self.stop_requested = False
        self.log(
            f"Training stopped ({reason})." if reason != "error"
            else "Training stopped after an error.",
            "error" if reason == "error" else "info",
        )
        self.refresh_stats()
        self._sync_controls()
        self._update_status()
        self._save_state()

    def _on_train_error(self, text: str) -> None:
        self.training = False
        self.stop_requested = False
        self.log("TRAIN ERROR:\n" + text, "error")
        self._sync_controls()
        self._update_status()

    # --------------------------------------------------------------- evaluation
    def evaluate(self) -> None:
        if self.training:
            messagebox.showinfo("Busy", "Stop DQN training before evaluating.",
                                parent=self.root)
            return
        if self.evaluating:
            return
        mode = self.agent_mode.get()
        if mode == AGENT_NTUPLE and self.ntuple is None:
            messagebox.showinfo("N-tuple required", "Load an N-tuple checkpoint first.",
                                parent=self.root)
            return
        total = int(self.settings["eval_games"])
        seed_start = int(self.settings["eval_seed"])
        settings = dict(self.settings)
        self.stop_autoplay("evaluation")
        self.evaluating = True
        self.eval_cancel.clear()
        self.eval_records = []
        self.eval_report = None
        self.eval_total = total
        for item in self.eval_tree.get_children():
            self.eval_tree.delete(item)
        self.eval_progress.configure(maximum=total, value=0)
        self.eval_progress_label.config(text=f"0/{total}")
        self.eval_labels["Agent"].config(text=mode)
        for key in ("Games", "Avg score", "Median", "P(2048)", "P(4096)",
                    "P(8192)", "Nodes/game", "Elapsed"):
            self.eval_labels[key].config(text="—")
        self.eval_labels["Games"].config(text=f"0/{total}")
        self.log(
            f"Evaluation started: {mode} · {total} games · seeds {seed_start}.."
            f"{seed_start + total - 1}."
        )
        thread = threading.Thread(
            target=self._eval_worker,
            args=(mode, total, settings, self.ntuple, seed_start),
            daemon=True,
        )
        thread.start()
        self._sync_controls()
        self._update_status()

    def _eval_worker(self, mode, total, settings, ntuple, seed_start) -> None:
        try:
            report = evaluate(
                mode,
                total,
                seed_start=seed_start,
                dqn_agent=self.agent if mode == AGENT_DQN else None,
                ntuple=ntuple,
                depth=int(settings["search_depth"]),
                node_budget=int(settings["node_budget"]),
                time_budget_ms=float(settings["time_budget_ms"]),
                cancel=self.eval_cancel,
                on_game=lambda index, requested, record: self.uiq.put(
                    ("eval_game", index, requested, record)
                ),
            )
            self.uiq.put(("eval_done", report))
        except Exception:
            self.uiq.put(("eval_error", traceback.format_exc()))

    def _on_eval_game(self, index: int, total: int, record) -> None:
        self.eval_records.append(record)
        tag = "odd" if index % 2 else ""
        self.eval_tree.insert(
            "", "end",
            values=(
                record.seed,
                f"{record.score:,}",
                f"{record.max_tile:,}",
                record.moves,
                f"{record.search_nodes:,}",
            ),
            tags=(tag,) if tag else (),
        )
        self.eval_progress.configure(maximum=total, value=index)
        self.eval_progress_label.config(text=f"{index}/{total}")
        self._update_eval_summary(self.eval_records, total)
        if index % 10 == 0 or index == total:
            successes = sum(item.max_tile >= 2048 for item in self.eval_records)
            self.log(f"EVAL progress {index}/{total} · P(2048)={successes}/{index}")
        self._update_status()

    def _update_eval_summary(self, records, total: int, report=None) -> None:
        if not records:
            self.eval_labels["Games"].config(text=f"0/{total}")
            return
        scores = [record.score for record in records]
        tiles = [record.max_tile for record in records]
        count = len(records)
        self.eval_labels["Games"].config(text=f"{count}/{total}")
        self.eval_labels["Avg score"].config(text=f"{statistics.fmean(scores):,.0f}")
        self.eval_labels["Median"].config(text=f"{statistics.median(scores):,.0f}")
        self.eval_labels["P(2048)"].config(
            text=f"{sum(tile >= 2048 for tile in tiles) / count * 100:.1f}%")
        self.eval_labels["P(4096)"].config(
            text=f"{sum(tile >= 4096 for tile in tiles) / count * 100:.1f}%")
        self.eval_labels["P(8192)"].config(
            text=f"{sum(tile >= 8192 for tile in tiles) / count * 100:.1f}%")
        self.eval_labels["Nodes/game"].config(
            text=f"{statistics.fmean(record.search_nodes for record in records):,.0f}")
        if report is not None:
            summary = report.summary()
            low, high = summary.get("p2048_wilson_95", (0.0, 0.0))
            self.eval_labels["P(2048)"].config(
                text=f"{summary['tile_rates']['2048'] * 100:.1f}% "
                     f"[{low * 100:.1f}–{high * 100:.1f}]")
            self.eval_labels["Elapsed"].config(
                text=f"{summary['elapsed_seconds']:.1f}s")
        else:
            self.eval_labels["Elapsed"].config(text="…")

    def _on_eval_done(self, report) -> None:
        self.evaluating = False
        self.eval_cancel.clear()
        self.eval_report = report
        self.eval_progress.configure(maximum=report.games_requested,
                                     value=report.games)
        label = f"{report.games}/{report.games_requested}"
        if report.cancelled:
            label += " cancelled"
        self.eval_progress_label.config(text=label)
        self._update_eval_summary(report.records, report.games_requested, report)
        summary = report.summary()
        if report.games:
            try:
                self.records = self.records.with_completed_games(
                    RECORDS_PATH,
                    ((record.score, record.max_tile) for record in report.records),
                )
                self.high_score = self.records.best_score
                self.high_tile = self.records.best_tile
            except Exception as error:
                self.log(f"Could not persist evaluation records: {error!r}", "warn")
            self.log(
                f"EVAL {report.agent}: {report.games} games · "
                f"avg={summary['mean_score']:,.1f} · "
                f"median={summary['median_score']:,.1f} · "
                f"P(2048)={summary['tile_rates']['2048'] * 100:.1f}% · "
                f"P(4096)={summary['tile_rates']['4096'] * 100:.1f}% · "
                f"P(8192)={summary['tile_rates']['8192'] * 100:.1f}% · "
                f"{summary['elapsed_seconds']:.1f}s"
            )
        else:
            self.log("Evaluation finished without completed games.", "warn")
        self._auto_save_report()
        self._sync_controls()
        self._update_status()
        self._save_state()

    def _on_eval_error(self, text: str) -> None:
        self.evaluating = False
        self.eval_cancel.clear()
        self.eval_progress_label.config(text="failed")
        self.log("EVALUATION FAILED:\n" + text, "error")
        self._sync_controls()
        self._update_status()

    def cancel_evaluation(self) -> None:
        if not self.evaluating:
            return
        self.eval_cancel.set()
        self.log("Evaluation cancel requested; finishing the current game…", "warn")
        self._update_status()

    def _auto_save_report(self) -> None:
        if self.eval_report is None or not self.eval_report.records:
            return
        slug = re.sub(r"[^a-z0-9]+", "_", self.eval_report.agent.lower()).strip("_") or "agent"
        name = (
            f"gui_eval_{slug}_{self.eval_report.games}_seed{self.eval_report.seed_start}_"
            f"{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        try:
            path = self.eval_report.save(REPORTS_DIR / name)
            self.last_eval_path = path
            self.log(f"Evaluation report saved: {path}")
        except Exception as error:
            self.log(f"Could not save evaluation report: {error}", "warn")

    def save_eval_report(self) -> None:
        if self.eval_report is None:
            return
        initial = (
            self.last_eval_path.name if self.last_eval_path is not None
            else f"evaluation_{self.eval_report.agent.lower().replace(' ', '_')}.json"
        )
        path = filedialog.asksaveasfilename(
            parent=self.root,
            initialdir=str(REPORTS_DIR),
            initialfile=initial,
            defaultextension=".json",
            filetypes=[("JSON report", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            saved = self.eval_report.save(path)
            self.last_eval_path = saved
            self.log(f"Evaluation report saved: {saved}")
        except Exception as error:
            messagebox.showerror("Save failed", str(error), parent=self.root)

    def open_reports(self) -> None:
        self._open_path(REPORTS_DIR)

    def _open_path(self, path: Path) -> None:
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as error:
            messagebox.showerror("Cannot open folder", str(error), parent=self.root)

    # --------------------------------------------------------------- event loop
    def poll(self) -> None:
        if self._closing:
            return
        try:
            while True:
                item = self.uiq.get_nowait()
                kind = item[0]
                if kind == "log":
                    self.log(item[1], item[2] if len(item) > 2 else "info")
                elif kind == "episode":
                    self._on_training_episode(item[1])
                elif kind == "progress":
                    self._on_training_progress(*item[1:])
                elif kind == "train_stopped":
                    self._on_train_stopped(item[1])
                elif kind == "train_error":
                    self._on_train_error(item[1])
                elif kind == "eval_game":
                    self._on_eval_game(item[1], item[2], item[3])
                elif kind == "eval_done":
                    self._on_eval_done(item[1])
                elif kind == "eval_error":
                    self._on_eval_error(item[1])
        except queue.Empty:
            pass
        if self._closing:
            return
        self.refresh_stats()
        self._update_status()
        try:
            self.poll_job = self.root.after(80, self.poll)
        except tk.TclError:
            pass

    # --------------------------------------------------------------- settings dialog
    def _center_over_root(self, win: tk.Toplevel, width: int, height: int) -> None:
        """Place a dialog over the main window, clamped to the screen."""
        try:
            self.root.update_idletasks()
            root_x = self.root.winfo_rootx()
            root_y = self.root.winfo_rooty()
            root_w = self.root.winfo_width()
            root_h = self.root.winfo_height()
            x = root_x + (root_w - width) // 2
            y = root_y + (root_h - height) // 2
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            x = max(0, min(x, screen_w - width))
            y = max(0, min(y, screen_h - height))
            win.geometry(f"{width}x{height}+{x}+{y}")
        except tk.TclError:
            win.geometry(f"{width}x{height}")

    def open_settings(self) -> None:
        if self.training or self.evaluating:
            messagebox.showinfo(
                "Busy",
                "Stop training or evaluation before changing settings.",
                parent=self.root,
            )
            return
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.configure(bg=theme.BG)
        self._center_over_root(win, 580, 620)
        win.transient(self.root)
        win.resizable(False, False)
        win.grab_set()

        tk.Label(win, text="SETTINGS", bg=theme.BG, fg=theme.FG,
                 font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=18, pady=(16, 2))
        tk.Label(win, text="Search budgets, autoplay and evaluation defaults.",
                 bg=theme.BG, fg=theme.MUTED, font=("Consolas", 8)).pack(
            anchor="w", padx=18, pady=(0, 10))

        form = tk.Frame(win, bg=theme.BG)
        form.pack(fill="both", expand=True, padx=18)
        form.grid_columnconfigure(0, weight=1)

        depth_var = tk.IntVar(value=int(self.settings["search_depth"]))
        node_var = tk.IntVar(value=int(self.settings["node_budget"]))
        time_var = tk.DoubleVar(value=float(self.settings["time_budget_ms"]))
        autoplay_var = tk.IntVar(value=int(self.settings["autoplay_ms"]))
        restart_var = tk.BooleanVar(value=bool(self.settings["auto_restart"]))
        games_var = tk.IntVar(value=int(self.settings["eval_games"]))
        seed_var = tk.IntVar(value=int(self.settings["eval_seed"]))
        anim_var = tk.BooleanVar(value=bool(self.settings["animations"]))
        log_var = tk.IntVar(value=int(self.settings["max_log_lines"]))

        def section(title: str, row: int):
            frame = tk.LabelFrame(form, text=f" {title} ", bg=theme.BG, fg=theme.FG,
                                  padx=10, pady=8)
            frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
            frame.grid_columnconfigure(1, weight=1)
            return frame

        def add_spin(parent, row, label, variable, low, high, increment):
            tk.Label(parent, text=label, bg=theme.BG, fg=theme.FG,
                     font=("Segoe UI", 9)).grid(
                row=row, column=0, sticky="w", pady=3)
            ttk.Spinbox(parent, from_=low, to=high, increment=increment,
                        textvariable=variable, width=14).grid(
                row=row, column=1, sticky="e", pady=3)

        search = section("SEARCH", 0)
        add_spin(search, 0, "Depth limit (player decisions)", depth_var, 1, 8, 1)
        add_spin(search, 1, "Node budget per move", node_var, 1_000, 10_000_000, 50_000)
        add_spin(search, 2, "Time budget per move (ms)", time_var, 1, 2_000, 5)

        autoplay = section("AUTOPLAY", 1)
        add_spin(autoplay, 0, "Delay between moves (ms)", autoplay_var, 10, 2_000, 10)
        ttk.Checkbutton(autoplay, text="Restart automatically after game over",
                        variable=restart_var).grid(row=1, column=0, columnspan=2,
                                                   sticky="w", pady=(4, 0))

        evaluation = section("EVALUATION", 2)
        add_spin(evaluation, 0, "Games per evaluation", games_var, 1, 1_000, 10)
        add_spin(evaluation, 1, "First seed", seed_var, 0, 2_147_483_647, 1_000)

        interface = section("INTERFACE", 3)
        ttk.Checkbutton(interface, text="Animate tile spawns and board flashes",
                        variable=anim_var).grid(row=0, column=0, columnspan=2,
                                                sticky="w", pady=3)
        add_spin(interface, 1, "Debug log lines kept", log_var, 100, 5_000, 100)

        buttons = tk.Frame(win, bg=theme.BG)
        buttons.pack(fill="x", padx=18, pady=(4, 16))

        def reset_defaults():
            depth_var.set(SETTINGS_SPEC["search_depth"][3])
            node_var.set(SETTINGS_SPEC["node_budget"][3])
            time_var.set(SETTINGS_SPEC["time_budget_ms"][3])
            autoplay_var.set(SETTINGS_SPEC["autoplay_ms"][3])
            restart_var.set(SETTINGS_SPEC["auto_restart"][3])
            games_var.set(SETTINGS_SPEC["eval_games"][3])
            seed_var.set(SETTINGS_SPEC["eval_seed"][3])
            anim_var.set(SETTINGS_SPEC["animations"][3])
            log_var.set(SETTINGS_SPEC["max_log_lines"][3])

        def apply():
            try:
                updates = {
                    "search_depth": depth_var.get(),
                    "node_budget": node_var.get(),
                    "time_budget_ms": time_var.get(),
                    "autoplay_ms": autoplay_var.get(),
                    "auto_restart": restart_var.get(),
                    "eval_games": games_var.get(),
                    "eval_seed": seed_var.get(),
                    "animations": anim_var.get(),
                    "max_log_lines": log_var.get(),
                }
                for key, value in updates.items():
                    self.settings[key] = _coerce_setting(key, value)
            except Exception as error:
                messagebox.showerror("Invalid settings", str(error), parent=win)
                return
            self._apply_solver_settings()
            self.eval_button.configure(text=f"EVAL ×{self.settings['eval_games']}")
            self._save_state()
            self.log(
                "Settings updated: "
                f"depth={self.settings['search_depth']} "
                f"budget={self.settings['node_budget']:,} "
                f"time={self.settings['time_budget_ms']:.0f}ms "
                f"autoplay={self.settings['autoplay_ms']}ms "
                f"eval={self.settings['eval_games']}"
            )
            win.destroy()
            self.render(full=True)

        ttk.Button(buttons, text="RESET DEFAULTS", command=reset_defaults,
                   takefocus=False).pack(side="left")
        ttk.Button(buttons, text="CANCEL", command=win.destroy,
                   takefocus=False).pack(side="right", padx=(6, 0))
        ttk.Button(buttons, text="APPLY", style="Accent.TButton", command=apply,
                   takefocus=False).pack(side="right")
        win.protocol("WM_DELETE_WINDOW", win.destroy)

    # --------------------------------------------------------------- model manager
    def open_models(self) -> None:
        if self.training or self.evaluating:
            messagebox.showinfo("Busy", "Stop training or evaluation first.",
                                parent=self.root)
            return
        win = tk.Toplevel(self.root)
        win.title("Model manager")
        win.configure(bg=theme.BG)
        self._center_over_root(win, 760, 500)
        win.transient(self.root)
        win.grab_set()

        tk.Label(win, text="SAVED DQN MODELS", bg=theme.BG, fg=theme.FG,
                 font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=16, pady=(14, 2))
        tk.Label(win, text=str(MODELS_DIR), bg=theme.BG, fg=theme.MUTED,
                 font=("Consolas", 8)).pack(anchor="w", padx=16, pady=(0, 8))

        frame = tk.Frame(win, bg=theme.BG)
        frame.pack(fill="both", expand=True, padx=16)
        tree = ttk.Treeview(frame, columns=("name", "size", "modified"),
                            show="headings", selectmode="browse")
        for column, title, width in (
            ("name", "FILE", 320),
            ("size", "SIZE", 100),
            ("modified", "MODIFIED", 160),
        ):
            tree.heading(column, text=title)
            tree.column(column, width=width, anchor="w" if column == "name" else "center")
        tree.tag_configure("odd", background=theme.PANEL_ALT)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        include_replay = tk.BooleanVar(value=True)
        options = tk.Frame(win, bg=theme.BG)
        options.pack(fill="x", padx=16, pady=(8, 0))
        ttk.Checkbutton(options, text="Include replay buffer when saving",
                        variable=include_replay).pack(side="left")

        def refresh():
            for item in tree.get_children():
                tree.delete(item)
            for index, path in enumerate(sorted(MODELS_DIR.glob("*.pt"))):
                try:
                    stat = path.stat()
                    tree.insert(
                        "", "end", iid=str(path),
                        values=(
                            path.name,
                            f"{stat.st_size / 1024 / 1024:.2f} MB",
                            time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
                        ),
                        tags=("odd",) if index % 2 else (),
                    )
                except OSError:
                    continue

        def selected() -> Path | None:
            selection = tree.selection()
            return Path(selection[0]) if selection else None

        def save_named():
            name = simpledialog.askstring("Save model", "Model name:", parent=win)
            if not name:
                return
            safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._")
            if not safe:
                messagebox.showwarning("Invalid name", "Enter a valid model name.",
                                       parent=win)
                return
            path = MODELS_DIR / f"{safe}.pt"
            if path.exists() and not messagebox.askyesno(
                "Overwrite", f"{path.name} already exists. Replace it?", parent=win
            ):
                return
            try:
                self.agent.save(path, self.episode, include_replay=include_replay.get())
                self.current_model_name = path.name
                self._save_state()
                self.log(f"Saved model: {path.name}")
                refresh()
            except Exception as error:
                messagebox.showerror("Save failed", str(error), parent=win)

        def load_selected():
            path = selected()
            if path is None:
                return
            if self.training or self.evaluating:
                messagebox.showinfo("Busy", "Stop training or evaluation first.",
                                    parent=win)
                return
            try:
                episodes = self.agent.load(path)
                self.episode = int(episodes)
                self.current_model_name = path.name
                self.log(
                    f"Loaded model: {path.name} · episodes={episodes} · "
                    f"steps={self.agent.steps} · arch={self.agent.model_architecture}"
                )
                self._save_state()
                self.render(full=True)
                self.refresh_stats()
            except Exception as error:
                messagebox.showerror("Load failed", str(error), parent=win)

        def delete_selected():
            path = selected()
            if path is None:
                return
            if messagebox.askyesno("Delete model", f"Delete {path.name}?", parent=win):
                try:
                    path.unlink()
                    self.log(f"Deleted model: {path.name}")
                    refresh()
                except Exception as error:
                    messagebox.showerror("Delete failed", str(error), parent=win)

        def export_selected():
            path = selected()
            if path is None:
                return
            target = filedialog.asksaveasfilename(
                parent=win,
                defaultextension=".pt",
                initialfile=path.name,
                filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
            )
            if not target:
                return
            try:
                import shutil
                shutil.copy2(path, target)
                self.log(f"Exported model: {target}")
            except Exception as error:
                messagebox.showerror("Export failed", str(error), parent=win)

        tree.bind("<Double-1>", lambda _event: load_selected())

        buttons = tk.Frame(win, bg=theme.BG)
        buttons.pack(fill="x", padx=16, pady=12)
        ttk.Button(buttons, text="SAVE CURRENT", command=save_named,
                   takefocus=False).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="LOAD", style="Accent.TButton", command=load_selected,
                   takefocus=False).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="DELETE", style="Danger.TButton", command=delete_selected,
                   takefocus=False).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="EXPORT", command=export_selected,
                   takefocus=False).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="OPEN FOLDER", command=lambda: self._open_path(MODELS_DIR),
                   takefocus=False).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="CLOSE", command=win.destroy,
                   takefocus=False).pack(side="right")
        refresh()
        win.protocol("WM_DELETE_WINDOW", win.destroy)

    # --------------------------------------------------------------- n-tuple dialog
    def open_ntuple(self) -> None:
        if self.training or self.evaluating:
            messagebox.showinfo("Busy", "Stop training or evaluation first.",
                                parent=self.root)
            return
        win = tk.Toplevel(self.root)
        win.title("N-tuple evaluator")
        win.configure(bg=theme.BG)
        self._center_over_root(win, 680, 430)
        win.transient(self.root)
        win.grab_set()

        tk.Label(win, text="N-TUPLE EVALUATOR", bg=theme.BG, fg=theme.FG,
                 font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=16, pady=(14, 2))
        tk.Label(
            win,
            text="D4-symmetric six-cell tables (format v4). The Python Expectimax "
                 "fallback is used when the optional native library is unavailable.",
            bg=theme.BG, fg=theme.MUTED, font=("Consolas", 8),
            wraplength=620, justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 10))

        info = tk.Frame(win, bg=theme.PANEL, highlightthickness=1,
                        highlightbackground=theme.BORDER, padx=12, pady=10)
        info.pack(fill="x", padx=16)
        info.grid_columnconfigure(1, weight=1)
        keys = ("File", "Evaluator backend", "Tables", "Search backend",
                "Depth / budget", "Native library")
        labels: dict[str, tk.Label] = {}
        for row, key in enumerate(keys):
            tk.Label(info, text=key, bg=theme.PANEL, fg=theme.MUTED,
                     font=("Consolas", 8), anchor="w").grid(
                row=row, column=0, sticky="w", padx=(0, 12), pady=2)
            value = tk.Label(info, text="—", bg=theme.PANEL, fg=theme.FG,
                             font=("Consolas", 8), anchor="w")
            value.grid(row=row, column=1, sticky="w", pady=2)
            labels[key] = value

        def refresh():
            if self.ntuple is None:
                labels["File"].config(text="not loaded")
                labels["Evaluator backend"].config(text="—")
                labels["Tables"].config(text="—")
                labels["Search backend"].config(text="—")
                labels["Depth / budget"].config(text="—")
            else:
                labels["File"].config(
                    text=os.path.basename(self.last_ntuple_path)
                    if self.last_ntuple_path else "loaded")
                labels["Evaluator backend"].config(text=self.ntuple.backend)
                labels["Tables"].config(
                    text=f"{self.ntuple.tables.nbytes / 1024 ** 2:.1f} MiB · "
                         f"shape {self.ntuple.tables.shape}")
                labels["Search backend"].config(
                    text=self.ntuple_solver.backend if self.ntuple_solver else "—")
                labels["Depth / budget"].config(
                    text=f"{self.settings['search_depth']}d · "
                         f"{self.settings['node_budget']:,} nodes · "
                         f"{self.settings['time_budget_ms']:.0f} ms")
            native = "available" if self.ntuple_solver and self.ntuple_solver.native_mode == "ntuple" \
                else "unavailable (Python fallback)"
            labels["Native library"].config(
                text=native,
                fg=theme.ACC if native == "available" else theme.WARN,
            )

        def load():
            path = filedialog.askopenfilename(
                parent=win,
                initialdir=str(MODELS_DIR),
                filetypes=[("DESTROYER N-tuple checkpoint", "*.npz"),
                           ("All files", "*.*")],
            )
            if not path:
                return
            self._load_ntuple_path(Path(path), parent=win)
            refresh()

        def unload():
            if self.ntuple is None:
                return
            if not messagebox.askyesno(
                "Unload N-tuple",
                "Unload the evaluator and switch back to DQN?",
                parent=win,
            ):
                return
            self._unload_ntuple()
            refresh()

        buttons = tk.Frame(win, bg=theme.BG)
        buttons.pack(fill="x", padx=16, pady=14)
        ttk.Button(buttons, text="LOAD .NPZ…", style="Accent.TButton",
                   command=load, takefocus=False).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="UNLOAD", style="Danger.TButton",
                   command=unload, takefocus=False).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="OPEN MODELS FOLDER",
                   command=lambda: self._open_path(MODELS_DIR),
                   takefocus=False).pack(side="left")
        ttk.Button(buttons, text="CLOSE", command=win.destroy,
                   takefocus=False).pack(side="right")

        def close():
            self.ntuple_dialog_refresh = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", close)
        self.ntuple_dialog_refresh = refresh
        refresh()

    def _load_ntuple_path(self, path: Path, parent=None) -> None:
        if self.training or self.evaluating:
            messagebox.showinfo("Busy", "Stop training or evaluation first.",
                                parent=parent or self.root)
            return
        try:
            candidate = NtupleValue.load(path)
            solver = self._make_solver(AGENT_NTUPLE, candidate)
        except Exception as error:
            messagebox.showerror("N-tuple load failed", str(error),
                                 parent=parent or self.root)
            return
        old_solver = self.ntuple_solver
        self.ntuple = candidate
        self.ntuple_solver = solver
        self.last_ntuple_path = str(path)
        self.agent_mode.set(AGENT_NTUPLE)
        self.current_mode = AGENT_NTUPLE
        if old_solver is not None:
            old_solver.close()
        self.log(
            f"Loaded N-tuple: {path.name} · evaluator={candidate.backend} · "
            f"tables={candidate.tables.nbytes / 1024 ** 2:.1f} MiB · "
            f"search={solver.backend}"
        )
        if solver.native_mode != "ntuple":
            self.log(
                "Native C++ search is unavailable; using the Python Expectimax "
                "fallback (functional, but slower).",
                "warn",
            )
        self._save_state()
        self.render(full=True)
        self._sync_controls()
        if callable(self.ntuple_dialog_refresh):
            self.ntuple_dialog_refresh()

    def _unload_ntuple(self) -> None:
        old_solver = self.ntuple_solver
        self.ntuple = None
        self.ntuple_solver = None
        if old_solver is not None:
            old_solver.close()
        if self.agent_mode.get() == AGENT_NTUPLE:
            self.agent_mode.set(AGENT_DQN)
            self.current_mode = AGENT_DQN
        self.last_ntuple_path = None
        self.log("N-tuple evaluator unloaded.")
        self._save_state()
        self.render(full=True)
        self._sync_controls()
        if callable(self.ntuple_dialog_refresh):
            self.ntuple_dialog_refresh()

    # --------------------------------------------------------------- shutdown
    def on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        try:
            self.stop_flag.set()
            self.eval_cancel.set()
            self.stop_autoplay("closing")
        except Exception:
            pass
        for job in (getattr(self, "poll_job", None), getattr(self, "auto_job", None)):
            if job is not None:
                try:
                    self.root.after_cancel(job)
                except Exception:
                    pass
        self.poll_job = None
        self.auto_job = None
        for view in (getattr(self, "board_view", None),
                     getattr(self, "train_preview", None)):
            if view is not None:
                try:
                    view.stop_animation()
                except Exception:
                    pass
        if self.game.game_over():
            self._finish_game_record()
        self._save_state()
        thread = self.train_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
            if not thread.is_alive():
                try:
                    self.agent.save(MODELS_DIR / "latest.pt", self.episode,
                                    include_replay=True)
                except Exception:
                    pass
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
