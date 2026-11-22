"""Custom Tk widgets used by the DESTROYER desktop lab."""

from __future__ import annotations

import time
import tkinter as tk
from typing import Iterable

from . import theme


def rounded_rect(canvas: tk.Canvas, x1, y1, x2, y2, radius=8, **kwargs):
    """Draw a smooth rounded rectangle and return its canvas item id."""

    radius = max(0.0, min(float(radius), abs(x2 - x1) / 2.0, abs(y2 - y1) / 2.0))
    points = (
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    )
    return canvas.create_polygon(points, smooth=True, splinesteps=12, **kwargs)


class BoardView(tk.Canvas):
    """Responsive 4x4 board renderer with spawn glow, flash and overlays."""

    def __init__(self, master, on_new_game=None, on_continue=None, **kwargs):
        super().__init__(
            master,
            bg=theme.BG,
            highlightthickness=0,
            bd=0,
            **kwargs,
        )
        self.on_new_game = on_new_game
        self.on_continue = on_continue
        self._board = [[0] * 4 for _ in range(4)]
        self._spawn: tuple[int, int, int] | None = None
        self._spawn_time = 0.0
        self._overlay: str | None = None
        self._flash: str | None = None
        self._flash_time = 0.0
        self._flash_duration = 0.35
        self._animations = True
        self._animation_job = None
        self._hover_tag: str | None = None
        self._overlay_button_colors: dict[str, tuple[str, str]] = {}

        self.bind("<Configure>", lambda _event: self.redraw())
        self.tag_bind("overlay_new", "<Button-1>", lambda _event: self._click_new())
        self.tag_bind("overlay_continue", "<Button-1>", lambda _event: self._click_continue())
        for tag in ("overlay_new", "overlay_continue"):
            self.tag_bind(tag, "<Enter>", lambda _event, t=tag: self._set_hover(t, True))
            self.tag_bind(tag, "<Leave>", lambda _event, t=tag: self._set_hover(t, False))

    # ------------------------------------------------------------------ state
    def set_state(
        self,
        board: Iterable[Iterable[int]],
        *,
        spawn: tuple[int, int, int] | None = None,
        overlay: str | None = None,
        flash: str | None = None,
        animations: bool | None = None,
    ) -> None:
        self._board = [list(row) for row in board]
        self._overlay = overlay
        if animations is not None:
            self._animations = bool(animations)
        if spawn is not None:
            self._spawn = tuple(spawn)
            self._spawn_time = time.perf_counter()
        else:
            self._spawn = None
        if flash is not None:
            self._flash = flash
            self._flash_time = time.perf_counter()
        else:
            self._flash = None
        self.redraw()
        if self._animations and (self._spawn is not None or self._flash is not None):
            self._schedule_animation()

    def flash(self, color: str = theme.DANGER, duration: float = 0.35) -> None:
        self._flash = color
        self._flash_time = time.perf_counter()
        self._flash_duration = max(0.05, float(duration))
        self.redraw()
        if self._animations:
            self._schedule_animation()
        else:
            self.after(int(self._flash_duration * 1000), self._clear_flash)

    def _clear_flash(self):
        self._flash = None
        self.redraw()

    def _schedule_animation(self) -> None:
        if self._animation_job is not None:
            try:
                self.after_cancel(self._animation_job)
            except tk.TclError:
                pass
            self._animation_job = None
        self._animation_job = self.after(40, self._animate_tick)

    def stop_animation(self) -> None:
        if self._animation_job is not None:
            try:
                self.after_cancel(self._animation_job)
            except tk.TclError:
                pass
            self._animation_job = None

    def _animate_tick(self):
        self._animation_job = None
        now = time.perf_counter()
        spawn_active = self._spawn is not None and now - self._spawn_time < 0.55
        flash_active = self._flash is not None and now - self._flash_time < self._flash_duration
        if not spawn_active:
            self._spawn = None
        if not flash_active:
            self._flash = None
        self.redraw()
        if spawn_active or flash_active:
            self._animation_job = self.after(40, self._animate_tick)

    # ------------------------------------------------------------------ drawing
    def redraw(self) -> None:
        self.delete("all")
        width = max(2, self.winfo_width())
        height = max(2, self.winfo_height())
        if width < 40 or height < 40:
            return

        size = min(width, height)
        margin = size * 0.038
        gap = max(4.0, size * 0.018)
        cell = (size - 2.0 * margin - 3.0 * gap) / 4.0
        origin_x = (width - size) / 2.0 + margin
        origin_y = (height - size) / 2.0 + margin
        radius = max(6.0, cell * 0.10)

        rounded_rect(
            self, 1, 1, width - 1, height - 1,
            radius=max(10.0, size * 0.035),
            fill=theme.BOARD_BG,
            outline=theme.BORDER,
            width=1,
        )

        spawn_age = time.perf_counter() - self._spawn_time if self._spawn is not None else 99.0
        for row in range(4):
            for column in range(4):
                x = origin_x + column * (cell + gap)
                y = origin_y + row * (cell + gap)
                value = int(self._board[row][column])
                if value <= 0:
                    rounded_rect(
                        self, x, y, x + cell, y + cell,
                        radius=radius,
                        fill=theme.EMPTY_TILE,
                        outline=theme.EMPTY_TILE_OUTLINE,
                        width=1,
                    )
                    continue

                rounded_rect(
                    self, x + 2, y + 3, x + cell + 2, y + cell + 3,
                    radius=radius,
                    fill="#080e14",
                    outline="",
                )
                rounded_rect(
                    self, x, y, x + cell, y + cell,
                    radius=radius,
                    fill=theme.tile_color(value),
                    outline=theme.tile_outline(value),
                    width=1,
                )
                font_size = theme.tile_font_size(value, cell)
                self.create_text(
                    x + cell / 2.0,
                    y + cell / 2.0 + 1,
                    text=str(value),
                    fill=theme.tile_text_color(value),
                    font=("Segoe UI", font_size, "bold"),
                )

                if (
                    self._animations
                    and self._spawn is not None
                    and (row, column) == (self._spawn[0], self._spawn[1])
                    and spawn_age < 0.55
                ):
                    fraction = min(1.0, spawn_age / 0.55)
                    glow = theme.blend(theme.ACC, theme.tile_color(value), fraction)
                    rounded_rect(
                        self,
                        x - 2, y - 2, x + cell + 2, y + cell + 2,
                        radius=radius + 2,
                        fill="",
                        outline=glow,
                        width=max(2, int(cell * 0.035)),
                    )

        if self._flash is not None:
            age = time.perf_counter() - self._flash_time
            if age < self._flash_duration:
                fraction = min(1.0, age / self._flash_duration)
                rounded_rect(
                    self, 2, 2, width - 2, height - 2,
                    radius=max(10.0, size * 0.035),
                    fill="",
                    outline=theme.blend(self._flash, theme.BORDER, fraction),
                    width=3,
                )

        if self._overlay is not None:
            self._draw_overlay(width, height)

    def _draw_overlay(self, width: int, height: int) -> None:
        self.create_rectangle(
            0, 0, width, height,
            fill="#05090e",
            outline="",
            tags=("overlay",),
        )
        if self._overlay == "gameover":
            title = "GAME OVER"
            title_color = theme.DANGER
            subtitle = "No legal moves remain."
            buttons = (
                ("NEW GAME", "overlay_new", theme.ACC_DARK, theme.ACC_HI, "#eafff4"),
            )
        else:
            title = "2048 REACHED"
            title_color = theme.GOLD
            subtitle = "Target tile unlocked. Keep going?"
            buttons = (
                ("KEEP PLAYING", "overlay_continue", theme.ACC_DARK, theme.ACC_HI, "#eafff4"),
                ("NEW GAME", "overlay_new", theme.PANEL_HI, theme.BORDER_HI, theme.FG),
            )
        self._overlay_button_colors = {
            tag: (base, hover) for _label, tag, base, hover, _foreground in buttons
        }

        title_size = max(16, int(min(width, height) * 0.075))
        center_y = height * 0.40
        button_width = min(230.0, width * 0.62)
        button_height = max(34.0, min(48.0, height * 0.085))
        gap = 10.0
        top = center_y + title_size * 2.0
        block_height = len(buttons) * button_height + (len(buttons) - 1) * gap
        card_width = min(width - 24.0, max(button_width + 48.0, 250.0))
        card_top = center_y - title_size * 0.95
        card_bottom = top + block_height + 20.0
        rounded_rect(
            self,
            (width - card_width) / 2.0, card_top,
            (width + card_width) / 2.0, card_bottom,
            radius=12,
            fill=theme.PANEL,
            outline=theme.BORDER_HI,
            width=1,
            tags=("overlay", "overlay_card"),
        )

        self.create_text(
            width / 2.0, center_y,
            text=title,
            fill=title_color,
            font=("Segoe UI", title_size, "bold"),
            tags=("overlay",),
        )
        self.create_text(
            width / 2.0, center_y + title_size * 1.15,
            text=subtitle,
            fill=theme.FG,
            font=("Consolas", max(9, int(title_size * 0.36))),
            tags=("overlay",),
        )

        for index, (label, tag, fill, hover_fill, foreground) in enumerate(buttons):
            y1 = top + index * (button_height + gap)
            y2 = y1 + button_height
            x1 = width / 2.0 - button_width / 2.0
            x2 = width / 2.0 + button_width / 2.0
            rounded_rect(
                self, x1, y1, x2, y2,
                radius=button_height * 0.28,
                fill=hover_fill if self._hover_tag == tag else fill,
                outline=theme.BORDER_HI,
                width=1,
                tags=(tag, f"{tag}_bg", "overlay"),
            )
            self.create_text(
                width / 2.0, (y1 + y2) / 2.0,
                text=label,
                fill=foreground,
                font=("Segoe UI", max(9, int(button_height * 0.30)), "bold"),
                tags=(tag, "overlay"),
            )

    def _set_hover(self, tag: str, active: bool) -> None:
        self._hover_tag = tag if active else None
        try:
            self.configure(cursor="hand2" if active else "")
        except tk.TclError:
            pass
        if active:
            colors = self._overlay_button_colors.get(tag)
            if colors is not None:
                self.itemconfigure(f"{tag}_bg", fill=colors[1])
        else:
            for other, colors in self._overlay_button_colors.items():
                self.itemconfigure(f"{other}_bg", fill=colors[0])

    def _click_new(self):
        if self.on_new_game is not None:
            self.on_new_game()

    def _click_continue(self):
        if self.on_continue is not None:
            self.on_continue()


class Sparkline(tk.Canvas):
    """Score history with a moving average and a highlighted best episode."""

    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            bg="#0b1118",
            highlightthickness=0,
            bd=0,
            **kwargs,
        )
        self._episodes: list[dict] = []
        self.bind("<Configure>", lambda _event: self.redraw())

    def set_episodes(self, episodes: list[dict]) -> None:
        self._episodes = list(episodes[-140:])
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        width = max(320, self.winfo_width())
        height = max(120, self.winfo_height())
        self.create_rectangle(0, 0, width, height, fill="#0b1118", outline="")

        episodes = self._episodes
        if len(episodes) < 2:
            self.create_text(
                width / 2.0, height / 2.0,
                text="Train episodes to populate the chart",
                fill=theme.DIM,
                font=("Consolas", 9),
            )
            return

        scores = [float(item.get("score", 0)) for item in episodes]
        window = min(10, len(scores))
        averages: list[float] = []
        running = 0.0
        for index, score in enumerate(scores):
            running += score
            if index >= window:
                running -= scores[index - window]
            averages.append(running / min(window, index + 1))

        left, right = 42.0, width - 12.0
        top, bottom = 26.0, height - 20.0
        low = min(min(scores), min(averages))
        high = max(max(scores), max(averages))
        if high - low < 1.0:
            high = low + 1.0
        padding = (high - low) * 0.08
        low -= padding
        high += padding

        for step in range(5):
            fraction = step / 4.0
            y = top + (bottom - top) * fraction
            value = high - (high - low) * fraction
            self.create_line(left, y, right, y, fill="#1b2735")
            self.create_text(
                left - 6, y,
                text=f"{value:,.0f}",
                fill=theme.DIM,
                anchor="e",
                font=("Consolas", 7),
            )

        count = len(scores)
        x_step = (right - left) / max(1, count - 1)
        score_points = []
        average_points = []
        best_index = max(range(count), key=lambda index: scores[index])
        for index, score in enumerate(scores):
            x = left + index * x_step
            y = bottom - (score - low) / (high - low) * (bottom - top)
            score_points.extend((x, y))
            y_avg = bottom - (averages[index] - low) / (high - low) * (bottom - top)
            average_points.extend((x, y_avg))

        if len(score_points) >= 4:
            self.create_line(*score_points, fill=theme.BLUE, width=2, smooth=False)
        if len(average_points) >= 4:
            self.create_line(*average_points, fill=theme.ACC, width=2, smooth=False)

        best_x = left + best_index * x_step
        best_y = bottom - (scores[best_index] - low) / (high - low) * (bottom - top)
        self.create_oval(
            best_x - 3, best_y - 3, best_x + 3, best_y + 3,
            fill=theme.GOLD, outline="",
        )
        self.create_text(
            best_x, best_y - 10,
            text=f"{scores[best_index]:,.0f}",
            fill=theme.GOLD,
            font=("Consolas", 7, "bold"),
        )

        self.create_rectangle(10, 8, 18, 12, fill=theme.BLUE, outline="")
        self.create_text(23, 10, text="score", fill=theme.MUTED, anchor="w", font=("Consolas", 7))
        self.create_rectangle(68, 8, 76, 12, fill=theme.ACC, outline="")
        self.create_text(81, 10, text="avg", fill=theme.MUTED, anchor="w", font=("Consolas", 7))
        self.create_text(
            right, 10,
            text=f"last {scores[-1]:,.0f}  ·  best {max(scores):,.0f}",
            fill=theme.MUTED,
            anchor="e",
            font=("Consolas", 7),
        )


class MetricCard(tk.Frame):
    """Small dark card with an uppercase caption and a large value."""

    def __init__(self, master, caption: str, value: str = "—", accent: str = theme.FG, **kwargs):
        super().__init__(
            master,
            bg=theme.PANEL,
            highlightthickness=1,
            highlightbackground=theme.BORDER,
            bd=0,
            **kwargs,
        )
        self._accent = accent
        self._value = value
        tk.Label(
            self,
            text=caption.upper(),
            bg=theme.PANEL,
            fg=theme.MUTED,
            font=("Consolas", 8),
            anchor="w",
        ).pack(fill="x", padx=9, pady=(6, 0))
        self.value_label = tk.Label(
            self,
            text=value,
            bg=theme.PANEL,
            fg=accent,
            font=("Segoe UI", 17, "bold"),
            anchor="w",
        )
        self.value_label.pack(fill="x", padx=9, pady=(0, 6))

    def set(self, value, flash: bool = False) -> None:
        text = str(value)
        if text != self._value:
            self._value = text
            self.value_label.config(text=text)
        if flash:
            self.value_label.config(fg=theme.ACC)
            try:
                self.after(260, lambda: self.value_label.config(fg=self._accent))
            except tk.TclError:
                pass


class Tooltip:
    """Lightweight hover tooltip."""

    def __init__(self, widget, text: str, delay: int = 450):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._job = None
        self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Destroy>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        try:
            self._job = self.widget.after(self.delay, self._show)
        except tk.TclError:
            pass

    def _cancel(self):
        if self._job is not None:
            try:
                self.widget.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None

    def _show(self):
        self._job = None
        if self._tip is not None or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 10
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        except tk.TclError:
            return
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        try:
            tip.attributes("-topmost", True)
        except tk.TclError:
            pass
        label = tk.Label(
            tip,
            text=self.text,
            bg=theme.PANEL_HI,
            fg=theme.FG,
            font=("Segoe UI", 8),
            padx=8,
            pady=4,
            justify="left",
            highlightthickness=1,
            highlightbackground=theme.BORDER_HI,
        )
        label.pack()
        tip.geometry(f"+{x}+{y}")
        self._tip = tip

    def _hide(self, _event=None):
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None
