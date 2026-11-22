"""Dark palette, tile colours and ttk style setup for the desktop UI."""

from __future__ import annotations

from tkinter import ttk

BG = "#0b1017"
BG_ALT = "#0f1620"
PANEL = "#151f2b"
PANEL_ALT = "#1b2735"
PANEL_HI = "#223243"
BORDER = "#26384a"
BORDER_HI = "#36506a"
FG = "#e9f0f6"
MUTED = "#8fa3b8"
DIM = "#5d7186"
ACC = "#4fd48c"
ACC_DARK = "#1f6f4a"
ACC_HI = "#2b8f61"
BLUE = "#57a8ff"
BLUE_DARK = "#1c4f80"
WARN = "#f2b84b"
DANGER = "#ef7a6b"
DANGER_DARK = "#5a2622"
GOLD = "#e8c34a"

EMPTY_TILE = "#1b2836"
EMPTY_TILE_OUTLINE = "#243444"
BOARD_BG = "#0e1721"

TILE_COLORS = {
    2: "#eee4da",
    4: "#ede0c8",
    8: "#f2b179",
    16: "#f59563",
    32: "#f67c5f",
    64: "#f65e3b",
    128: "#edcf72",
    256: "#edcc61",
    512: "#edc850",
    1024: "#edc53f",
    2048: "#edc22e",
    4096: "#3c3a32",
    8192: "#332f28",
    16384: "#29251f",
    32768: "#201c17",
    65536: "#171410",
}
TILE_OUTLINES = {
    2: "#f7efe6",
    4: "#f6ecda",
    8: "#f8c296",
    16: "#f8a87c",
    32: "#f99078",
    64: "#f97756",
    128: "#f4dd90",
    256: "#f4da80",
    512: "#f4d56e",
    1024: "#f4d15d",
    2048: "#f4ce4b",
    4096: "#5a574b",
    8192: "#4a4539",
    16384: "#3b362d",
    32768: "#2d2922",
    65536: "#221f1a",
}


def tile_color(value: int) -> str:
    if value <= 0:
        return EMPTY_TILE
    return TILE_COLORS.get(value, "#12100c")


def tile_outline(value: int) -> str:
    if value <= 0:
        return EMPTY_TILE_OUTLINE
    return TILE_OUTLINES.get(value, "#2a251d")


def tile_text_color(value: int) -> str:
    if value <= 0:
        return FG
    if value <= 4:
        return "#5b5146"
    return "#fffaf2"


def tile_font_size(value: int, cell: float) -> int:
    digits = len(str(max(1, value)))
    if digits <= 2:
        return max(11, int(cell * 0.42))
    if digits == 3:
        return max(10, int(cell * 0.34))
    if digits == 4:
        return max(9, int(cell * 0.28))
    return max(8, int(cell * 0.22))


def _blend(first: str, second: str, fraction: float) -> str:
    """Blend two #rrggbb colours; fraction 0 -> first, 1 -> second."""

    fraction = max(0.0, min(1.0, float(fraction)))
    a = [int(first[index:index + 2], 16) for index in (1, 3, 5)]
    b = [int(second[index:index + 2], 16) for index in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * fraction):02x}" for x, y in zip(a, b))


blend = _blend


def configure_styles(root) -> ttk.Style:
    """Configure the ttk widgets once for the whole application."""

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure("TFrame", background=BG)
    style.configure("Panel.TFrame", background=PANEL)
    style.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 9))
    style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Consolas", 9))
    style.configure("Panel.TLabel", background=PANEL, foreground=FG, font=("Segoe UI", 9))
    style.configure("PanelMuted.TLabel", background=PANEL, foreground=MUTED, font=("Consolas", 8))
    style.configure("Title.TLabel", background=BG, foreground=FG, font=("Segoe UI", 22, "bold"))
    style.configure("Accent.TLabel", background=BG, foreground=ACC, font=("Consolas", 9))

    style.configure(
        "TButton",
        background=PANEL_ALT,
        foreground=FG,
        borderwidth=0,
        focusthickness=0,
        focuscolor=PANEL_ALT,
        padding=(10, 7),
        font=("Segoe UI", 9, "bold"),
    )
    style.map(
        "TButton",
        background=[("disabled", "#121a23"), ("pressed", PANEL_HI), ("active", BORDER)],
        foreground=[("disabled", DIM), ("active", FG)],
    )
    style.configure(
        "Accent.TButton",
        background=ACC_DARK,
        foreground="#eafff4",
        padding=(10, 7),
        font=("Segoe UI", 9, "bold"),
    )
    style.map(
        "Accent.TButton",
        background=[("disabled", "#1a2a22"), ("pressed", "#1a6b48"), ("active", ACC_HI)],
        foreground=[("disabled", DIM)],
    )
    style.configure(
        "Danger.TButton",
        background=DANGER_DARK,
        foreground="#ffd9d3",
        padding=(10, 7),
        font=("Segoe UI", 9, "bold"),
    )
    style.map(
        "Danger.TButton",
        background=[("disabled", "#241413"), ("pressed", "#7a332c"), ("active", "#74302a")],
        foreground=[("disabled", DIM)],
    )
    style.configure(
        "Ghost.TButton",
        background=BG,
        foreground=MUTED,
        padding=(8, 6),
        font=("Segoe UI", 8, "bold"),
    )
    style.map(
        "Ghost.TButton",
        background=[("active", PANEL_ALT), ("pressed", PANEL)],
        foreground=[("active", FG)],
    )

    style.configure("TLabelframe", background=BG, bordercolor=BORDER, borderwidth=1, relief="solid")
    style.configure(
        "TLabelframe.Label",
        background=BG,
        foreground=MUTED,
        font=("Consolas", 9, "bold"),
    )

    style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 0, 0, 0))
    style.configure(
        "TNotebook.Tab",
        background=BG_ALT,
        foreground=MUTED,
        padding=(16, 7),
        borderwidth=0,
        font=("Segoe UI", 9, "bold"),
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", PANEL), ("active", PANEL_ALT)],
        foreground=[("selected", FG), ("active", FG)],
    )

    style.configure(
        "Treeview",
        background=PANEL,
        fieldbackground=PANEL,
        foreground=FG,
        borderwidth=0,
        rowheight=24,
        font=("Consolas", 9),
    )
    style.map(
        "Treeview",
        background=[("selected", ACC)],
        foreground=[("selected", "#07130c")],
    )
    style.configure(
        "Treeview.Heading",
        background=PANEL_ALT,
        foreground=MUTED,
        relief="flat",
        borderwidth=0,
        font=("Segoe UI", 8, "bold"),
        padding=(4, 5),
    )
    style.map("Treeview.Heading", background=[("active", BORDER)])

    style.configure(
        "TCombobox",
        fieldbackground=PANEL,
        background=PANEL,
        foreground=FG,
        arrowcolor=MUTED,
        bordercolor=BORDER,
        lightcolor=PANEL,
        darkcolor=PANEL,
        padding=(7, 5),
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", PANEL), ("disabled", BG_ALT)],
        foreground=[("readonly", FG), ("disabled", DIM)],
        bordercolor=[("focus", ACC), ("active", BORDER_HI)],
    )

    style.configure(
        "TEntry",
        fieldbackground=PANEL,
        foreground=FG,
        insertcolor=FG,
        bordercolor=BORDER,
        lightcolor=PANEL,
        darkcolor=PANEL,
        padding=5,
    )
    style.configure(
        "TSpinbox",
        fieldbackground=PANEL,
        background=PANEL_ALT,
        foreground=FG,
        arrowcolor=MUTED,
        bordercolor=BORDER,
        lightcolor=PANEL,
        darkcolor=PANEL,
        padding=4,
    )
    style.map("TSpinbox", fieldbackground=[("disabled", BG_ALT)], foreground=[("disabled", DIM)])

    style.configure(
        "TCheckbutton",
        background=BG,
        foreground=FG,
        indicatorbackground=PANEL,
        indicatorforeground=ACC,
        font=("Segoe UI", 9),
    )
    style.map(
        "TCheckbutton",
        background=[("active", BG)],
        foreground=[("disabled", DIM)],
        indicatorbackground=[("selected", ACC_DARK), ("active", PANEL_HI)],
    )

    style.configure(
        "Accent.Horizontal.TProgressbar",
        troughcolor=PANEL,
        background=ACC,
        bordercolor=PANEL,
        lightcolor=ACC,
        darkcolor=ACC,
        thickness=6,
    )
    style.configure(
        "Blue.Horizontal.TProgressbar",
        troughcolor=PANEL,
        background=BLUE,
        bordercolor=PANEL,
        lightcolor=BLUE,
        darkcolor=BLUE,
        thickness=6,
    )
    style.configure(
        "TScrollbar",
        background=PANEL_ALT,
        troughcolor=BG,
        arrowcolor=MUTED,
        bordercolor=BG,
        lightcolor=PANEL_ALT,
        darkcolor=PANEL_ALT,
    )
    style.map("TScrollbar", background=[("active", BORDER_HI)])

    root.option_add("*TCombobox*Listbox.background", PANEL)
    root.option_add("*TCombobox*Listbox.foreground", FG)
    root.option_add("*TCombobox*Listbox.selectBackground", ACC)
    root.option_add("*TCombobox*Listbox.selectForeground", "#07130c")
    root.option_add("*TCombobox*Listbox.font", "{Consolas} 9")
    return style
