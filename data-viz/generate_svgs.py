from __future__ import annotations

import math
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent

ROWS = 80
COLS = 81
CELL = 7
PANEL_PAD = 18
TITLE_H = 34
SUBTITLE_H = 18
LEGEND_H = 34
PANEL_W = COLS * CELL
PANEL_H = ROWS * CELL
LOOP_SECONDS = 17.6
HOLD_PCT = 88.0


def evolve(rule: int, rows: int = ROWS, cols: int = COLS) -> list[list[int]]:
    """Elementary cellular automaton from a single live center cell."""
    state = [0] * cols
    state[cols // 2] = 1
    history = [state[:]]
    for _ in range(rows - 1):
        nxt = [0] * cols
        for i in range(cols):
            left = state[i - 1] if i > 0 else 0
            center = state[i]
            right = state[i + 1] if i < cols - 1 else 0
            idx = (left << 2) | (center << 1) | right
            nxt[i] = (rule >> idx) & 1
        state = nxt
        history.append(state[:])
    return history


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def local_entropy(history: list[list[int]], radius: int = 4) -> list[list[float]]:
    """Local Shannon entropy of the binary occupancy in a square neighborhood."""
    rows = len(history)
    cols = len(history[0])
    out = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        r0 = max(0, r - radius)
        r1 = min(rows, r + radius + 1)
        for c in range(cols):
            c0 = max(0, c - radius)
            c1 = min(cols, c + radius + 1)
            total = 0
            live = 0
            for rr in range(r0, r1):
                row = history[rr]
                for cc in range(c0, c1):
                    total += 1
                    live += row[cc]
            out[r][c] = binary_entropy(live / total)
    return out


def entropy_color(v: float) -> str:
    """Map entropy to a display scale that preserves detail near 1.0."""
    # Most local 9x9 entropies for Rule 30 sit very close to 1 bit, so a
    # naive linear 0..1 mapping collapses the interesting structure into a
    # near-white plateau. Clip to the visually informative upper band and use
    # a non-linear expansion so 0.90..1.00 spreads across the palette.
    lo = 0.82
    hi = 1.0
    if v <= lo:
        x = 0.0
    elif v >= hi:
        x = 1.0
    else:
        x = (v - lo) / (hi - lo)
    x = x ** 2.8

    stops = [
        (0.0, (5, 8, 20)),
        (0.18, (23, 37, 84)),
        (0.40, (22, 119, 158)),
        (0.62, (72, 187, 120)),
        (0.82, (247, 168, 43)),
        (1.0, (255, 214, 64)),
    ]
    if x <= stops[0][0]:
        r, g, b = stops[0][1]
        return f"#{r:02x}{g:02x}{b:02x}"
    if x >= stops[-1][0]:
        r, g, b = stops[-1][1]
        return f"#{r:02x}{g:02x}{b:02x}"
    for (a_t, a_rgb), (b_t, b_rgb) in zip(stops, stops[1:]):
        if a_t <= x <= b_t:
            t = (x - a_t) / (b_t - a_t)
            r = round(a_rgb[0] + t * (b_rgb[0] - a_rgb[0]))
            g = round(a_rgb[1] + t * (b_rgb[1] - a_rgb[1]))
            b = round(a_rgb[2] + t * (b_rgb[2] - a_rgb[2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#ffffff"


def row_styles(prefix: str, rows: int = ROWS) -> str:
    parts = []
    for r in range(rows):
        start = (HOLD_PCT * r) / max(1, rows - 1)
        parts.append(
            f".{prefix}r{r}" "{"
            f"animation:{prefix}r{r}a {LOOP_SECONDS:.2f}s step-end infinite"
            "}"
            f"@keyframes {prefix}r{r}a"
            "{"
            f"0%,{start:.3f}%{{opacity:0}}"
            f"{start + 0.001:.3f}%,{HOLD_PCT:.3f}%{{opacity:1}}"
            f"{HOLD_PCT + 0.001:.3f}%,100%{{opacity:0}}"
            "}"
        )
    return "".join(parts)


def sequential_scene_styles(prefix: str, n_scenes: int, rows: int = ROWS) -> str:
    parts = []
    scene_span = 100.0 / n_scenes
    active_span = scene_span * 0.92
    for s in range(n_scenes):
        start = s * scene_span
        end = start + active_span
        parts.append(
            f".{prefix}bg{s}" "{"
            f"animation:{prefix}bg{s}a {LOOP_SECONDS:.2f}s step-start infinite"
            "}"
            f"@keyframes {prefix}bg{s}a"
            "{"
            f"0%,{start:.3f}%{{opacity:0}}"
            f"{start + 0.001:.3f}%,{end:.3f}%{{opacity:1}}"
            f"{end + 0.001:.3f}%,100%{{opacity:0}}"
            "}"
        )
        parts.append(
            f".{prefix}lb{s}" "{"
            f"animation:{prefix}lb{s}a {LOOP_SECONDS:.2f}s step-start infinite"
            "}"
            f"@keyframes {prefix}lb{s}a"
            "{"
            f"0%,{start:.3f}%{{opacity:0}}"
            f"{start + 0.001:.3f}%,{end:.3f}%{{opacity:1}}"
            f"{end + 0.001:.3f}%,100%{{opacity:0}}"
            "}"
        )
        for r in range(rows):
            row_start = start + (active_span * r) / max(1, rows - 1)
            parts.append(
                f".{prefix}s{s}r{r}" "{"
                f"animation:{prefix}s{s}r{r}a {LOOP_SECONDS:.2f}s step-start infinite"
                "}"
                f"@keyframes {prefix}s{s}r{r}a"
                "{"
                f"0%,{row_start:.3f}%{{opacity:0}}"
                f"{row_start + 0.001:.3f}%,{end:.3f}%{{opacity:1}}"
                f"{end + 0.001:.3f}%,100%{{opacity:0}}"
                "}"
            )
    return "".join(parts)


def svg_header(width: int, height: int, bg: str, extra_style: str = "") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect width="{width}" height="{height}" fill="{bg}"/>'
        "<defs><style>"
        "text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}"
        ".title{font-size:18px;font-weight:700;letter-spacing:0.6px}"
        ".sub{font-size:11px;letter-spacing:0.3px}"
        ".legend{font-size:10px;letter-spacing:0.2px}"
        ".panel{fill:#111827;stroke:#334155;stroke-width:1.2}"
        f"{extra_style}"
        "</style></defs>"
    )


def panel_frame(x: int, y: int, title: str, subtitle: str) -> str:
    return (
        f'<rect class="panel" x="{x}" y="{y}" width="{PANEL_W + PANEL_PAD * 2}" '
        f'height="{PANEL_H + TITLE_H + SUBTITLE_H + PANEL_PAD * 2}" rx="16"/>'
        f'<text class="title" x="{x + PANEL_PAD}" y="{y + 24}" fill="#e5e7eb">{title}</text>'
        f'<text class="sub" x="{x + PANEL_PAD}" y="{y + 42}" fill="#94a3b8">{subtitle}</text>'
    )


def draw_rule_panel(
    history: list[list[int]],
    x: int,
    y: int,
    prefix: str,
    fg: str,
    bg: str,
) -> str:
    gx = x + PANEL_PAD
    gy = y + TITLE_H + SUBTITLE_H + PANEL_PAD
    parts = [f'<rect x="{gx}" y="{gy}" width="{PANEL_W}" height="{PANEL_H}" fill="{bg}" rx="8"/>']
    for r, row in enumerate(history):
        parts.append(f'<g class="{prefix}r{r}" opacity="0">')
        parts.append(f'<rect x="{gx}" y="{gy + r * CELL}" width="{PANEL_W}" height="{CELL}" fill="{bg}"/>')
        for c, bit in enumerate(row):
            if bit:
                parts.append(
                    f'<rect x="{gx + c * CELL}" y="{gy + r * CELL}" '
                    f'width="{CELL}" height="{CELL}" fill="{fg}"/>'
                )
        parts.append("</g>")
    return "".join(parts)


def draw_entropy_panel(
    values: list[list[float]],
    x: int,
    y: int,
    prefix: str,
) -> str:
    gx = x + PANEL_PAD
    gy = y + TITLE_H + SUBTITLE_H + PANEL_PAD
    parts = [f'<rect x="{gx}" y="{gy}" width="{PANEL_W}" height="{PANEL_H}" fill="#09090b" rx="8"/>']
    for r, row in enumerate(values):
        parts.append(f'<g class="{prefix}r{r}" opacity="0">')
        for c, v in enumerate(row):
            parts.append(
                f'<rect x="{gx + c * CELL}" y="{gy + r * CELL}" '
                f'width="{CELL}" height="{CELL}" fill="{entropy_color(v)}"/>'
            )
        parts.append("</g>")
    legend_y = gy + PANEL_H + 16
    legend_x = gx
    legend_colors = [entropy_color(v / 20.0) for v in range(21)]
    for i, color in enumerate(legend_colors):
        parts.append(
            f'<rect x="{legend_x + i * 12}" y="{legend_y}" width="12" height="10" fill="{color}"/>'
        )
    parts.append(
        f'<text class="legend" x="{legend_x}" y="{legend_y + 24}" fill="#94a3b8">0.0 bits</text>'
        f'<text class="legend" x="{legend_x + 196}" y="{legend_y + 24}" fill="#94a3b8">1.0 bit</text>'
    )
    return "".join(parts)


def write_rule30_pyramid() -> None:
    history = evolve(30)
    width = PANEL_W + PANEL_PAD * 2
    height = PANEL_H + TITLE_H + SUBTITLE_H + PANEL_PAD * 2 + 20
    style = row_styles("p")
    parts = [
        svg_header(width, height, "#0b1020", style),
        panel_frame(
            0,
            0,
            "Rule 30",
            "80 steps from a single live cell",
        ),
        draw_rule_panel(history, 0, 0, "p", "#f97316", "#140c05"),
        (
            f'<text class="sub" x="{PANEL_PAD}" y="{height - 10}" fill="#64748b">'
            "Elementary cellular automaton, fixed-zero boundary"
            "</text>"
        ),
        "</svg>",
    ]
    (OUT_DIR / "rule30_pyramid.svg").write_text("".join(parts), encoding="utf-8")


def write_rule30_entropy() -> None:
    history = evolve(30)
    entropy = local_entropy(history, radius=4)
    width = PANEL_W + PANEL_PAD * 2
    height = PANEL_H + TITLE_H + SUBTITLE_H + PANEL_PAD * 2 + LEGEND_H + 24
    style = row_styles("e")
    parts = [
        svg_header(width, height, "#080b14", style),
        panel_frame(
            0,
            0,
            "Rule 30 Local Entropy",
            "Binary Shannon entropy over each cell's 9x9 neighborhood",
        ),
        draw_entropy_panel(entropy, 0, 0, "e"),
        (
            f'<text class="sub" x="{PANEL_PAD}" y="{height - 10}" fill="#64748b">'
            "Color encodes H2(p), where p is local live-cell density"
            "</text>"
        ),
        "</svg>",
    ]
    (OUT_DIR / "rule30_entropy.svg").write_text("".join(parts), encoding="utf-8")


def write_showcase() -> None:
    scenes = [
        ("Rule 30", "Chaotic asymmetric growth", evolve(30), "#39d353", "#081109", "binary"),
        ("Rule 110", "Complex localized structures", evolve(110), "#38bdf8", "#07131c", "binary"),
        ("Rule 90", "Additive XOR / Sierpinski triangle", evolve(90), "#c084fc", "#140a1f", "binary"),
        ("Rule 54", "Particles and collisions", evolve(54), "#fb923c", "#1a0d02", "binary"),
        ("Rule 30 Local Entropy", "Binary Shannon entropy over each cell's 9x9 neighborhood", local_entropy(evolve(30), radius=4), None, "#09090b", "entropy"),
    ]

    width = PANEL_W + PANEL_PAD * 2 + 16
    height = PANEL_H + PANEL_PAD * 2 + 54
    style = sequential_scene_styles("seq", len(scenes))
    parts = [svg_header(width, height, "#0d1117", style)]

    content_x = 8
    content_y = 8
    frame_w = PANEL_W + PANEL_PAD * 2
    frame_h = PANEL_H + PANEL_PAD * 2 + 30
    grid_x = content_x + PANEL_PAD
    grid_y = content_y + 10

    for idx, (title, subtitle, values, fg, bg, mode) in enumerate(scenes):
        parts.append(
            f'<g class="seqbg{idx}" opacity="0">'
            f'<rect x="{content_x}" y="{content_y}" width="{frame_w}" height="{frame_h}" rx="16" fill="#111827" stroke="#334155" stroke-width="1.2"/>'
            f'<text class="title" x="{content_x + PANEL_PAD}" y="{content_y + 24}" fill="{fg or "#f8fafc"}">{title}</text>'
            f'<text class="sub" x="{content_x + PANEL_PAD}" y="{content_y + 42}" fill="#94a3b8">{subtitle}</text>'
            f'</g>'
        )
        parts.append(f'<g class="seqlb{idx}" opacity="0">')
        parts.append(f'<rect x="{grid_x}" y="{grid_y + 44}" width="{PANEL_W}" height="{PANEL_H}" fill="{bg}" rx="8"/>')
        if mode == "entropy":
            legend_y = grid_y + 44 + PANEL_H + 12
            for i in range(21):
                color = entropy_color(i / 20.0)
                parts.append(f'<rect x="{grid_x + i * 12}" y="{legend_y}" width="12" height="8" fill="{color}"/>')
            parts.append(f'<text class="legend" x="{grid_x}" y="{legend_y + 20}" fill="#94a3b8">0.0 bits</text>')
            parts.append(f'<text class="legend" x="{grid_x + 196}" y="{legend_y + 20}" fill="#94a3b8">1.0 bit</text>')
        else:
            parts.append(
                f'<text class="legend" x="{grid_x}" y="{grid_y + 44 + PANEL_H + 20}" fill="#64748b">'
                "single centered seed · fixed-zero boundary · 80 steps"
                "</text>"
            )
        parts.append("</g>")

        for r, row in enumerate(values):
            parts.append(f'<g class="seqs{idx}r{r}" opacity="0">')
            if mode == "entropy":
                for c, v in enumerate(row):
                    parts.append(
                        f'<rect x="{grid_x + c * CELL}" y="{grid_y + 44 + r * CELL}" '
                        f'width="{CELL}" height="{CELL}" fill="{entropy_color(v)}"/>'
                    )
            else:
                parts.append(
                    f'<rect x="{grid_x}" y="{grid_y + 44 + r * CELL}" width="{PANEL_W}" height="{CELL}" fill="{bg}"/>'
                )
                for c, bit in enumerate(row):
                    if bit:
                        parts.append(
                            f'<rect x="{grid_x + c * CELL}" y="{grid_y + 44 + r * CELL}" '
                            f'width="{CELL}" height="{CELL}" fill="{fg}"/>'
                        )
            parts.append("</g>")

    parts.append("</svg>")
    (OUT_DIR / "ca_showcase.svg").write_text("".join(parts), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_rule30_pyramid()
    write_rule30_entropy()
    write_showcase()
    print("Wrote:", OUT_DIR / "rule30_pyramid.svg")
    print("Wrote:", OUT_DIR / "rule30_entropy.svg")
    print("Wrote:", OUT_DIR / "ca_showcase.svg")


if __name__ == "__main__":
    main()
