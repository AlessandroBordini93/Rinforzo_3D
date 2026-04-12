# app.py — RebarCA API
#
# NOTE:
# - Input JSON: aggiunta opzionale di un campo per selezionare il sistema di rinforzo:
#     payload["reinforcementSystem"] (oppure payload["reinforcement_system"], payload["reinforcementType"])
#     valori: "resisto5.9" (default) | "sismagrid"
# - Output JSON: STESSA STRUTTURA di sempre.
#   Per "sismagrid":
#     * genera SOLO V/H (no diagonali) con clipping su finestre (come oggi)
#     * non esegue calcoli Aeq/Keq
#     * results.stats_table contiene SOLO le chiavi richieste (orizzontali/verticali + passi medi + aree)
#     * /export NON disponibile (HTTP 400)
#
# AGGIORNAMENTO IMPORTANTE:
# - Le "secondarie" sono SOLO le linee vicino alle aperture
# - Le "intermedie" vengono costruite successivamente a partire da primarie + secondarie
# - Distinzione chiara:
#     Xsec/Ysec = secondarie vere
#     Xint/Yint = intermedie di riempimento
#
# IMPORTANTISSIMO:
# - Resisto5.9: logica invariata dove non necessario
# - Nessuna modifica alla struttura JSON input/output esistente, a parte la lettura del nuovo campo di selezione.

from __future__ import annotations

import math
import bisect
import textwrap
import re
import zipfile
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict, deque
from pathlib import Path
from io import BytesIO

# --- headless matplotlib (Render / server) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import RootModel

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors

import pandas as pd

# optional
try:
    import ezdxf
except ImportError:
    ezdxf = None

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    PdfReader = PdfWriter = None


# ============================================================
# CONFIG
# ============================================================
FONT_REG, FONT_BOLD = "Helvetica", "Helvetica-Bold"
DEFAULT_SCHEMA_SCALE = 1.0  # default se non arriva query param

ALLOWED_POSSIBLE_STEPS = {25, 50, 75, 100}
DEFAULT_POSSIBLE_STEPS = [25, 50, 75, 100]

# NUOVO: sistemi ammessi
DEFAULT_REINFORCEMENT_SYSTEM = "resisto5.9"
ALLOWED_REINFORCEMENT_SYSTEMS = {"resisto5.9", "sismagrid"}

app = FastAPI(
    title="RebarCA API",
    version="2.15 (secondarie vere separate dalle intermedie)",
)

class Payload(RootModel[Dict[str, Any]]):
    pass


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class Beam:
    y_axis: float
    spess: float

@dataclass
class Column:
    x_axis: float
    spess: float

@dataclass
class Window:
    x: float
    y_rel: float
    w: float
    h: float
    y_abs: float = 0.0


# ============================================================
# VALIDATION + NAMING
# ============================================================
def _must(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)

def slugify(name: str) -> str:
    name = (name or "").strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "x"

def safe_suffix(s: str) -> str:
    s = (s or "").strip()
    _must(len(s) > 0, "meta.suffix mancante o vuoto")
    _must(len(s) <= 64, "meta.suffix troppo lungo (max 64)")
    _must(re.fullmatch(r"[A-Za-z0-9._-]+", s) is not None, "meta.suffix non valido (usa solo A-Z a-z 0-9 . _ -)")
    return s

ORIENT_MAP = {
    "n": "Nord", "nord": "Nord",
    "s": "Sud", "sud": "Sud",
    "e": "Est", "est": "Est",
    "o": "Ovest", "ovest": "Ovest", "w": "Ovest",
    "ne": "Nord-Est", "nordest": "Nord-Est", "nord-est": "Nord-Est",
    "no": "Nord-Ovest", "nordovest": "Nord-Ovest", "nord-ovest": "Nord-Ovest",
    "se": "Sud-Est", "sudest": "Sud-Est", "sud-est": "Sud-Est",
    "so": "Sud-Ovest", "sudovest": "Sud-Ovest", "sud-ovest": "Sud-Ovest",
}
ALLOWED_ORIENTATIONS = {"Nord", "Sud", "Est", "Ovest", "Nord-Est", "Nord-Ovest", "Sud-Est", "Sud-Ovest"}

def normalize_orientation(s: str) -> str:
    raw = (s or "").strip().lower().replace(" ", "")
    raw = raw.replace("–", "-").replace("—", "-").replace("_", "-")
    raw_no_dash = raw.replace("-", "")
    norm = ORIENT_MAP.get(raw_no_dash)
    if norm is not None:
        return norm
    s2 = (s or "").strip().replace("–", "-").replace("—", "-")
    if s2 in ALLOWED_ORIENTATIONS:
        return s2
    raise ValueError(f"meta.wall_orientation non valida. Valori ammessi: {sorted(ALLOWED_ORIENTATIONS)}")

def make_job_id(project_name: str, location_name: str, wall_orientation: str, suffix: str) -> Tuple[str, str]:
    o = normalize_orientation(wall_orientation)
    job_id = f"{slugify(project_name)}__{slugify(location_name)}__{slugify(o)}__{safe_suffix(suffix)}"
    return job_id, o


# ============================================================
# PDF HELPERS
# ============================================================
def _wrap(txt: str, width=92) -> str:
    return "\n".join(textwrap.fill(p.strip(), width) for p in txt.strip().splitlines())

def _footer(c: canvas.Canvas, W, H):
    h5 = H / 15
    c.setFont(FONT_REG, 11)
    c.drawCentredString(W / 2, 0.75 * h5, "Ing. Alessandro Bordini")
    c.drawCentredString(W / 2, 0.35 * h5, "Phone: 3451604706 - ✉: alessandro_bordini@outlook.com")

def _crop_png_whitespace(png_path: Path, pad_px: int = 12) -> Optional[BytesIO]:
    try:
        img = mpimg.imread(str(png_path))
        if img.ndim != 3:
            return None

        if img.shape[2] == 4:
            rgb = img[:, :, :3]
            alpha = img[:, :, 3]
        else:
            rgb = img[:, :, :3]
            alpha = np.ones(rgb.shape[:2], dtype=rgb.dtype)

        tol = 0.98
        nonwhite = (alpha > 0.05) & (np.min(rgb, axis=2) < tol)
        if not np.any(nonwhite):
            return None

        ys, xs = np.where(nonwhite)
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())

        y1 = max(0, y1 - pad_px)
        x1 = max(0, x1 - pad_px)
        y2 = min(img.shape[0] - 1, y2 + pad_px)
        x2 = min(img.shape[1] - 1, x2 + pad_px)

        cropped = img[y1:y2 + 1, x1:x2 + 1, :]
        buf = BytesIO()
        plt.imsave(buf, cropped)
        buf.seek(0)
        return buf
    except Exception:
        return None


# ============================================================
# GEOMETRY UTILITIES
# ============================================================
def win_box(w: Window, pad: float = 0.0) -> Tuple[float, float, float, float]:
    return (w.x - pad, w.y_abs - pad, w.x + w.w + pad, w.y_abs + w.h + pad)

def _box_intersect_strict(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float], eps: float = 1e-9) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 + eps or ax1 >= bx2 - eps or ay2 <= by1 + eps or ay1 >= by2 - eps)

def ok_seg(x1, y1, x2, y2, wins: List[Window], *, DISTF: float = 0.0, eps: float = 1e-9) -> bool:
    xmin, xmax = sorted((x1, x2))
    ymin, ymax = sorted((y1, y2))
    seg_box = (xmin, ymin, xmax, ymax)
    for w in wins:
        if _box_intersect_strict(seg_box, win_box(w, pad=DISTF), eps=eps):
            return False
    return True

def primarie(nodes, *, vertical: bool, PASSO: int, CLEAR: int):
    lo = nodes[0].x_axis if vertical else nodes[0].y_axis
    hi = nodes[-1].x_axis if vertical else nodes[-1].y_axis
    low = lo - nodes[0].spess / 2 + CLEAR
    high = hi + nodes[-1].spess / 2 - CLEAR

    best, full = 1e9, []
    for z0 in range(int(low), int(low) + PASSO):
        s = ((low - z0 + PASSO - 1) // PASSO) * PASSO + z0
        e = ((high - z0) // PASSO) * PASSO + z0
        if s > e:
            continue
        g = list(range(int(s), int(e) + 1, PASSO))

        if any(
            not any(a - n.spess / 2 + CLEAR <= v <= a + n.spess / 2 - CLEAR for v in g)
            for n, a in (((n, n.x_axis) if vertical else (n, n.y_axis)) for n in nodes)
        ):
            continue

        scrt = (g[0] - low) + (high - g[-1])
        if scrt < best:
            best, full = scrt, g

    if not full:
        raise ValueError("Maglia primaria da passo fisso impossibile (controlla geometria/spessori/CLEAR).")

    base = []
    for n in nodes:
        a, sp = (n.x_axis, n.spess) if vertical else (n.y_axis, n.spess)
        base.append(min((v for v in full if a - sp / 2 + CLEAR <= v <= a + sp / 2 - CLEAR), key=lambda v: abs(v - a)))
    return full, base


# ============================================================
# HELPERS PER PASSI MEDI
# ============================================================
def _sorted_unique(nums: List[float], eps: float = 1e-9) -> List[float]:
    if not nums:
        return []
    s = sorted(float(x) for x in nums)
    out = [s[0]]
    for v in s[1:]:
        if abs(v - out[-1]) > eps:
            out.append(v)
    return out

def _mean_diffs(vals_sorted: List[float]) -> float:
    if len(vals_sorted) < 2:
        return 0.0
    return sum(vals_sorted[i + 1] - vals_sorted[i] for i in range(len(vals_sorted) - 1)) / (len(vals_sorted) - 1)

def _mean_step_from_grid_coords(coords: List[float]) -> float:
    vals = _sorted_unique(coords)
    return float(_mean_diffs(vals))

def _subset_in_range(coords_sorted_unique: List[float], a: float, b: float, eps: float = 1e-9) -> List[float]:
    lo, hi = (a, b) if a <= b else (b, a)
    return [v for v in coords_sorted_unique if (lo - eps) <= v <= (hi + eps)]

def _panel_step_mean_mm_from_grid(
    i: int,
    j: int,
    *,
    Xall: List[float],
    Yall: List[float],
    Xbase: List[float],
    Ybase: List[float],
) -> float:
    xs_u = _sorted_unique(Xall)
    ys_u = _sorted_unique(Yall)

    x1, x2 = float(Xbase[j]), float(Xbase[j + 1])
    y1, y2 = float(Ybase[i]), float(Ybase[i + 1])

    xs = _subset_in_range(xs_u, x1, x2)
    ys = _subset_in_range(ys_u, y1, y2)

    mx = _mean_diffs(xs) if len(xs) >= 2 else 0.0
    my = _mean_diffs(ys) if len(ys) >= 2 else 0.0

    pm_cm = (mx + my) / 2.0 if (mx > 0 and my > 0) else (mx or my or 0.0)
    return float(round(pm_cm * 10.0, 3))


# ============================================================
# DISTF CLAMPED
# ============================================================
def inflate_window_clamped_to_panel(
    w: Window,
    i: int,
    j: int,
    cols: List[Column],
    beams: List[Beam],
    DISTF: float,
    eps: float = 1e-9,
) -> Window:
    if DISTF <= 0:
        return w

    xmin = cols[j].x_axis + cols[j].spess / 2.0
    xmax = cols[j + 1].x_axis - cols[j + 1].spess / 2.0
    ymin = beams[i].y_axis + beams[i].spess / 2.0
    ymax = beams[i + 1].y_axis - beams[i + 1].spess / 2.0

    xa = w.x
    ya = w.y_abs
    xb = xa + w.w
    yb = ya + w.h

    pad_l = max(0.0, min(DISTF, xa - xmin))
    pad_r = max(0.0, min(DISTF, xmax - xb))
    pad_b = max(0.0, min(DISTF, ya - ymin))
    pad_t = max(0.0, min(DISTF, ymax - yb))

    if xa - pad_l < xmin - eps:
        pad_l = max(0.0, xa - xmin)
    if xb + pad_r > xmax + eps:
        pad_r = max(0.0, xmax - xb)
    if ya - pad_b < ymin - eps:
        pad_b = max(0.0, ya - ymin)
    if yb + pad_t > ymax + eps:
        pad_t = max(0.0, ymax - yb)

    return Window(
        x=xa - pad_l,
        y_rel=w.y_rel,
        w=w.w + pad_l + pad_r,
        h=w.h + pad_b + pad_t,
        y_abs=ya - pad_b,
    )

def build_inflated_windows_clamped(
    win_data: Dict[Tuple[int, int], List[Window]],
    cols: List[Column],
    beams: List[Beam],
    DISTF: float,
) -> Tuple[List[Window], Dict[Tuple[int, int], List[Window]]]:
    all_w_infl: List[Window] = []
    infl_by_panel: Dict[Tuple[int, int], List[Window]] = {}

    for (i, j), lst in win_data.items():
        out_lst: List[Window] = []
        for w in lst:
            wi = inflate_window_clamped_to_panel(w, i=i, j=j, cols=cols, beams=beams, DISTF=DISTF)
            out_lst.append(wi)
            all_w_infl.append(wi)
        infl_by_panel[(i, j)] = out_lst

    return all_w_infl, infl_by_panel


# ============================================================
# CANDIDATE-DRIVEN LINES NEAR WINDOWS
# ============================================================
def _overlap_strict(a1: float, a2: float, b1: float, b2: float, eps: float = 1e-9) -> bool:
    lo1, hi1 = (a1, a2) if a1 <= a2 else (a2, a1)
    lo2, hi2 = (b1, b2) if b1 <= b2 else (b2, b1)
    return not (hi1 <= lo2 + eps or hi2 <= lo1 + eps)

def linee_finestre_candidate_driven(grid: List[float], wins_gonf: List[Window], asse: str, eps: float = 1e-9) -> List[float]:
    grid = sorted(grid)
    if not grid or not wins_gonf:
        return []

    def x_inside(w: Window, x: float) -> bool:
        return (w.x + eps) < x < (w.x + w.w - eps)

    def y_inside(w: Window, y: float) -> bool:
        return (w.y_abs + eps) < y < (w.y_abs + w.h - eps)

    def competitors_for_x(wA: Window) -> List[Window]:
        out = []
        Ay1, Ay2 = wA.y_abs, wA.y_abs + wA.h
        for wB in wins_gonf:
            if wB is wA:
                continue
            By1, By2 = wB.y_abs, wB.y_abs + wB.h
            if _overlap_strict(Ay1, Ay2, By1, By2, eps=eps):
                out.append(wB)
        return out

    def competitors_for_y(wA: Window) -> List[Window]:
        out = []
        Ax1, Ax2 = wA.x, wA.x + wA.w
        for wB in wins_gonf:
            if wB is wA:
                continue
            Bx1, Bx2 = wB.x, wB.x + wB.w
            if _overlap_strict(Ax1, Ax2, Bx1, Bx2, eps=eps):
                out.append(wB)
        return out

    extra: List[float] = []

    for wA in wins_gonf:
        if asse == "x":
            L = wA.x
            R = wA.x + wA.w
            comps = competitors_for_x(wA)

            i_sx = bisect.bisect_right(grid, L) - 1
            while i_sx >= 0 and any(x_inside(wB, grid[i_sx]) for wB in comps):
                i_sx -= 1
            if i_sx >= 0:
                extra.append(grid[i_sx])

            i_dx = bisect.bisect_left(grid, R)
            while i_dx < len(grid) and any(x_inside(wB, grid[i_dx]) for wB in comps):
                i_dx += 1
            if i_dx < len(grid):
                extra.append(grid[i_dx])

        else:
            B = wA.y_abs
            T = wA.y_abs + wA.h
            comps = competitors_for_y(wA)

            i_giu = bisect.bisect_right(grid, B) - 1
            while i_giu >= 0 and any(y_inside(wB, grid[i_giu]) for wB in comps):
                i_giu -= 1
            if i_giu >= 0:
                extra.append(grid[i_giu])

            i_su = bisect.bisect_left(grid, T)
            while i_su < len(grid) and any(y_inside(wB, grid[i_su]) for wB in comps):
                i_su += 1
            if i_su < len(grid):
                extra.append(grid[i_su])

    return sorted(set(extra))


# ============================================================
# INTERMEDIE (RESISTO 5.9)
# ============================================================
def _normalize_possible_steps(possible_steps: Optional[List[int]]) -> List[int]:
    if possible_steps is None:
        possible_steps = DEFAULT_POSSIBLE_STEPS
    try:
        steps = sorted(set(int(v) for v in possible_steps))
    except Exception:
        raise ValueError("settings.possible_steps non valido (deve essere una lista di interi)")

    if not steps:
        raise ValueError("settings.possible_steps vuoto: impossibile generare linee intermedie")

    bad = [v for v in steps if v not in ALLOWED_POSSIBLE_STEPS]
    if bad:
        raise ValueError(f"settings.possible_steps contiene valori non ammessi: {bad}. Ammessi: {sorted(ALLOWED_POSSIBLE_STEPS)}")

    return steps

def intermedie(lines: List[float], PASSO: int, possible_steps: Optional[List[int]] = None) -> List[float]:
    out: List[float] = []
    steps = _normalize_possible_steps(possible_steps)

    for s in steps:
        if s % PASSO != 0:
            raise ValueError(f"settings.possible_steps incoerente: step {s} non è multiplo del PASSO base {PASSO}")

    if PASSO == 25 and steps == [25, 50, 75, 100]:
        for a, b in zip(lines[:-1], lines[1:]):
            rem = b - a
            if rem < 2 * PASSO:
                continue
            pos = a
            while rem > PASSO:
                step = 100 if rem >= 100 else 75 if rem >= 75 else 50 if rem >= 50 else PASSO
                if rem == 125:
                    step = 75
                pos += step
                out.append(pos)
                rem = b - pos
        return out

    eps = 1e-9
    steps_desc = sorted(steps, reverse=True)

    for a, b in zip(lines[:-1], lines[1:]):
        rem = b - a
        if rem <= 0:
            continue

        q = rem / PASSO
        if abs(q - round(q)) > 1e-6:
            raise ValueError(f"Intervallo non multiplo del PASSO base {PASSO}: rem={rem}")

        target_units = int(round(rem / PASSO))
        step_units = [(s, s // PASSO) for s in steps_desc]

        INF = (10**9, 10**9, 10**18)
        dp_cost: List[Tuple[int, int, int]] = [INF] * (target_units + 1)
        prev: List[Optional[Tuple[int, int]]] = [None] * (target_units + 1)

        dp_cost[0] = (0, 0, 0)

        def better(a_cost, b_cost) -> bool:
            return a_cost < b_cost

        for u in range(target_units + 1):
            if dp_cost[u] == INF:
                continue
            n25, nseg, neg_score = dp_cost[u]
            for s_cm, s_u in step_units:
                uu = u + s_u
                if uu > target_units:
                    continue
                n25_2 = n25 + (1 if s_cm == 25 else 0)
                nseg_2 = nseg + 1
                neg_score_2 = neg_score - (s_cm * s_cm)
                cand = (n25_2, nseg_2, neg_score_2)
                if better(cand, dp_cost[uu]):
                    dp_cost[uu] = cand
                    prev[uu] = (u, s_cm)

        if dp_cost[target_units] == INF:
            raise ValueError(
                f"Impossibile generare linee intermedie: intervallo {rem:.3f} cm non decomponibile con possible_steps={steps}"
            )

        used_steps: List[int] = []
        cur = target_units
        while cur != 0:
            p = prev[cur]
            if p is None:
                raise ValueError("Errore interno DP (ricostruzione passi fallita)")
            prev_u, s_cm = p
            used_steps.append(s_cm)
            cur = prev_u
        used_steps.reverse()

        if len(used_steps) <= 1:
            continue

        pos = a
        for s in used_steps[:-1]:
            pos += s
            if pos < b - eps:
                out.append(pos)

    return out


# ============================================================
# INTERMEDIE GENERICHE (SISMAGRID)
# ============================================================
def intermedie_generic_steps(lines: List[float], PASSO: int, steps_allowed: List[int]) -> List[float]:
    out: List[float] = []
    if not steps_allowed:
        raise ValueError("steps_allowed vuoto")
    steps = sorted(set(int(s) for s in steps_allowed))
    for s in steps:
        if s <= 0:
            raise ValueError("steps_allowed contiene step <= 0")
        if s % PASSO != 0:
            raise ValueError(f"steps_allowed incoerente: step {s} non è multiplo del PASSO base {PASSO}")

    eps = 1e-9
    steps_desc = sorted(steps, reverse=True)

    for a, b in zip(lines[:-1], lines[1:]):
        rem = b - a
        if rem <= 0:
            continue

        q = rem / PASSO
        if abs(q - round(q)) > 1e-6:
            raise ValueError(f"Intervallo non multiplo del PASSO base {PASSO}: rem={rem}")

        target_units = int(round(rem / PASSO))
        step_units = [(s, s // PASSO) for s in steps_desc]

        INF = (10**9, 10**9, 10**18)
        dp_cost: List[Tuple[int, int, int]] = [INF] * (target_units + 1)
        prev: List[Optional[Tuple[int, int]]] = [None] * (target_units + 1)
        dp_cost[0] = (0, 0, 0)

        def better(a_cost, b_cost) -> bool:
            return a_cost < b_cost

        for u in range(target_units + 1):
            if dp_cost[u] == INF:
                continue
            n25, nseg, neg_score = dp_cost[u]
            for s_cm, s_u in step_units:
                uu = u + s_u
                if uu > target_units:
                    continue
                n25_2 = n25 + (1 if s_cm == 25 else 0)
                nseg_2 = nseg + 1
                neg_score_2 = neg_score - (s_cm * s_cm)
                cand = (n25_2, nseg_2, neg_score_2)
                if better(cand, dp_cost[uu]):
                    dp_cost[uu] = cand
                    prev[uu] = (u, s_cm)

        if dp_cost[target_units] == INF:
            raise ValueError(
                f"Impossibile generare linee intermedie: intervallo {rem:.3f} cm non decomponibile con steps_allowed={steps}"
            )

        used_steps: List[int] = []
        cur = target_units
        while cur != 0:
            p = prev[cur]
            if p is None:
                raise ValueError("Errore interno DP (ricostruzione passi fallita)")
            prev_u, s_cm = p
            used_steps.append(s_cm)
            cur = prev_u
        used_steps.reverse()

        if len(used_steps) <= 1:
            continue

        pos = a
        for s in used_steps[:-1]:
            pos += s
            if pos < b - eps:
                out.append(pos)

    return out


# ============================================================
# INTERVAL HELPERS
# ============================================================
def _merge_intervals(ints: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not ints:
        return []
    ints = sorted((min(a, b), max(a, b)) for a, b in ints)
    out = [ints[0]]
    for a, b in ints[1:]:
        la, lb = out[-1]
        if a <= lb:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out

def _subtract_intervals(base: Tuple[float, float], cuts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    a, b = min(base), max(base)
    cuts = _merge_intervals([(max(a, c1), min(b, c2)) for c1, c2 in cuts if not (c2 <= a or c1 >= b)])
    if not cuts:
        return [(a, b)]
    out: List[Tuple[float, float]] = []
    cur = a
    for c1, c2 in cuts:
        if c1 > cur:
            out.append((cur, c1))
        cur = max(cur, c2)
    if cur < b:
        out.append((cur, b))
    return [(x1, x2) for x1, x2 in out if x2 > x1]

def clip_vertical_segment(x: float, y1: float, y2: float, wins: List[Window], DISTF: float, eps: float = 1e-9) -> List[Tuple[float, float]]:
    ya, yb = min(y1, y2), max(y1, y2)
    cuts: List[Tuple[float, float]] = []
    for w in wins:
        xmin, ymin, xmax, ymax = win_box(w, pad=DISTF)
        if (xmin + eps) < x < (xmax - eps):
            cuts.append((ymin, ymax))
    return _subtract_intervals((ya, yb), cuts)

def clip_horizontal_segment(y: float, x1: float, x2: float, wins: List[Window], DISTF: float, eps: float = 1e-9) -> List[Tuple[float, float]]:
    xa, xb = min(x1, x2), max(x1, x2)
    cuts: List[Tuple[float, float]] = []
    for w in wins:
        xmin, ymin, xmax, ymax = win_box(w, pad=DISTF)
        if (ymin + eps) < y < (ymax - eps):
            cuts.append((xmin, xmax))
    return _subtract_intervals((xa, xb), cuts)


# ============================================================
# PRUNE “TRATTINI” (dangling)
# ============================================================
def _k(v: float, nd: int = 6) -> float:
    return round(float(v), nd)

def split_segments_at_intersections(
    v_segs: List[Tuple[float, float, float]],
    h_segs: List[Tuple[float, float, float]],
    nd: int = 6,
) -> Tuple[List[Tuple[Tuple[float, float], Tuple[float, float], str]], Dict[Tuple[float, float], set]]:
    v_norm = [(_k(x, nd), _k(min(y1, y2), nd), _k(max(y1, y2), nd)) for x, y1, y2 in v_segs if max(y1, y2) > min(y1, y2)]
    h_norm = [(_k(y, nd), _k(min(x1, x2), nd), _k(max(x1, x2), nd)) for y, x1, x2 in h_segs if max(x1, x2) > min(x1, x2)]

    v_points: Dict[Tuple[float, float, float], set] = {}
    h_points: Dict[Tuple[float, float, float], set] = {}

    for x, y1, y2 in v_norm:
        v_points[(x, y1, y2)] = {y1, y2}
    for y, x1, x2 in h_norm:
        h_points[(y, x1, x2)] = {x1, x2}

    def between(a, b, x):
        return a <= x <= b

    for x, y1, y2 in v_norm:
        for y, x1, x2 in h_norm:
            if between(x1, x2, x) and between(y1, y2, y):
                v_points[(x, y1, y2)].add(y)
                h_points[(y, x1, x2)].add(x)

    edges: List[Tuple[Tuple[float, float], Tuple[float, float], str]] = []
    adj: Dict[Tuple[float, float], set] = defaultdict(set)

    def add_edge(u, v, typ):
        if u == v:
            return
        eid = len(edges)
        edges.append((u, v, typ))
        adj[u].add(eid)
        adj[v].add(eid)

    for (x, y1, y2), ys in v_points.items():
        ys2 = sorted(ys)
        for a, b in zip(ys2[:-1], ys2[1:]):
            add_edge((x, a), (x, b), "v")

    for (y, x1, x2), xs in h_points.items():
        xs2 = sorted(xs)
        for a, b in zip(xs2[:-1], xs2[1:]):
            add_edge((a, y), (b, y), "h")

    return edges, adj

def prune_dangling(
    edges: List[Tuple[Tuple[float, float], Tuple[float, float], str]],
    adj: Dict[Tuple[float, float], set],
    protected_nodes: set,
) -> List[bool]:
    alive = [True] * len(edges)

    def degree(node):
        return sum(1 for eid in adj.get(node, set()) if alive[eid])

    q = deque([n for n in adj.keys() if degree(n) == 1 and n not in protected_nodes])

    while q:
        n = q.popleft()
        if n in protected_nodes:
            continue
        if degree(n) != 1:
            continue
        eids = [eid for eid in adj[n] if alive[eid]]
        if not eids:
            continue
        eid = eids[0]
        alive[eid] = False
        u, v, _t = edges[eid]
        other = v if u == n else u
        if other not in protected_nodes and degree(other) == 1:
            q.append(other)

    return alive

def merge_atomic_edges(
    edges: List[Tuple[Tuple[float, float], Tuple[float, float], str]],
    alive: List[bool],
    nd: int = 6,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    v_parts = []
    h_parts = []
    for ok, (u, v, t) in zip(alive, edges):
        if not ok:
            continue
        (x1, y1), (x2, y2) = u, v
        if t == "v":
            x = x1
            ylo, yhi = sorted((y1, y2))
            v_parts.append((_k(x, nd), _k(ylo, nd), _k(yhi, nd)))
        else:
            y = y1
            xlo, xhi = sorted((x1, x2))
            h_parts.append((_k(y, nd), _k(xlo, nd), _k(xhi, nd)))

    v_by_x = defaultdict(list)
    for x, y1, y2 in v_parts:
        v_by_x[x].append((y1, y2))
    v_out = []
    for x, lst in v_by_x.items():
        lst = sorted(lst)
        cur_a, cur_b = lst[0]
        for a, b in lst[1:]:
            if a <= cur_b:
                cur_b = max(cur_b, b)
            else:
                v_out.append((x, cur_a, cur_b))
                cur_a, cur_b = a, b
        v_out.append((x, cur_a, cur_b))

    h_by_y = defaultdict(list)
    for y, x1, x2 in h_parts:
        h_by_y[y].append((x1, x2))
    h_out = []
    for y, lst in h_by_y.items():
        lst = sorted(lst)
        cur_a, cur_b = lst[0]
        for a, b in lst[1:]:
            if a <= cur_b:
                cur_b = max(cur_b, b)
            else:
                h_out.append((y, cur_a, cur_b))
                cur_a, cur_b = a, b
        h_out.append((y, cur_a, cur_b))

    return v_out, h_out

def prune_vh_segments(
    Xall: List[float],
    Yall: List[float],
    Xbase: List[float],
    Ybase: List[float],
    v_segs: List[Tuple[float, float, float]],
    h_segs: List[Tuple[float, float, float]],
    nd: int = 6,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    protected = set()
    Xbase_set = {_k(x, nd) for x in Xbase}
    Ybase_set = {_k(y, nd) for y in Ybase}
    xmin, xmax = _k(min(Xall), nd), _k(max(Xall), nd)
    ymin, ymax = _k(min(Yall), nd), _k(max(Yall), nd)
    for x in Xall:
        for y in Yall:
            xx, yy = _k(x, nd), _k(y, nd)
            if (xx in Xbase_set) or (yy in Ybase_set) or (xx in (xmin, xmax)) or (yy in (ymin, ymax)):
                protected.add((xx, yy))
    edges, adj = split_segments_at_intersections(v_segs, h_segs, nd=nd)
    alive = prune_dangling(edges, adj, protected_nodes=protected)
    v2, h2 = merge_atomic_edges(edges, alive, nd=nd)
    return v2, h2


# ============================================================
# STIFFNESS (RESISTO 5.9)
# ============================================================
def diagonali_rigidezze(
    Xall: List[float],
    Yall: List[float],
    cols: List[Column],
    beams: List[Beam],
    wins: List[Window],
    *,
    EA: float,
    CLEAR: int,
    DISTF: int,
) -> Dict[Tuple[int, int], List[List[float]]]:
    Xstr = [x for x in Xall if any(abs(x - c.x_axis) <= c.spess / 2 - CLEAR + 1e-6 for c in cols)]
    Ystr = [y for y in Yall if any(abs(y - b.y_axis) <= b.spess / 2 - CLEAR + 1e-6 for b in beams)]
    Xstr.sort()
    Ystr.sort()

    pannelli = defaultdict(list)
    for ix in range(len(Xall) - 1):
        for iy in range(len(Yall) - 1):
            x1, x2 = Xall[ix], Xall[ix + 1]
            y1, y2 = Yall[iy], Yall[iy + 1]
            if not ok_seg(x1, y1, x2, y2, wins, DISTF=DISTF):
                continue
            j = bisect.bisect_right(Xstr, x1) - 1
            i = bisect.bisect_right(Ystr, y1) - 1
            b_h = x2 - x1
            L = math.hypot(x2 - x1, y2 - y1)
            k = EA * (b_h * 10) / ((L * 10) * (L * 10))
            pannelli[(i, j)].append((y1, x1, k))

    rig: Dict[Tuple[int, int], List[List[float]]] = {}
    for (i, j), diag in pannelli.items():
        if not diag:
            rig[(i, j)] = [[0.0]]
            continue
        diag.sort(key=lambda t: (t[0], t[1]))
        cols_ref = sorted({x for _, x, _ in diag})
        rows_ref = sorted({y for y, _, _ in diag})
        mat = [[0.0] * len(cols_ref) for _ in range(len(rows_ref))]
        c_idx = {x: ii for ii, x in enumerate(cols_ref)}
        r_idx = {y: ii for ii, y in enumerate(rows_ref)}
        for y1, x1, k in diag:
            mat[r_idx[y1]][c_idx[x1]] = k
        rig[(i, j)] = mat

    for i in range(len(beams) - 1):
        for j in range(len(cols) - 1):
            rig.setdefault((i, j), [[0.0]])
    return rig


# ============================================================
# PDF GENERATORS
# ============================================================
def _first_page(schema_png: Path, stats: List[str], out_pdf: Path, header_lines: List[str]):
    W, H = A4
    c = canvas.Canvas(str(out_pdf), pagesize=A4)

    c.setFont(FONT_BOLD, 14)
    c.drawCentredString(W / 2, H - 0.55 * cm, "Calcolo Automatizzato – Schema di posa Resisto 5.9")

    c.setFont(FONT_REG, 11)
    y = H - 1.45 * cm
    for ln in header_lines:
        c.drawCentredString(W / 2, y, ln)
        y -= 0.50 * cm

    footer_space = 2.1 * cm
    bottom_limit = footer_space + 0.55 * cm
    top_limit = y - 0.35 * cm
    avail_h = max(10.0, top_limit - bottom_limit)
    avail_w = W - 1.2 * cm

    cropped_buf = _crop_png_whitespace(schema_png, pad_px=12)
    if cropped_buf is not None:
        img = ImageReader(cropped_buf)
    else:
        img = ImageReader(str(schema_png))

    iw, ih = img.getSize()
    scale = min(avail_w / iw, avail_h / ih)
    w_img = iw * scale
    h_img = ih * scale

    x_img = (W - w_img) / 2
    y_img = bottom_limit + (avail_h - h_img) / 2
    c.drawImage(img, x_img, y_img, w_img, h_img)

    _footer(c, W, H)
    c.save()

def _muratura_summary_page(
    out_pdf: Path,
    header_lines: List[str],
    *,
    passo_medio_x_cm: float,
    passo_medio_y_cm: float,
):
    W, H = A4
    c = canvas.Canvas(str(out_pdf), pagesize=A4)

    c.setFont(FONT_BOLD, 14)
    c.drawCentredString(W / 2, H - 0.55 * cm, "Calcolo Automatizzato – Schema di posa Resisto 5.9")

    c.setFont(FONT_REG, 11)
    y = H - 1.45 * cm
    for ln in header_lines:
        c.drawCentredString(W / 2, y, ln)
        y -= 0.50 * cm

    y -= 0.60 * cm
    left = 2.2 * cm

    def draw_block_title(txt: str, y: float) -> float:
        c.setFont(FONT_BOLD, 12)
        c.setFillColor(colors.black)
        c.drawString(left, y, txt)
        return y - 0.55 * cm

    def draw_kv(label: str, value: str, y: float) -> float:
        c.setFont(FONT_BOLD, 11)
        c.drawString(left, y, label)
        c.setFont(FONT_REG, 11)
        c.drawString(left + 5.2 * cm, y, value)
        return y - 0.50 * cm

    y = draw_block_title("Armatura eq Verticale:", y)
    y = draw_kv("Area:", "1,18 cm² [1Ø12]", y)
    y = draw_kv("Passo:", f"{passo_medio_x_cm:.1f} cm", y)

    y -= 0.35 * cm

    y = draw_block_title("Armatura eq Orizzontale:", y)
    y = draw_kv("Area:", "1,18 cm² [1Ø12]", y)
    y = draw_kv("Passo:", f"{passo_medio_y_cm:.1f} cm", y)

    y -= 0.55 * cm

    y = draw_block_title("Parametri:", y)
    y = draw_kv("Materiale:", "B450", y)
    y = draw_kv("Drift taglio:", "0,008", y)
    y = draw_kv("Drift P.F.:", "0,016", y)

    _footer(c, W, H)
    c.save()

def _extra_pages(
    matrices: Dict[Tuple[int, int], "pd.DataFrame"],
    Aeq: Dict[Tuple[int, int], float],
    Keq: Dict[Tuple[int, int], float],
    grafico1: Path,
    grafico2: Path,
    area_uni: float,
    out_pdf: Path,
):
    W, H = A4
    MARGX = 2 * cm
    c = canvas.Canvas(str(out_pdf), pagesize=A4)

    def _draw_equations(y0: float) -> float:
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(MARGX, y0, "Calcolo rigidezze equivalenti – riepilogo formule")
        y = y0 - 12
        eqs = [
            r"$K_{d,i}= \dfrac{E\,A}{L_i^{2}}\,b_i$",
            r"$K_{\text{or}}= \dfrac{1}{\sum K_{d,i,x}}$",
            r"$K_{\text{eq}}= \dfrac{1}{\sum K_{\text{or}}}$",
            r"$A_{\text{eq}}= \dfrac{K_{\text{eq}}\,l^{2}}{E\,b}$",
        ]
        for eq in eqs:
            buf = BytesIO()
            fig = plt.figure(figsize=(0.01, 0.01))
            fig.text(0, 0, eq, fontsize=6)
            fig.patch.set_alpha(0)
            plt.axis("off")
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.02, transparent=True)
            plt.close(fig)
            buf.seek(0)
            img = ImageReader(buf)
            iw, ih = img.getSize()
            scale = 0.5
            c.drawImage(img, MARGX, y - ih * scale, iw * scale, ih * scale, mask="auto")
            y -= ih * scale + 2
        return y

    top_y = H - 1.6 * cm
    y = _draw_equations(top_y) - 50

    for (i, j) in sorted(matrices):
        df = matrices[(i, j)]
        lines = df.to_string().splitlines()
        blocco_h = (len(lines) + 3) * 0.32 * cm
        if y - blocco_h < 2 * cm:
            _footer(c, W, H)
            c.showPage()
            y = H - 2.5 * cm

        c.setFont(FONT_BOLD, 11)
        c.drawString(MARGX, y, f"Piano: {i} – Tamponamento: {j+1}")
        y -= 0.45 * cm

        c.setFont("Courier", 8)
        for ln in lines:
            c.drawString(MARGX, y, ln)
            y -= 0.32 * cm

        y -= 0.5 * cm
        c.setFont(FONT_BOLD, 9)
        c.setFillColor(colors.red)
        c.drawString(MARGX, y, f"Aeq = {Aeq[(i,j)]:.0f} mm² || Keq = {Keq[(i,j)]:.0f} N/mm")
        y -= 0.7 * cm
        c.setFillColor(colors.black)

    testo = _wrap(f"Aunivoca = {area_uni:.0f} mm²", 80)
    if y - 2 * cm < 2 * cm:
        _footer(c, W, H)
        c.showPage()
        y = H - 2.5 * cm

    c.setFont(FONT_BOLD, 11)
    c.setFillColor(colors.blue)
    c.drawString(MARGX, y, "Calcolo Area Equivalente univoca")
    y -= 0.45 * cm
    c.setFont(FONT_BOLD, 9)
    c.setFillColor(colors.red)
    for ln in testo.splitlines():
        c.drawString(MARGX, y, ln)
        y -= 0.32 * cm
    c.setFillColor(colors.black)

    _footer(c, W, H)
    c.showPage()

    img1, img2 = ImageReader(str(grafico1)), ImageReader(str(grafico2))
    iw1, ih1 = img1.getSize()
    iw2, ih2 = img2.getSize()
    slot_h = (H - 5 * cm) / 2

    def _place(img, iw, ih, y_top):
        h = min(slot_h, ih)
        w = h * iw / ih
        if w > W - 3 * cm:
            w = W - 3 * cm
            h = w * ih / iw
        c.drawImage(img, (W - w) / 2, y_top - h, w, h)

    c.setFont(FONT_BOLD, 13)
    c.drawCentredString(W / 2, H - 1.5 * cm, "Grafici – Diagonali Equivalenti")
    _place(img1, iw1, ih1, H - 2.7 * cm)
    _place(img2, iw2, ih2, H - 2.7 * cm - slot_h - 0.8 * cm)
    _footer(c, W, H)
    c.save()


# ============================================================
# DXF EXPORT
# ============================================================
def _export_dxf(
    cols: List[Column],
    beams: List[Beam],
    finestre_real: List[Window],
    finestre_infl: List[Window],
    X: List[float],
    Y: List[float],
    *,
    Xbase: List[float],
    Ybase: List[float],
    path: Path,
) -> bool:
    if ezdxf is None:
        return False

    doc = ezdxf.new(setup=True)
    m = doc.modelspace()
    doc.layers.new("Struttura", dxfattribs={"color": 1})
    doc.layers.new("Resisto", dxfattribs={"color": 7})

    def rect(x1, y1, x2, y2):
        m.add_lwpolyline([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], dxfattribs={"layer": "Struttura"})

    y_min = beams[0].y_axis - beams[0].spess / 2
    y_max = beams[-1].y_axis + beams[-1].spess / 2
    for c in cols:
        rect(c.x_axis - c.spess / 2, y_min, c.x_axis + c.spess / 2, y_max)

    x_min = cols[0].x_axis - cols[0].spess / 2
    x_max = cols[-1].x_axis + cols[-1].spess / 2
    for b in beams:
        rect(x_min, b.y_axis - b.spess / 2, x_max, b.y_axis + b.spess / 2)

    for w in finestre_real:
        rect(w.x, w.y_abs, w.x + w.w, w.y_abs + w.h)

    add = lambda a, b: m.add_line(a, b, dxfattribs={"layer": "Resisto"})

    v_raw: List[Tuple[float, float, float]] = []
    h_raw: List[Tuple[float, float, float]] = []

    for x in X:
        for y1, y2 in zip(Y[:-1], Y[1:]):
            for ya, yb in clip_vertical_segment(x, y1, y2, finestre_infl, DISTF=0):
                if yb > ya:
                    v_raw.append((x, ya, yb))

    for y in Y:
        for x1, x2 in zip(X[:-1], X[1:]):
            for xa, xb in clip_horizontal_segment(y, x1, x2, finestre_infl, DISTF=0):
                if xb > xa:
                    h_raw.append((y, xa, xb))

    v_segs, h_segs = prune_vh_segments(X, Y, Xbase, Ybase, v_raw, h_raw, nd=6)

    for x, y1, y2 in v_segs:
        add((x, y1), (x, y2))
    for y, x1, x2 in h_segs:
        add((x1, y), (x2, y))

    for i in range(len(X) - 1):
        for j in range(len(Y) - 1):
            a, b = X[i], X[i + 1]
            c_, d_ = Y[j], Y[j + 1]
            if ok_seg(a, c_, b, d_, finestre_infl, DISTF=0):
                add((a, c_), (b, d_))
            if ok_seg(a, d_, b, c_, finestre_infl, DISTF=0):
                add((a, d_), (b, c_))

    doc.saveas(str(path))
    return path.exists()


# ============================================================
# PARSE PAYLOAD
# ============================================================
def _normalize_reinforcement_system(payload: Dict[str, Any]) -> str:
    raw = (
        payload.get("reinforcementSystem")
        or payload.get("reinforcement_system")
        or payload.get("reinforcementType")
        or payload.get("reinforcement_type")
        or payload.get("system")
        or DEFAULT_REINFORCEMENT_SYSTEM
    )
    sys = str(raw).strip().lower()
    if sys not in ALLOWED_REINFORCEMENT_SYSTEMS:
        sys = DEFAULT_REINFORCEMENT_SYSTEM
    return sys

def parse_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    building_type = (payload.get("buildingType") or payload.get("building_type") or "telaio").strip().lower()
    if building_type not in {"telaio", "muratura", "cemento_armato"}:
        building_type = "telaio"

    reinforcement_system = _normalize_reinforcement_system(payload)

    try:
        schema_scale = float(payload.get("__schema_scale", DEFAULT_SCHEMA_SCALE))
    except Exception:
        schema_scale = DEFAULT_SCHEMA_SCALE
    schema_scale = max(0.2, min(schema_scale, 5.0))

    meta = payload.get("meta", {})
    project_name = meta.get("project_name", "")
    location_name = meta.get("location_name", "")
    suffix = meta.get("suffix", "")
    wall_orientation_raw = meta.get("wall_orientation", "")

    _must(project_name.strip() != "", "meta.project_name mancante o vuoto")
    _must(location_name.strip() != "", "meta.location_name mancante o vuoto")
    job_id, wall_orientation = make_job_id(project_name, location_name, wall_orientation_raw, suffix)

    settings = payload.get("settings", {})

    possible_steps_raw = settings.get("possible_steps", None)
    possible_steps = _normalize_possible_steps(possible_steps_raw)
    PASSO = int(min(possible_steps))

    CLEAR = int(settings.get("CLEAR", 5))
    DISTF = int(settings.get("DISTF", 10))
    E = float(settings.get("E_MPa", 210000))
    A = float(settings.get("A_mm2", 150))

    PASSO = max(1, PASSO)
    CLEAR = max(5, CLEAR)
    DISTF = max(0, DISTF)
    EA = E * A

    grid = payload["grid"]
    np_ = int(grid["np"])
    nt_ = int(grid["nt"])

    spB = list(map(float, grid["beams"]["spessori_cm"]))
    intB = list(map(float, grid["beams"]["interassi_cm"]))
    spC = list(map(float, grid["columns"]["spessori_cm"]))
    intC = list(map(float, grid["columns"]["interassi_cm"]))

    _must(np_ > 0 and nt_ > 0, "np/nt devono essere >0")
    _must(len(spB) == nt_ + 1, "beams.spessori_cm deve avere lunghezza nt+1")
    _must(len(intB) == nt_, "beams.interassi_cm deve avere lunghezza nt")
    _must(len(spC) == np_ + 1, "columns.spessori_cm deve avere lunghezza np+1")
    _must(len(intC) == np_, "columns.interassi_cm deve avere lunghezza np")
    _must(all(v > 0 for v in spB + intB + spC + intC), "spessori/interassi devono essere >0")

    spB = list(reversed(spB))
    intB = list(reversed(intB))

    y = [0.0]
    for v in intB:
        y.append(y[-1] + v)
    x = [0.0]
    for v in intC:
        x.append(x[-1] + v)

    beams = [Beam(yy, sp) for yy, sp in zip(y, spB)]
    cols = [Column(xx, sp) for xx, sp in zip(x, spC)]

    win_data: Dict[Tuple[int, int], List[Window]] = defaultdict(list)

    export = payload.get("export", {})
    export_png = bool(export.get("png", False))
    export_pdf = bool(export.get("pdf", False))
    export_dxf = bool(export.get("dxf", False))

    EPSG = 1e-9

    for item in payload.get("openings", []):
        i = int(item["panel"]["i"])
        j = int(item["panel"]["j"])
        _must(0 <= i < nt_, f"panel.i fuori range: {i}")
        _must(0 <= j < np_, f"panel.j fuori range: {j}")

        for w in item.get("windows", []):
            dx = float(w["dx_cm"])
            dy = float(w["dy_cm"])
            ww = float(w["w_cm"])
            hh = float(w["h_cm"])
            _must(ww > 0 and hh > 0, "w/h finestra devono essere >0")

            x_pil_sx = cols[j].x_axis
            x_pil_dx = cols[j + 1].x_axis
            y_trave_inf = beams[i].y_axis
            y_trave_sup = beams[i + 1].y_axis

            sp_pil_sx = cols[j].spess
            sp_pil_dx = cols[j + 1].spess
            sp_tr_inf = beams[i].spess
            sp_tr_sup = beams[i + 1].spess

            xa = x_pil_sx + dx
            ya = y_trave_inf + dy
            xb = xa + ww
            yb = ya + hh

            xmin = x_pil_sx + sp_pil_sx / 2.0
            xmax = x_pil_dx - sp_pil_dx / 2.0
            ymin = y_trave_inf + sp_tr_inf / 2.0
            ymax = y_trave_sup - sp_tr_sup / 2.0

            _must(
                not (xa < xmin - EPSG or xb > xmax + EPSG or ya < ymin - EPSG or yb > ymax + EPSG),
                f"Finestra esce dal pannello (i={i}, j={j}).",
            )

            win_data[(i, j)].append(Window(x=xa, y_rel=dy, w=ww, h=hh, y_abs=ya))

    return {
        "job_id": job_id,
        "building_type": building_type,
        "reinforcement_system": reinforcement_system,
        "schema_scale": schema_scale,
        "meta_norm": {
            "project_name": project_name,
            "location_name": location_name,
            "suffix": suffix,
            "wall_orientation": wall_orientation,
        },
        "export_png": export_png,
        "export_pdf": export_pdf,
        "export_dxf": export_dxf,
        "PASSO": PASSO,
        "possible_steps": possible_steps,
        "CLEAR": CLEAR,
        "DISTF": DISTF,
        "E": E,
        "A": A,
        "EA": EA,
        "np": np_,
        "nt": nt_,
        "beams": beams,
        "cols": cols,
        "win_data": dict(win_data),
    }


# ============================================================
# OVERLAY HELPERS
# ============================================================
def _line(a, b, layer, stroke="#111", width=1, dash=None, panel=None):
    return {
        "type": "line",
        "layer": layer,
        "a": [float(a[0]), float(a[1])],
        "b": [float(b[0]), float(b[1])],
        **({"panel": panel} if panel is not None else {}),
        "style": {"stroke": stroke, "width": width, "dash": dash or []},
    }

def _text(pos, text, layer="label", fill="#111", size=10, panel=None):
    return {
        "type": "text",
        "layer": layer,
        "pos": [float(pos[0]), float(pos[1])],
        "text": str(text),
        **({"panel": panel} if panel is not None else {}),
        "style": {"fill": fill, "size": size},
    }


# ============================================================
# COMPUTE — DISPATCHER
# ============================================================
def compute(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = parse_payload(payload)
    system = cfg.get("reinforcement_system", DEFAULT_REINFORCEMENT_SYSTEM)

    if system == "sismagrid":
        return compute_sismagrid(payload, cfg)

    return compute_resisto(payload, cfg)


# ============================================================
# COMPUTE RESISTO5.9
# ============================================================
def compute_resisto(payload: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    PASSO, CLEAR, DISTF = cfg["PASSO"], cfg["CLEAR"], cfg["DISTF"]
    possible_steps = cfg.get("possible_steps", DEFAULT_POSSIBLE_STEPS)

    E, EA = cfg["E"], cfg["EA"]

    cols: List[Column] = cfg["cols"]
    beams: List[Beam] = cfg["beams"]
    win_data: Dict[Tuple[int, int], List[Window]] = cfg["win_data"]
    all_w_real = [w for lst in win_data.values() for w in lst]

    all_w_infl, infl_by_panel = build_inflated_windows_clamped(win_data, cols, beams, DISTF)

    # primarie
    Xfull, Xbase = primarie(cols, vertical=True, PASSO=PASSO, CLEAR=CLEAR)
    Yfull, Ybase = primarie(beams, vertical=False, PASSO=PASSO, CLEAR=CLEAR)

    # secondarie vere = SOLO linee vicino aperture
    Xsec_all: List[float] = []
    Ysec_all: List[float] = []
    for (_ij, wins_panel) in infl_by_panel.items():
        if wins_panel:
            Xsec_all.extend(linee_finestre_candidate_driven(Xfull, wins_panel, "x"))
            Ysec_all.extend(linee_finestre_candidate_driven(Yfull, wins_panel, "y"))

    Xsec = sorted(set(Xsec_all))
    Ysec = sorted(set(Ysec_all))

    EPSG = 1e-6

    def _inside_any_column_inclusive(x: float) -> bool:
        return any(abs(x - c.x_axis) <= (c.spess / 2 + EPSG) for c in cols)

    def _inside_any_beam_inclusive(y: float) -> bool:
        return any(abs(y - b.y_axis) <= (b.spess / 2 + EPSG) for b in beams)

    # le secondarie non devono stare dentro pilastri/travi e non devono coincidere con le primarie
    Xsec = [x for x in Xsec if not _inside_any_column_inclusive(x) and all(abs(x - xb) > EPSG for xb in Xbase)]
    Ysec = [y for y in Ysec if not _inside_any_beam_inclusive(y) and all(abs(y - yb) > EPSG for yb in Ybase)]

    # intermedie = riempimento successivo tra primarie + secondarie
    Xint = intermedie(sorted(set(Xbase + Xsec)), PASSO=PASSO, possible_steps=possible_steps)
    Yint = intermedie(sorted(set(Ybase + Ysec)), PASSO=PASSO, possible_steps=possible_steps)

    Xint = [x for x in Xint if not _inside_any_column_inclusive(x) and all(abs(x - v) > EPSG for v in (Xbase + Xsec))]
    Yint = [y for y in Yint if not _inside_any_beam_inclusive(y) and all(abs(y - v) > EPSG for v in (Ybase + Ysec))]

    Xall = sorted(set(Xbase + Xsec + Xint))
    Yall = sorted(set(Ybase + Ysec + Yint))

    # V/H raw con clipping vs finestre gonfiate clampate
    v_raw: List[Tuple[float, float, float]] = []
    h_raw: List[Tuple[float, float, float]] = []

    for y in Yall:
        for x1, x2 in zip(Xall[:-1], Xall[1:]):
            for xa, xb in clip_horizontal_segment(y, x1, x2, all_w_infl, DISTF=0):
                if xb > xa:
                    h_raw.append((y, xa, xb))

    for x in Xall:
        for y1, y2 in zip(Yall[:-1], Yall[1:]):
            for ya, yb in clip_vertical_segment(x, y1, y2, all_w_infl, DISTF=0):
                if yb > ya:
                    v_raw.append((x, ya, yb))

    v_segs, h_segs = prune_vh_segments(Xall, Yall, Xbase, Ybase, v_raw, h_raw, nd=6)

    # ============================================================
    # STATS
    # ============================================================
    Lh_cm = sum(x2 - x1 for (y, x1, x2) in h_segs)
    Lv_cm = sum(y2 - y1 for (x, y1, y2) in v_segs)

    edges_atomic, _adj_atomic = split_segments_at_intersections(v_segs, h_segs, nd=6)
    n_o = sum(1 for (_u, _v, t) in edges_atomic if t == "h")
    n_v = sum(1 for (_u, _v, t) in edges_atomic if t == "v")

    Ld_cm = 0.0
    n_d = 0
    for i in range(len(Xall) - 1):
        for j in range(len(Yall) - 1):
            a, b = Xall[i], Xall[i + 1]
            c_, d_ = Yall[j], Yall[j + 1]
            dlen = math.hypot(b - a, d_ - c_)
            if ok_seg(a, c_, b, d_, all_w_infl, DISTF=0):
                n_d += 1
                Ld_cm += dlen
            if ok_seg(a, d_, b, c_, all_w_infl, DISTF=0):
                n_d += 1
                Ld_cm += dlen

    width_cm = cols[-1].x_axis
    height_cm = beams[-1].y_axis - beams[0].y_axis
    area_tot_m2 = (width_cm * height_cm) / 10_000
    area_open_m2 = sum(w.w * w.h for w in all_w_real) / 10_000
    area_pieno_m2 = max(area_tot_m2 - area_open_m2, 1e-6)

    def inc(n, a):
        return n / 2.75 / a

    passo_medio_x = _mean_step_from_grid_coords(Xall)
    passo_medio_y = _mean_step_from_grid_coords(Yall)
    p_medio = (passo_medio_x + passo_medio_y) / 2.0 if (passo_medio_x > 0 and passo_medio_y > 0) else (passo_medio_x or passo_medio_y or 0.0)

    stats = [
        f"Orizzontali : L = {Lh_cm/100:.2f} m | n = {n_o} | Inc. P {inc(n_o,area_pieno_m2):.2f} T {inc(n_o,area_tot_m2):.2f}",
        f"Verticali : L = {Lv_cm/100:.2f} m | n = {n_v} | Inc. P {inc(n_v,area_pieno_m2):.2f} T {inc(n_v,area_tot_m2):.2f}",
        f"Diagonali : L = {Ld_cm/100:.2f} m | n = {n_d} --> 2 x n°: {n_d/2:.0f} | Inc. P {inc(n_d,area_pieno_m2):.2f} T {inc(n_d,area_tot_m2):.2f}",
        "======================================================================",
        f"Passo medio X = {passo_medio_x:.1f} cm",
        f"Passo medio Y = {passo_medio_y:.1f} cm",
        f"Passo medio = {p_medio:.1f} cm",
        f"Area pieno (senza aperture) : {area_pieno_m2:.2f} m²",
        f"Area totale : {area_tot_m2:.2f} m²",
    ]

    stats_table = {
        "orizzontali": {"L_m": float(round(Lh_cm / 100.0, 4)), "n": int(n_o), "inc_p": float(round(inc(n_o, area_pieno_m2), 4)), "inc_t": float(round(inc(n_o, area_tot_m2), 4))},
        "verticali": {"L_m": float(round(Lv_cm / 100.0, 4)), "n": int(n_v), "inc_p": float(round(inc(n_v, area_pieno_m2), 4)), "inc_t": float(round(inc(n_v, area_tot_m2), 4))},
        "diagonali": {"L_m": float(round(Ld_cm / 100.0, 4)), "n": int(n_d), "n_x2": float(round(n_d / 2.0, 4)), "inc_p": float(round(inc(n_d, area_pieno_m2), 4)), "inc_t": float(round(inc(n_d, area_tot_m2), 4))},
        "passo_medio_x_cm": float(round(passo_medio_x, 3)),
        "passo_medio_y_cm": float(round(passo_medio_y, 3)),
        "passo_medio_cm": float(round(p_medio, 3)),
        "area_pieno_m2": float(round(area_pieno_m2, 6)),
        "area_tot_m2": float(round(area_tot_m2, 6)),
    }

    # stiffness OPEN
    rig = diagonali_rigidezze(Xall, Yall, cols, beams, all_w_infl, EA=EA, CLEAR=CLEAR, DISTF=0)

    Aeq_dict: Dict[Tuple[int, int], float] = {}
    Keq_dict: Dict[Tuple[int, int], float] = {}
    Kad_dict: Dict[Tuple[int, int], float] = {}
    matrices_for_pdf: Dict[Tuple[int, int], "pd.DataFrame"] = {}

    for (i, j), mat in rig.items():
        df = pd.DataFrame(reversed(mat)).round(0).astype(int)
        df_tmp = df.copy()
        df_tmp["Kor"] = df_tmp.apply(lambda row: 1 / (row.sum()) if row.sum() else 0.0, axis=1)
        K_eq = 1 / df_tmp["Kor"].sum() if df_tmp["Kor"].sum() else 0.0

        Lx = cols[j + 1].x_axis - cols[j].x_axis
        Ly = beams[i + 1].y_axis - beams[i].y_axis
        Ld_mm = math.hypot(Lx * 10, Ly * 10)
        Aeq = (K_eq * (Ld_mm**2) / (Lx * 10) / E) if (K_eq > 0 and Lx > 0) else 0.0
        Kad = ((E * (Lx * 10)) / (Ld_mm**2)) if Ld_mm > 0 else 0.0

        Aeq_dict[(i, j)] = float(round(Aeq, 2))
        Keq_dict[(i, j)] = float(round(K_eq, 2))
        Kad_dict[(i, j)] = float(round(Kad, 2))

        df_pdf = df.copy()
        df_pdf["Kor"] = df_tmp["Kor"].map(lambda v: f"{v:.2e}")
        matrices_for_pdf[(i, j)] = df_pdf

    # univoca OPEN
    s_keq = pd.Series(Keq_dict, name="Keq")
    s_keq.index = pd.MultiIndex.from_tuples(s_keq.index, names=["i_trave", "j_pilastro"])
    df_keq = s_keq.unstack(level="j_pilastro").sort_index(axis=0, ascending=False).fillna(0.0)
    df_keq["Kor"] = df_keq.apply(lambda row: 1 / (row.sum()) if row.sum() else 0.0, axis=1)
    K_eq_u = 1 / df_keq["Kor"].sum() if df_keq["Kor"].sum() else 0.0

    s_kad = pd.Series(Kad_dict, name="Kad")
    s_kad.index = pd.MultiIndex.from_tuples(s_kad.index, names=["i_trave", "j_pilastro"])
    df_kad = s_kad.unstack(level="j_pilastro").sort_index(axis=0, ascending=False).fillna(0.0)
    df_kad["Kor"] = df_kad.apply(lambda row: 1 / (row.sum()) if row.sum() else 0.0, axis=1)
    K_eq_u_adi = 1 / df_kad["Kor"].sum() if df_kad["Kor"].sum() else 0.0

    Aeq_univoca = (K_eq_u / K_eq_u_adi) if K_eq_u_adi else 0.0

    # overlays OPEN
    grid_entities = []
    for x, y1, y2 in v_segs:
        grid_entities.append(_line((x, y1), (x, y2), layer="grid_v"))
    for y, x1, x2 in h_segs:
        grid_entities.append(_line((x1, y), (x2, y), layer="grid_h"))
    for ii in range(len(Xall) - 1):
        for jj in range(len(Yall) - 1):
            a, b = Xall[ii], Xall[ii + 1]
            c_, d_ = Yall[jj], Yall[jj + 1]
            if ok_seg(a, c_, b, d_, all_w_infl, DISTF=0):
                grid_entities.append(_line((a, c_), (b, d_), layer="grid_d"))
            if ok_seg(a, d_, b, c_, all_w_infl, DISTF=0):
                grid_entities.append(_line((a, d_), (b, c_), layer="grid_d"))

    aeq_entities, uni_entities = [], []
    for (i, j), Aeqv in Aeq_dict.items():
        xL, xR = cols[j].x_axis, cols[j + 1].x_axis
        yB, yT = beams[i].y_axis, beams[i + 1].y_axis
        pid = f"{i},{j}"
        aeq_entities.append(_line((xL, yB), (xR, yT), layer="diag_eq", stroke="#d00", width=2, dash=[6, 4], panel=pid))
        aeq_entities.append(_line((xL, yT), (xR, yB), layer="diag_eq", stroke="#d00", width=2, dash=[6, 4], panel=pid))
        xc, yc = (xL + xR) / 2, (yB + yT) / 2
        aeq_entities.append(_text((xc + 3, yc + 2), f"Aeq={Aeqv:.0f} mm²", fill="#d00", size=10, panel=pid))

        uni_entities.append(_line((xL, yB), (xR, yT), layer="diag_uni", stroke="#0a0", width=2, dash=[6, 4], panel=pid))
        uni_entities.append(_line((xL, yT), (xR, yB), layer="diag_uni", stroke="#0a0", width=2, dash=[6, 4], panel=pid))

    if Yall:
        uni_entities.append(_text((Xall[0], max(Yall) + 15), f"Aunivoca={Aeq_univoca:.0f} mm²", fill="#0a0", size=12))

    overlays = [
        {"id": "grid", "title": "Schema di posa Resisto 5.9", "entities": grid_entities},
        {"id": "aeq_by_panel", "title": "Diagonali equivalenti – Aeq per pannello", "entities": aeq_entities},
        {"id": "aeq_univoca", "title": "Diagonali equivalenti – Aeq univoca", "entities": uni_entities},
    ]

    passo_medio_dict: Dict[Tuple[int, int], float] = {}
    nt = len(beams) - 1
    np__ = len(cols) - 1

    for i in range(nt):
        for j in range(np__):
            passo_medio_dict[(i, j)] = _panel_step_mean_mm_from_grid(
                i, j,
                Xall=Xall, Yall=Yall,
                Xbase=Xbase, Ybase=Ybase,
            )

    panels = []
    for i in range(nt):
        for j in range(np__):
            xL, xR = cols[j].x_axis, cols[j + 1].x_axis
            yB, yT = beams[i].y_axis, beams[i + 1].y_axis
            pid = f"{i},{j}"
            panels.append(
                {
                    "id": pid,
                    "i": i,
                    "j": j,
                    "bounds": {"xmin": xL, "xmax": xR, "ymin": yB, "ymax": yT},
                    "center": {"x": (xL + xR) / 2, "y": (yB + yT) / 2},
                    "Aeq_mm2": Aeq_dict.get((i, j), 0.0),
                    "Keq_N_per_mm": Keq_dict.get((i, j), 0.0),
                    "openings": [{"x": w.x, "y": w.y_abs, "w": w.w, "h": w.h} for w in win_data.get((i, j), [])],
                }
            )

    full_block: Optional[Dict[str, Any]] = None
    if cfg.get("building_type") == "cemento_armato":
        v_full_raw = [(x, y1, y2) for x in Xall for y1, y2 in zip(Yall[:-1], Yall[1:]) if y2 > y1]
        h_full_raw = [(y, x1, x2) for y in Yall for x1, x2 in zip(Xall[:-1], Xall[1:]) if x2 > x1]
        v_full, h_full = prune_vh_segments(Xall, Yall, Xbase, Ybase, v_full_raw, h_full_raw, nd=6)

        rig_full = diagonali_rigidezze(Xall, Yall, cols, beams, wins=[], EA=EA, CLEAR=CLEAR, DISTF=0)
        Aeq_full: Dict[Tuple[int, int], float] = {}
        Keq_full: Dict[Tuple[int, int], float] = {}
        Kad_full: Dict[Tuple[int, int], float] = {}

        for (i, j), mat in rig_full.items():
            df = pd.DataFrame(reversed(mat)).round(0).astype(int)
            df_tmp = df.copy()
            df_tmp["Kor"] = df_tmp.apply(lambda row: 1 / (row.sum()) if row.sum() else 0.0, axis=1)
            K_eq = 1 / df_tmp["Kor"].sum() if df_tmp["Kor"].sum() else 0.0

            Lx = cols[j + 1].x_axis - cols[j].x_axis
            Ly = beams[i + 1].y_axis - beams[i].y_axis
            Ld_mm = math.hypot(Lx * 10, Ly * 10)
            Aeq = (K_eq * (Ld_mm**2) / (Lx * 10) / E) if (K_eq > 0 and Lx > 0) else 0.0
            Kad = ((E * (Lx * 10)) / (Ld_mm**2)) if Ld_mm > 0 else 0.0

            Aeq_full[(i, j)] = float(round(Aeq, 2))
            Keq_full[(i, j)] = float(round(K_eq, 2))
            Kad_full[(i, j)] = float(round(Kad, 2))

        s_keq = pd.Series(Keq_full, name="Keq")
        s_keq.index = pd.MultiIndex.from_tuples(s_keq.index, names=["i_trave", "j_pilastro"])
        df_keq = s_keq.unstack(level="j_pilastro").sort_index(axis=0, ascending=False).fillna(0.0)
        df_keq["Kor"] = df_keq.apply(lambda row: 1 / (row.sum()) if row.sum() else 0.0, axis=1)
        K_eq_u = 1 / df_keq["Kor"].sum() if df_keq["Kor"].sum() else 0.0

        s_kad = pd.Series(Kad_full, name="Kad")
        s_kad.index = pd.MultiIndex.from_tuples(s_kad.index, names=["i_trave", "j_pilastro"])
        df_kad = s_kad.unstack(level="j_pilastro").sort_index(axis=0, ascending=False).fillna(0.0)
        df_kad["Kor"] = df_kad.apply(lambda row: 1 / (row.sum()) if row.sum() else 0.0, axis=1)
        K_eq_u_adi = 1 / df_kad["Kor"].sum() if df_kad["Kor"].sum() else 0.0

        Aeq_univoca_full = (K_eq_u / K_eq_u_adi) if K_eq_u_adi else 0.0

        ents_full = []
        for x, y1, y2 in v_full:
            ents_full.append(_line((x, y1), (x, y2), layer="grid_v"))
        for y, x1, x2 in h_full:
            ents_full.append(_line((x1, y), (x2, y), layer="grid_h"))
        for ii in range(len(Xall) - 1):
            for jj in range(len(Yall) - 1):
                a, b = Xall[ii], Xall[ii + 1]
                c_, d_ = Yall[jj], Yall[jj + 1]
                ents_full.append(_line((a, c_), (b, d_), layer="grid_d"))
                ents_full.append(_line((a, d_), (b, c_), layer="grid_d"))

        overlays.append({"id": "grid_full", "title": "Schema FULL (senza aperture)", "entities": ents_full})

        full_block = {
            "Xall": Xall,
            "Yall": Yall,
            "Xbase": Xbase,
            "Ybase": Ybase,
            "v_segs": v_full,
            "h_segs": h_full,
            "Aeq_by_panel_mm2": {f"{i},{j}": v for (i, j), v in Aeq_full.items()},
            "Keq_by_panel_N_per_mm": {f"{i},{j}": v for (i, j), v in Keq_full.items()},
            "Aeq_univoca_mm2": float(round(Aeq_univoca_full, 2)),
        }

    out = {
        "job_id": cfg["job_id"],
        "units": "cm",
        "meta": cfg["meta_norm"],
        "geometry": {
            "structure": {
                "beams": [{"y": b.y_axis, "sp": b.spess} for b in beams],
                "columns": [{"x": c.x_axis, "sp": c.spess} for c in cols],
                "windows": [{"x": w.x, "y": w.y_abs, "w": w.w, "h": w.h} for w in all_w_real],
            },
            "panels": panels,
        },
        "results": {
            "Aeq_univoca_mm2": float(round(Aeq_univoca, 2)),
            "Aeq_by_panel_mm2": {f"{i},{j}": v for (i, j), v in Aeq_dict.items()},
            "Keq_by_panel_N_per_mm": {f"{i},{j}": v for (i, j), v in Keq_dict.items()},
            "stats": stats,
            "stats_table": stats_table,
            "passo_medio_mm": {f"{i},{j}": v for (i, j), v in passo_medio_dict.items()},
        },
        "overlays": overlays,
        "internals": {
            "Xall": Xall,
            "Yall": Yall,
            "Xbase": Xbase,
            "Ybase": Ybase,
            "Xsec": Xsec,
            "Ysec": Ysec,
            "Xint": Xint,
            "Yint": Yint,
            "matrices_for_pdf": matrices_for_pdf,
            "Aeq_dict": Aeq_dict,
            "Keq_dict": Keq_dict,
            "beams": beams,
            "cols": cols,
            "all_w_real": all_w_real,
            "all_w_infl": all_w_infl,
            "win_data": win_data,
            "DISTF": DISTF,
            "v_segs_pruned": v_segs,
            "h_segs_pruned": h_segs,
        },
    }

    if full_block is not None:
        out["results"]["full"] = full_block

    return out


# ============================================================
# COMPUTE SISMAGRID
# ============================================================
def compute_sismagrid(payload: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    PASSO = 40
    CLEAR = cfg["CLEAR"]
    DISTF = cfg["DISTF"]

    cols: List[Column] = cfg["cols"]
    beams: List[Beam] = cfg["beams"]
    win_data: Dict[Tuple[int, int], List[Window]] = cfg["win_data"]
    all_w_real = [w for lst in win_data.values() for w in lst]

    all_w_infl, infl_by_panel = build_inflated_windows_clamped(win_data, cols, beams, DISTF)

    # primarie a passo 40
    Xfull, Xbase = primarie(cols, vertical=True, PASSO=PASSO, CLEAR=CLEAR)
    Yfull, Ybase = primarie(beams, vertical=False, PASSO=PASSO, CLEAR=CLEAR)

    # secondarie vere = solo linee vicino aperture
    Xsec_all: List[float] = []
    Ysec_all: List[float] = []
    for (_ij, wins_panel) in infl_by_panel.items():
        if wins_panel:
            Xsec_all.extend(linee_finestre_candidate_driven(Xfull, wins_panel, "x"))
            Ysec_all.extend(linee_finestre_candidate_driven(Yfull, wins_panel, "y"))

    Xsec = sorted(set(Xsec_all))
    Ysec = sorted(set(Ysec_all))

    EPSG = 1e-6

    def _inside_any_column_inclusive(x: float) -> bool:
        return any(abs(x - c.x_axis) <= (c.spess / 2 + EPSG) for c in cols)

    def _inside_any_beam_inclusive(y: float) -> bool:
        return any(abs(y - b.y_axis) <= (b.spess / 2 + EPSG) for b in beams)

    Xsec = [x for x in Xsec if not _inside_any_column_inclusive(x) and all(abs(x - xb) > EPSG for xb in Xbase)]
    Ysec = [y for y in Ysec if not _inside_any_beam_inclusive(y) and all(abs(y - yb) > EPSG for yb in Ybase)]

    # intermedie con solo passo 40
    Xint = intermedie_generic_steps(sorted(set(Xbase + Xsec)), PASSO=PASSO, steps_allowed=[40])
    Yint = intermedie_generic_steps(sorted(set(Ybase + Ysec)), PASSO=PASSO, steps_allowed=[40])

    Xint = [x for x in Xint if not _inside_any_column_inclusive(x) and all(abs(x - v) > EPSG for v in (Xbase + Xsec))]
    Yint = [y for y in Yint if not _inside_any_beam_inclusive(y) and all(abs(y - v) > EPSG for v in (Ybase + Ysec))]

    Xall = sorted(set(Xbase + Xsec + Xint))
    Yall = sorted(set(Ybase + Ysec + Yint))

    # SOLO V/H
    v_raw: List[Tuple[float, float, float]] = []
    h_raw: List[Tuple[float, float, float]] = []

    for y in Yall:
        for x1, x2 in zip(Xall[:-1], Xall[1:]):
            for xa, xb in clip_horizontal_segment(y, x1, x2, all_w_infl, DISTF=0):
                if xb > xa:
                    h_raw.append((y, xa, xb))

    for x in Xall:
        for y1, y2 in zip(Yall[:-1], Yall[1:]):
            for ya, yb in clip_vertical_segment(x, y1, y2, all_w_infl, DISTF=0):
                if yb > ya:
                    v_raw.append((x, ya, yb))

    v_segs, h_segs = prune_vh_segments(Xall, Yall, Xbase, Ybase, v_raw, h_raw, nd=6)

    # stats_table
    Lh_cm = sum(x2 - x1 for (y, x1, x2) in h_segs)
    Lv_cm = sum(y2 - y1 for (x, y1, y2) in v_segs)

    edges_atomic, _ = split_segments_at_intersections(v_segs, h_segs, nd=6)
    n_o = sum(1 for (_u, _v, t) in edges_atomic if t == "h")
    n_v = sum(1 for (_u, _v, t) in edges_atomic if t == "v")

    width_cm = cols[-1].x_axis
    height_cm = beams[-1].y_axis - beams[0].y_axis
    area_tot_m2 = (width_cm * height_cm) / 10_000
    area_open_m2 = sum(w.w * w.h for w in all_w_real) / 10_000
    area_pieno_m2 = max(area_tot_m2 - area_open_m2, 1e-6)

    def inc(n, a):
        return n / 2.75 / a

    passo_medio_x = _mean_step_from_grid_coords(Xall)
    passo_medio_y = _mean_step_from_grid_coords(Yall)
    p_medio = (passo_medio_x + passo_medio_y) / 2.0 if (passo_medio_x > 0 and passo_medio_y > 0) else (passo_medio_x or passo_medio_y or 0.0)

    stats_table = {
        "orizzontali": {"L_m": float(round(Lh_cm / 100.0, 4)), "n": int(n_o), "inc_p": float(round(inc(n_o, area_pieno_m2), 4)), "inc_t": float(round(inc(n_o, area_tot_m2), 4))},
        "verticali": {"L_m": float(round(Lv_cm / 100.0, 4)), "n": int(n_v), "inc_p": float(round(inc(n_v, area_pieno_m2), 4)), "inc_t": float(round(inc(n_v, area_tot_m2), 4))},
        "passo_medio_x_cm": float(round(passo_medio_x, 3)),
        "passo_medio_y_cm": float(round(passo_medio_y, 3)),
        "passo_medio_cm": float(round(p_medio, 3)),
        "area_pieno_m2": float(round(area_pieno_m2, 6)),
        "area_tot_m2": float(round(area_tot_m2, 6)),
    }

    grid_entities = []
    for x, y1, y2 in v_segs:
        grid_entities.append(_line((x, y1), (x, y2), layer="grid_v"))
    for y, x1, x2 in h_segs:
        grid_entities.append(_line((x1, y), (x2, y), layer="grid_h"))

    overlays = [
        {"id": "grid", "title": "Schema di posa SismaGrid", "entities": grid_entities},
    ]

    nt = len(beams) - 1
    np__ = len(cols) - 1
    panels = []
    for i in range(nt):
        for j in range(np__):
            xL, xR = cols[j].x_axis, cols[j + 1].x_axis
            yB, yT = beams[i].y_axis, beams[i + 1].y_axis
            pid = f"{i},{j}"
            panels.append(
                {
                    "id": pid,
                    "i": i,
                    "j": j,
                    "bounds": {"xmin": xL, "xmax": xR, "ymin": yB, "ymax": yT},
                    "center": {"x": (xL + xR) / 2, "y": (yB + yT) / 2},
                    "Aeq_mm2": 0.0,
                    "Keq_N_per_mm": 0.0,
                    "openings": [{"x": w.x, "y": w.y_abs, "w": w.w, "h": w.h} for w in win_data.get((i, j), [])],
                }
            )

    out = {
        "job_id": cfg["job_id"],
        "units": "cm",
        "meta": cfg["meta_norm"],
        "geometry": {
            "structure": {
                "beams": [{"y": b.y_axis, "sp": b.spess} for b in beams],
                "columns": [{"x": c.x_axis, "sp": c.spess} for c in cols],
                "windows": [{"x": w.x, "y": w.y_abs, "w": w.w, "h": w.h} for w in all_w_real],
            },
            "panels": panels,
        },
        "results": {
            "Aeq_univoca_mm2": 0.0,
            "Aeq_by_panel_mm2": {},
            "Keq_by_panel_N_per_mm": {},
            "stats": [],
            "stats_table": stats_table,
            "passo_medio_mm": {},
        },
        "overlays": overlays,
        "internals": {
            "Xall": Xall,
            "Yall": Yall,
            "Xbase": Xbase,
            "Ybase": Ybase,
            "Xsec": Xsec,
            "Ysec": Ysec,
            "Xint": Xint,
            "Yint": Yint,
            "beams": beams,
            "cols": cols,
            "all_w_real": all_w_real,
            "all_w_infl": all_w_infl,
            "win_data": win_data,
            "DISTF": DISTF,
            "v_segs_pruned": v_segs,
            "h_segs_pruned": h_segs,
        },
    }
    return out


# ============================================================
# EXPORTS TO DIRECTORY (TEMP) + ZIP IN RAM
# ============================================================
def render_exports_to_dir(payload: Dict[str, Any], computed: Dict[str, Any], out_dir: Path) -> Dict[str, Optional[Path]]:
    cfg = parse_payload(payload)

    export_png = cfg["export_png"]
    export_pdf = cfg["export_pdf"]
    export_dxf = cfg["export_dxf"]
    if not (export_png or export_pdf or export_dxf):
        export_png = export_pdf = export_dxf = True

    job_id = cfg["job_id"]
    meta_norm = cfg["meta_norm"]
    building_type = cfg.get("building_type", "telaio")
    schema_scale = float(cfg.get("schema_scale", DEFAULT_SCHEMA_SCALE))

    Xall = computed["internals"]["Xall"]
    Yall = computed["internals"]["Yall"]
    Xbase = computed["internals"]["Xbase"]
    Ybase = computed["internals"]["Ybase"]

    cols = computed["internals"]["cols"]
    beams = computed["internals"]["beams"]

    all_w_real = computed["internals"]["all_w_real"]
    all_w_infl = computed["internals"]["all_w_infl"]

    matrices_for_pdf = computed["internals"].get("matrices_for_pdf", {})
    Aeq_dict = computed["internals"].get("Aeq_dict", {})
    Keq_dict = computed["internals"].get("Keq_dict", {})
    v_segs = computed["internals"].get("v_segs_pruned", [])
    h_segs = computed["internals"].get("h_segs_pruned", [])

    Aeq_univoca = computed["results"].get("Aeq_univoca_mm2", 0.0)
    stats = computed["results"].get("stats", [])
    stats_table = computed["results"].get("stats_table", {}) or {}
    passo_medio_x_cm = float(stats_table.get("passo_medio_x_cm", 0.0))
    passo_medio_y_cm = float(stats_table.get("passo_medio_y_cm", 0.0))

    out_dir.mkdir(parents=True, exist_ok=True)

    schema_png = out_dir / f"{job_id}_rinforzo.png"
    grafico1_png = out_dir / f"{job_id}_diag_eq.png"
    grafico2_png = out_dir / f"{job_id}_diag_eq_uni.png"
    dxf_path = out_dir / f"{job_id}_schema_posa_resisto59.dxf"
    first_pdf = out_dir / f"{job_id}_00_schema_statistiche.pdf"
    extra_pdf = out_dir / f"{job_id}_01_extra.pdf"
    muratura_pdf = out_dir / f"{job_id}_01_muratura.pdf"
    final_pdf = out_dir / f"{job_id}_Report_resisto59_completo.pdf"

    header_lines = [
        f"Progetto: {meta_norm['project_name']}",
        f"Posizione: {meta_norm['location_name']}",
        f"Parete: {meta_norm['wall_orientation']} | Revisione: {meta_norm['suffix']}",
    ]

    written: Dict[str, Optional[Path]] = {
        "schema_png": None,
        "grafico1_png": None,
        "grafico2_png": None,
        "pdf_final": None,
        "dxf": None,
    }

    if export_png or export_pdf:
        def base_axes(ax):
            x_min = cols[0].x_axis - cols[0].spess / 2
            x_max = cols[-1].x_axis + cols[-1].spess / 2
            y_min = beams[0].y_axis - beams[0].spess / 2
            y_max = beams[-1].y_axis + beams[-1].spess / 2

            for b in beams:
                ax.add_patch(plt.Rectangle((x_min, b.y_axis - b.spess / 2), x_max - x_min, b.spess, fc="#d0d0d0", ec="none"))
            for c in cols:
                ax.add_patch(plt.Rectangle((c.x_axis - c.spess / 2, y_min), c.spess, y_max - y_min, fc="#a0a0a0", ec="none"))
            for w in all_w_real:
                ax.add_patch(plt.Rectangle((w.x, w.y_abs), w.w, w.h, fill=False, ec="blue", lw=1.4))

            ax.set_xlim(x_min - 10, x_max + 10)
            ax.set_ylim(y_min - 10, y_max + 10)
            ax.set_xlabel("X [cm]")
            ax.set_ylabel("Y [cm]")
            ax.grid(True)

        fig, ax = plt.subplots(figsize=(7 * schema_scale, 4 * schema_scale))
        ax.set_aspect("equal")
        base_axes(ax)
        ax.set_title("Schema di Posa Resisto 5.9")
        for spine in ax.spines.values():
            spine.set_visible(False)

        for x, y1, y2 in v_segs:
            ax.plot([x, x], [y1, y2], "k", lw=0.7)
        for y, x1, x2 in h_segs:
            ax.plot([x1, x2], [y, y], "k", lw=0.7)

        for i in range(len(Xall) - 1):
            for j in range(len(Yall) - 1):
                a, b = Xall[i], Xall[i + 1]
                c_, d_ = Yall[j], Yall[j + 1]
                if ok_seg(a, c_, b, d_, all_w_infl, DISTF=0):
                    ax.plot([a, b], [c_, d_], "k", lw=0.7)
                if ok_seg(a, d_, b, c_, all_w_infl, DISTF=0):
                    ax.plot([a, b], [d_, c_], "k", lw=0.7)

        fig.tight_layout()
        fig.savefig(schema_png, dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_aspect("equal")
        base_axes(ax)
        ax.set_title("Diagonali equivalenti – Aeq per pannello")
        for (i, j), Aeqv in Aeq_dict.items():
            xL, xR = cols[j].x_axis, cols[j + 1].x_axis
            yB, yT = beams[i].y_axis, beams[i + 1].y_axis
            ax.plot([xL, xR], [yB, yT], "r--", lw=1.2)
            ax.plot([xL, xR], [yT, yB], "r--", lw=1.2)
            xc, yc = (xL + xR) / 2, (yB + yT) / 2
            ax.text(xc + 3, yc + 2, f"Aeq={Aeqv:.0f} mm²", fontsize=8, color="red")
        fig.tight_layout()
        fig.savefig(grafico1_png, dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_aspect("equal")
        base_axes(ax)
        ax.set_title("Diagonali equivalenti – Aeq Univoca per pannello")
        for (i, j) in Aeq_dict.keys():
            xL, xR = cols[j].x_axis, cols[j + 1].x_axis
            yB, yT = beams[i].y_axis, beams[i + 1].y_axis
            ax.plot([xL, xR], [yB, yT], "g--", lw=1.2)
            ax.plot([xL, xR], [yT, yB], "g--", lw=1.2)
            xc, yc = (xL + xR) / 2, (yB + yT) / 2
            ax.text(xc + 3, yc + 2, f"Aeq={Aeq_univoca:.0f} mm²", fontsize=8, color="green")
        fig.tight_layout()
        fig.savefig(grafico2_png, dpi=300)
        plt.close(fig)

        if export_png:
            written["schema_png"] = schema_png
            written["grafico1_png"] = grafico1_png
            written["grafico2_png"] = grafico2_png

        if export_pdf:
            _first_page(schema_png, stats, first_pdf, header_lines=header_lines)

            merged = False
            if PdfWriter is not None:
                wr = PdfWriter()
                for p in PdfReader(str(first_pdf)).pages:
                    wr.add_page(p)

                if building_type == "muratura":
                    _muratura_summary_page(
                        muratura_pdf,
                        header_lines=header_lines,
                        passo_medio_x_cm=passo_medio_x_cm,
                        passo_medio_y_cm=passo_medio_y_cm,
                    )
                    for p in PdfReader(str(muratura_pdf)).pages:
                        wr.add_page(p)
                else:
                    _extra_pages(matrices_for_pdf, Aeq_dict, Keq_dict, grafico1_png, grafico2_png, Aeq_univoca, extra_pdf)
                    for p in PdfReader(str(extra_pdf)).pages:
                        wr.add_page(p)

                with final_pdf.open("wb") as f:
                    wr.write(f)
                merged = True

            if merged:
                written["pdf_final"] = final_pdf

    if export_dxf:
        dxf_ok = _export_dxf(
            cols, beams,
            finestre_real=all_w_real,
            finestre_infl=all_w_infl,
            X=Xall, Y=Yall,
            Xbase=Xbase, Ybase=Ybase,
            path=dxf_path
        )
        if dxf_ok:
            written["dxf"] = dxf_path

    return written


# ============================================================
# API ENDPOINTS
# ============================================================
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/preview")
def preview(
    payload: Payload,
    schema_scale: float = Query(default=1.0, ge=0.2, le=5.0),
):
    try:
        data = dict(payload.root)
        data["__schema_scale"] = float(schema_scale)
        data = {**data, "export": {"png": False, "pdf": False, "dxf": False}}
        computed = compute(data)
        computed.pop("internals", None)
        return {"ok": True, **computed}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/export")
def export(
    payload: Payload,
    schema_scale: float = Query(default=1.0, ge=0.2, le=5.0),
):
    try:
        data = dict(payload.root)
        data["__schema_scale"] = float(schema_scale)

        sys = _normalize_reinforcement_system(data)
        if sys == "sismagrid":
            raise HTTPException(status_code=400, detail="Export (PDF/DXF) non disponibile per il sistema 'sismagrid'.")

        computed = compute(data)
        job_id = computed["job_id"]

        with tempfile.TemporaryDirectory(prefix="rebarca_") as tmp:
            out_dir = Path(tmp)
            written = render_exports_to_dir(data, computed, out_dir=out_dir)

            mem = BytesIO()
            with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
                pdf_final = written.get("pdf_final")
                dxf = written.get("dxf")

                if pdf_final and Path(pdf_final).is_file():
                    z.write(pdf_final, arcname=Path(pdf_final).name)

                if dxf and Path(dxf).is_file():
                    z.write(dxf, arcname=Path(dxf).name)

            mem.seek(0)
            filename = f"{job_id}_allegati.zip"
            return StreamingResponse(
                mem,
                media_type="application/zip",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))