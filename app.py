# main.py
# =========================================================
# FASTAPI SERVICE - RINFORZO TAMPONAMENTI DA JSON FRONT-END
# READY FOR RENDER + n8n
#
# ENDPOINTS:
# - GET  /health
# - POST /compute
#
# INPUT:
#   stesso payload del front-end (lista o dict)
#
# OUTPUT:
#   JSON finale del calcolo, equivalente a OUTPUT_Rinforzo_Lovable.json
#
# NOTE:
# - Mstar è assunto in TONNELLATE nel JSON/modal
# - verifica su ΔK reale MDOF (+X, -X, +Y, -Y)
# - export ADRS pulito: k_1GDL [g/mm], Se(Fy) [g]
# - plotting disabilitato lato API
#
# LOGICA FINALE SU SUDDIVISIONI:
# - le primarie sono univoche
# - le secondarie sono SOLO quelle vicino alle aperture
# - in Y le secondarie vengono prima calcolate facciata per facciata,
#   poi rese globali/univoche e riportate su tutte le facciate
# - le intermedie vengono generate DOPO, a partire da primarie + secondarie
# =========================================================

import math
import bisect
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# =========================================================
# APP
# =========================================================
app = FastAPI(
    title="Muratura Reinforcement API",
    version="1.2.0",
    description="API per il calcolo del rinforzo dei tamponamenti murari da JSON front-end."
)

# =========================================================
# COSTANTI
# =========================================================
DEFAULT_LAST_HEIGHT = 3.0
GROUPING_TOL_S = 0.40
EPS = 1e-9
EDGE_ANCHOR_TOL = 0.12
QTOP_GROUP_TOL = 1e-6
GROUND_BEAM_THICKNESS_CM = 30.0
PASSO_BASE_CM = 25
G_ACCEL = 9.81

FAMILY_PRIORITY = {
    "100_75_50_25": 1,
    "75_50_25": 2,
    "50_25": 3,
}

DIAGONAL_PRIORITY = {
    "PIATTE": 1,
    "IBRIDE": 2,
    "AC": 3,
}

SYSTEM_FAMILIES = {
    "100_75_50_25": [25, 50, 75, 100],
    "75_50_25": [25, 50, 75],
    "50_25": [25, 50],
}

DIAGONAL_LABEL = {
    "PIATTE": "DIAGONALI PIATTE",
    "IBRIDE": "DIAGONALI IBRIDE",
    "AC": "DIAGONALI A C",
}

DIAGONAL_COEFFS = {
    "DIAGONALI PIATTE": {
        "POSITIVO": {
            "Kel": {"a_s": 50237, "b_s": -1.25, "c_s": -0.967},
            "Fh_y_Hd": {"a_s": 94.8, "b_s": -1.21, "c_s": -1.02},
            "dHu_Hd": {"a_s": 0.304, "b_s": -0.605, "c_s": 1.34},
        },
        "NEGATIVO": {
            "Kel": {"a_s": 50237, "b_s": -1.25, "c_s": -0.967},
            "Fh_y_Hd": {"a_s": 94.8, "b_s": -1.21, "c_s": -1.02},
            "dHu_Hd": {"a_s": 0.304, "b_s": -0.605, "c_s": 1.34},
        },
    },
    "DIAGONALI IBRIDE": {
        "POSITIVO": {
            "Kel": {"a_s": 74600, "b_s": -1.2, "c_s": -0.928},
            "Fh_y_Hd": {"a_s": 506, "b_s": -1.39, "c_s": -1.09},
            "dHu_Hd": {"a_s": 0.00165, "b_s": 0.171, "c_s": 1.07},
        },
        "NEGATIVO": {
            "Kel": {"a_s": 47504, "b_s": -1.12, "c_s": -1.062},
            "Fh_y_Hd": {"a_s": 691, "b_s": -1.44, "c_s": -1.11},
            "dHu_Hd": {"a_s": 0.00204, "b_s": 0.129, "c_s": 1.12},
        },
    },
    "DIAGONALI A C": {
        "POSITIVO": {
            "Kel": {"a_s": 62683, "b_s": -1.11, "c_s": -1.03},
            "Fh_y_Hd": {"a_s": 973, "b_s": -1.44, "c_s": -1.12},
            "dHu_Hd": {"a_s": 0.00119, "b_s": 0.21, "c_s": 1.08},
        },
        "NEGATIVO": {
            "Kel": {"a_s": 62683, "b_s": -1.11, "c_s": -1.03},
            "Fh_y_Hd": {"a_s": 973, "b_s": -1.44, "c_s": -1.12},
            "dHu_Hd": {"a_s": 0.00119, "b_s": 0.21, "c_s": 1.08},
        },
    },
}

# =========================================================
# MODELS
# =========================================================
class ComputeRequest(BaseModel):
    payload: Any


@dataclass
class BeamR:
    y_axis: float
    spess: float


@dataclass
class ColumnR:
    x_axis: float
    spess: float


@dataclass
class WindowR:
    x: float
    y_rel: float
    w: float
    h: float
    y_abs: float = 0.0


# =========================================================
# HELPERS BASE
# =========================================================
def clamp(value: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(value, vmax))


def safe_parallel_sum(values: List[float], eps: float = 1e-12) -> float:
    vals = [float(v) for v in values if v is not None and float(v) > eps]
    return sum(vals) if vals else 0.0


def safe_series_sum(values: List[float], eps: float = 1e-12) -> float:
    vals = [float(v) for v in values if v is not None and float(v) > eps]
    if not vals:
        return 0.0
    denom = sum(1.0 / v for v in vals)
    if denom <= eps:
        return 0.0
    return 1.0 / denom


def safe_mean(values: List[float], eps: float = 1e-12) -> float:
    vals = [float(v) for v in values if v is not None and float(v) > eps]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _sorted_unique(nums: List[float], eps: float = 1e-9) -> List[float]:
    if not nums:
        return []
    s = sorted(float(x) for x in nums)
    out = [s[0]]
    for v in s[1:]:
        if abs(v - out[-1]) > eps:
            out.append(v)
    return out


# =========================================================
# INPUT NORMALIZATION
# =========================================================
def normalize_frontend_payload(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, list):
        if not raw:
            raise ValueError("Payload list vuota.")
        raw = raw[0]

    if not isinstance(raw, dict):
        raise ValueError("Payload non valido: atteso dict o list[dict].")

    body = raw.get("body", raw)
    if not isinstance(body, dict):
        raise ValueError("Payload non valido: campo 'body' assente o non valido.")

    geometry = body.get("geometry")
    deltaK = body.get("deltaK")
    modal = body.get("modal")

    if geometry is None or deltaK is None or modal is None:
        raise ValueError("Payload non valido: richiesti geometry, deltaK, modal.")

    settings = geometry.get("settings", {})

    return {
        "geometry": geometry,
        "deltaK": deltaK,
        "modal": modal,
        "settings": settings,
    }


# =========================================================
# LIVELLI
# =========================================================
def get_level_height(level_obj: Dict[str, Any], default_last_height: float = DEFAULT_LAST_HEIGHT) -> float:
    h = level_obj.get("height_m")
    return float(h) if h is not None else float(default_last_height)


def get_level_top(level_obj: Dict[str, Any], default_last_height: float = DEFAULT_LAST_HEIGHT) -> float:
    return float(level_obj["quota_m"])


def get_level_bottom(level_obj: Dict[str, Any], default_last_height: float = DEFAULT_LAST_HEIGHT) -> float:
    return get_level_top(level_obj, default_last_height) - get_level_height(level_obj, default_last_height)


def get_total_height(global_levels: List[Dict[str, Any]], default_last_height: float = DEFAULT_LAST_HEIGHT) -> float:
    max_top = 0.0
    for lvl in global_levels:
        max_top = max(max_top, get_level_top(lvl, default_last_height))
    return max_top


# =========================================================
# GEOMETRIA BASE
# =========================================================
def get_column_width_m(col: Optional[Dict[str, Any]]) -> float:
    if not col:
        return 0.0
    if col.get("section_along_facade_cm") is not None:
        return float(col["section_along_facade_cm"]) / 100.0
    return float(col.get("width_cm", 30)) / 100.0


def get_opening_geometry(op: Dict[str, Any]) -> Tuple[float, float, float]:
    width_m = float(op["width_cm"]) / 100.0
    if op.get("height_cm") is not None:
        height_m = float(op["height_cm"]) / 100.0
    elif op.get("top_cm") is not None and op.get("sill_height_cm") is not None:
        height_m = (float(op["top_cm"]) - float(op["sill_height_cm"])) / 100.0
    else:
        height_m = 1.20
    sill_m = float(op["sill_height_cm"]) / 100.0 if op.get("sill_height_cm") is not None else 0.0
    return width_m, height_m, sill_m


def get_beam_height_m(beam: Dict[str, Any]) -> float:
    return float(beam.get("height_cm", 30)) / 100.0


def find_matching_qtop_key(keys, q_top: float, tol: float = QTOP_GROUP_TOL):
    for k in keys:
        if abs(k - q_top) <= tol:
            return k
    return None


def get_facade_local_axes(facade: Dict[str, Any]):
    x0 = float(facade["start"]["x"])
    y0 = float(facade["start"]["y"])
    x1 = float(facade["end"]["x"])
    y1 = float(facade["end"]["y"])

    dx = x1 - x0
    dy = y1 - y0
    L = math.hypot(dx, dy)

    if L <= EPS:
        tx, ty = 1.0, 0.0
    else:
        tx, ty = dx / L, dy / L

    nx = float(facade["normal"]["x"])
    ny = float(facade["normal"]["y"])

    return (x0, y0), (tx, ty), (nx, ny)


def get_facade_unit_vectors(facade: Dict[str, Any]):
    x0 = float(facade["start"]["x"])
    y0 = float(facade["start"]["y"])
    x1 = float(facade["end"]["x"])
    y1 = float(facade["end"]["y"])

    dx = x1 - x0
    dy = y1 - y0
    L = math.hypot(dx, dy)
    if L <= EPS:
        raise ValueError(f"Facciata {facade.get('index', '?')} di lunghezza nulla")

    tx, ty = dx / L, dy / L

    n = facade.get("normal", {})
    nx = float(n.get("x", -ty))
    ny = float(n.get("y", tx))
    nL = math.hypot(nx, ny)
    if nL <= EPS:
        nx, ny = -ty, tx
        nL = math.hypot(nx, ny)
    nx /= nL
    ny /= nL

    return (tx, ty), (nx, ny)


def project_world_to_facade_s(x: float, y: float, facade: Dict[str, Any]) -> float:
    (x0, y0), (tx, ty), _ = get_facade_local_axes(facade)
    vx = x - x0
    vy = y - y0
    return vx * tx + vy * ty


def local_to_world_on_facade(facade: Dict[str, Any], s_m: float, z_m: float) -> Tuple[float, float, float]:
    (x0, y0), (tx, ty), _ = get_facade_local_axes(facade)
    x = x0 + s_m * tx
    y = y0 + s_m * ty
    z = z_m
    return x, y, z


def get_column_reference_s(col: Dict[str, Any], facade: Dict[str, Any], effective_role: str = "internal") -> float:
    if effective_role in ("start", "end"):
        if col.get("distance_along_facade_m") is not None:
            return float(col["distance_along_facade_m"])

    pw = col.get("position_world")
    if pw and "x" in pw and "y" in pw:
        return project_world_to_facade_s(float(pw["x"]), float(pw["y"]), facade)

    return float(col.get("distance_along_facade_m", 0.0))


def get_column_interval(col: Dict[str, Any], facade: Dict[str, Any], effective_role: str = "internal") -> Tuple[float, float]:
    s = get_column_reference_s(col, facade, effective_role=effective_role)
    w = get_column_width_m(col)

    if effective_role == "start":
        return s, s + w
    if effective_role == "end":
        return s - w, s
    return s - w / 2.0, s + w / 2.0


def get_facade_frame_key(facade: Dict[str, Any], axis: str, nd: int = 6):
    x0 = float(facade["start"]["x"])
    y0 = float(facade["start"]["y"])
    x1 = float(facade["end"]["x"])
    y1 = float(facade["end"]["y"])

    axis = axis.upper().strip()
    if axis == "X":
        return round((y0 + y1) / 2.0, nd)
    if axis == "Y":
        return round((x0 + x1) / 2.0, nd)
    raise ValueError("axis deve essere X o Y")


# =========================================================
# PILASTRATE CONTINUE
# =========================================================
def extract_facade_column_slices(facade: Dict[str, Any]) -> List[Dict[str, Any]]:
    slices = []
    facade_length = float(facade["length_m"])

    for lvl in facade.get("levels", []):
        q_top = get_level_top(lvl, DEFAULT_LAST_HEIGHT)
        q_base = get_level_bottom(lvl, DEFAULT_LAST_HEIGHT)

        start_col = lvl.get("start_column")
        end_col = lvl.get("end_column")

        start_id = start_col.get("id") if start_col else None
        end_id = end_col.get("id") if end_col else None

        for col in lvl.get("columns", []):
            col_id = col.get("id")

            if start_id is not None and col_id == start_id:
                role_semantic = "start"
            elif end_id is not None and col_id == end_id:
                role_semantic = "end"
            else:
                role_semantic = "internal"

            d = col.get("distance_along_facade_m", None)

            if role_semantic == "start" and d is not None and abs(float(d) - 0.0) <= EDGE_ANCHOR_TOL:
                effective_role = "start"
            elif role_semantic == "end" and d is not None and abs(float(d) - facade_length) <= EDGE_ANCHOR_TOL:
                effective_role = "end"
            else:
                effective_role = "internal"

            s_ref = get_column_reference_s(col, facade, effective_role=effective_role)
            s_min, s_max = get_column_interval(col, facade, effective_role=effective_role)
            s_center = 0.5 * (s_min + s_max)

            slices.append({
                "level_name": lvl["level_name"],
                "q_base": q_base,
                "q_top": q_top,
                "role": role_semantic,
                "effective_role": effective_role,
                "s": s_ref,
                "s_min": s_min,
                "s_max": s_max,
                "s_center": s_center,
                "width": s_max - s_min,
            })

    slices.sort(key=lambda x: (x["q_base"], x["s_center"]))
    return slices


def cluster_column_slices_vertical(slices: List[Dict[str, Any]], grouping_tol: float = GROUPING_TOL_S):
    groups = []

    for sl in slices:
        best_group = None
        best_score = None

        for grp in groups:
            overlap_len = min(sl["s_max"], grp["env_max"]) - max(sl["s_min"], grp["env_min"])
            center_dist = abs(sl["s_center"] - grp["center_ref"])

            compatible = (overlap_len >= -grouping_tol) or (center_dist <= grouping_tol)
            if not compatible:
                continue

            score = (center_dist, -overlap_len)
            if best_score is None or score < best_score:
                best_score = score
                best_group = grp

        if best_group is not None:
            best_group["members"].append(sl)
            best_group["env_min"] = min(best_group["env_min"], sl["s_min"])
            best_group["env_max"] = max(best_group["env_max"], sl["s_max"])
            best_group["center_ref"] = sum(m["s_center"] for m in best_group["members"]) / len(best_group["members"])
        else:
            groups.append({
                "members": [sl],
                "env_min": sl["s_min"],
                "env_max": sl["s_max"],
                "center_ref": sl["s_center"],
            })

    return groups


def build_vertical_pilastrate_for_facade(facade: Dict[str, Any], global_levels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    facade_length = float(facade["length_m"])
    total_height = get_total_height(global_levels, DEFAULT_LAST_HEIGHT)

    slices = extract_facade_column_slices(facade)
    if not slices:
        return []

    raw_groups = cluster_column_slices_vertical(slices, GROUPING_TOL_S)

    bands = []
    for i, grp in enumerate(raw_groups, start=1):
        members = grp["members"]
        s_common_min = max(m["s_min"] for m in members)
        s_common_max = min(m["s_max"] for m in members)

        if s_common_max - s_common_min <= EPS:
            continue

        vis_min = max(0.0, s_common_min)
        vis_max = min(facade_length, s_common_max)
        if vis_max - vis_min <= EPS:
            continue

        bands.append({
            "group_index": i,
            "x_left": vis_min,
            "x_right": vis_max,
            "x_center": 0.5 * (vis_min + vis_max),
            "width": vis_max - vis_min,
            "y_bottom": 0.0,
            "height": total_height,
            "member_count": len(members),
        })

    bands.sort(key=lambda b: b["x_left"])
    return bands


# =========================================================
# TRAVI CONTINUE
# =========================================================
def build_global_min_beam_height_by_qtop(geometry: Dict[str, Any]) -> Dict[float, float]:
    grouped_heights = {}

    for facade in geometry.get("facades", []):
        for lvl in facade.get("levels", []):
            q_top = get_level_top(lvl, DEFAULT_LAST_HEIGHT)
            for beam in lvl.get("beams", []):
                h = get_beam_height_m(beam)
                existing_key = find_matching_qtop_key(grouped_heights.keys(), q_top, QTOP_GROUP_TOL)
                if existing_key is None:
                    grouped_heights[q_top] = [h]
                else:
                    grouped_heights[existing_key].append(h)

    min_beam_height_by_qtop = {}
    for q_top, heights in grouped_heights.items():
        if heights:
            min_beam_height_by_qtop[q_top] = min(heights)

    return min_beam_height_by_qtop


def get_min_beam_height_for_qtop(q_top: float, min_beam_height_by_qtop: Dict[float, float], tol: float = QTOP_GROUP_TOL):
    for k, v in min_beam_height_by_qtop.items():
        if abs(k - q_top) <= tol:
            return v
    return None


def build_continuous_beam_bands_for_facade(
    facade: Dict[str, Any],
    min_beam_height_by_qtop: Dict[float, float],
    pilastrate_bands: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    facade_length = float(facade["length_m"])

    if pilastrate_bands:
        x_global_min = min(b["x_left"] for b in pilastrate_bands)
        x_global_max = max(b["x_right"] for b in pilastrate_bands)
    else:
        x_global_min = 0.0
        x_global_max = facade_length

    bands = []

    ground_h = GROUND_BEAM_THICKNESS_CM / 100.0
    if x_global_max - x_global_min > EPS:
        bands.append({
            "level_name": "GROUND",
            "q_top": ground_h,
            "x_left": x_global_min,
            "x_right": x_global_max,
            "x_center": 0.5 * (x_global_min + x_global_max),
            "beam_len": x_global_max - x_global_min,
            "beam_height_m": ground_h,
            "y_bottom": 0.0,
            "y_top": ground_h,
            "y_center": 0.5 * ground_h,
        })

    for lvl in facade.get("levels", []):
        q_top = get_level_top(lvl, DEFAULT_LAST_HEIGHT)
        beam_h = get_min_beam_height_for_qtop(q_top, min_beam_height_by_qtop, QTOP_GROUP_TOL)
        if beam_h is None:
            continue

        y_bottom = q_top - beam_h
        x_left = clamp(x_global_min, 0.0, facade_length)
        x_right = clamp(x_global_max, 0.0, facade_length)

        if x_right - x_left <= EPS:
            continue

        bands.append({
            "level_name": lvl["level_name"],
            "q_top": q_top,
            "x_left": x_left,
            "x_right": x_right,
            "x_center": 0.5 * (x_left + x_right),
            "beam_len": x_right - x_left,
            "beam_height_m": beam_h,
            "y_bottom": y_bottom,
            "y_top": q_top,
            "y_center": 0.5 * (y_bottom + q_top),
        })

    bands.sort(key=lambda b: b["y_center"])
    return bands


# =========================================================
# MODELLO FACCIATA
# =========================================================
def build_facade_opening_windows_R(facade: Dict[str, Any]) -> List[WindowR]:
    windows = []

    for lvl in facade.get("levels", []):
        q_base_m = get_level_bottom(lvl, DEFAULT_LAST_HEIGHT)

        for op in lvl.get("openings", []):
            width_m, height_m, sill_m = get_opening_geometry(op)
            s_center_m = float(op["distance_along_facade_m"])
            x_left_m = s_center_m - width_m / 2.0
            y_abs_m = q_base_m + sill_m

            windows.append(
                WindowR(
                    x=x_left_m * 100.0,
                    y_rel=sill_m * 100.0,
                    w=width_m * 100.0,
                    h=height_m * 100.0,
                    y_abs=y_abs_m * 100.0,
                )
            )
    return windows


def facade_to_reinforcement_model(
    facade: Dict[str, Any],
    global_levels: List[Dict[str, Any]],
    min_beam_height_by_qtop: Dict[float, float]
) -> Dict[str, Any]:
    pilastrate_bands = build_vertical_pilastrate_for_facade(facade, global_levels)
    beam_bands = build_continuous_beam_bands_for_facade(
        facade=facade,
        min_beam_height_by_qtop=min_beam_height_by_qtop,
        pilastrate_bands=pilastrate_bands,
    )

    cols = [ColumnR(x_axis=band["x_center"] * 100.0, spess=band["width"] * 100.0) for band in pilastrate_bands]
    cols.sort(key=lambda c: c.x_axis)

    beams = [BeamR(y_axis=band["y_center"] * 100.0, spess=band["beam_height_m"] * 100.0) for band in beam_bands]
    beams.sort(key=lambda b: b.y_axis)

    windows = build_facade_opening_windows_R(facade)

    return {
        "pilastrate_bands": pilastrate_bands,
        "beam_bands": beam_bands,
        "cols": cols,
        "beams": beams,
        "windows": windows,
    }


# =========================================================
# LINEE VICINO APERTURE + FINESTRE GONFIATE CLAMPATE
# =========================================================
def _overlap_strict(a1: float, a2: float, b1: float, b2: float, eps: float = 1e-9) -> bool:
    lo1, hi1 = (a1, a2) if a1 <= a2 else (a2, a1)
    lo2, hi2 = (b1, b2) if b1 <= b2 else (b2, b1)
    return not (hi1 <= lo2 + eps or hi2 <= lo1 + eps)


def linee_finestre_candidate_driven(grid: List[float], wins_gonf: List[WindowR], asse: str, eps: float = 1e-9) -> List[float]:
    grid = sorted(grid)
    if not grid or not wins_gonf:
        return []

    def x_inside(w: WindowR, x: float) -> bool:
        return (w.x + eps) < x < (w.x + w.w - eps)

    def y_inside(w: WindowR, y: float) -> bool:
        return (w.y_abs + eps) < y < (w.y_abs + w.h - eps)

    def competitors_for_x(wA: WindowR) -> List[WindowR]:
        out = []
        Ay1, Ay2 = wA.y_abs, wA.y_abs + wA.h
        for wB in wins_gonf:
            if wB is wA:
                continue
            By1, By2 = wB.y_abs, wB.y_abs + wB.h
            if _overlap_strict(Ay1, Ay2, By1, By2, eps=eps):
                out.append(wB)
        return out

    def competitors_for_y(wA: WindowR) -> List[WindowR]:
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


def inflate_window_clamped_to_panel(
    w: WindowR,
    i: int,
    j: int,
    cols: List[ColumnR],
    beams: List[BeamR],
    DISTF: float,
    eps: float = 1e-9,
) -> WindowR:
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

    return WindowR(
        x=xa - pad_l,
        y_rel=w.y_rel,
        w=w.w + pad_l + pad_r,
        h=w.h + pad_b + pad_t,
        y_abs=ya - pad_b,
    )


# =========================================================
# GRIGLIA RINFORZO
# =========================================================
def primarie(nodes, vertical: bool, PASSO: int, CLEAR: int):
    lo = nodes[0].x_axis if vertical else nodes[0].y_axis
    hi = nodes[-1].x_axis if vertical else nodes[-1].y_axis
    low = lo - nodes[0].spess / 2 + CLEAR
    high = hi + nodes[-1].spess / 2 - CLEAR

    best = 1e9
    full = []

    for z0 in range(int(low), int(low) + PASSO):
        s = ((low - z0 + PASSO - 1) // PASSO) * PASSO + z0
        e = ((high - z0) // PASSO) * PASSO + z0
        if s > e:
            continue
        g = list(range(int(s), int(e) + 1, PASSO))

        ok_all = True
        for n in nodes:
            a = n.x_axis if vertical else n.y_axis
            if not any(a - n.spess / 2 + CLEAR <= v <= a + n.spess / 2 - CLEAR for v in g):
                ok_all = False
                break

        if not ok_all:
            continue

        scarto = (g[0] - low) + (high - g[-1])
        if scarto < best:
            best = scarto
            full = g

    if not full:
        raise ValueError("Maglia primaria impossibile per questa facciata.")

    base = []
    for n in nodes:
        a = n.x_axis if vertical else n.y_axis
        vals = [v for v in full if a - n.spess / 2 + CLEAR <= v <= a + n.spess / 2 - CLEAR]
        base.append(min(vals, key=lambda v: abs(v - a)))

    return full, base


def intermedie(lines: List[float], PASSO: int, possible_steps=None):
    if possible_steps is None:
        possible_steps = [25, 50, 75, 100]

    steps = sorted(set(int(v) for v in possible_steps))
    out = []

    for s in steps:
        if s % PASSO != 0:
            raise ValueError(f"step {s} non multiplo del passo base {PASSO}")

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
            raise ValueError(f"Intervallo non multiplo del passo base {PASSO}: rem={rem}")

        target_units = int(round(rem / PASSO))
        step_units = [(s, s // PASSO) for s in steps_desc]

        INF = (10**9, 10**9, 10**18)
        dp_cost = [INF] * (target_units + 1)
        prev = [None] * (target_units + 1)
        dp_cost[0] = (0, 0, 0)

        for u in range(target_units + 1):
            if dp_cost[u] == INF:
                continue
            n25, nseg, neg_score = dp_cost[u]
            for s_cm, s_u in step_units:
                uu = u + s_u
                if uu > target_units:
                    continue
                cand = (
                    n25 + (1 if s_cm == 25 else 0),
                    nseg + 1,
                    neg_score - (s_cm * s_cm),
                )
                if cand < dp_cost[uu]:
                    dp_cost[uu] = cand
                    prev[uu] = (u, s_cm)

        if dp_cost[target_units] == INF:
            raise ValueError(f"Impossibile decomporre intervallo {rem} con steps {steps}")

        used_steps = []
        cur = target_units
        while cur != 0:
            p = prev[cur]
            if p is None:
                raise ValueError("Errore interno ricostruzione DP")
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


# =========================================================
# INTERSEZIONI / CLIPPING
# =========================================================
def win_box(w: WindowR, pad: float = 0.0):
    return (w.x - pad, w.y_abs - pad, w.x + w.w + pad, w.y_abs + w.h + pad)


def _box_intersect_strict(a, b, eps: float = 1e-9):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 + eps or ax1 >= bx2 - eps or ay2 <= by1 + eps or ay1 >= by2 - eps)


def ok_seg(x1: float, y1: float, x2: float, y2: float, wins: List[WindowR], DISTF: float = 0.0, eps: float = 1e-9) -> bool:
    xmin, xmax = sorted((x1, x2))
    ymin, ymax = sorted((y1, y2))
    seg_box = (xmin, ymin, xmax, ymax)
    for w in wins:
        if _box_intersect_strict(seg_box, win_box(w, pad=DISTF), eps=eps):
            return False
    return True


def _merge_intervals(ints):
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


def _subtract_intervals(base, cuts):
    a, b = min(base), max(base)
    cuts = _merge_intervals([(max(a, c1), min(b, c2)) for c1, c2 in cuts if not (c2 <= a or c1 >= b)])
    if not cuts:
        return [(a, b)]
    out = []
    cur = a
    for c1, c2 in cuts:
        if c1 > cur:
            out.append((cur, c1))
        cur = max(cur, c2)
    if cur < b:
        out.append((cur, b))
    return [(x1, x2) for x1, x2 in out if x2 > x1]


def clip_vertical_segment(x: float, y1: float, y2: float, wins: List[WindowR], DISTF: float, eps: float = 1e-9):
    ya, yb = min(y1, y2), max(y1, y2)
    cuts = []
    for w in wins:
        xmin, ymin, xmax, ymax = win_box(w, pad=DISTF)
        if (xmin + eps) < x < (xmax - eps):
            cuts.append((ymin, ymax))
    return _subtract_intervals((ya, yb), cuts)


def clip_horizontal_segment(y: float, x1: float, x2: float, wins: List[WindowR], DISTF: float, eps: float = 1e-9):
    xa, xb = min(x1, x2), max(x1, x2)
    cuts = []
    for w in wins:
        xmin, ymin, xmax, ymax = win_box(w, pad=DISTF)
        if (ymin + eps) < y < (ymax - eps):
            cuts.append((xmin, xmax))
    return _subtract_intervals((xa, xb), cuts)


def _k(v: float, nd: int = 6):
    return round(float(v), nd)


def split_segments_at_intersections(v_segs, h_segs, nd: int = 6):
    v_norm = [(_k(x, nd), _k(min(y1, y2), nd), _k(max(y1, y2), nd)) for x, y1, y2 in v_segs if max(y1, y2) > min(y1, y2)]
    h_norm = [(_k(y, nd), _k(min(x1, x2), nd), _k(max(x1, x2), nd)) for y, x1, x2 in h_segs if max(x1, x2) > min(x1, x2)]

    v_points = {}
    h_points = {}

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

    edges = []
    adj = defaultdict(set)

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


def prune_dangling(edges, adj, protected_nodes):
    alive = [True] * len(edges)

    def degree(node):
        return sum(1 for eid in adj.get(node, set()) if alive[eid])

    queue = deque([n for n in adj.keys() if degree(n) == 1 and n not in protected_nodes])

    while queue:
        n = queue.popleft()
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
            queue.append(other)

    return alive


def merge_atomic_edges(edges, alive, nd: int = 6):
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


def prune_vh_segments(Xall, Yall, Xbase, Ybase, v_segs, h_segs, nd: int = 6):
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


# =========================================================
# NUOVO: Y SECONDARIE GLOBALI
# =========================================================
def build_global_unique_y_secondary(
    geometry: Dict[str, Any],
    min_beam_height_by_qtop: Dict[float, float],
    clear_cm: int,
    distf_cm: int,
    forced_Ybase: List[float],
) -> List[float]:
    """
    Le secondarie Y sono SOLO le linee vicino alle aperture.
    Logica:
    1. per ogni facciata costruisco il modello locale
    2. ricavo Yfull locale
    3. ricavo Ysec locale = linee candidate vicino finestre
    4. tolgo coincidenze con Ybase globale
    5. faccio l'unione globale
    """

    global_levels = geometry["levels"]
    all_ysec = []

    for facade in geometry["facades"]:
        model = facade_to_reinforcement_model(
            facade=facade,
            global_levels=global_levels,
            min_beam_height_by_qtop=min_beam_height_by_qtop,
        )

        beams = model["beams"]
        windows = model["windows"]

        if len(beams) < 2:
            continue

        Yfull_local, _ = primarie(
            beams,
            vertical=False,
            PASSO=PASSO_BASE_CM,
            CLEAR=clear_cm,
        )

        windows_infl = []
        for w in windows:
            windows_infl.append(
                WindowR(
                    x=w.x - distf_cm,
                    y_rel=w.y_rel,
                    w=w.w + 2 * distf_cm,
                    h=w.h + 2 * distf_cm,
                    y_abs=w.y_abs - distf_cm,
                )
            )

        Ysec_local = sorted(set(linee_finestre_candidate_driven(Yfull_local, windows_infl, "y")))
        Ysec_local = [y for y in Ysec_local if all(abs(y - yb) > EPS for yb in forced_Ybase)]

        all_ysec.extend(Ysec_local)

    return sorted(set(all_ysec))


# =========================================================
# BUILD REINFORCEMENT PER FACCIATA
# =========================================================
def build_reinforcement_for_facade(
    facade: Dict[str, Any],
    global_levels: List[Dict[str, Any]],
    min_beam_height_by_qtop: Dict[float, float],
    possible_steps_cm: List[int],
    clear_cm: int,
    distf_cm: int,
    forced_Ybase=None,
    forced_Ysec=None,
) -> Dict[str, Any]:
    model = facade_to_reinforcement_model(facade, global_levels, min_beam_height_by_qtop)

    cols = model["cols"]
    beams = model["beams"]
    windows = model["windows"]

    if len(cols) < 2 or len(beams) < 2:
        return {
            **model,
            "Xall": [],
            "Yall": [],
            "Xbase": [],
            "Ybase": [],
            "Xsec": [],
            "Ysec": [],
            "Xint": [],
            "Yint": [],
            "v_segs": [],
            "h_segs": [],
            "d_segs": [],
            "v_full": [],
            "h_full": [],
            "d_full": [],
        }

    # -----------------------------------------------------
    # 1) PRIMARIE X
    # -----------------------------------------------------
    Xfull, Xbase = primarie(cols, vertical=True, PASSO=PASSO_BASE_CM, CLEAR=clear_cm)

    # -----------------------------------------------------
    # 2) PRIMARIE Y
    # -----------------------------------------------------
    Yfull_local, Ybase_local = primarie(
        beams,
        vertical=False,
        PASSO=PASSO_BASE_CM,
        CLEAR=clear_cm,
    )

    if forced_Ybase is None:
        Ybase = Ybase_local
    else:
        Ybase = sorted(set(float(y) for y in forced_Ybase))

    Yfull = Yfull_local

    # -----------------------------------------------------
    # 3) FINESTRE INFLUENZATE
    # -----------------------------------------------------
    windows_infl = []
    for w in windows:
        windows_infl.append(
            WindowR(
                x=w.x - distf_cm,
                y_rel=w.y_rel,
                w=w.w + 2 * distf_cm,
                h=w.h + 2 * distf_cm,
                y_abs=w.y_abs - distf_cm,
            )
        )

    # -----------------------------------------------------
    # 4) SECONDARIE VERE
    # -----------------------------------------------------
    Xsec = sorted(set(linee_finestre_candidate_driven(Xfull, windows_infl, "x")))
    Ysec_local = sorted(set(linee_finestre_candidate_driven(Yfull, windows_infl, "y")))

    Xsec = [x for x in Xsec if all(abs(x - xb) > EPS for xb in Xbase)]

    if forced_Ysec is None:
        Ysec = [y for y in Ysec_local if all(abs(y - yb) > EPS for yb in Ybase)]
    else:
        Ysec = sorted(set(float(y) for y in forced_Ysec if all(abs(float(y) - yb) > EPS for yb in Ybase)))

    # -----------------------------------------------------
    # 5) INTERMEDIE
    # -----------------------------------------------------
    Xint = intermedie(
        sorted(set(Xbase + Xsec)),
        PASSO=PASSO_BASE_CM,
        possible_steps=possible_steps_cm,
    )

    Yint = intermedie(
        sorted(set(Ybase + Ysec)),
        PASSO=PASSO_BASE_CM,
        possible_steps=possible_steps_cm,
    )

    Xint = [x for x in Xint if all(abs(x - xb) > EPS for xb in (Xbase + Xsec))]
    Yint = [y for y in Yint if all(abs(y - yb) > EPS for yb in (Ybase + Ysec))]

    # -----------------------------------------------------
    # 6) GRIGLIE FINALI
    # -----------------------------------------------------
    Xall = sorted(set(Xbase + Xsec + Xint))
    Yall = sorted(set(Ybase + Ysec + Yint))

    # -----------------------------------------------------
    # 7) V/H
    # -----------------------------------------------------
    v_raw = []
    h_raw = []

    for y in Yall:
        for x1, x2 in zip(Xall[:-1], Xall[1:]):
            for xa, xb in clip_horizontal_segment(y, x1, x2, windows_infl, DISTF=0):
                if xb > xa:
                    h_raw.append((y, xa, xb))

    for x in Xall:
        for y1, y2 in zip(Yall[:-1], Yall[1:]):
            for ya, yb in clip_vertical_segment(x, y1, y2, windows_infl, DISTF=0):
                if yb > ya:
                    v_raw.append((x, ya, yb))

    v_segs, h_segs = prune_vh_segments(Xall, Yall, Xbase, Ybase, v_raw, h_raw, nd=6)

    # -----------------------------------------------------
    # 8) DIAGONALI
    # -----------------------------------------------------
    d_segs = []
    for i in range(len(Xall) - 1):
        for j in range(len(Yall) - 1):
            a, b = Xall[i], Xall[i + 1]
            c, d = Yall[j], Yall[j + 1]

            if ok_seg(a, c, b, d, windows_infl, DISTF=0):
                d_segs.append((a, c, b, d))
            if ok_seg(a, d, b, c, windows_infl, DISTF=0):
                d_segs.append((a, d, b, c))

    v_full = []
    h_full = []
    d_full = []

    for x in Xall:
        for y1, y2 in zip(Yall[:-1], Yall[1:]):
            if y2 > y1:
                v_full.append((x, y1, y2))

    for y in Yall:
        for x1, x2 in zip(Xall[:-1], Xall[1:]):
            if x2 > x1:
                h_full.append((y, x1, x2))

    for i in range(len(Xall) - 1):
        for j in range(len(Yall) - 1):
            a, b = Xall[i], Xall[i + 1]
            c, d = Yall[j], Yall[j + 1]
            d_full.append((a, c, b, d))
            d_full.append((a, d, b, c))

    return {
        **model,
        "Xall": Xall,
        "Yall": Yall,
        "Xbase": Xbase,
        "Ybase": Ybase,
        "Xsec": Xsec,
        "Ysec": Ysec,
        "Xint": Xint,
        "Yint": Yint,
        "v_segs": v_segs,
        "h_segs": h_segs,
        "d_segs": d_segs,
        "v_full": v_full,
        "h_full": h_full,
        "d_full": d_full,
    }


# =========================================================
# PANNELLI / RAPPORTO APERTURE
# =========================================================
def _panel_step_mean_from_grid(Xall: List[float], Yall: List[float], x1: float, x2: float, y1: float, y2: float) -> Tuple[float, float, float]:
    xs_u = _sorted_unique(Xall)
    ys_u = _sorted_unique(Yall)

    x_lo, x_hi = min(x1, x2), max(x1, x2)
    y_lo, y_hi = min(y1, y2), max(y1, y2)

    xs = [v for v in xs_u if x_lo - EPS <= v <= x_hi + EPS]
    ys = [v for v in ys_u if y_lo - EPS <= v <= y_hi + EPS]

    xs = _sorted_unique(xs)
    ys = _sorted_unique(ys)

    dxs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    dys = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]

    mx = sum(dxs) / len(dxs) if dxs else 0.0
    my = sum(dys) / len(dys) if dys else 0.0
    pm = (mx + my) / 2.0 if (mx > 0 and my > 0) else (mx or my or 0.0)

    return mx, my, pm


def opening_overlap_area_cm2(panel_x1: float, panel_x2: float, panel_y1: float, panel_y2: float, window: WindowR) -> float:
    x1 = max(min(panel_x1, panel_x2), window.x)
    x2 = min(max(panel_x1, panel_x2), window.x + window.w)
    y1 = max(min(panel_y1, panel_y2), window.y_abs)
    y2 = min(max(panel_y1, panel_y2), window.y_abs + window.h)

    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def compute_panel_opening_ratio(panel_x1: float, panel_x2: float, panel_y1: float, panel_y2: float, windows: List[WindowR]) -> float:
    gross = abs(panel_x2 - panel_x1) * abs(panel_y2 - panel_y1)
    if gross <= EPS:
        return 0.0

    open_area = 0.0
    for w in windows:
        open_area += opening_overlap_area_cm2(panel_x1, panel_x2, panel_y1, panel_y2, w)

    ratio = max(0.0, min(1.0, (gross - open_area) / gross))
    return ratio


def build_real_panel_dataframe_for_system(
    geometry: Dict[str, Any],
    min_beam_height_by_qtop: Dict[float, float],
    possible_steps_cm: List[int],
    clear_cm: int,
    distf_cm: int,
    forced_Ybase: List[float],
    forced_Ysec: List[float],
):
    global_levels = geometry["levels"]
    rows = []
    reinf_by_facade = {}

    for facade in geometry["facades"]:
        reinf = build_reinforcement_for_facade(
            facade=facade,
            global_levels=global_levels,
            min_beam_height_by_qtop=min_beam_height_by_qtop,
            possible_steps_cm=possible_steps_cm,
            clear_cm=clear_cm,
            distf_cm=distf_cm,
            forced_Ybase=forced_Ybase,
            forced_Ysec=forced_Ysec,
        )
        reinf_by_facade[facade["index"]] = reinf

        cols = reinf["cols"]
        beams = reinf["beams"]
        Xall = reinf["Xall"]
        Yall = reinf["Yall"]
        windows = reinf["windows"]

        if len(cols) < 2 or len(beams) < 2 or len(Xall) < 2 or len(Yall) < 2:
            continue

        for i in range(len(beams) - 1):
            for j in range(len(cols) - 1):
                x1 = cols[j].x_axis
                x2 = cols[j + 1].x_axis
                y1 = beams[i].y_axis
                y2 = beams[i + 1].y_axis

                H_cm = y2 - y1
                B_cm = x2 - x1

                if H_cm <= EPS or B_cm <= EPS:
                    continue

                mx, my, pm = _panel_step_mean_from_grid(Xall, Yall, x1, x2, y1, y2)
                panel_tag = f"F{facade['index']}_P{i+1}_C{j+1}"
                ratio = compute_panel_opening_ratio(x1, x2, y1, y2, windows)

                rows.append({
                    "Facciata": int(facade["index"]),
                    "Piano": i + 1,
                    "Campata": j + 1,
                    "Tamponamento": panel_tag,
                    "H [cm]": H_cm,
                    "B [cm]": B_cm,
                    "Pm [cm]": pm,
                    "mx [cm]": mx,
                    "my [cm]": my,
                    "ratio": ratio,
                })

    return rows, reinf_by_facade


def build_global_unique_y_primary(geometry: Dict[str, Any], min_beam_height_by_qtop: Dict[float, float], clear_cm: int):
    global_levels = geometry["levels"]
    arr_base = []

    for facade in geometry["facades"]:
        model = facade_to_reinforcement_model(facade, global_levels, min_beam_height_by_qtop)
        cols = model["cols"]
        beams = model["beams"]

        if len(cols) < 2 or len(beams) < 2:
            continue

        _, Ybase = primarie(beams, vertical=False, PASSO=PASSO_BASE_CM, CLEAR=clear_cm)
        arr_base.extend(Ybase)

    return sorted(set(arr_base))


# =========================================================
# BILINEARI PANNELLO
# =========================================================
def _calc_diag_formula(p_d_mm: float, H_d_cm: float, B_d_cm: float, coeff: Dict[str, float]) -> float:
    a_s = coeff["a_s"]
    b_s = coeff["b_s"]
    c_s = coeff["c_s"]

    if p_d_mm <= 0 or H_d_cm <= 0 or B_d_cm <= 0:
        return 0.0

    return a_s * (p_d_mm ** b_s) * ((H_d_cm / B_d_cm) ** c_s)


def build_bilinear_for_panel(H_cm: float, B_cm: float, Pm_cm: float, diagonale_tipo: str, ratio: float = 1.0) -> Dict[str, Any]:
    coeffs = DIAGONAL_COEFFS[diagonale_tipo]
    p_d_mm = Pm_cm * 10.0
    H_mm = H_cm * 10.0

    out = {}
    for ramo in ["POSITIVO", "NEGATIVO"]:
        ramo_coeff = coeffs[ramo]

        Kel_kNmm = _calc_diag_formula(p_d_mm, H_cm, B_cm, ramo_coeff["Kel"])
        Fh_y_Hd_kNmm = _calc_diag_formula(p_d_mm, H_cm, B_cm, ramo_coeff["Fh_y_Hd"])
        dHu_Hd = _calc_diag_formula(p_d_mm, H_cm, B_cm, ramo_coeff["dHu_Hd"])

        Fy_kN = Fh_y_Hd_kNmm * H_mm
        dH_u_mm = dHu_Hd * H_mm
        dH_y_mm = Fy_kN / Kel_kNmm if abs(Kel_kNmm) > EPS else 0.0
        if dH_u_mm > 0 and dH_y_mm > dH_u_mm:
            dH_y_mm = dH_u_mm

        out[ramo] = {
            "Kel_ratio_kNmm": Kel_kNmm * ratio,
            "Fy_ratio_kN": Fy_kN * ratio,
            "dH_y_ratio_mm": dH_y_mm,
            "dH_u_ratio_mm": dH_u_mm,
        }

    return {
        "tipo_diagonale": diagonale_tipo,
        "positivo": out["POSITIVO"],
        "negativo": out["NEGATIVO"],
    }


def project_local_panel_to_global_direction(K_local_Nmm: float, Fy_local_kN: float, facade: Dict[str, Any], axis: str) -> Tuple[float, float]:
    (tx, ty), (_nx, _ny) = get_facade_unit_vectors(facade)
    axis = axis.upper().strip()

    if axis == "X":
        c = abs(tx)
    elif axis == "Y":
        c = abs(ty)
    else:
        raise ValueError(f"Asse non valido: {axis}")

    K_dir = K_local_Nmm * (c ** 2)
    Fy_dir = Fy_local_kN * c
    return K_dir, Fy_dir


def direction_to_axis_and_branch(direction: str) -> Tuple[str, str]:
    d = direction.upper().strip()
    if d == "+X":
        return "X", "POSITIVO"
    if d == "-X":
        return "X", "NEGATIVO"
    if d == "+Y":
        return "Y", "POSITIVO"
    if d == "-Y":
        return "Y", "NEGATIVO"
    raise ValueError(f"Direzione non valida: {direction}")


# =========================================================
# AGGREGAZIONE MDOF
# =========================================================
def build_directional_panel_dataframe(df_panels: List[Dict[str, Any]], geometry: Dict[str, Any], direction: str, diag_key: str):
    axis, branch = direction_to_axis_and_branch(direction)
    facades_by_idx = {f["index"]: f for f in geometry["facades"]}
    rows = []

    for row in df_panels:
        facade = facades_by_idx[int(row["Facciata"])]
        H_cm = float(row["H [cm]"])
        B_cm = float(row["B [cm]"])
        Pm_cm = float(row["Pm [cm]"])
        ratio = float(row["ratio"])
        panel_tag = row["Tamponamento"]

        bil = build_bilinear_for_panel(
            H_cm=H_cm,
            B_cm=B_cm,
            Pm_cm=Pm_cm,
            diagonale_tipo=DIAGONAL_LABEL[diag_key],
            ratio=ratio,
        )

        ramo = bil["positivo"] if branch == "POSITIVO" else bil["negativo"]
        Fy_loc_kN = abs(float(ramo["Fy_ratio_kN"]))
        dHy_mm = abs(float(ramo["dH_y_ratio_mm"]))
        K_loc_kNmm = Fy_loc_kN / dHy_mm if dHy_mm > EPS else 0.0
        K_loc_Nmm = K_loc_kNmm * 1000.0

        K_dir, Fy_dir = project_local_panel_to_global_direction(
            K_local_Nmm=K_loc_Nmm,
            Fy_local_kN=Fy_loc_kN,
            facade=facade,
            axis=axis,
        )

        frame_key = get_facade_frame_key(facade, axis)

        rows.append({
            "Facciata": int(row["Facciata"]),
            "Tamponamento": panel_tag,
            "Piano": int(row["Piano"]),
            "Campata": int(row["Campata"]),
            "FrameKey": frame_key,
            "Direction": direction,
            "K_panel_dir [N/mm]": K_dir,
            "Fy_panel_dir [kN]": Fy_dir,
        })

    return rows


def combine_directional_panels_series_parallel(df_dir_panels: List[Dict[str, Any]]) -> Dict[str, float]:
    if not df_dir_panels:
        return {
            "K_MDOF [N/mm]": 0.0,
            "Fy_MDOF [kN]": 0.0,
        }

    floor_group = defaultdict(list)
    for row in df_dir_panels:
        floor_group[(row["FrameKey"], row["Piano"])].append(row)

    floor_rows = []
    for (frame_key, piano), sub in floor_group.items():
        K_floor = safe_parallel_sum([r["K_panel_dir [N/mm]"] for r in sub])
        floor_rows.append({
            "FrameKey": frame_key,
            "Piano": int(piano),
            "K_floor_parallel [N/mm]": K_floor,
        })

    frame_group_K = defaultdict(list)
    for row in floor_rows:
        frame_group_K[row["FrameKey"]].append(row["K_floor_parallel [N/mm]"])

    frameK_rows = []
    for frame_key, values in frame_group_K.items():
        K_frame = safe_series_sum(values)
        frameK_rows.append({
            "FrameKey": frame_key,
            "K_frame [N/mm]": K_frame,
        })

    bay_group = defaultdict(list)
    for row in df_dir_panels:
        bay_group[(row["FrameKey"], row["Campata"])].append(row["Fy_panel_dir [kN]"])

    bay_rows = []
    for (frame_key, campata), values in bay_group.items():
        Fy_bay = safe_mean(values)
        bay_rows.append({
            "FrameKey": frame_key,
            "Campata": int(campata),
            "Fy_bay_mean [kN]": Fy_bay,
        })

    frame_group_F = defaultdict(list)
    for row in bay_rows:
        frame_group_F[row["FrameKey"]].append(row["Fy_bay_mean [kN]"])

    frameF_rows = []
    for frame_key, values in frame_group_F.items():
        Fy_frame = safe_parallel_sum(values)
        frameF_rows.append({
            "FrameKey": frame_key,
            "Fy_frame [kN]": Fy_frame,
        })

    K_R = safe_parallel_sum([r["K_frame [N/mm]"] for r in frameK_rows]) if frameK_rows else 0.0
    Fy_R = safe_parallel_sum([r["Fy_frame [kN]"] for r in frameF_rows]) if frameF_rows else 0.0

    return {
        "K_MDOF [N/mm]": float(K_R),
        "Fy_MDOF [kN]": float(Fy_R),
    }


# =========================================================
# CONVERSIONI FINALI PER IL FRONT-END
# =========================================================
def convert_real_deltaK_to_sdf(deltaK_real_kNmm: float, gamma: float, mstar_ton: float) -> float:
    denom = float(mstar_ton) * G_ACCEL * float(gamma)
    return float(deltaK_real_kNmm) / denom if abs(denom) > EPS else 0.0


def convert_real_force_to_sdf(Fy_real_kN: float, gamma: float, mstar_ton: float) -> Dict[str, float]:
    denom = float(gamma) * float(mstar_ton) * G_ACCEL
    Se_g = float(Fy_real_kN) / denom if abs(denom) > EPS else 0.0
    return {
        "Se_g": Se_g,
    }


# =========================================================
# CALCOLO SISTEMA SU 4 DIREZIONI
# =========================================================
def evaluate_system_4dirs(
    geometry: Dict[str, Any],
    modal: Dict[str, Any],
    deltaK_targets: Dict[str, float],
    df_real_panels: List[Dict[str, Any]],
    diag_key: str
):
    directions = ["+X", "-X", "+Y", "-Y"]
    out = {}

    for direction in directions:
        axis, _branch = direction_to_axis_and_branch(direction)

        df_dir = build_directional_panel_dataframe(
            df_panels=df_real_panels,
            geometry=geometry,
            direction=direction,
            diag_key=diag_key,
        )
        comb = combine_directional_panels_series_parallel(df_dir)

        deltaK_real_kNmm = comb["K_MDOF [N/mm]"] / 1000.0
        Fy_MDOF = comb["Fy_MDOF [kN]"]

        gamma = float(modal[axis]["gamma"])
        mstar_ton = float(modal[axis]["Mstar"])

        target = float(deltaK_targets[direction])
        verifica = deltaK_real_kNmm >= target - 1e-12
        overshoot = max(0.0, deltaK_real_kNmm - target)

        k_1gdl_gmm = convert_real_deltaK_to_sdf(
            deltaK_real_kNmm=deltaK_real_kNmm,
            gamma=gamma,
            mstar_ton=mstar_ton,
        )
        force_conv = convert_real_force_to_sdf(
            Fy_real_kN=Fy_MDOF,
            gamma=gamma,
            mstar_ton=mstar_ton,
        )

        out[direction] = {
            "deltaK_target_real": target,
            "deltaK_real_MDOF": deltaK_real_kNmm,
            "deltaK_1GDL_g_per_mm": k_1gdl_gmm,
            "verifica": bool(verifica),
            "Fy_MDOF": Fy_MDOF,
            "Se_g": force_conv["Se_g"],
            "overshoot": overshoot,
        }

    verifica_globale = all(out[d]["verifica"] for d in directions)
    overshoot_total = sum(out[d]["overshoot"] for d in directions)

    return {
        "+X": out["+X"],
        "-X": out["-X"],
        "+Y": out["+Y"],
        "-Y": out["-Y"],
        "verifica_globale": verifica_globale,
        "overshoot_total": overshoot_total,
    }


def build_all_systems_table(
    geometry: Dict[str, Any],
    modal: Dict[str, Any],
    deltaK_targets: Dict[str, float],
    min_beam_height_by_qtop: Dict[float, float],
    clear_cm: int,
    distf_cm: int,
    forced_Ybase: List[float],
    forced_Ysec: List[float],
):
    rows = []
    reinf_cache = {}
    panels_cache = {}

    for family_name in ["100_75_50_25", "75_50_25", "50_25"]:
        step_family = SYSTEM_FAMILIES[family_name]

        df_real_panels, reinf_by_facade = build_real_panel_dataframe_for_system(
            geometry=geometry,
            min_beam_height_by_qtop=min_beam_height_by_qtop,
            possible_steps_cm=step_family,
            clear_cm=clear_cm,
            distf_cm=distf_cm,
            forced_Ybase=forced_Ybase,
            forced_Ysec=forced_Ysec,
        )
        panels_cache[family_name] = df_real_panels
        reinf_cache[family_name] = reinf_by_facade

        for diag_key in ["PIATTE", "IBRIDE", "AC"]:
            res = evaluate_system_4dirs(
                geometry=geometry,
                modal=modal,
                deltaK_targets=deltaK_targets,
                df_real_panels=df_real_panels,
                diag_key=diag_key,
            )

            rows.append({
                "famiglia": family_name,
                "diagonale": diag_key,

                "deltaK_target_+X": res["+X"]["deltaK_target_real"],
                "deltaK_real_+X_MDOF": res["+X"]["deltaK_real_MDOF"],
                "k_1GDL_+X_gmm": res["+X"]["deltaK_1GDL_g_per_mm"],
                "verifica_+X": res["+X"]["verifica"],

                "deltaK_target_-X": res["-X"]["deltaK_target_real"],
                "deltaK_real_-X_MDOF": res["-X"]["deltaK_real_MDOF"],
                "k_1GDL_-X_gmm": res["-X"]["deltaK_1GDL_g_per_mm"],
                "verifica_-X": res["-X"]["verifica"],

                "deltaK_target_+Y": res["+Y"]["deltaK_target_real"],
                "deltaK_real_+Y_MDOF": res["+Y"]["deltaK_real_MDOF"],
                "k_1GDL_+Y_gmm": res["+Y"]["deltaK_1GDL_g_per_mm"],
                "verifica_+Y": res["+Y"]["verifica"],

                "deltaK_target_-Y": res["-Y"]["deltaK_target_real"],
                "deltaK_real_-Y_MDOF": res["-Y"]["deltaK_real_MDOF"],
                "k_1GDL_-Y_gmm": res["-Y"]["deltaK_1GDL_g_per_mm"],
                "verifica_-Y": res["-Y"]["verifica"],

                "Fy_MDOF_+X": res["+X"]["Fy_MDOF"],
                "Se_+X_g": res["+X"]["Se_g"],

                "Fy_MDOF_-X": res["-X"]["Fy_MDOF"],
                "Se_-X_g": res["-X"]["Se_g"],

                "Fy_MDOF_+Y": res["+Y"]["Fy_MDOF"],
                "Se_+Y_g": res["+Y"]["Se_g"],

                "Fy_MDOF_-Y": res["-Y"]["Fy_MDOF"],
                "Se_-Y_g": res["-Y"]["Se_g"],

                "verifica_globale": res["verifica_globale"],
                "overshoot_total": res["overshoot_total"],
                "family_priority": FAMILY_PRIORITY[family_name],
                "diag_priority": DIAGONAL_PRIORITY[diag_key],
            })

    return rows, panels_cache, reinf_cache


def select_final_system(df_all: List[Dict[str, Any]]):
    valid = [r for r in df_all if r["verifica_globale"] is True]
    if not valid:
        raise RuntimeError("Nessun sistema soddisfa contemporaneamente i target +X, -X, +Y, -Y.")

    valid.sort(key=lambda r: (r["family_priority"], r["diag_priority"], r["overshoot_total"]))
    return valid, valid[0]


# =========================================================
# GEOMETRIA 3D ESPORTO FRONT-END
# =========================================================
def build_segment_records_for_facade(facade: Dict[str, Any], reinforcement: Dict[str, Any]) -> Dict[str, Any]:
    facade_index = int(facade["index"])
    (tx, ty), (nx, ny) = get_facade_unit_vectors(facade)

    nodes_map = {}
    node_seq = 0
    segments = []

    def get_or_create_node(s_cm: float, z_cm: float):
        nonlocal node_seq
        key = (round(float(s_cm), 6), round(float(z_cm), 6))
        if key in nodes_map:
            return nodes_map[key]

        s_m = float(s_cm) / 100.0
        z_m = float(z_cm) / 100.0
        x_w, y_w, z_w = local_to_world_on_facade(facade, s_m=s_m, z_m=z_m)

        node_seq += 1
        node_id = f"F{facade_index}_N{node_seq}"

        node_obj = {
            "node_id": node_id,
            "facade_index": facade_index,
            "local": {
                "s_m": s_m,
                "z_m": z_m,
            },
            "world": {
                "x_m": x_w,
                "y_m": y_w,
                "z_m": z_w,
            }
        }
        nodes_map[key] = node_obj
        return node_obj

    seg_seq = 0

    def add_segment(seg_type: str, s1_cm: float, z1_cm: float, s2_cm: float, z2_cm: float):
        nonlocal seg_seq
        n1 = get_or_create_node(s1_cm, z1_cm)
        n2 = get_or_create_node(s2_cm, z2_cm)
        seg_seq += 1

        s1_m = float(s1_cm) / 100.0
        z1_m = float(z1_cm) / 100.0
        s2_m = float(s2_cm) / 100.0
        z2_m = float(z2_cm) / 100.0

        x1_w, y1_w, z1_w = n1["world"]["x_m"], n1["world"]["y_m"], n1["world"]["z_m"]
        x2_w, y2_w, z2_w = n2["world"]["x_m"], n2["world"]["y_m"], n2["world"]["z_m"]

        length_m = math.sqrt((x2_w - x1_w) ** 2 + (y2_w - y1_w) ** 2 + (z2_w - z1_w) ** 2)

        segments.append({
            "segment_id": f"F{facade_index}_S{seg_seq}",
            "facade_index": facade_index,
            "type": seg_type,
            "node_start_id": n1["node_id"],
            "node_end_id": n2["node_id"],
            "start_local": {"s_m": s1_m, "z_m": z1_m},
            "end_local": {"s_m": s2_m, "z_m": z2_m},
            "start_world": {"x_m": x1_w, "y_m": y1_w, "z_m": z1_w},
            "end_world": {"x_m": x2_w, "y_m": y2_w, "z_m": z2_w},
            "length_m": length_m,
        })

    for x_cm, y1_cm, y2_cm in reinforcement["v_segs"]:
        add_segment("vertical", x_cm, y1_cm, x_cm, y2_cm)

    for y_cm, x1_cm, x2_cm in reinforcement["h_segs"]:
        add_segment("horizontal", x1_cm, y_cm, x2_cm, y_cm)

    for x1_cm, y1_cm, x2_cm, y2_cm in reinforcement["d_segs"]:
        add_segment("diagonal", x1_cm, y1_cm, x2_cm, y2_cm)

    nodes = sorted(nodes_map.values(), key=lambda n: n["node_id"])

    return {
        "facade_index": facade_index,
        "start": {
            "x_m": float(facade["start"]["x"]),
            "y_m": float(facade["start"]["y"]),
            "z_m": 0.0,
        },
        "end": {
            "x_m": float(facade["end"]["x"]),
            "y_m": float(facade["end"]["y"]),
            "z_m": 0.0,
        },
        "length_m": float(facade["length_m"]),
        "orientation_deg": float(facade.get("orientation_deg", 0.0)),
        "direction": facade.get("direction"),
        "fixed_axis": facade.get("fixed_axis"),
        "fixed_coordinate_m": float(facade.get("fixed_coordinate_m", 0.0)),
        "tangent_unit_vector": {"x": tx, "y": ty, "z": 0.0},
        "normal_unit_vector": {"x": nx, "y": ny, "z": 0.0},
        "local_axes_definition": {
            "s_axis": "ascissa locale lungo la facciata",
            "z_axis": "quota verticale",
        },
        "nodes": nodes,
        "segments": segments,
    }


# =========================================================
# JSON OUTPUT
# =========================================================
def build_output_json(
    deltaK_targets: Dict[str, float],
    modal: Dict[str, Any],
    df_all_systems: List[Dict[str, Any]],
    df_valid: List[Dict[str, Any]],
    best_system: Dict[str, Any],
    reinf_by_facade: Dict[int, Dict[str, Any]],
    geometry: Dict[str, Any],
) -> Dict[str, Any]:
    facade_geometries = []
    global_nodes = []
    global_segments = []

    for facade in geometry["facades"]:
        idx = int(facade["index"])
        facade_geom = build_segment_records_for_facade(facade, reinf_by_facade[idx])
        facade_geometries.append(facade_geom)
        global_nodes.extend(facade_geom["nodes"])
        global_segments.extend(facade_geom["segments"])

    final_results = {
        "sistema_scelto": {
            "famiglia": best_system["famiglia"],
            "diagonale": best_system["diagonale"],
        },
        "+X": {
            "deltaK_target_real": float(best_system["deltaK_target_+X"]),
            "deltaK_real_MDOF": float(best_system["deltaK_real_+X_MDOF"]),
            "k_1GDL_g_per_mm": float(best_system["k_1GDL_+X_gmm"]),
            "Fy_MDOF": float(best_system["Fy_MDOF_+X"]),
            "Se_g": float(best_system["Se_+X_g"]),
        },
        "-X": {
            "deltaK_target_real": float(best_system["deltaK_target_-X"]),
            "deltaK_real_MDOF": float(best_system["deltaK_real_-X_MDOF"]),
            "k_1GDL_g_per_mm": float(best_system["k_1GDL_-X_gmm"]),
            "Fy_MDOF": float(best_system["Fy_MDOF_-X"]),
            "Se_g": float(best_system["Se_-X_g"]),
        },
        "+Y": {
            "deltaK_target_real": float(best_system["deltaK_target_+Y"]),
            "deltaK_real_MDOF": float(best_system["deltaK_real_+Y_MDOF"]),
            "k_1GDL_g_per_mm": float(best_system["k_1GDL_+Y_gmm"]),
            "Fy_MDOF": float(best_system["Fy_MDOF_+Y"]),
            "Se_g": float(best_system["Se_+Y_g"]),
        },
        "-Y": {
            "deltaK_target_real": float(best_system["deltaK_target_-Y"]),
            "deltaK_real_MDOF": float(best_system["deltaK_real_-Y_MDOF"]),
            "k_1GDL_g_per_mm": float(best_system["k_1GDL_-Y_gmm"]),
            "Fy_MDOF": float(best_system["Fy_MDOF_-Y"]),
            "Se_g": float(best_system["Se_-Y_g"]),
        },
    }

    return {
        "target": {
            "+X": float(deltaK_targets["+X"]),
            "-X": float(deltaK_targets["-X"]),
            "+Y": float(deltaK_targets["+Y"]),
            "-Y": float(deltaK_targets["-Y"]),
            "X_max": float(deltaK_targets["X_max"]),
            "Y_max": float(deltaK_targets["Y_max"]),
        },
        "modal": {
            "X": {
                "gamma": float(modal["X"]["gamma"]),
                "Mstar_ton": float(modal["X"]["Mstar"]),
            },
            "Y": {
                "gamma": float(modal["Y"]["gamma"]),
                "Mstar_ton": float(modal["Y"]["Mstar"]),
            },
        },
        "risultati_9_sistemi": [
            {
                "famiglia": r["famiglia"],
                "diagonale": r["diagonale"],
                "deltaK_target_+X": r["deltaK_target_+X"],
                "deltaK_real_+X_MDOF": r["deltaK_real_+X_MDOF"],
                "k_1GDL_+X_gmm": r["k_1GDL_+X_gmm"],
                "verifica_+X": r["verifica_+X"],
                "deltaK_target_-X": r["deltaK_target_-X"],
                "deltaK_real_-X_MDOF": r["deltaK_real_-X_MDOF"],
                "k_1GDL_-X_gmm": r["k_1GDL_-X_gmm"],
                "verifica_-X": r["verifica_-X"],
                "deltaK_target_+Y": r["deltaK_target_+Y"],
                "deltaK_real_+Y_MDOF": r["deltaK_real_+Y_MDOF"],
                "k_1GDL_+Y_gmm": r["k_1GDL_+Y_gmm"],
                "verifica_+Y": r["verifica_+Y"],
                "deltaK_target_-Y": r["deltaK_target_-Y"],
                "deltaK_real_-Y_MDOF": r["deltaK_real_-Y_MDOF"],
                "k_1GDL_-Y_gmm": r["k_1GDL_-Y_gmm"],
                "verifica_-Y": r["verifica_-Y"],
                "Fy_MDOF_+X": r["Fy_MDOF_+X"],
                "Se_+X_g": r["Se_+X_g"],
                "Fy_MDOF_-X": r["Fy_MDOF_-X"],
                "Se_-X_g": r["Se_-X_g"],
                "Fy_MDOF_+Y": r["Fy_MDOF_+Y"],
                "Se_+Y_g": r["Se_+Y_g"],
                "Fy_MDOF_-Y": r["Fy_MDOF_-Y"],
                "Se_-Y_g": r["Se_-Y_g"],
                "verifica_globale": r["verifica_globale"],
            }
            for r in df_all_systems
        ],
        "sistemi_validi": [
            {
                "famiglia": r["famiglia"],
                "diagonale": r["diagonale"],
                "deltaK_target_+X": r["deltaK_target_+X"],
                "deltaK_real_+X_MDOF": r["deltaK_real_+X_MDOF"],
                "verifica_+X": r["verifica_+X"],
                "deltaK_target_-X": r["deltaK_target_-X"],
                "deltaK_real_-X_MDOF": r["deltaK_real_-X_MDOF"],
                "verifica_-X": r["verifica_-X"],
                "deltaK_target_+Y": r["deltaK_target_+Y"],
                "deltaK_real_+Y_MDOF": r["deltaK_real_+Y_MDOF"],
                "verifica_+Y": r["verifica_+Y"],
                "deltaK_target_-Y": r["deltaK_target_-Y"],
                "deltaK_real_-Y_MDOF": r["deltaK_real_-Y_MDOF"],
                "verifica_-Y": r["verifica_-Y"],
                "verifica_globale": r["verifica_globale"],
                "overshoot_total": r["overshoot_total"],
            }
            for r in df_valid
        ],
        "sistema_finale": {
            "famiglia": best_system["famiglia"],
            "diagonale": best_system["diagonale"],
            "family_priority": int(best_system["family_priority"]),
            "diag_priority": int(best_system["diag_priority"]),
            "overshoot_total": float(best_system["overshoot_total"]),
        },
        "risultati_finali": final_results,
        "geometria_rinforzo": {
            "by_facade": facade_geometries,
            "all_nodes_world": global_nodes,
            "all_segments_world": global_segments,
        },
    }


# =========================================================
# CORE COMPUTE
# =========================================================
def compute_reinforcement(payload: Any) -> Dict[str, Any]:
    data = normalize_frontend_payload(payload)
    geometry = data["geometry"]
    deltaK = data["deltaK"]
    modal = data["modal"]
    settings = data["settings"]

    clear_cm = int(settings.get("CLEAR", 5))
    distf_cm = int(settings.get("DISTF", 10))

    deltaK_targets = {
        "+X": float(deltaK["+X"]),
        "-X": float(deltaK["-X"]),
        "+Y": float(deltaK["+Y"]),
        "-Y": float(deltaK["-Y"]),
        "X_max": float(deltaK["X_max"]),
        "Y_max": float(deltaK["Y_max"]),
    }

    min_beam_height_by_qtop = build_global_min_beam_height_by_qtop(geometry)

    forced_Ybase = build_global_unique_y_primary(
        geometry=geometry,
        min_beam_height_by_qtop=min_beam_height_by_qtop,
        clear_cm=clear_cm,
    )

    forced_Ysec = build_global_unique_y_secondary(
        geometry=geometry,
        min_beam_height_by_qtop=min_beam_height_by_qtop,
        clear_cm=clear_cm,
        distf_cm=distf_cm,
        forced_Ybase=forced_Ybase,
    )

    df_all, _panels_cache, reinf_cache = build_all_systems_table(
        geometry=geometry,
        modal=modal,
        deltaK_targets=deltaK_targets,
        min_beam_height_by_qtop=min_beam_height_by_qtop,
        clear_cm=clear_cm,
        distf_cm=distf_cm,
        forced_Ybase=forced_Ybase,
        forced_Ysec=forced_Ysec,
    )

    df_valid, best_system = select_final_system(df_all)

    final_family = best_system["famiglia"]
    final_reinf_by_facade = reinf_cache[final_family]

    output_json = build_output_json(
        deltaK_targets=deltaK_targets,
        modal=modal,
        df_all_systems=df_all,
        df_valid=df_valid,
        best_system=best_system,
        reinf_by_facade=final_reinf_by_facade,
        geometry=geometry,
    )

    return output_json


# =========================================================
# ENDPOINTS
# =========================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "muratura-reinforcement-api",
        "version": "1.2.0",
    }


@app.post("/compute")
def compute(req: ComputeRequest):
    try:
        result = compute_reinforcement(req.payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# =========================================================
# NOTE DEPLOY
# =========================================================
# requirements.txt
# fastapi
# uvicorn[standard]
# pydantic
#
# render start command:
# uvicorn main:app --host 0.0.0.0 --port $PORT
#
# n8n HTTP Request node:
# POST https://<tuo-render-url>/compute
# body JSON:
# {
#   "payload": <qui il JSON del front-end>
# }