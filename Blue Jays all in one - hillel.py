"""Blue Jays 2025 postseason pitcher charts: one file per pitcher, all in one script.

Run it (no arguments needed):  python blue_jays_all_in_one.py

It downloads the Kaggle pitch-by-pitch data, then writes everything to an `updated_graphs`
folder next to this script:
  pitcher_charts/<pitcher>_all_charts.pdf   one 4-page PDF per pitcher:
                                            whiff rate, count-adjusted run value per 100
                                            pitches, CSW%, xwOBA (pitch type x Statcast zone)
  <pitcher>_whiff_by_zone.png               original whiff-rate PNG for each pitcher
  all_pitcher_graphs.pdf                    original combined whiff-rate PDF
  pitcher_pitch_zone_whiff_rates.csv        pitcher x pitch type x zone table
  pitcher_summary.csv                       one row per pitcher
and a *_SHARE_WITH_TEAM.zip next to that folder.

Options:
  --input FILE        use a CSV you already downloaded instead of Kaggle
  --output FOLDER     where to write results
  --min-swings N      whiff chart: minimum swings in a cell (default 3)
  --min-pitches N     run value and CSW% charts: minimum pitches in a cell (default 3)
  --min-bip N         xwOBA chart: minimum balls in play in a cell (default 2)
  --first-pitch-only  only 0-0 counts
  --no-open           don't open the output folder when finished

Source: MLB Postseason 2025 Pitch-by-Pitch Data, janus137 (Kaggle).
Whiff rate = whiffs / swings, NOT whiffs / pitches.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only; no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

# MLB Statcast zones 1-9 are inside the zone; 11-14 are outside.
ZONES = list(range(1, 10)) + [11, 12, 13, 14]

WHIFF_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
}
SWING_DESCRIPTIONS = WHIFF_DESCRIPTIONS | {
    "foul",
    "foul_tip",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
    "foul_pitchout",
    "swinging_pitchout",
}
BUNT_DESCRIPTIONS = {
    "foul_bunt",
    "missed_bunt",
    "bunt_foul_tip",
    "foul_bunt_batter_out",
}
PITCH_NAMES = {
    "FF": "4-Seam Fastball", "SI": "Sinker", "FC": "Cutter",
    "SL": "Slider", "ST": "Sweeper", "CU": "Curveball",
    "KC": "Knuckle Curve", "CH": "Changeup", "FS": "Split-Finger",
    "FO": "Forkball", "KN": "Knuckleball", "SV": "Slurve",
    "EP": "Eephus", "SC": "Screwball", "CS": "Slow Curve",
}

# Run value by count, from batter's perspective (positive = good for the hitter).
# (balls, strikes): (count run value vs 0-0, value of a strike, value of a ball)
# A strike at two strikes is a strikeout, and a ball at three balls is a walk,
# so those entries include the end of the at-bat.
COUNT_VALUES = {
    (0, 0): (0.000, -0.043, 0.036),
    (0, 1): (-0.043, -0.062, 0.027),
    (0, 2): (-0.105, -0.185, 0.021),
    (1, 0): (0.036, -0.052, 0.059),
    (1, 1): (-0.017, -0.067, 0.049),
    (1, 2): (-0.084, -0.206, 0.040),
    (2, 0): (0.095, -0.062, 0.098),
    (2, 1): (0.033, -0.077, 0.099),
    (2, 2): (-0.044, -0.246, 0.090),
    (3, 0): (0.193, -0.061, 0.126),
    (3, 1): (0.132, -0.085, 0.187),
    (3, 2): (0.047, -0.336, 0.272),
}
# For the count-adjusted run value, every pitch is valued as if it were thrown in this
# count. (0, 0) = the start of the at-bat. Any key in COUNT_VALUES works.
STANDARD_COUNT = (0, 0)
STRIKE_DESCRIPTIONS = {"called_strike", "swinging_strike", "swinging_strike_blocked",
                       "foul_tip", "foul", "foul_pitchout"}
BALL_DESCRIPTIONS = {"ball", "blocked_ball"}


def get_column(df: pd.DataFrame, names: list[str], required: bool = True) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    if required:
        raise ValueError(f"Required column not found. Expected one of: {names}")
    return None


def count_adjusted_run_exp(df: pd.DataFrame, raw: pd.Series) -> pd.Series:
    """delta_run_exp (batter perspective) with the effect of the count removed.

    - Balls and strikes: swap the count's table value for the standard-count value.
      Whatever is left over (base/out situation) is kept.
    - Fouls count as strikes, even with two strikes (where the raw value is ~0).
    - Balls in play and hit-by-pitch: shift the baseline from this count to the standard
      count, so a single is worth the same at 0-2 as at 3-0.
    - Anything else is left as is. Rows with an unknown count come back blank.
    """
    if "balls" not in df.columns or "strikes" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    table = pd.DataFrame(
        [(b, s, *vals) for (b, s), vals in COUNT_VALUES.items()],
        columns=["balls", "strikes", "count_rv", "strike_val", "ball_val"],
    ).astype(float)
    keys = pd.DataFrame({"balls": pd.to_numeric(df["balls"], errors="coerce"),
                         "strikes": pd.to_numeric(df["strikes"], errors="coerce")})
    look = keys.merge(table, on=["balls", "strikes"], how="left")
    look.index = df.index

    std_rv, std_strike, std_ball = COUNT_VALUES[STANDARD_COUNT]
    d = df["description_out"].fillna("")
    is_strike = d.isin(STRIKE_DESCRIPTIONS)
    is_ball = d.isin(BALL_DESCRIPTIONS)
    is_end = (df["is_in_play"] == 1) | (d == "hit_by_pitch")
    two_strike_foul = d.isin({"foul", "foul_pitchout"}) & (look["strikes"] >= 2)

    adj = raw.copy()
    adj.loc[is_strike] = raw[is_strike] - look.loc[is_strike, "strike_val"] + std_strike
    adj.loc[two_strike_foul] = std_strike
    adj.loc[is_ball] = raw[is_ball] - look.loc[is_ball, "ball_val"] + std_ball
    adj.loc[is_end] = raw[is_end] + look.loc[is_end, "count_rv"] - std_rv
    return adj.where(raw.notna())


def tor_mask(df: pd.DataFrame) -> pd.Series:
    team_col = get_column(df, ["pitching_team", "fielding_team"], False)
    if team_col:
        return df[team_col].astype(str).str.strip().str.upper().isin(
            {"TOR", "TORONTO", "TORONTO BLUE JAYS"}
        )
    home_col = get_column(df, ["home_team", "home_team_abbr"])
    away_col = get_column(df, ["away_team", "away_team_abbr"])
    half_col = get_column(df, ["inning_topbot", "inning_half", "top_bottom"])
    home = df[home_col].astype(str).str.strip().str.upper()
    away = df[away_col].astype(str).str.strip().str.upper()
    half = df[half_col].astype(str).str.strip().str.lower()
    return ((home == "TOR") & half.isin(["top", "t"])) | (
        (away == "TOR") & half.isin(["bot", "bottom", "b"])
    )


def load_data(path: Path, first_pitch_only: bool = False) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, low_memory=False)
    raw.columns = raw.columns.str.strip()
    source_rows = len(raw)
    df = raw.loc[tor_mask(raw)].copy()
    tor_rows = len(df)
    if df.empty:
        raise ValueError("No TOR pitching rows were found. Check team column names/values.")

    p_name = get_column(df, ["pitcher_name", "player_name", "pitcher_full_name"], False)
    p_id = get_column(df, ["pitcher", "pitcher_id", "pitcher_mlbam_id"], False)
    if p_name is None and p_id is None:
        raise ValueError("Need pitcher_name, player_name, or pitcher ID column.")
    if p_id:
        df["pitcher_id_out"] = df[p_id].astype("string").fillna("unknown")
        df["pitcher_id_out"] = df["pitcher_id_out"].str.replace(r"\.0$", "", regex=True)
    else:
        df["pitcher_id_out"] = df[p_name].astype("string").fillna("unknown")
    if p_name:
        df["pitcher_label"] = df[p_name].astype("string").str.strip().fillna("")
        df["pitcher_label"] = df["pitcher_label"].replace("", pd.NA)
        df["pitcher_label"] = df["pitcher_label"].fillna(df["pitcher_id_out"])
    else:
        df["pitcher_label"] = df["pitcher_id_out"]

    pitch_col = get_column(df, ["pitch_name", "pitch_type"])
    df["pitch_type_out"] = df[pitch_col].astype("string").str.strip()
    if pitch_col == "pitch_type":
        df["pitch_type_out"] = df["pitch_type_out"].map(
            lambda x: PITCH_NAMES.get(x, x) if pd.notna(x) else pd.NA
        )
    zone_col = get_column(df, ["zone", "statcast_zone"])
    df["zone_out"] = pd.to_numeric(df[zone_col], errors="coerce")
    desc_col = get_column(df, ["description", "pitch_description", "call_description"])
    df["description_out"] = df[desc_col].astype("string").str.strip().str.lower()

    if first_pitch_only:
        balls_col = get_column(df, ["balls"])
        strikes_col = get_column(df, ["strikes"])
        df = df.loc[(pd.to_numeric(df[balls_col], errors="coerce") == 0) &
                    (pd.to_numeric(df[strikes_col], errors="coerce") == 0)].copy()

    # Match the report's general rule: exclude bunts and unknown pitch/zone records.
    df = df.loc[
        df["pitch_type_out"].notna()
        & df["pitch_type_out"].ne("")
        & df["zone_out"].isin(ZONES)
        & ~df["description_out"].isin(BUNT_DESCRIPTIONS)
    ].copy()
    if df.empty:
        raise ValueError("No valid rows remain after filtering. Inspect source column names.")

    df["is_whiff"] = df["description_out"].isin(WHIFF_DESCRIPTIONS).astype(int)
    df["is_swing"] = df["description_out"].isin(SWING_DESCRIPTIONS).astype(int)
    df["zone_out"] = df["zone_out"].astype(int)

    # --- pitch-level flags and measures (aggregated later) ---
    d = df["description_out"].fillna("")
    df["is_called_strike"] = (d == "called_strike").astype(int)
    df["is_ball_taken"] = d.isin({"ball", "blocked_ball"}).astype(int)
    df["is_foul"] = d.isin({"foul", "foul_pitchout"}).astype(int)
    df["is_in_play"] = d.str.startswith("hit_into_play").astype(int)

    for src, dst in [("launch_speed", "launch_speed_out"),
                     ("launch_angle", "launch_angle_out"),
                     ("estimated_woba_using_speedangle", "xwoba_out")]:
        df[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else np.nan

    # Contact quality counts for balls in play only (fouls can carry tracking data too).
    df.loc[df["is_in_play"] == 0, ["launch_speed_out", "launch_angle_out", "xwoba_out"]] = np.nan

    # Run value. Statcast's delta_run_exp is from the batter's perspective; flip the sign
    # so positive = good for the pitcher. The adjusted version also removes the count's effect.
    if "delta_run_exp" in df.columns:
        raw_rv = pd.to_numeric(df["delta_run_exp"], errors="coerce")
    else:
        raw_rv = pd.Series(np.nan, index=df.index)
    df["run_exp_out"] = -raw_rv
    df["run_exp_adj_out"] = -count_adjusted_run_exp(df, raw_rv)

    if "events" in df.columns:
        events = df["events"].astype("string").str.lower().fillna("")
    else:
        events = pd.Series("", index=df.index, dtype="string")
    df["is_hit"] = events.isin({"single", "double", "triple", "home_run"}).astype(int)
    df["is_hard_hit"] = (df["launch_speed_out"] >= 95).astype(int)
    bad_desc = sorted(set(df["description_out"].dropna()) - SWING_DESCRIPTIONS)
    print("Non-swing descriptions (normal; excluded from swing denominator):", bad_desc)
    meta = {
        "source_rows": source_rows,
        "tor_rows_before_cleaning": tor_rows,
        "tor_rows_after_cleaning": len(df),
        "pitchers": df["pitcher_id_out"].nunique(),
    }
    return df, meta


# Shared by the pitcher x pitch type x zone table and the per-pitcher summary.
MEASURE_AGG = dict(
    pitches=("is_swing", "size"),
    swings=("is_swing", "sum"),
    whiffs=("is_whiff", "sum"),
    called_strikes=("is_called_strike", "sum"),
    balls_taken=("is_ball_taken", "sum"),
    fouls=("is_foul", "sum"),
    balls_in_play=("is_in_play", "sum"),
    hits=("is_hit", "sum"),
    hard_hit=("is_hard_hit", "sum"),
    avg_launch_speed=("launch_speed_out", "mean"),
    avg_launch_angle=("launch_angle_out", "mean"),
    avg_xwoba=("xwoba_out", "mean"),
    run_value=("run_exp_out", lambda x: x.sum(min_count=1)),
    run_value_count_adj=("run_exp_adj_out", lambda x: x.sum(min_count=1)),
    avg_run_exp=("run_exp_out", "mean"),
    avg_run_exp_adj=("run_exp_adj_out", "mean"),
)


def add_derived_columns(result: pd.DataFrame) -> pd.DataFrame:
    """Rates computed from the aggregated counts."""
    result["whiff_rate_pct"] = 100 * result["whiffs"] / result["swings"].replace(0, np.nan)
    result["csw_pct"] = 100 * (result["called_strikes"] + result["whiffs"]) / result["pitches"]
    # Run values are already pitcher-perspective (positive = good for the pitcher).
    result["rv_per_100"] = 100 * result["avg_run_exp"]
    result["rv_per_100_count_adj"] = 100 * result["avg_run_exp_adj"]
    return result.drop(columns=["avg_run_exp", "avg_run_exp_adj"])


def calculate_rates(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["pitcher_id_out", "pitcher_label", "pitch_type_out", "zone_out"]
    result = (
        df.groupby(group_cols, dropna=False)
        .agg(**MEASURE_AGG)
        .reset_index()
    )
    result = add_derived_columns(result)
    return result.rename(columns={
        "pitcher_id_out": "pitcher_id",
        "pitcher_label": "pitcher_name",
        "pitch_type_out": "pitch_type",
        "zone_out": "zone",
    }).sort_values(["pitcher_name", "pitch_type", "zone"]).reset_index(drop=True)


def safe_filename(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", label).strip("_")
    return cleaned[:65] or "unknown_pitcher"


# --- Heatmap charts -------------------------------------------------------------------
# Every chart uses the same layout (pitch type on the left, Statcast zone along the bottom).
# Colours run from light blue (worse) to dark Blue Jays blue (better), so the darkest cell is
# the best result on every chart. White cells have too little data (too few pitches, swings or
# balls in play), and the cell shows the sample size instead.
BLUE_JAYS_BLUES = LinearSegmentedColormap.from_list(
    "blue_jays", ["#C9DDF2", "#6FA3D8", "#1F5FA8", "#134A8E", "#0A2352"])
CHART_SPECS = {
    "whiff": dict(
        value="whiff_rate_pct", gate_col="swings", n_col="swings",
        what="Whiff rate", fmt="{:.0f}%", cmap=BLUE_JAYS_BLUES, vmin=0, vmax=100, extend="neither",
        cbar="Whiff rate (%) = whiffs / swings × 100"),
    "rv": dict(
        value="rv_per_100_count_adj", gate_col="pitches", n_col="pitches",
        what="Count-adjusted run value per 100 pitches", fmt="{:+.1f}", cmap=BLUE_JAYS_BLUES,
        vmin=-20, vmax=20, extend="both",
        cbar="Run value per 100 pitches, count-adjusted\n(higher = better for the pitcher)"),
    "csw": dict(
        value="csw_pct", gate_col="pitches", n_col="pitches",
        what="CSW% (called strikes + whiffs)", fmt="{:.0f}%", cmap=BLUE_JAYS_BLUES,
        vmin=0, vmax=100, extend="neither",
        cbar="CSW% = (called strikes + whiffs) / pitches × 100\n(higher = better)"),
    "xwoba": dict(
        value="avg_xwoba", gate_col="balls_in_play", n_col="pitches",
        what="Average xwOBA on balls in play", fmt="{:.3f}", cmap=BLUE_JAYS_BLUES.reversed(),
        vmin=0, vmax=1.0, extend="max",
        cbar="Average xwOBA on balls in play\n(lower = better, so the scale runs downward)"),
}
CHART_ORDER = ["whiff", "rv", "csw", "xwoba"]


def _text_color(rgba) -> str:
    """Black or white, whichever is easier to read on this background colour."""
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgba[:3]]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "black" if (lum + 0.05) / 0.05 > 1.05 / (lum + 0.05) else "white"


def plot_metric(group: pd.DataFrame, key: str, min_swings: int = 3, min_pitches: int = 3,
                min_bip: int = 2, first_pitch_only: bool = False):
    spec = CHART_SPECS[key]
    min_n = {"whiff": min_swings, "rv": min_pitches, "csw": min_pitches, "xwoba": min_bip}[key]
    name = str(group["pitcher_name"].iloc[0])
    pitcher_id = str(group["pitcher_id"].iloc[0])
    # Same row order on every chart: most-thrown pitch on top.
    types = (group.groupby("pitch_type")["pitches"].sum()
             .sort_values(ascending=False, kind="stable").index.tolist())
    shape = (len(types), len(ZONES))
    vals = np.full(shape, np.nan)
    gate = np.zeros(shape, dtype=int)
    pitches = np.zeros(shape, dtype=int)
    bip = np.zeros(shape, dtype=int)
    for row in group.itertuples(index=False):
        i = types.index(row.pitch_type)
        j = ZONES.index(row.zone)
        pitches[i, j] = int(row.pitches)
        bip[i, j] = int(row.balls_in_play)
        gate[i, j] = int(getattr(row, spec["gate_col"]))
        v = getattr(row, spec["value"])
        if gate[i, j] >= min_n and pd.notna(v):
            vals[i, j] = v

    fig_height = max(4.5, 1.1 * len(types) + 2.5)
    fig, ax = plt.subplots(figsize=(13.5, fig_height), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.045, 1, 0.955))  # leave room for the footnote
    im = ax.imshow(np.ma.masked_invalid(vals), aspect="auto", cmap=spec["cmap"],
                   vmin=spec["vmin"], vmax=spec["vmax"])
    cbar = fig.colorbar(im, ax=ax, shrink=0.92, extend=spec["extend"])
    cbar.set_label(spec["cbar"])
    if key == "xwoba":
        cbar.ax.invert_yaxis()  # best (low xwOBA) at the top, like the other charts
    ax.set_xticks(np.arange(len(ZONES)), [str(z) for z in ZONES])
    ax.set_yticks(np.arange(len(types)), types)
    ax.set_xlabel("Statcast zone (1–9 inside; 11–14 outside strike zone)")
    ax.set_ylabel("Pitch type")
    ax.set_xticks(np.arange(-.5, len(ZONES), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(types), 1), minor=True)
    ax.grid(which="minor", linewidth=0.6, alpha=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(types)):
        for j in range(len(ZONES)):
            n = pitches[i, j] if spec["n_col"] == "pitches" else gate[i, j]
            if np.isfinite(vals[i, j]):
                lines = [spec["fmt"].format(vals[i, j])]
                if key == "xwoba":
                    lines += [f"BIP={bip[i, j]}", f"n={pitches[i, j]}"]
                else:
                    lines.append(f"n={n}")
                color = _text_color(im.cmap(im.norm(vals[i, j])))
            elif n > 0:
                if key == "xwoba":
                    lines = [f"n={pitches[i, j]}", f"BIP={bip[i, j]} (low)"]
                else:
                    lines = [f"n={n}", "(low)"]
                color = "black"
            else:
                lines, color = ["—"], "black"
            ax.text(j, i, "\n".join(lines), ha="center", va="center", fontsize=8, color=color)

    context = "0–0 count only" if first_pitch_only else "all counts"
    ax.set_title(f"2025 Postseason · Toronto Blue Jays · {name} (ID {pitcher_id})\n"
                 f"{spec['what']} by pitch type and zone · {context}", fontsize=13)
    notes = {
        "whiff": f"Each cell = whiffs / swings. n = swings. Low means fewer than {min_swings} swings; "
                 "rate hidden. Gray/empty cells have no reliable shown rate.",
        "rv": "Each cell = average count-adjusted run value per 100 pitches. n = pitches. "
              f"Low means fewer than {min_pitches} pitches; value hidden. Colour capped at ±20.",
        "csw": "Each cell = (called strikes + whiffs) / pitches. n = pitches. "
               f"Low means fewer than {min_pitches} pitches; rate hidden.",
        "xwoba": "Each cell = average xwOBA on balls in play. n = pitches, BIP = balls in play. "
                 f"Low means fewer than {min_bip} balls in play; value hidden. Colour capped at 1.000.",
    }
    fig.text(0.5, 0.012, notes[key], ha="center", fontsize=9)
    return fig


def plot_pitcher(group: pd.DataFrame, min_swings: int, first_pitch_only: bool):
    """The original whiff-rate chart."""
    return plot_metric(group, "whiff", min_swings=min_swings, first_pitch_only=first_pitch_only)


def save_pitcher_pdf(group: pd.DataFrame, path: Path, min_swings: int, min_pitches: int,
                     min_bip: int, first_pitch_only: bool) -> None:
    """One PDF per pitcher: whiff rate, run value, CSW% and xwOBA charts, one per page."""
    with PdfPages(path) as pdf:
        for key in CHART_ORDER:
            fig = plot_metric(group, key, min_swings, min_pitches, min_bip, first_pitch_only)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def run(input_path: Path, output_dir: Path, min_swings: int, first_pitch_only: bool,
        min_pitches: int = 3, min_bip: int = 2) -> None:
    if min_swings < 1:
        raise ValueError("--min-swings must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    df, meta = load_data(input_path, first_pitch_only)
    rate = calculate_rates(df)
    rate.to_csv(output_dir / "pitcher_pitch_zone_whiff_rates.csv", index=False, float_format="%.2f")

    pitcher_totals = (
        df.groupby(["pitcher_id_out", "pitcher_label"])
        .agg(**MEASURE_AGG)
        .reset_index()
    )
    pitcher_totals = add_derived_columns(pitcher_totals).rename(
        columns={"pitcher_id_out": "pitcher_id", "pitcher_label": "pitcher_name",
                 "pitches": "total_pitches"})
    pitcher_totals.to_csv(output_dir / "pitcher_summary.csv", index=False, float_format="%.2f")

    pdf_path = output_dir / "all_pitcher_graphs.pdf"
    pitcher_pdf_dir = output_dir / "pitcher_charts"
    pitcher_pdf_dir.mkdir(exist_ok=True)
    n = 0
    with PdfPages(pdf_path) as pdf:
        for (pitcher_id, label), g in rate.groupby(["pitcher_id", "pitcher_name"], sort=True):
            fig = plot_pitcher(g, min_swings, first_pitch_only)
            name = safe_filename(f"{label}_{pitcher_id}")
            save_pitcher_pdf(g, pitcher_pdf_dir / f"{name}_all_charts.pdf",
                             min_swings, min_pitches, min_bip, first_pitch_only)
            fig.savefig(output_dir / f"{name}_whiff_by_zone.png", dpi=180, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            n += 1

    print("Dataset rows:", meta["source_rows"])
    print("Toronto pitches before filtering:", meta["tor_rows_before_cleaning"])
    print("Toronto pitches after filtering:", meta["tor_rows_after_cleaning"])
    print("Pitchers:", n)
    print("Wrote:", pdf_path)
    print("Wrote:", output_dir / "pitcher_pitch_zone_whiff_rates.csv")
    print("Wrote:", output_dir / "pitcher_summary.csv")
    print("Wrote one PNG chart per pitcher")
    print("Wrote one 4-chart PDF per pitcher to:", pitcher_pdf_dir)
    if not first_pitch_only:
        print("Note: The earlier report used 2,784 cleaned Toronto pitches. ")
        print("A mismatch can result from different raw CSV versions or cleaning rules; inspect before sharing.")


KAGGLE_SLUG = "janus137/mlb-postseason-2025-pitch-by-pitch-data"


def download_csv() -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install numpy pandas matplotlib kagglehub") from exc
    print("Downloading Kaggle dataset:", KAGGLE_SLUG)
    try:
        folder = Path(kagglehub.dataset_download(KAGGLE_SLUG))
    except Exception as exc:
        raise SystemExit(
            "Could not download Kaggle data. Check internet access and Kaggle permissions. "
            "You can instead download the CSV manually and run this script with --input YOUR_FILE.csv. "
            f"Original error: {exc}"
        ) from exc
    csv_files = sorted(folder.rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csv_files:
        raise SystemExit(f"No CSV file was found in {folder}. Download the dataset CSV manually.")
    for file in csv_files:
        try:
            cols = set(pd.read_csv(file, nrows=2).columns.str.strip())
        except Exception:
            continue
        if {"home_team", "away_team", "zone"}.issubset(cols) and ("pitch_name" in cols or "pitch_type" in cols):
            print("Using:", file)
            return file
    raise SystemExit(
        "The dataset downloaded, but no CSV has the expected Statcast columns. "
        "Open the CSV and confirm it contains home_team, away_team, zone and pitch_name/pitch_type."
    )


def package_results(output: Path) -> Path:
    zip_path = output.parent / f"{output.name}_SHARE_WITH_TEAM.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for file in sorted(output.rglob("*")):
            if file.is_file():
                bundle.write(file, arcname=str(file.relative_to(output.parent)))
        script = Path(__file__).resolve()
        bundle.write(script, arcname=script.name)
    print("Ready to share with teammates:", zip_path)
    return zip_path


def open_folder(path: Path) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        pass  # opening the folder is a convenience only


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="Use an existing CSV instead of downloading from Kaggle")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "updated_graphs")
    parser.add_argument("--min-swings", default=3, type=int)
    parser.add_argument("--min-pitches", default=3, type=int)
    parser.add_argument("--min-bip", default=2, type=int)
    parser.add_argument("--first-pitch-only", action="store_true")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    source = args.input if args.input is not None else download_csv()
    run(source, args.output, args.min_swings, args.first_pitch_only, args.min_pitches, args.min_bip)
    package_results(args.output)
    print("\nAll charts and tables are in:", args.output.resolve())
    if not args.no_open:
        open_folder(args.output)


if __name__ == "__main__":
    main()