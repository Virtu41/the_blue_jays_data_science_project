import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. LOAD DATA & FILTER BLUE JAYS PITCHES
# ---------------------------------------------------------
# Load data (adjust file path if necessary)
df = pd.read_csv("Data_MLB_2025_StatcastPostseason_PitchByPitch_20251102a.csv")

# Identify pitching team and keep Blue Jays (TOR) pitches
df["pitching_team"] = np.where(
    df["inning_topbot"] == "Top", df["home_team"], df["away_team"]
)
jays = df[df["pitching_team"] == "TOR"].copy()
print("Blue Jays pitches:", len(jays))

# ---------------------------------------------------------
# 2. CREATE PLATE APPEARANCE (PA) LEVEL METRICS
# ---------------------------------------------------------
keys = ["game_pk", "pitcher", "batter", "at_bat_number"]

# Group by plate appearance to get final result (event)
pa = (
    jays.groupby(keys, as_index=False)
    .agg(
        events=(
            "events",
            lambda x: x.dropna().iloc[-1] if len(x.dropna()) > 0 else np.nan,
        )
    )
    .sort_values(["game_pk", "pitcher", "at_bat_number"])
)

# Count how many times the same pitcher faced the same batter in a game
pa["times_faced"] = pa.groupby(["game_pk", "pitcher", "batter"]).cumcount() + 1

# Group into 1st, 2nd, and 3rd+ encounters
pa["times_faced_group"] = np.where(
    pa["times_faced"] >= 3,
    "3rd+",
    np.where(pa["times_faced"] == 2, "2nd", "1st"),
)

order = ["1st", "2nd", "3rd+"]

# Sample counts per group
sample_counts = pa["times_faced_group"].value_counts().reindex(order)
print("\nSample Counts:")
print(sample_counts)

# ---------------------------------------------------------
# 3. OVERALL WHIFF & STRIKEOUT RATE CALCULATIONS
# ---------------------------------------------------------
# Merge times-faced group back to pitch-level data
jays = jays.merge(pa[keys + ["times_faced_group"]], on=keys, how="left")

# Define swings and whiffs
swing_events = [
    "hit_into_play",
    "foul",
    "swinging_strike",
    "swinging_strike_blocked",
    "foul_tip",
    "foul_bunt",
    "missed_bunt",
]

whiff_events = ["swinging_strike", "swinging_strike_blocked", "missed_bunt"]

jays["swing"] = jays["description"].isin(swing_events).astype(int)
jays["whiff"] = jays["description"].isin(whiff_events).astype(int)

# Group level whiff rates
whiff = jays.groupby("times_faced_group").agg(
    Swings=("swing", "sum"), Whiffs=("whiff", "sum")
)
whiff["Whiff Rate (%)"] = (whiff["Whiffs"] / whiff["Swings"]) * 100

# Group level strikeout rates
pa["strikeout"] = pa["events"] == "strikeout"
k_rate = pa.groupby("times_faced_group").agg(
    PA=("events", "size"), Strikeouts=("strikeout", "sum")
)
k_rate["Strikeout Rate (%)"] = (k_rate["Strikeouts"] / k_rate["PA"]) * 100

# Final summary table
results = pd.DataFrame(
    {
        "Plate Appearances": sample_counts,
        "Whiff Rate (%)": whiff["Whiff Rate (%)"],
        "Strikeout Rate (%)": k_rate["Strikeout Rate (%)"],
    }
).reindex(order)

print("\nFinal Results:")
print(results.round(2))

# ---------------------------------------------------------
# 4. GRAPH OVERALL WHIFF AND K RATES
# ---------------------------------------------------------
# Graph 1: Whiff Rate
results["Whiff Rate (%)"].plot(kind="bar")
plt.title("Blue Jays Whiff Rate by Times Faced")
plt.ylabel("Whiff Rate (%)")
plt.xticks(rotation=0)
plt.show()

# Graph 2: Strikeout Rate
results["Strikeout Rate (%)"].plot(kind="bar")
plt.title("Blue Jays Strikeout Rate by Times Faced")
plt.ylabel("Strikeout Rate (%)")
plt.xticks(rotation=0)
plt.show()

# ---------------------------------------------------------
# 5. PITCHER INDIVIDUAL PERFORMANCE ANALYSIS
# ---------------------------------------------------------
pitcher_names = jays.groupby("pitcher")["player_name"].first()

# Strikeouts per pitcher
pitcher_pa = pa.groupby(["pitcher", "times_faced_group"]).agg(
    Plate_Appearances=("strikeout", "size"), Strikeouts=("strikeout", "sum")
)
pitcher_pa["K_Rate (%)"] = (
    pitcher_pa["Strikeouts"] / pitcher_pa["Plate_Appearances"]
) * 100

# Whiffs per pitcher
pitcher_whiff = jays.groupby(["pitcher", "times_faced_group"]).agg(
    Swings=("swing", "sum"), Whiffs=("whiff", "sum")
)
pitcher_whiff["Whiff_Rate (%)"] = (
    pitcher_whiff["Whiffs"] / pitcher_whiff["Swings"]
) * 100

# Combine pitcher statistics
pitcher_stats = pitcher_pa.join(pitcher_whiff).reset_index()
pitcher_stats["Pitcher_Name"] = pitcher_stats["pitcher"].map(pitcher_names)

pitcher_stats = pitcher_stats[
    [
        "Pitcher_Name",
        "times_faced_group",
        "Plate_Appearances",
        "Strikeouts",
        "K_Rate (%)",
        "Swings",
        "Whiffs",
        "Whiff_Rate (%)",
    ]
].round(2)

pitcher_stats["times_faced_group"] = pd.Categorical(
    pitcher_stats["times_faced_group"],
    categories=["1st", "2nd", "3rd+"],
    ordered=True,
)
pitcher_stats = pitcher_stats.sort_values(["Pitcher_Name", "times_faced_group"])

# ---------------------------------------------------------
# 6. COMPARE 1ST VS 2ND ENCOUNTER FOR QUALIFIED PITCHERS
# ---------------------------------------------------------
comparison = pitcher_stats.pivot(
    index="Pitcher_Name",
    columns="times_faced_group",
    values=["Plate_Appearances", "K_Rate (%)", "Whiff_Rate (%)"],
)

# Keep pitchers with both 1st and 2nd encounter data
comparison = comparison.dropna(
    subset=[("K_Rate (%)", "1st"), ("K_Rate (%)", "2nd")]
)

comparison["K_Change"] = (
    comparison[("K_Rate (%)", "2nd")] - comparison[("K_Rate (%)", "1st")]
)
comparison["Whiff_Change"] = (
    comparison[("Whiff_Rate (%)", "2nd")]
    - comparison[("Whiff_Rate (%)", "1st")]
)

changes = pd.DataFrame(
    {
        "Pitcher": comparison.index,
        "1st PA": comparison[("Plate_Appearances", "1st")].values,
        "2nd PA": comparison[("Plate_Appearances", "2nd")].values,
        "K Change (%)": comparison["K_Change"].values,
        "Whiff Change (%)": comparison["Whiff_Change"].values,
    }
).sort_values("K Change (%)")

print("\nPitcher Performance Changes (1st to 2nd Encounter):")
print(changes.round(2))

# Graph: Change in Strikeout Rate
plt.figure(figsize=(10, 5))
plt.bar(changes["Pitcher"], changes["K Change (%)"])
plt.axhline(0, color="black", linewidth=1)
plt.title("Change in Strikeout Rate: 1st vs 2nd Encounter")
plt.xlabel("Blue Jays Pitcher")
plt.ylabel("Change in Strikeout Rate (Percentage Points)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 7. OUT RATE METRICS (GROUNDOUTS, FLYOUTS, ETC.)
# ---------------------------------------------------------
out_events = [
    "strikeout",
    "field_out",
    "force_out",
    "grounded_into_double_play",
    "strikeout_double_play",
    "double_play",
    "triple_play",
    "fielders_choice_out",
    "other_out",
]

pa["is_out"] = pa["events"].isin(out_events)

out_stats = pa.groupby("times_faced_group").agg(
    Plate_Appearances=("is_out", "size"), Outs=("is_out", "sum")
)
out_stats["Out Rate (%)"] = (
    out_stats["Outs"] / out_stats["Plate_Appearances"]
) * 100
out_stats = out_stats.reindex(["1st", "2nd", "3rd+"])

# Graph: Out Percentage
plt.figure(figsize=(7, 5))
plt.bar(out_stats.index, out_stats["Out Rate (%)"])
plt.title("Blue Jays Out Percentage by Times Faced")
plt.xlabel("Times Facing the Same Batter")
plt.ylabel("Out Rate (%)")
plt.ylim(0, 100)
plt.show()

# ---------------------------------------------------------
# 8. STARTING PITCHER BREAKDOWN BY PITCH TYPE
# ---------------------------------------------------------
starters = [
    "Gausman, Kevin",
    "Yesavage, Trey",
    "Bieber, Shane",
    "Scherzer, Max",
    "Lauer, Eric",
]

starter_pitches = jays[jays["player_name"].isin(starters)].copy()
starter_pa = pa[pa["pitcher"].isin(starter_pitches["pitcher"])].copy()
starter_pa["is_out"] = starter_pa["events"].isin(out_events)

pitch_names = {
    "FF": "4-Seam Fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "CH": "Changeup",
    "FS": "Splitter",
    "SV": "Slurve",
}

starter_pitches["Pitch_Type"] = (
    starter_pitches["pitch_type"]
    .map(pitch_names)
    .fillna(starter_pitches["pitch_type"])
)

# Whiff rate by pitch type
whiff_by_pitch = (
    starter_pitches.groupby(["player_name", "Pitch_Type", "times_faced_group"])
    .agg(Swings=("swing", "sum"), Whiffs=("whiff", "sum"))
    .reset_index()
)
whiff_by_pitch["Whiff_Rate (%)"] = (
    whiff_by_pitch["Whiffs"]
    / whiff_by_pitch["Swings"].replace(0, np.nan)
    * 100
)

# Assign pitch type of final pitch in PA for Out Rate calculations
pa_pitch_type = (
    starter_pitches[
        [
            "game_pk",
            "pitcher",
            "batter",
            "at_bat_number",
            "pitch_type",
            "pitch_number",
        ]
    ]
    .sort_values("pitch_number")
    .drop_duplicates(
        ["game_pk", "pitcher", "batter", "at_bat_number"], keep="last"
    )
)

pa_pitch_type["Pitch_Type"] = (
    pa_pitch_type["pitch_type"]
    .map(pitch_names)
    .fillna(pa_pitch_type["pitch_type"])
)

starter_pa = starter_pa.merge(
    pa_pitch_type[
        ["game_pk", "pitcher", "batter", "at_bat_number", "Pitch_Type"]
    ],
    on=["game_pk", "pitcher", "batter", "at_bat_number"],
    how="left",
)

starter_pa["player_name"] = starter_pa["pitcher"].map(pitcher_names)

outs_by_pitch = (
    starter_pa.groupby(["player_name", "Pitch_Type", "times_faced_group"])
    .agg(Plate_Appearances=("is_out", "size"), Outs=("is_out", "sum"))
    .reset_index()
)
outs_by_pitch["Out_Rate (%)"] = (
    outs_by_pitch["Outs"] / outs_by_pitch["Plate_Appearances"] * 100
)

# Merge Whiff & Out Rate data
pitch_analysis = whiff_by_pitch.merge(
    outs_by_pitch,
    on=["player_name", "Pitch_Type", "times_faced_group"],
    how="outer",
).sort_values(["player_name", "Pitch_Type", "times_faced_group"])

# ---------------------------------------------------------
# 9. PLOT INDIVIDUAL STARTER CHARTS BY PITCH TYPE
# ---------------------------------------------------------
order = ["1st", "2nd", "3rd+"]

for pitcher in starters:
    data = pitch_analysis[pitch_analysis["player_name"] == pitcher].copy()
    if data.empty:
        continue

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Whiff rate plot
    whiff_chart = data.pivot(
        index="times_faced_group",
        columns="Pitch_Type",
        values="Whiff_Rate (%)",
    ).reindex(order)
    whiff_chart.plot(kind="bar", ax=axes[0])
    axes[0].set_title(f"{pitcher} - Whiff Rate")
    axes[0].set_ylabel("Whiff Rate (%)")
    axes[0].set_xlabel("Times Faced")
    axes[0].tick_params(axis="x", rotation=0)

    # Out rate plot
    out_chart = data.pivot(
        index="times_faced_group",
        columns="Pitch_Type",
        values="Out_Rate (%)",
    ).reindex(order)
    out_chart.plot(kind="bar", ax=axes[1])
    axes[1].set_title(f"{pitcher} - Out Percentage")
    axes[1].set_ylabel("Out Rate (%)")
    axes[1].set_xlabel("Times Faced")
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.show()