import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#cleaan the toronto pitching dataset
df = pd.read_csv("mlb_postseason_data.csv", low_memory=False)
#clean corrupted game year values
df["game_year"] = pd.to_numeric(df["game_year"], errors="coerce")
#keep actual 2025 rows
df = df[df["game_year"] == 2025].copy()
print("Clean shape:",  df.shape)
#find toronto games
toronto_games = df [
    (df["home_team"] == "TOR") |
    (df["away_team"] == "TOR")
]

print("\n Toronto games:")
print(
    toronto_games[
        ["game_date" , "home_team" , "away_team" , "game_pk"]
    ]
    .drop_duplicates()
    .sort_values("game_date")
    .to_string(index=False)
)

#isolate only blue jays pitching
tor_pitching = df[
    ((df["home_team"] == "TOR") & (df["inning_topbot"] == "Top")) |
    ((df["away_team"] == "TOR") & (df["inning_topbot"] == "Bot"))
].copy()
print("\n Toronto pitching rows:", len(tor_pitching))

#add the opponent
tor_pitching["opponent"] = np.where(
    tor_pitching["home_team"] == "TOR",
    tor_pitching["away_team"],
    tor_pitching["home_team"]
)
print(tor_pitching["opponent"])

series_map = {
    "BOS": "Wild Card",
    "NYY": "ALDS",
    "SEA": "ALCS",
    "LAD": "World Series"
}
tor_pitching["series"] = tor_pitching["opponent"].map(series_map)
print(
    tor_pitching[["game_date", "opponent", "series"]]
    .drop_duplicates()
    .sort_values("game_date")
    .to_string(index=False)
)

#calculate strikeout and walk rate
pa = tor_pitching[tor_pitching["events"].notna()].copy()

series_summary = (
    pa.groupby("series")
    .agg(
        plate_appearances=("events", "count"),
        strikeouts=("events", lambda x: (x == "strikeout").sum()),
        walks=("events", lambda x: (x == "walk").sum())
    )
    .reset_index()
)
series_summary["K_rate"] = (
    series_summary["strikeouts"] /
    series_summary["plate_appearances"]
)
series_summary["BB_rate"] = (
    series_summary["walks"] /
    series_summary["plate_appearances"]
)
print(series_summary)
series_summary["K_rate"] *=100
series_summary["BB_rate"] *=100
print(series_summary)

#calculate whiff rate
swing_descriptions = [
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_bunt",
    "missed_bunt",
    "bunt_foul_tip",
    "foul_tip",
    "hit_into_play"
]

whiff_descriptions = [
    "swinging_strike",
    "swinging_strike_blocked"
]

tor_pitching["is_swing"] = (
    tor_pitching["description"].isin(swing_descriptions)
)

tor_pitching["is_whiff"] = (
    tor_pitching["description"].isin(whiff_descriptions)
)
fastball_types = ["FF", "SI", "FC"]

tor_pitching["pitch_group"] = np.where(
    tor_pitching["pitch_type"].isin(fastball_types),
    "Fastball",
    "Other pitches"
)
#Compare whiff rates by serites and pitch group
whiff_by_pitch = (
    tor_pitching
    .groupby(["series", "pitch_group"])
    .agg(
        pitches=("pitch_type", "count"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum")
    )
    .reset_index()
)
whiff_by_pitch["whiff_rate_pct"] = (
    whiff_by_pitch["whiffs"]
    / whiff_by_pitch["swings"] * 100
)
print("\nWhiffrate by series and pitch group:")
print(whiff_by_pitch.round(2).to_string(index=False))
whiff_summary = (
    tor_pitching
    .groupby("series")
    .agg(
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum")
    )
    .reset_index()
)
whiff_summary["whiff_rate"] = (
    whiff_summary["whiffs"] /
    whiff_summary["swings"] * 100
)

print(whiff_summary)

#calculate fastball velocity
fastballs = tor_pitching[
    tor_pitching["pitch_type"].isin(["FF", "SI", "FC"])
].copy()

velocity_summary = (
    fastballs
    .groupby("series")["release_speed"]
    .mean()
    .reset_index(name="avg_fastball_velocity")
)
print(velocity_summary)
velocity_summary = (
    fastballs
    .groupby("series")["release_speed"]
    .agg(["mean", "median", "count"])
    .reset_index()
)
print(velocity_summary)
# Fastball velocity and whiff rate by series
fastball_summary = (
    tor_pitching[
        tor_pitching["pitch_type"].isin(fastball_types)
    ]
    .groupby("series")
    .agg(
        avg_velocity=("release_speed", "mean"),
        median_velocity=("release_speed", "median"),
        pitches=("pitch_type", "count"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum")
    )
    .reset_index()
)
# INDIVIDUAL BLUE JAYS PITCHER ANALYSIS

# Use the actual pitcher's name column when available
pitcher_col = (
    "pitcher_name"
    if "pitcher_name" in tor_pitching.columns
    else "player_name"
)

print("\nChecking pitcher names:")
print(tor_pitching[pitcher_col].dropna().unique())

fastball_types = ["FF", "SI", "FC"]

player_results = []

# Repeat the analysis for each pitcher
for player in tor_pitching[pitcher_col].dropna().unique():

    player_df = tor_pitching[
        tor_pitching[pitcher_col] == player
    ]

    # Calculate statistics separately for each postseason series
    for series, series_df in player_df.groupby("series"):

        # Plate appearances completed by an outcome
        pa = series_df[series_df["events"].notna()]
        total_pa = len(pa)

        strikeouts = (pa["events"] == "strikeout").sum()
        walks = (pa["events"] == "walk").sum()

        # Pitch-level swing and whiff statistics
        swings = series_df["is_swing"].sum()
        whiffs = series_df["is_whiff"].sum()

        whiff_rate = (
            whiffs / swings * 100 if swings > 0 else np.nan
        )

        # Fastball-specific statistics
        fastballs = series_df[
            series_df["pitch_type"].isin(fastball_types)
        ]

        avg_velocity = fastballs["release_speed"].mean()

        # Store this pitcher's results for this series
        player_results.append({
            "pitcher": player,
            "series": series,
            "plate_appearances": total_pa,
            "strikeouts": strikeouts,
            "K_rate_pct": (
                strikeouts / total_pa * 100
                if total_pa > 0 else np.nan
            ),
            "walks": walks,
            "BB_rate_pct": (
                walks / total_pa * 100
                if total_pa > 0 else np.nan
            ),
            "pitches": len(series_df),
            "swings": swings,
            "whiffs": whiffs,
            "whiff_rate_pct": whiff_rate,
            "fastball_pitches": len(fastballs),
            "avg_fastball_velocity": avg_velocity
        })

# Turn the results into a readable table
player_summary = pd.DataFrame(player_results)

player_summary = player_summary.sort_values(
    ["pitcher", "series"]
)

print("\nINDIVIDUAL PITCHER PERFORMANCE BY SERIES")
print(player_summary.round(2).to_string(index=False))

# Optional: save the results to a CSV for the group
player_summary.to_csv(
    "blue_jays_individual_pitcher_summary.csv",
    index=False
)

print("\nSaved individual pitcher summary to CSV.")

fastball_summary["whiff_rate_pct"] = (
    fastball_summary["whiffs"]
    / fastball_summary["swings"] * 100
)

print("\nFastball performance by series:")
print(fastball_summary.round(2).to_string(index=False))
#combine the results into a table
results = (
    series_summary[
        ["series", "plate_appearances", "K_rate", "BB_rate"]
    ]
    .merge(
        whiff_summary[["series", "whiff_rate"]],
        on="series",
    )
    .merge(
        velocity_summary[["series", "mean"]],
        on="series"
    )
    .rename(columns={
        "mean": "avg_fastball_velocity",
    })
)
print(results)
series_order = ["Wild Card" , "ALDS" , "ALCS" , "World Series"]
results["series"] = pd.Categorical(
    results["series"],
    categories=series_order,
    ordered=True
)

results = results.sort_values("series")
#make visuals using matplotlib
# graph 1 K% by series
plt.figure(figsize=(8,5))

plt.plot(
    results["series"],
    results["K_rate"],
    marker="o"
)

plt.xlabel("Postseason Series")
plt.ylabel("Strikeout Rate (%)")
plt.title("Blue Jays pitcher Strikeout Rate by Postseason Series")
plt.grid(True)

plt.show()

#Graph 2 Whiff rate
plt.figure(figsize=(8,5))

plt.plot(
    results["series"],
    results["whiff_rate"],
    marker="o"
)

plt.xlabel("Postseason Series")
plt.ylabel("Whiff Rate (%)")
plt.title("Blue Jays Pitcher Whiff Rate by Postseason Series")
plt.grid(True)

plt.show()

#Graph 3 Fastball Velocity
plt.figure(figsize=(8,5))

plt.plot(
    results["series"],
    results["avg_fastball_velocity"],
    marker="o"
)

plt.xlabel("Postseason Series")
plt.ylabel("Average Fastball Velocity (mph)")
plt.title("Blue Jays Fastball Velocity by Postseason Series")
plt.grid(True)

plt.show()

# INDIVIDUAL PITCHER CHANGES BY PITCH TYPE

# Put postseason rounds in the correct chronological order
series_order = ["Wild Card", "ALDS", "ALCS", "World Series"]

# Use the pitcher column already identified above
pitch_data = tor_pitching[
    tor_pitching[pitcher_col].notna()
    & tor_pitching["series"].notna()
    & tor_pitching["pitch_type"].notna()
].copy()

pitch_data["series"] = pd.Categorical(
    pitch_data["series"],
    categories=series_order,
    ordered=True
)

# Summarize each pitcher's performance with each pitch type
pitcher_pitch_summary = (
    pitch_data
    .groupby(
        [pitcher_col, "pitch_type", "series"],
        observed=True
    )
    .agg(
        pitches=("pitch_type", "size"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        avg_velocity=("release_speed", "mean")
    )
    .reset_index()
)

# Calculate whiff rate for each pitcher and pitch type
pitcher_pitch_summary["whiff_rate_pct"] = np.where(
    pitcher_pitch_summary["swings"] > 0,
    pitcher_pitch_summary["whiffs"]
    / pitcher_pitch_summary["swings"] * 100,
    np.nan
)

# Only compare whiff rates where there are enough swings
# Keep the raw counts in the table so sample sizes are visible
pitcher_pitch_summary["reliable_whiff_rate"] = (
    pitcher_pitch_summary["whiff_rate_pct"].where(
        pitcher_pitch_summary["swings"] >= 10
    )
)

print("\nINDIVIDUAL PITCHER PERFORMANCE BY PITCH TYPE")
print(
    pitcher_pitch_summary
    .sort_values([pitcher_col, "pitch_type", "series"])
    .round(2)
    .to_string(index=False)
)


# Calculate change from one round to the next


# Pivot each metric so the postseason rounds become columns
def calculate_series_changes(metric):
    wide = pitcher_pitch_summary.pivot_table(
        index=[pitcher_col, "pitch_type"],
        columns="series",
        values=metric,
        observed=False
    )

    # Ensure all rounds appear in chronological order
    wide = wide.reindex(columns=series_order)

    # Difference between consecutive rounds
    changes = wide.diff(axis=1)

    changes.columns = [
        f"{col}_change" for col in changes.columns
    ]

    return changes.reset_index()


whiff_changes = calculate_series_changes("reliable_whiff_rate")
velocity_changes = calculate_series_changes("avg_velocity")

print("\nCHANGE IN WHIFF RATE: PERCENTAGE POINTS")
print(whiff_changes.round(2).to_string(index=False))

print("\nCHANGE IN AVERAGE VELOCITY: MPH")
print(velocity_changes.round(2).to_string(index=False))

# Save both detailed results and changes for the group
pitcher_pitch_summary.to_csv(
    "blue_jays_pitcher_by_pitch_type.csv",
    index=False
)

whiff_changes.to_csv(
    "blue_jays_whiff_rate_changes.csv",
    index=False
)

velocity_changes.to_csv(
    "blue_jays_velocity_changes.csv",
    index=False
)

# GRAPH 1: WHIFF RATE BY PITCH TYPE
# One graph per pitch type, one line per pitcher


for pitch in sorted(pitcher_pitch_summary["pitch_type"].dropna().unique()):

    subset = pitcher_pitch_summary[
        pitcher_pitch_summary["pitch_type"] == pitch
    ]

    whiff_plot = subset.pivot(
        index="series",
        columns=pitcher_col,
        values="reliable_whiff_rate"
    ).reindex(series_order)

    # Skip pitch types with no usable rates
    if whiff_plot.notna().sum().sum() == 0:
        continue

    ax = whiff_plot.plot(
        kind="line",
        marker="o",
        figsize=(9, 5)
    )

    ax.set_title(f"Individual Pitcher Whiff Rate — {pitch}")
    ax.set_xlabel("Postseason Series")
    ax.set_ylabel("Whiff Rate (%)")
    ax.grid(True)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
# GRAPH 2: VELOCITY BY PITCH TYPE
# One graph per pitch type, one line per pitcher

for pitch in sorted(pitcher_pitch_summary["pitch_type"].dropna().unique()):

    subset = pitcher_pitch_summary[
        pitcher_pitch_summary["pitch_type"] == pitch
    ]

    velocity_plot = subset.pivot(
        index="series",
        columns=pitcher_col,
        values="avg_velocity"
    ).reindex(series_order)

    if velocity_plot.notna().sum().sum() == 0:
        continue

    ax = velocity_plot.plot(
        kind="line",
        marker="o",
        figsize=(9, 5)
    )

    ax.set_title(f"Individual Pitcher Average Velocity — {pitch}")
    ax.set_xlabel("Postseason Series")
    ax.set_ylabel("Average Velocity (mph)")
    ax.grid(True)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# ==========================================
# OUT PERCENTAGE BY PITCHER AND SERIES
# ==========================================

out_events = [
    "field_out",
    "strikeout",
    "grounded_into_double_play",
    "fielders_choice_out",
    "force_out",
    "double_play"
]

# Keep only completed plate appearances
completed_pa = tor_pitching[
    tor_pitching["events"].notna()
    & tor_pitching[pitcher_col].notna()
    & tor_pitching["series"].notna()
].copy()

# Identify whether each plate appearance ended in a listed out event
completed_pa["is_out_event"] = (
    completed_pa["events"].isin(out_events)
)

# Summarize each pitcher's results in each postseason series
out_summary = (
    completed_pa
    .groupby([pitcher_col, "series"], observed=True)
    .agg(
        plate_appearances=("events", "count"),
        out_events=("is_out_event", "sum")
    )
    .reset_index()
)

out_summary["out_percentage"] = (
    out_summary["out_events"]
    / out_summary["plate_appearances"]
    * 100
)

print("\nOUT PERCENTAGE BY PITCHER AND SERIES")
print(
    out_summary
    .sort_values([pitcher_col, "series"])
    .round(2)
    .to_string(index=False)
)

# Compare each pitcher against their own previous series
series_order = ["Wild Card", "ALDS", "ALCS", "World Series"]

out_summary["series"] = pd.Categorical(
    out_summary["series"],
    categories=series_order,
    ordered=True
)

out_changes = out_summary.pivot(
    index=pitcher_col,
    columns="series",
    values="out_percentage"
).reindex(columns=series_order)

out_changes.columns.name = None

# Difference in percentage points between consecutive series
out_changes["ALDS change (pp)"] = (
    out_changes["ALDS"] - out_changes["Wild Card"]
)
out_changes["ALCS change (pp)"] = (
    out_changes["ALCS"] - out_changes["ALDS"]
)
out_changes["World Series change (pp)"] = (
    out_changes["World Series"] - out_changes["ALCS"]
)

print("\nCHANGE IN OUT PERCENTAGE BY PITCHER")
print(out_changes.round(2).to_string())

# Save results for the group
out_summary.to_csv(
    "blue_jays_out_percentage_by_pitcher.csv",
    index=False
)

out_changes.to_csv(
    "blue_jays_out_percentage_changes.csv"
)

# GRAPH: OUT PERCENTAGE OVER THE POSTSEASON


out_plot = out_summary.pivot(
    index="series",
    columns=pitcher_col,
    values="out_percentage"
).reindex(series_order)

ax = out_plot.plot(
    kind="line",
    marker="o",
    figsize=(11, 6)
)

ax.set_title("Blue Jays Pitcher Out Percentage by Postseason Series")
ax.set_xlabel("Postseason Series")
ax.set_ylabel("Plate Appearances Ending in Listed Out Events (%)")
ax.grid(True)
plt.tight_layout()
plt.show()
