# tennis_dashboard.py
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path

# =========================================================
# Load data
# =========================================================
HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "Adi_1.csv"  # keep Adi_1.csv in same folder
df = pd.read_csv(DATA_PATH)

# ---- Required columns for full functionality ----
required_cols = {
    "Player", "Shot", "Type", "Stroke", "Spin", "Speed (MPH)", "Winner",
    "Point", "Result", "Game", "Set",
    "bounce_x_std", "bounce_y_std", "hit_x_std", "hit_y_std",
    "Direction", "Hit Zone", "Bounce Zone"
}
missing = required_cols - set(df.columns)
if missing:
    raise KeyError(f"Missing columns in Adi_1.csv: {sorted(missing)}")

# Sort to ensure within-point ordering works
df = df.sort_values(["Set", "Game", "Point", "Shot"]).reset_index(drop=True)

# Robust unique point id
df["_point_id"] = df[["Set", "Game", "Point"]].astype(str).agg("-".join, axis=1)

players = sorted(df["Player"].dropna().unique().tolist())

def pick_opponent(selected_player: str) -> str:
    # Prefer explicit "Opponent" label if present
    if "Opponent" in players and selected_player != "Opponent":
        return "Opponent"
    # else first other player
    others = [p for p in players if p != selected_player]
    return others[0] if others else selected_player

# =========================================================
# Court dimensions (meters)
# =========================================================
Y_NEAR = 0.0
Y_NET = 11.885
Y_FAR = 23.77

X_SINGLES = 4.115
X_DOUBLES = 5.485

SERVICE_TO_NET = 6.40
Y_NEAR_SERVICE = Y_NET - SERVICE_TO_NET
Y_FAR_SERVICE = Y_NET + SERVICE_TO_NET

# =========================================================
# Calculate Hit Region based on hit_x_std and hit_y_std
# =========================================================
X = df["hit_x_std"]
Y = df["hit_y_std"]

# Define region masks
mask_service_box = (
    X.notna() & Y.notna() &
    (Y >= Y_NEAR_SERVICE) & (Y <= Y_NET) &
    (X.abs() <= X_SINGLES)
)

mask_baseline_area = (
    X.notna() & Y.notna() &
    (Y >= Y_NEAR) & (Y < Y_NEAR_SERVICE) &
    (X.abs() <= X_SINGLES)
)

mask_behind_baseline = (
    X.notna() & Y.notna() &
    (Y < Y_NEAR)
)

mask_out_right_singles = (
    X.notna() & Y.notna() &
    (X > X_SINGLES)
)

mask_out_left_singles = (
    X.notna() & Y.notna() &
    (X < -X_SINGLES)
)

# Assign hit_region column
df["hit_region"] = "unknown"
df.loc[mask_service_box, "hit_region"] = "made_inside_service_box"
df.loc[mask_baseline_area, "hit_region"] = "made_inside_baseline"
df.loc[mask_behind_baseline, "hit_region"] = "made_behind_baseline"
df.loc[mask_out_right_singles, "hit_region"] = "made_outside_right_singles"
df.loc[mask_out_left_singles, "hit_region"] = "made_outside_left_singles"

# =========================================================
# Styling
# =========================================================
colors = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "success": "#2A9D8F",
    "warning": "#E76F51",
    "bg": "#F8F9FA",
    "text": "#212529"
}

def kpi_card(title, value):
    return html.Div(
        [
            html.Div(title, style={"fontSize": "13px", "opacity": 0.8}),
            html.Div(value, style={"fontSize": "26px", "fontWeight": "700"}),
        ],
        style={
            "background": "white",
            "padding": "14px 16px",
            "borderRadius": "12px",
            "boxShadow": "0 2px 6px rgba(0,0,0,0.08)",
        },
    )

def render_table(df_in, page_size=12):
    return dash_table.DataTable(
        data=df_in.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df_in.columns],
        page_size=page_size,
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": "Arial",
            "fontSize": 12,
            "padding": "6px",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_header={"fontWeight": "700", "backgroundColor": "#f1f3f5"},
    )

def options_from_col(col):
    vals = [v for v in df[col].dropna().unique().tolist()]
    vals = sorted(vals, key=lambda x: str(x))
    return [{"label": str(v), "value": v} for v in vals]

# =========================================================
# Notebook-style helpers
# =========================================================
def normalize_side_from_hit_zone(hit_zone):
    s = str(hit_zone).strip().lower()
    if s.startswith("deuce") or s == "deuce":
        return "Deuce"
    if s.startswith("ad") or s == "ad":
        return "Ad"
    return "Unknown"

def fmt_record(won, total):
    pct = round((won / total) * 100, 1) if total else 0
    return f"{int(won)}/{int(total)} ({pct}%)"

def fmt_made(made, total):
    if total == 0:
        return "0/0 —"
    pct = 100 * made / total
    return f"{int(made)}/{int(total)} {pct:.1f}%"

def pct_from_str(s):
    try:
        return float(str(s).split()[-1].replace("%", ""))
    except Exception:
        return np.nan

# =========================================================
# Point-level tables
# =========================================================
shots_per_point = df.groupby("_point_id").size().rename("Shot_Count").reset_index()
winner_per_point = df.groupby("_point_id")["Winner"].first().reset_index()
points = shots_per_point.merge(winner_per_point, on="_point_id", how="left").rename(columns={"Winner": "Point_Winner"})
df = df.merge(shots_per_point, on="_point_id", how="left")

last_shots = df.groupby("_point_id").tail(1).copy()

# =========================================================
# Serve summary (notebook-style)
# =========================================================
one_row_points = df.groupby("_point_id").size()
one_row_point_ids = set(one_row_points[one_row_points == 1].index.tolist())

aces = df[
    (df["_point_id"].isin(one_row_point_ids)) &
    (df["Type"].eq("first_serve")) &
    (df["Result"].eq("In"))
]
ace_counts = aces.groupby("Player").size().reset_index(name="Aces")

double_faults = df[(df["Type"] == "second_serve") & (df["Result"] != "In")]
df_counts = double_faults.groupby("Player").size().reset_index(name="Double_Faults")

first = df[df["Type"] == "first_serve"]
first_serve_pct = (first["Result"].eq("In").groupby(first["Player"]).mean() * 100).reset_index(name="First_Serve_%")

second = df[df["Type"] == "second_serve"]
second_serve_pct = (second["Result"].eq("In").groupby(second["Player"]).mean() * 100).reset_index(name="Second_Serve_%")

first_in = df[(df["Type"] == "first_serve") & (df["Result"] == "In")]
first_serve_won = (first_in["Winner"].eq(first_in["Player"]).groupby(first_in["Player"]).mean() * 100).reset_index(name="First_Serve_Won_%")

second_in = df[(df["Type"] == "second_serve") & (df["Result"] == "In")]
second_serve_won = (second_in["Winner"].eq(second_in["Player"]).groupby(second_in["Player"]).mean() * 100).reset_index(name="Second_Serve_Won_%")

all_players = df[["Player"]].drop_duplicates()

summary_serve = (
    all_players
    .merge(ace_counts, on="Player", how="left")
    .merge(df_counts, on="Player", how="left")
    .merge(first_serve_pct, on="Player", how="left")
    .merge(second_serve_pct, on="Player", how="left")
    .merge(first_serve_won, on="Player", how="left")
    .merge(second_serve_won, on="Player", how="left")
).fillna(0)

summary_serve["Aces"] = summary_serve["Aces"].astype(int)
summary_serve["Double_Faults"] = summary_serve["Double_Faults"].astype(int)
for c in ["First_Serve_%", "Second_Serve_%", "First_Serve_Won_%", "Second_Serve_Won_%"]:
    summary_serve[c] = summary_serve[c].astype(float).round(1)

# =========================================================
# Return summary (notebook-style)
# =========================================================
return_winners = last_shots[
    (last_shots["Type"].isin(["first_return", "second_return"])) &
    (last_shots["Result"] == "In")
]
return_winner_counts = return_winners.groupby("Player").size().reset_index(name="Return_Winners")

df["_Next_Result"] = df.groupby("_point_id")["Result"].shift(-1)
return_forced_errors = df[
    (df["Type"].isin(["first_return", "second_return"])) &
    (df["_Next_Result"].notna()) &
    (df["_Next_Result"] != "In")
]
rfe_counts = return_forced_errors.groupby("Player").size().reset_index(name="Return_Forced_Errors")

second_return_missed = df[(df["Type"] == "second_return") & (df["Result"] != "In")]
srm_counts = second_return_missed.groupby("Player").size().reset_index(name="Second_Return_Missed")

first_return_points = df[df["Type"] == "first_return"]
first_return_won = (first_return_points["Winner"].eq(first_return_points["Player"]).groupby(first_return_points["Player"]).mean() * 100).reset_index(name="First_Return_Won_%")

second_return_points = df[df["Type"] == "second_return"]
second_return_won = (second_return_points["Winner"].eq(second_return_points["Player"]).groupby(second_return_points["Player"]).mean() * 100).reset_index(name="Second_Return_Won_%")

return_summary = (
    all_players
    .merge(return_winner_counts, on="Player", how="left")
    .merge(rfe_counts, on="Player", how="left")
    .merge(srm_counts, on="Player", how="left")
    .merge(first_return_won, on="Player", how="left")
    .merge(second_return_won, on="Player", how="left")
).fillna(0)

for c in ["Return_Winners", "Return_Forced_Errors", "Second_Return_Missed"]:
    return_summary[c] = return_summary[c].astype(int)
for c in ["First_Return_Won_%", "Second_Return_Won_%"]:
    return_summary[c] = return_summary[c].astype(float).round(1)

# =========================================================
# Winners / Net / All points won / Unforced errors
# =========================================================
winners = last_shots[(last_shots["Result"] == "In") & (last_shots["Stroke"] != "Serve")]
winner_counts = winners.groupby("Player").size().reset_index(name="Winners")

# Net points (no groupby.apply warning)
net_points = df[df["Stroke"].isin(["Volley", "Overhead"])].copy()
if len(net_points) > 0:
    net_points["is_net_won"] = (net_points["Winner"] == net_points["Player"]).astype(int)
    net_point_stats = (
        net_points.groupby("Player", as_index=False)
        .agg(Net_Won=("is_net_won", "sum"), Net_Total=("is_net_won", "size"))
    )
    net_point_stats["Net_Point_Won_%"] = net_point_stats.apply(
        lambda r: f'{int(r["Net_Won"])}/{int(r["Net_Total"])} ({(r["Net_Won"]/r["Net_Total"]*100):.1f}%)'
        if r["Net_Total"] else "0/0 (0.0%)",
        axis=1
    )
else:
    net_point_stats = pd.DataFrame({"Player": players, "Net_Point_Won_%": ["0/0 (0.0%)"] * len(players)})

# All points won (point winners)
all_points_total = len(points)
all_points_won = points.groupby("Point_Winner").size().reset_index(name="Points_Won").rename(columns={"Point_Winner": "Player"})
all_points_won["All_Points_Won"] = all_points_won.apply(
    lambda r: f'{int(r["Points_Won"])}/{all_points_total} ({(r["Points_Won"]/all_points_total*100):.1f}%)'
    if all_points_total else "0/0 (0.0%)",
    axis=1
)

# Unforced errors proxy (speed-relative)
avg_speeds = (
    df[df["Stroke"] != "Serve"]
    .groupby(["Player", "Stroke"])["Speed (MPH)"]
    .mean()
    .reset_index()
    .rename(columns={"Speed (MPH)": "Avg_Speed"})
)

df_with_avg = df.copy()
df_with_avg["Prev_Player"] = df_with_avg["Player"].shift(1)
df_with_avg["Prev_Stroke"] = df_with_avg["Stroke"].shift(1)
df_with_avg["Prev_Speed"] = df_with_avg["Speed (MPH)"].shift(1)
df_with_avg["Prev_PointID"] = df_with_avg["_point_id"].shift(1)

df_with_avg = df_with_avg.merge(
    avg_speeds,
    left_on=["Prev_Player", "Prev_Stroke"],
    right_on=["Player", "Stroke"],
    how="left",
    suffixes=("", "_prev")
)

unforced_errors = df_with_avg[
    (df_with_avg["Result"] != "In") &
    (df_with_avg["Stroke"] != "Serve") &
    (df_with_avg["Prev_Player"] != df_with_avg["Player"]) &
    (df_with_avg["Prev_PointID"] == df_with_avg["_point_id"]) &
    (df_with_avg["Prev_Speed"] < df_with_avg["Avg_Speed"]) &
    (df_with_avg["Prev_Stroke"] != "Serve")
]
ue_counts = unforced_errors.groupby("Player").size().reset_index(name="Unforced_Errors")

winners_summary = (
    all_players
    .merge(winner_counts, on="Player", how="left")
    .merge(net_point_stats[["Player", "Net_Point_Won_%"]], on="Player", how="left")
    .merge(all_points_won[["Player", "All_Points_Won"]], on="Player", how="left")
    .merge(ue_counts, on="Player", how="left")
).fillna(0)

winners_summary["Winners"] = winners_summary["Winners"].astype(int)
winners_summary["Unforced_Errors"] = winners_summary["Unforced_Errors"].astype(int)

# =========================================================
# Rally analysis
# =========================================================
rally_dist = pd.DataFrame({
    "Category": ["1 shot", "2 shots", "3 shots", "4 shots", "5-6 shots", "7+ shots"],
    "Count": [
        (points["Shot_Count"] == 1).sum(),
        (points["Shot_Count"] == 2).sum(),
        (points["Shot_Count"] == 3).sum(),
        (points["Shot_Count"] == 4).sum(),
        ((points["Shot_Count"] >= 5) & (points["Shot_Count"] <= 6)).sum(),
        (points["Shot_Count"] >= 7).sum()
    ]
})
rally_dist["Percentage"] = (rally_dist["Count"] / rally_dist["Count"].sum() * 100).round(1)

rally_mode_by_player = []
for p in players:
    won = points[points["Point_Winner"] == p]
    lost = points[points["Point_Winner"] != p]
    most_won = int(won["Shot_Count"].mode().iloc[0]) if len(won) else 0
    most_lost = int(lost["Shot_Count"].mode().iloc[0]) if len(lost) else 0
    rally_mode_by_player.append({"Player": p, "Most_Won_Shots": most_won, "Most_Lost_Shots": most_lost})
rally_mode_df = pd.DataFrame(rally_mode_by_player)

# =========================================================
# Serve+1 / Return+1 (tables + point-level df for donut)
# =========================================================
def two_shot_point_df_by_role(df_in, anchor_type, plus_one_type, hit_zone_col="Hit Zone"):
    def qualifies_point(g):
        anchor = g[g["Type"] == anchor_type]
        if len(anchor) != 1:
            return False
        focal_player = anchor["Player"].iloc[0]
        focal_rows = g[g["Player"] == focal_player]
        return (len(focal_rows) == 2) and (set(focal_rows["Type"]) == {anchor_type, plus_one_type})

    pts = df_in.groupby("_point_id", group_keys=False).filter(qualifies_point)

    rows = []
    for pid, g in pts.groupby("_point_id"):
        anchor_row = g.loc[g["Type"] == anchor_type].iloc[0]
        focal_player = anchor_row["Player"]
        side = normalize_side_from_hit_zone(anchor_row[hit_zone_col])
        winner = g["Winner"].iloc[0]
        rows.append({"_point_id": pid, "Player": focal_player, "Side": side, "Won": (winner == focal_player)})
    return pd.DataFrame(rows)

def two_shot_table_by_role(df_in, anchor_type, plus_one_type, role_label, hit_zone_col="Hit Zone"):
    point_df = two_shot_point_df_by_role(df_in, anchor_type, plus_one_type, hit_zone_col)
    if point_df.empty:
        return pd.DataFrame({"Player": players})

    rows = []
    for p in players:
        pp = point_df[point_df["Player"] == p]
        dd = pp[pp["Side"] == "Deuce"]
        aa = pp[pp["Side"] == "Ad"]
        rows.append({
            "Player": p,
            f"{role_label} (All)": fmt_record(pp["Won"].sum(), len(pp)),
            f"{role_label} (Deuce)": fmt_record(dd["Won"].sum(), len(dd)),
            f"{role_label} (Ad)": fmt_record(aa["Won"].sum(), len(aa)),
        })
    return pd.DataFrame(rows)

serve_plus_one_tbl = two_shot_table_by_role(df, "first_serve", "serve_plus_one", "Serve+1")
return_plus_one_tbl = two_shot_table_by_role(df, "first_return", "return_plus_one", "Return+1")

serve_plus_one_points = two_shot_point_df_by_role(df, "first_serve", "serve_plus_one", "Hit Zone")
return_plus_one_points = two_shot_point_df_by_role(df, "first_return", "return_plus_one", "Hit Zone")

# =========================================================
# Two-ring donut (Win/Loss × Deuce/Ad) — Plotly version
# =========================================================
def two_ring_donut_winloss_deucead(point_df, player, title, drop_unknown=True):
    d = point_df[point_df["Player"] == player].copy()
    if drop_unknown:
        d = d[d["Side"].isin(["Deuce", "Ad"])]

    if d.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{title}<br><sup>No qualifying Deuce/Ad points</sup>", height=420)
        return fig

    win_deuce = int(((d["Won"] == True) & (d["Side"] == "Deuce")).sum())
    win_ad = int(((d["Won"] == True) & (d["Side"] == "Ad")).sum())
    loss_deuce = int(((d["Won"] == False) & (d["Side"] == "Deuce")).sum())
    loss_ad = int(((d["Won"] == False) & (d["Side"] == "Ad")).sum())

    outer_labels = ["Win–Deuce", "Win–Ad", "Loss–Deuce", "Loss–Ad"]
    outer_values = [win_deuce, win_ad, loss_deuce, loss_ad]
    inner_labels = ["Win", "Loss"]
    inner_values = [win_deuce + win_ad, loss_deuce + loss_ad]

    if sum(outer_values) == 0:
        fig = go.Figure()
        fig.update_layout(title=f"{title}<br><sup>Counts are all zero</sup>", height=420)
        return fig

    fig = go.Figure()

    # Outer ring
    fig.add_trace(go.Pie(
        labels=outer_labels,
        values=outer_values,
        hole=0.55,
        sort=False,
        direction="clockwise",
        textinfo="label+percent",
        textposition="outside",
        showlegend=False
    ))

    # Inner ring (smaller domain so it sits inside)
    fig.add_trace(go.Pie(
        labels=inner_labels,
        values=inner_values,
        hole=0.55,
        sort=False,
        direction="clockwise",
        textinfo="label+percent",
        textposition="inside",
        showlegend=False,
        domain={"x": [0.22, 0.78], "y": [0.22, 0.78]}
    ))

    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="white"
    )
    return fig

# =========================================================
# Region table + Movement
# =========================================================
def summarize_region(df_in, mask, colname):
    tmp = df_in.loc[mask, ["Player", "Result"]].copy()
    tmp["made"] = (tmp["Result"] == "In")
    s = tmp.groupby("Player")["made"].agg(made="sum", total="count").reset_index()
    s[colname] = [fmt_made(m, t) for m, t in zip(s["made"], s["total"])]
    return s[["Player", colname]]

X = pd.to_numeric(df["hit_x_std"], errors="coerce")
Y = pd.to_numeric(df["hit_y_std"], errors="coerce")

mask_service_box = X.notna() & Y.notna() & (Y >= Y_NEAR_SERVICE) & (Y <= Y_NET) & (X.abs() <= X_SINGLES)
mask_baseline_area = X.notna() & Y.notna() & (Y >= Y_NEAR) & (Y < Y_NEAR_SERVICE) & (X.abs() <= X_SINGLES)
mask_behind_baseline = X.notna() & Y.notna() & (Y < Y_NEAR)
mask_out_right_singles = X.notna() & Y.notna() & (X > X_SINGLES)
mask_out_left_singles = X.notna() & Y.notna() & (X < -X_SINGLES)

svc = summarize_region(df, mask_service_box, "service_box")
base = summarize_region(df, mask_baseline_area, "baseline_area")
back = summarize_region(df, mask_behind_baseline, "behind_baseline")
right = summarize_region(df, mask_out_right_singles, "outside_right_singles")
left = summarize_region(df, mask_out_left_singles, "outside_left_singles")

region_tbl = svc
for t in [base, back, right, left]:
    region_tbl = pd.merge(region_tbl, t, on="Player", how="outer")

fill_cols = ["service_box", "baseline_area", "behind_baseline", "outside_right_singles", "outside_left_singles"]
region_tbl = region_tbl.fillna({c: "0/0 —" for c in fill_cols})

region_tbl["pct_service"] = region_tbl["service_box"].apply(pct_from_str)
region_tbl["pct_baseline"] = region_tbl["baseline_area"].apply(pct_from_str)
region_tbl = region_tbl.sort_values(["pct_service", "pct_baseline"], ascending=[False, False], na_position="last")

region_tbl_display = region_tbl.rename(columns={
    "service_box": "made_inside_service_box",
    "baseline_area": "made_inside_baseline",
    "behind_baseline": "made_behind_baseline",
    "outside_right_singles": "made_outside_right_singles",
    "outside_left_singles": "made_outside_left_singles"
})[[
    "Player",
    "made_inside_service_box",
    "made_inside_baseline",
    "made_behind_baseline",
    "made_outside_right_singles",
    "made_outside_left_singles"
]]

def add_distance_within_point(df_in, group_cols=("_point_id", "Player"), x_col="hit_x_std", y_col="hit_y_std"):
    d = df_in.copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d["_row"] = np.arange(len(d))
    d = d.sort_values(list(group_cols) + ["_row"])
    g = d.groupby(list(group_cols), sort=False)
    prev_x = g[x_col].shift(1)
    prev_y = g[y_col].shift(1)
    d["move_step"] = np.sqrt((d[x_col] - prev_x) ** 2 + (d[y_col] - prev_y) ** 2)
    return d.drop(columns=["_row"])

df_dist = add_distance_within_point(df)
move_summary = (
    df_dist.groupby("Player")["move_step"]
    .agg(avg_distance="mean", total_distance="sum")
    .reset_index()
    .sort_values("total_distance", ascending=False)
)
move_summary[["avg_distance", "total_distance"]] = move_summary[["avg_distance", "total_distance"]].round(2)

# =========================================================
# Court plotting
# =========================================================
def add_court_shapes(fig, scope="half"):
    if scope == "half":
        # near baseline to net
        fig.add_shape(type="line", x0=-X_SINGLES, y0=0, x1=-X_SINGLES, y1=Y_NET, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0= X_SINGLES, y0=0, x1= X_SINGLES, y1=Y_NET, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=-X_DOUBLES, y0=0, x1=-X_DOUBLES, y1=Y_NET, line=dict(color="black", width=1, dash="dash"))
        fig.add_shape(type="line", x0= X_DOUBLES, y0=0, x1= X_DOUBLES, y1=Y_NET, line=dict(color="black", width=1, dash="dash"))
        fig.add_shape(type="line", x0=-X_DOUBLES, y0=0, x1= X_DOUBLES, y1=0, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=-X_DOUBLES, y0=Y_NET, x1= X_DOUBLES, y1=Y_NET, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=-X_SINGLES, y0=Y_NEAR_SERVICE, x1=X_SINGLES, y1=Y_NEAR_SERVICE, line=dict(color="black", width=1.5))
        fig.add_shape(type="line", x0=0, y0=Y_NEAR_SERVICE, x1=0, y1=Y_NET, line=dict(color="black", width=1.5))
        fig.update_yaxes(range=[-4, Y_NET + 1])
    else:
        # full court
        fig.add_shape(type="line", x0=-X_SINGLES, y0=0, x1=-X_SINGLES, y1=Y_FAR, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0= X_SINGLES, y0=0, x1= X_SINGLES, y1=Y_FAR, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=-X_DOUBLES, y0=0, x1=-X_DOUBLES, y1=Y_FAR, line=dict(color="black", width=1, dash="dash"))
        fig.add_shape(type="line", x0= X_DOUBLES, y0=0, x1= X_DOUBLES, y1=Y_FAR, line=dict(color="black", width=1, dash="dash"))
        fig.add_shape(type="line", x0=-X_DOUBLES, y0=0, x1= X_DOUBLES, y1=0, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=-X_DOUBLES, y0=Y_FAR, x1= X_DOUBLES, y1=Y_FAR, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=-X_DOUBLES, y0=Y_NET, x1= X_DOUBLES, y1=Y_NET, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=-X_SINGLES, y0=Y_NEAR_SERVICE, x1=X_SINGLES, y1=Y_NEAR_SERVICE, line=dict(color="black", width=1.5))
        fig.add_shape(type="line", x0=-X_SINGLES, y0=Y_FAR_SERVICE, x1=X_SINGLES, y1=Y_FAR_SERVICE, line=dict(color="black", width=1.5))
        fig.add_shape(type="line", x0=0, y0=Y_NEAR_SERVICE, x1=0, y1=Y_NET, line=dict(color="black", width=1.5))
        fig.add_shape(type="line", x0=0, y0=Y_NET, x1=0, y1=Y_FAR_SERVICE, line=dict(color="black", width=1.5))
        fig.update_yaxes(range=[-4, Y_FAR + 4])

    fig.update_xaxes(range=[-X_DOUBLES - 0.6, X_DOUBLES + 0.6])
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

def build_court_figure(filtered_df, view="both", scope="half", title="Court Explorer"):
    fig = go.Figure()
    add_court_shapes(fig, scope=scope)
    fig.update_layout(plot_bgcolor="lightgreen", paper_bgcolor="white")

    if view in ["hit", "both"]:
        h = filtered_df[["hit_x_std", "hit_y_std", "_point_id", "Shot", "Player", "Result"]].copy()
        h["hit_x_std"] = pd.to_numeric(h["hit_x_std"], errors="coerce")
        h["hit_y_std"] = pd.to_numeric(h["hit_y_std"], errors="coerce")
        h = h.dropna(subset=["hit_x_std", "hit_y_std"])
        
        # Separate made (In) vs missed shots
        h_made = h[h["Result"] == "In"]
        h_missed = h[h["Result"] != "In"]
        
        # Add made shots (green)
        if not h_made.empty:
            fig.add_trace(go.Scatter(
                x=h_made["hit_x_std"], y=h_made["hit_y_std"],
                mode="markers",
                name="Hit (Made)",
                marker=dict(size=7, opacity=0.6, color="#2A9D8F"),  # green for made shots
                customdata=np.stack([h_made["_point_id"], h_made["Shot"], h_made["Player"], h_made["Result"]], axis=1),
                hovertemplate="Player=%{customdata[2]}<br>Point=%{customdata[0]}<br>Shot=%{customdata[1]}<br>Result=%{customdata[3]}<br>x=%{x:.2f}, y=%{y:.2f}<extra></extra>"
            ))
        
        # Add missed shots (red)
        if not h_missed.empty:
            fig.add_trace(go.Scatter(
                x=h_missed["hit_x_std"], y=h_missed["hit_y_std"],
                mode="markers",
                name="Hit (Missed)",
                marker=dict(size=7, opacity=0.6, color="#E76F51"),  # red/orange for missed shots
                customdata=np.stack([h_missed["_point_id"], h_missed["Shot"], h_missed["Player"], h_missed["Result"]], axis=1),
                hovertemplate="Player=%{customdata[2]}<br>Point=%{customdata[0]}<br>Shot=%{customdata[1]}<br>Result=%{customdata[3]}<br>x=%{x:.2f}, y=%{y:.2f}<extra></extra>"
            ))

    if view in ["bounce", "both"]:
        b = filtered_df[["bounce_x_std", "bounce_y_std", "_point_id", "Shot", "Player", "Result"]].copy()
        b["bounce_x_std"] = pd.to_numeric(b["bounce_x_std"], errors="coerce")
        b["bounce_y_std"] = pd.to_numeric(b["bounce_y_std"], errors="coerce")
        b = b.dropna(subset=["bounce_x_std", "bounce_y_std"])
        if scope == "half":
            b["bounce_y_std"] = b["bounce_y_std"].clip(upper=Y_NET)
        
        # Separate made (In) vs missed shots
        b_made = b[b["Result"] == "In"]
        b_missed = b[b["Result"] != "In"]
        
        # Add made shots (purple)
        if not b_made.empty:
            fig.add_trace(go.Scatter(
                x=b_made["bounce_x_std"], y=b_made["bounce_y_std"],
                mode="markers",
                name="Bounce (Made)",
                marker=dict(size=7, opacity=0.6, color="#6A4C93"),  # purple for made bounces
                customdata=np.stack([b_made["_point_id"], b_made["Shot"], b_made["Player"], b_made["Result"]], axis=1),
                hovertemplate="Player=%{customdata[2]}<br>Point=%{customdata[0]}<br>Shot=%{customdata[1]}<br>Result=%{customdata[3]}<br>x=%{x:.2f}, y=%{y:.2f}<extra></extra>"
            ))
        
        # Add missed shots (orange)
        if not b_missed.empty:
            fig.add_trace(go.Scatter(
                x=b_missed["bounce_x_std"], y=b_missed["bounce_y_std"],
                mode="markers",
                name="Bounce (Missed)",
                marker=dict(size=7, opacity=0.6, color="#F4A261"),  # orange for missed bounces
                customdata=np.stack([b_missed["_point_id"], b_missed["Shot"], b_missed["Player"], b_missed["Result"]], axis=1),
                hovertemplate="Player=%{customdata[2]}<br>Point=%{customdata[0]}<br>Shot=%{customdata[1]}<br>Result=%{customdata[3]}<br>x=%{x:.2f}, y=%{y:.2f}<extra></extra>"
            ))

    fig.update_layout(
        title=title,
        height=820 if scope == "half" else 900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig

# =========================================================
# Comparison plotting helpers (Spin / Direction)
# =========================================================
def grouped_pct_df(base_df, compare_players, category_col):
    rows = []
    for p in compare_players:
        d = base_df[base_df["Player"] == p].copy()
        d = d.dropna(subset=[category_col])
        total = len(d)
        if total == 0:
            continue
        counts = d[category_col].value_counts()
        pct = (counts / total * 100).round(1)
        for cat, cnt in counts.items():
            rows.append({
                "Player": p,
                category_col: str(cat),
                "Count": int(cnt),
                "Percentage": float(pct.loc[cat])
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = out.groupby(category_col)["Count"].sum().sort_values(ascending=False).index.tolist()
    out[category_col] = pd.Categorical(out[category_col], categories=order, ordered=True)
    return out.sort_values(category_col)

def grouped_bar(fig_df, xcol, title):
    if fig_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{title}<br><sup>No data</sup>", height=420)
        return fig
    fig_df = fig_df.copy()
    fig_df["label"] = fig_df.apply(lambda r: f'{r["Count"]}<br>({r["Percentage"]}%)', axis=1)

    fig = px.bar(
        fig_df,
        x=xcol,
        y="Percentage",
        color="Player",
        barmode="group",
        text="label",
        title=title
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        yaxis_title="Percentage (%)",
        xaxis_title=xcol,
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=480,
        margin=dict(t=60, b=40, l=20, r=20)
    )
    return fig

# =========================================================
# Dash app
# =========================================================
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Tennis Analytics Dashboard - SwingVision"

app.layout = html.Div([
    html.Div([
        html.H1("🎾 Tennis Analytics Dashboard - SwingVision",
                style={"textAlign": "center", "color": colors["primary"], "margin": "8px 0"}),
        html.Div("Notebook-equivalent stats + interactive court explorer (Hit/Bounce) with filters + comparison plots",
                 style={"textAlign": "center", "color": colors["text"], "opacity": 0.85}),
    ], style={"background": "white", "padding": "14px", "borderRadius": "12px",
              "boxShadow": "0 2px 6px rgba(0,0,0,0.08)", "margin": "12px"}),

    html.Div([
        html.Label("Select Player", style={"fontWeight": "700", "marginRight": "10px"}),
        dcc.Dropdown(
            id="player-selector",
            options=[{"label": p, "value": p} for p in players],
            value=players[0],
            clearable=False,
            style={"width": "320px"}
        )
    ], style={"background": "white", "padding": "12px", "borderRadius": "12px",
              "boxShadow": "0 2px 6px rgba(0,0,0,0.08)", "margin": "0 12px 12px 12px"}),

    dcc.Tabs(id="tabs", value="serve", children=[
        dcc.Tab(label="Serve", value="serve"),
        dcc.Tab(label="Return", value="return"),
        dcc.Tab(label="Winners / Net / UE", value="winners"),
        dcc.Tab(label="Speed (Comparison)", value="speed"),
        dcc.Tab(label="Rally", value="rally"),
        dcc.Tab(label="Spin & Direction (Comparison)", value="stroke"),
        dcc.Tab(label="Serve+1 / Return+1 (Donut + Table)", value="plus1"),
        dcc.Tab(label="Court Explorer (Filters)", value="court"),
        dcc.Tab(label="Movement + Regions", value="move"),
    ], style={"margin": "0 12px"}),

    html.Div(id="tab-content", style={"margin": "12px"}),
], style={"background": colors["bg"], "minHeight": "100vh", "paddingBottom": "24px"})


# =========================================================
# Tab renderers
# =========================================================
def render_serve_tab(player):
    row = summary_serve.loc[summary_serve["Player"] == player].iloc[0]
    cards = html.Div([
        kpi_card("Aces", str(int(row["Aces"]))),
        kpi_card("Double Faults", str(int(row["Double_Faults"]))),
        kpi_card("1st Serve %", f'{row["First_Serve_%"]:.1f}%'),
        kpi_card("1st Serve Won %", f'{row["First_Serve_Won_%"]:.1f}%'),
        kpi_card("2nd Serve %", f'{row["Second_Serve_%"]:.1f}%'),
        kpi_card("2nd Serve Won %", f'{row["Second_Serve_Won_%"]:.1f}%'),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "10px", "marginBottom": "12px"})

    return html.Div([
        html.H2("Serve Statistics", style={"color": colors["primary"]}),
        cards,
        html.Div([html.H4("Serve Summary (All Players)"), render_table(summary_serve)],
                 style={"background": "white", "padding": "12px", "borderRadius": "12px",
                        "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"})
    ])

def render_return_tab(player):
    row = return_summary.loc[return_summary["Player"] == player].iloc[0]
    cards = html.Div([
        kpi_card("Return Winners", str(int(row["Return_Winners"]))),
        kpi_card("Return Forced Errors", str(int(row["Return_Forced_Errors"]))),
        kpi_card("2nd Return Missed", str(int(row["Second_Return_Missed"]))),
        kpi_card("1st Return Won %", f'{row["First_Return_Won_%"]:.1f}%'),
        kpi_card("2nd Return Won %", f'{row["Second_Return_Won_%"]:.1f}%'),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "10px", "marginBottom": "12px"})

    return html.Div([
        html.H2("Return Statistics", style={"color": colors["primary"]}),
        cards,
        html.Div([html.H4("Return Summary (All Players)"), render_table(return_summary)],
                 style={"background": "white", "padding": "12px", "borderRadius": "12px",
                        "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"})
    ])

def render_winners_tab(player):
    row = winners_summary.loc[winners_summary["Player"] == player].iloc[0]
    cards = html.Div([
        kpi_card("Winners", str(int(row["Winners"]))),
        kpi_card("Unforced Errors", str(int(row["Unforced_Errors"]))),
        kpi_card("Net Point Won %", row["Net_Point_Won_%"]),
        kpi_card("All Points Won", row["All_Points_Won"]),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "10px", "marginBottom": "12px"})

    return html.Div([
        html.H2("Winners / Net / All Points / Unforced Errors", style={"color": colors["primary"]}),
        cards,
        html.Div([html.H4("Winners Summary (All Players)"), render_table(winners_summary)],
                 style={"background": "white", "padding": "12px", "borderRadius": "12px",
                        "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"})
    ])

def render_speed_tab(player):
    opponent = pick_opponent(player)
    compare_players = [player, opponent] if opponent != player else [player]

    all_strokes = []
    for p in compare_players:
        player_df = df[df["Player"] == p].copy()
        player_df["Speed (MPH)"] = pd.to_numeric(player_df["Speed (MPH)"], errors="coerce")

        serve_stats = (player_df[player_df["Stroke"] == "Serve"]
                       .groupby("Type")["Speed (MPH)"]
                       .agg(Mean="mean", Min="min", Max="max")
                       .round(1).reset_index()
                       .rename(columns={"Type": "Stroke"}))

        rally_stats = (player_df[player_df["Stroke"] != "Serve"]
                       .groupby("Stroke")["Speed (MPH)"]
                       .agg(Mean="mean", Min="min", Max="max")
                       .round(1).reset_index())

        combined = pd.concat([serve_stats, rally_stats], ignore_index=True)
        combined["Player"] = p
        all_strokes.append(combined)

    plot_df = pd.concat(all_strokes, ignore_index=True)
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Speed Distribution (No data)", height=520)
        return html.Div([html.H2("Speed (Comparison)", style={"color": colors["primary"]}), dcc.Graph(figure=fig)])

    strokes = plot_df["Stroke"].dropna().unique().tolist()

    fig = go.Figure()
    for p in compare_players:
        d = plot_df[plot_df["Player"] == p].set_index("Stroke").reindex(strokes)
        fig.add_trace(go.Bar(
            x=strokes,
            y=d["Mean"].values,
            name=p,
            offsetgroup=p,
            error_y=dict(
                type="data",
                symmetric=False,
                array=(d["Max"] - d["Mean"]).fillna(0).clip(lower=0).values,
                arrayminus=(d["Mean"] - d["Min"]).fillna(0).clip(lower=0).values,
            )
        ))

    fig.update_layout(
        title="Speed Distribution by Stroke/Type — Player vs Opponent",
        xaxis_title="Stroke/Type",
        yaxis_title="Speed (MPH)",
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=540
    )

    return html.Div([
        html.H2("Speed (Comparison)", style={"color": colors["primary"]}),
        html.Div([dcc.Graph(figure=fig)],
                 style={"background": "white", "padding": "12px", "borderRadius": "12px",
                        "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"})
    ])

def render_rally_tab(player):
    fig = px.pie(rally_dist, values="Count", names="Category", hole=0.35, title="Rally Length Distribution (All Points)")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=520)

    row = rally_mode_df.loc[rally_mode_df["Player"] == player].iloc[0]

    return html.Div([
        html.H2("Rally Analysis", style={"color": colors["primary"]}),
        html.Div([
            kpi_card("Most Common (Won)", str(int(row["Most_Won_Shots"]))),
            kpi_card("Most Common (Lost)", str(int(row["Most_Lost_Shots"]))),
            kpi_card("Total Points", str(len(points))),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "10px", "marginBottom": "12px"}),

        html.Div([dcc.Graph(figure=fig)],
                 style={"background": "white", "padding": "12px", "borderRadius": "12px",
                        "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"}),

        html.Div([html.H4("Rally Mode by Player"), render_table(rally_mode_df)],
                 style={"background": "white", "padding": "12px", "borderRadius": "12px",
                        "boxShadow": "0 2px 6px rgba(0,0,0,0.08)", "marginTop": "12px"})
    ])

def render_stroke_tab(player):
    opponent = pick_opponent(player)
    compare_players = [player, opponent] if opponent != player else [player]

    blocks = []
    for stroke_name in ["Forehand", "Backhand"]:
        base = df[df["Stroke"] == stroke_name].copy()

        spin_df = grouped_pct_df(base, compare_players, "Spin")
        dir_df = grouped_pct_df(base, compare_players, "Direction")

        fig_spin = grouped_bar(spin_df, "Spin", f"{stroke_name} Spin — Player vs Opponent")
        fig_dir = grouped_bar(dir_df, "Direction", f"{stroke_name} Direction — Player vs Opponent")

        blocks.append(html.Div([
            html.H3(stroke_name, style={"marginTop": 0}),
            dcc.Graph(figure=fig_spin),
            dcc.Graph(figure=fig_dir),
        ], style={"background": "white", "padding": "12px", "borderRadius": "12px",
                  "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"}))

    return html.Div([
        html.H2("Spin & Direction (Comparison)", style={"color": colors["primary"]}),
        html.Div(blocks, style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"})
    ])

def render_plus1_tab(player):
    opponent = pick_opponent(player)

    fig_s1_p = two_ring_donut_winloss_deucead(serve_plus_one_points, player, f"{player} — Serve Plus 1")
    fig_s1_o = two_ring_donut_winloss_deucead(serve_plus_one_points, opponent, f"{opponent} — Serve Plus 1")

    fig_r1_p = two_ring_donut_winloss_deucead(return_plus_one_points, player, f"{player} — Return Plus 1")
    fig_r1_o = two_ring_donut_winloss_deucead(return_plus_one_points, opponent, f"{opponent} — Return Plus 1")

    return html.Div([
        html.H2("Serve+1 / Return+1 (Donut + Table)", style={"color": colors["primary"]}),

        html.H3("Serve +1", style={"marginTop": "6px"}),
        html.Div([
            html.Div([dcc.Graph(figure=fig_s1_p)], style={"background":"white","padding":"10px","borderRadius":"12px",
                                                         "boxShadow":"0 2px 6px rgba(0,0,0,0.08)"}),
            html.Div([dcc.Graph(figure=fig_s1_o)], style={"background":"white","padding":"10px","borderRadius":"12px",
                                                         "boxShadow":"0 2px 6px rgba(0,0,0,0.08)"}),
        ], style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"12px"}),

        html.Div([html.H4("Serve+1 Table (All Players)"), render_table(serve_plus_one_tbl)],
                 style={"background":"white","padding":"12px","borderRadius":"12px",
                        "boxShadow":"0 2px 6px rgba(0,0,0,0.08)", "marginTop":"12px"}),

        html.H3("Return +1", style={"marginTop": "18px"}),
        html.Div([
            html.Div([dcc.Graph(figure=fig_r1_p)], style={"background":"white","padding":"10px","borderRadius":"12px",
                                                         "boxShadow":"0 2px 6px rgba(0,0,0,0.08)"}),
            html.Div([dcc.Graph(figure=fig_r1_o)], style={"background":"white","padding":"10px","borderRadius":"12px",
                                                         "boxShadow":"0 2px 6px rgba(0,0,0,0.08)"}),
        ], style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"12px"}),

        html.Div([html.H4("Return+1 Table (All Players)"), render_table(return_plus_one_tbl)],
                 style={"background":"white","padding":"12px","borderRadius":"12px",
                        "boxShadow":"0 2px 6px rgba(0,0,0,0.08)", "marginTop":"12px"}),
    ])

def render_move_tab():
    return html.Div([
        html.H2("Movement + Regions", style={"color": colors["primary"]}),
        html.Div([
            html.Div([html.H4("Movement Summary (from hit_x_std/hit_y_std)"), render_table(move_summary)],
                     style={"background": "white", "padding": "12px", "borderRadius": "12px",
                            "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"}),
            html.Div([html.H4("Region Table (made % by hit-position regions)"), render_table(region_tbl_display)],
                     style={"background": "white", "padding": "12px", "borderRadius": "12px",
                            "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"}),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"})
    ])

def render_court_tab(player):
    opponent = pick_opponent(player)
    ctx = {"player": player, "opponent": opponent}

    # Point dropdown options based on selected player
    pts = df[df["Player"] == player]["_point_id"].dropna().unique().tolist()
    pts = sorted(pts, key=lambda x: tuple(map(int, x.split("-"))))
    point_opts = [{"label": "All", "value": "__ALL__"}] + [{"label": pid, "value": pid} for pid in pts]

    # Speed slider bounds
    speed_series = pd.to_numeric(df["Speed (MPH)"], errors="coerce")
    speed_min = float(np.nanmin(speed_series)) if np.isfinite(np.nanmin(speed_series)) else 0
    speed_max = float(np.nanmax(speed_series)) if np.isfinite(np.nanmax(speed_series)) else 100

    # Initial figure
    init_df = df[df["Player"] == player].copy()
    init_fig = build_court_figure(init_df, view="both", scope="half", title=f"Court Explorer — {player} (Half court)")

    return html.Div([
        # store so court callbacks never depend on global player-selector
        dcc.Store(id="court-context", data=ctx),

        html.H2("Court Explorer (Hit + Bounce) — with Filters", style={"color": colors["primary"]}),

        html.Div([
            html.Div([
                html.H4("Filters", style={"marginTop": 0}),

                dcc.RadioItems(
                    id="court-view",
                    options=[
                        {"label": "Hit", "value": "hit"},
                        {"label": "Bounce", "value": "bounce"},
                        {"label": "Hit + Bounce", "value": "both"},
                    ],
                    value="both",
                    inline=True
                ),
                html.Div(style={"height": "8px"}),

                dcc.RadioItems(
                    id="court-scope",
                    options=[
                        {"label": "Half court (near)", "value": "half"},
                        {"label": "Full court", "value": "full"},
                    ],
                    value="half",
                    inline=True
                ),

                html.Hr(),

                dcc.Checklist(
                    id="include-opponent",
                    options=[{"label": f"Include opponent: {opponent}", "value": "yes"}],
                    value=[]
                ),

                html.Label("Type", style={"fontWeight": "700", "marginTop": "10px"}),
                dcc.Dropdown(id="f-type", options=options_from_col("Type"), value=None, multi=True),

                html.Label("Stroke", style={"fontWeight": "700", "marginTop": "10px"}),
                dcc.Dropdown(id="f-stroke", options=options_from_col("Stroke"), value=None, multi=True),

                html.Label("Spin", style={"fontWeight": "700", "marginTop": "10px"}),
                dcc.Dropdown(id="f-spin", options=options_from_col("Spin"), value=None, multi=True),

                html.Label("Direction", style={"fontWeight": "700", "marginTop": "10px"}),
                dcc.Dropdown(id="f-direction", options=options_from_col("Direction"), value=None, multi=True),

                html.Label("Result", style={"fontWeight": "700", "marginTop": "10px"}),
                dcc.Dropdown(id="f-result", options=options_from_col("Result"), value=None, multi=True),

                html.Label("Winner", style={"fontWeight": "700", "marginTop": "10px"}),
                dcc.Dropdown(
                    id="f-winner",
                    options=[{"label": "All", "value": "__ALL__"}] + [{"label": p, "value": p} for p in sorted(df["Winner"].dropna().unique())],
                    value="__ALL__",
                    clearable=False
                ),

                html.Label("Hit Zone", style={"fontWeight": "700", "marginTop": "10px"}),
                dcc.Dropdown(id="f-hitzone", options=options_from_col("Hit Zone"), value=None, multi=True),

                html.Label("Bounce Zone", style={"fontWeight": "700", "marginTop": "10px"}),
                dcc.Dropdown(id="f-bouncezone", options=options_from_col("Bounce Zone"), value=None, multi=True),

                html.Label("Hit Region", style={"fontWeight": "700", "marginTop": "10px"}),
                dcc.Dropdown(id="f-hitregion", options=options_from_col("hit_region"), value=None, multi=True),

                html.Label("Rally length (Shot_Count per point)", style={"fontWeight": "700", "marginTop": "10px"}),
                dcc.RangeSlider(
                    id="f-rallylen",
                    min=int(points["Shot_Count"].min()),
                    max=int(points["Shot_Count"].max()),
                    step=1,
                    value=[int(points["Shot_Count"].min()), int(points["Shot_Count"].max())],
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": False}
                ),

                html.Label("Speed (MPH)", style={"fontWeight": "700", "marginTop": "10px"}),
                dcc.RangeSlider(
                    id="f-speed",
                    min=float(np.floor(speed_min)),
                    max=float(np.ceil(speed_max)),
                    step=1,
                    value=[float(np.floor(speed_min)), float(np.ceil(speed_max))],
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": False}
                ),

                html.Label("Point (optional) — click any dot to auto-select", style={"fontWeight": "700", "marginTop": "10px"}),
                dcc.Dropdown(id="f-point", options=point_opts, value="__ALL__", searchable=True),

                html.Div(id="court-count", style={"marginTop": "10px", "opacity": 0.85}),
            ], style={"background": "white", "padding": "12px", "borderRadius": "12px",
                      "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"}),

            html.Div([
                dcc.Graph(id="court-graph", figure=init_fig),
                html.Div([
                    html.H4("Selected Point — Shot-by-shot"),
                    dash_table.DataTable(
                        id="point-table",
                        page_size=20,
                        style_table={"overflowX": "auto"},
                        style_cell={"fontFamily": "Arial", "fontSize": 12, "padding": "6px", "whiteSpace": "normal", "height": "auto"},
                        style_header={"fontWeight": "700", "backgroundColor": "#f1f3f5"},
                    )
                ], style={"background": "white", "padding": "12px", "borderRadius": "12px",
                          "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"})
            ])
        ], style={"display": "grid", "gridTemplateColumns": "420px 1fr", "gap": "12px"})
    ])

# =========================================================
# Main tab content callback
# =========================================================
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("player-selector", "value")
)
def render_tab(tab, player):
    if tab == "serve":
        return render_serve_tab(player)
    if tab == "return":
        return render_return_tab(player)
    if tab == "winners":
        return render_winners_tab(player)
    if tab == "speed":
        return render_speed_tab(player)
    if tab == "rally":
        return render_rally_tab(player)
    if tab == "stroke":
        return render_stroke_tab(player)
    if tab == "plus1":
        return render_plus1_tab(player)
    if tab == "court":
        return render_court_tab(player)
    if tab == "move":
        return render_move_tab()
    return html.Div()

# =========================================================
# Court callbacks (ONLY depend on court-tab components)
# =========================================================
@app.callback(
    Output("court-graph", "figure"),
    Output("court-count", "children"),
    Input("court-context", "data"),
    Input("court-view", "value"),
    Input("court-scope", "value"),
    Input("include-opponent", "value"),
    Input("f-type", "value"),
    Input("f-stroke", "value"),
    Input("f-spin", "value"),
    Input("f-direction", "value"),
    Input("f-result", "value"),
    Input("f-winner", "value"),
    Input("f-hitzone", "value"),
    Input("f-bouncezone", "value"),
    Input("f-hitregion", "value"),
    Input("f-rallylen", "value"),
    Input("f-speed", "value"),
    Input("f-point", "value"),
)
def update_court(ctx, view, scope, include_opp, f_type, f_stroke, f_spin, f_dir, f_res, f_winner,
                 f_hz, f_bz, f_hitregion, f_rallylen, f_speed, f_point):

    player = ctx["player"]
    opponent = ctx["opponent"]

    d = df.copy()

    # Player filter
    if include_opp and "yes" in include_opp:
        d = d[d["Player"].isin([player, opponent])]
    else:
        d = d[d["Player"] == player]

    # Filters
    if f_type:
        d = d[d["Type"].isin(f_type)]
    if f_stroke:
        d = d[d["Stroke"].isin(f_stroke)]
    if f_spin:
        d = d[d["Spin"].isin(f_spin)]
    if f_dir:
        d = d[d["Direction"].isin(f_dir)]
    if f_res:
        d = d[d["Result"].isin(f_res)]
    if f_winner and f_winner != "__ALL__":
        d = d[d["Winner"] == f_winner]
    if f_hz:
        d = d[d["Hit Zone"].isin(f_hz)]
    if f_bz:
        d = d[d["Bounce Zone"].isin(f_bz)]
    if f_hitregion:
        d = d[d["hit_region"].isin(f_hitregion)]

    lo, hi = f_rallylen
    d = d[(d["Shot_Count"] >= lo) & (d["Shot_Count"] <= hi)]

    sp = pd.to_numeric(d["Speed (MPH)"], errors="coerce")
    d = d[(sp >= f_speed[0]) & (sp <= f_speed[1])]

    if f_point and f_point != "__ALL__":
        d = d[d["_point_id"] == f_point]

    title = f"Court Explorer — {player} ({'Half' if scope == 'half' else 'Full'} court)"
    fig = build_court_figure(d, view=view, scope=scope, title=title)

    count_txt = f"Filtered rows: {len(d)} | Points: {d['_point_id'].nunique()}"
    return fig, count_txt

@app.callback(
    Output("f-point", "value"),
    Input("court-graph", "clickData"),
    State("f-point", "value"),
    prevent_initial_call=True
)
def click_select_point(clickData, current_value):
    if not clickData or "points" not in clickData or len(clickData["points"]) == 0:
        return current_value
    cd = clickData["points"][0].get("customdata")
    if cd and len(cd) >= 1:
        return cd[0]
    return current_value

@app.callback(
    Output("point-table", "data"),
    Output("point-table", "columns"),
    Input("f-point", "value"),
)
def update_point_table(point_id):
    if not point_id or point_id == "__ALL__":
        return [], []

    d = df[df["_point_id"] == point_id].copy().sort_values(["Shot"]).reset_index(drop=True)
    keep = [
        "Player", "Shot", "Type", "Stroke", "Spin", "Speed (MPH)", "Result", "Winner",
        "Direction", "Hit Zone", "Bounce Zone",
        "hit_x_std", "hit_y_std", "bounce_x_std", "bounce_y_std", "Shot_Count"
    ]
    keep = [c for c in keep if c in d.columns]
    d = d[keep]
    return d.to_dict("records"), [{"name": c, "id": c} for c in d.columns]

# =========================================================
# Run (Dash v3+ uses app.run)
# =========================================================
if __name__ == "__main__":
    app.run(debug=True, port=8050)