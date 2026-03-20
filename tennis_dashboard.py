import re
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# =========================================================
# Paths / loading
# =========================================================
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


def is_match_file(path: Path) -> bool:
    return re.match(r"^.+_\d+\.csv$", path.name) is not None


def discover_csv_files():
    print("\n=== CSV discovery ===")
    print(f"Script directory : {HERE}")
    print(f"Data directory   : {DATA_DIR.resolve()}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    files = []
    for f in sorted(DATA_DIR.glob("*.csv")):
        if not is_match_file(f):
            print(f"  skipping non-match csv: {f.name}")
            continue
        files.append(f)
        print(f"  found match file: {f.resolve()}")

    print(f"Total valid match csv files found: {len(files)}")
    print("=====================\n")
    return files

REQUIRED_COLS = {
    "Player", "Shot", "Type", "Stroke", "Spin", "Speed (MPH)", "Winner",
    "Point", "Result", "Game", "Set",
    "bounce_x_std", "bounce_y_std", "hit_x_std", "hit_y_std",
    "Direction", "Hit Zone", "Bounce Zone"
}

# Court dimensions (meters)
Y_NEAR = 0.0
Y_NET = 11.885
Y_FAR = 23.77
X_SINGLES = 4.115
X_DOUBLES = 5.485
SERVICE_TO_NET = 6.40
Y_NEAR_SERVICE = Y_NET - SERVICE_TO_NET
Y_FAR_SERVICE = Y_NET + SERVICE_TO_NET

colors = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "success": "#2A9D8F",
    "warning": "#E76F51",
    "bg": "#F8F9FA",
    "text": "#212529",
}


def parse_file_info(path: Path):
    stem = path.stem
    m = re.match(r"(.+?)_(\d+)$", stem)
    if m:
        return m.group(1), m.group(2)
    return stem, "1"


def normalize_text(x):
    return str(x).strip() if pd.notna(x) else ""


def normalize_side(zone_value):
    s = str(zone_value).strip().lower()
    if s in {"", "nan", "none", "---"}:
        return "Unknown"
    if "ad" in s:
        return "Ad"
    if "deuce" in s:
        return "Deuce"
    return "Unknown"


def add_hit_region(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()
    X = pd.to_numeric(d.get("hit_x_std"), errors="coerce")
    Y = pd.to_numeric(d.get("hit_y_std"), errors="coerce")

    d["hit_region"] = "unknown"
    d.loc[(X.notna() & Y.notna() & (Y >= Y_NEAR_SERVICE) & (Y <= Y_NET) & (X.abs() <= X_SINGLES)), "hit_region"] = "inside_service_box"
    d.loc[(X.notna() & Y.notna() & (Y >= Y_NEAR) & (Y < Y_NEAR_SERVICE) & (X.abs() <= X_SINGLES)), "hit_region"] = "inside_baseline"
    d.loc[(X.notna() & Y.notna() & (Y < Y_NEAR)), "hit_region"] = "behind_baseline"
    d.loc[(X.notna() & Y.notna() & (X > X_SINGLES)), "hit_region"] = "outside_right_singles"
    d.loc[(X.notna() & Y.notna() & (X < -X_SINGLES)), "hit_region"] = "outside_left_singles"
    return d


def load_all_match_files() -> pd.DataFrame:
    unique_files = discover_csv_files()
    if not unique_files:
        raise FileNotFoundError(
            f"No valid match CSV files found in {DATA_DIR}. Expected names like Adi_1.csv."
        )

    dfs = []
    for f in unique_files:
        temp = pd.read_csv(f)
        missing = REQUIRED_COLS - set(temp.columns)
        if missing:
            print(f"Skipping invalid schema: {f.name}")
            print(f"  missing columns: {sorted(missing)}")
            continue

        file_player, match_label = parse_file_info(f)
        print(f"Loading {f.name} | parsed player={file_player} | match={match_label} | rows={len(temp)}")
        temp = temp.copy()
        temp["source_file"] = f.name
        temp["file_player"] = file_player
        temp["match_label"] = match_label
        temp["match_key"] = f.stem
        dfs.append(temp)

    if not dfs:
        raise ValueError("No files in data/ matched the required dashboard schema.")

    df = pd.concat(dfs, ignore_index=True)
    for col in ["Set", "Game", "Point", "Shot", "Speed (MPH)", "hit_x_std", "hit_y_std", "bounce_x_std", "bounce_y_std"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["court_side"] = df["Hit Zone"].apply(normalize_side)
    df = add_hit_region(df)
    df = df.sort_values(["source_file", "Set", "Game", "Point", "Shot"]).reset_index(drop=True)
    df["_point_id"] = (
        df["source_file"].astype(str) + "|" +
        df["Set"].fillna(-1).astype(int).astype(str) + "-" +
        df["Game"].fillna(-1).astype(int).astype(str) + "-" +
        df["Point"].fillna(-1).astype(int).astype(str)
    )
    return df


ALL_DF = load_all_match_files()
PLAYER_OPTIONS = sorted(ALL_DF["file_player"].dropna().astype(str).unique().tolist())
print("Players detected from filenames:", PLAYER_OPTIONS)
DEFAULT_PLAYER = PLAYER_OPTIONS[0]


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
            "textAlign": "center",
        },
        style_header={"fontWeight": "700", "backgroundColor": "#f1f3f5"},
    )


def get_match_options(player):
    d = ALL_DF[ALL_DF["file_player"] == player]
    files = sorted(d["source_file"].dropna().astype(str).unique().tolist())
    return [{"label": "All files", "value": "__ALL__"}] + [{"label": f, "value": f} for f in files]


def filter_base_df(player, selected_file):
    d = ALL_DF[ALL_DF["file_player"] == player].copy()
    if selected_file and selected_file != "__ALL__":
        d = d[d["source_file"] == selected_file].copy()
    return d.sort_values(["source_file", "Set", "Game", "Point", "Shot"]).reset_index(drop=True)


def pick_opponent(player_label, d):
    explicit = sorted(d["Player"].dropna().astype(str).unique().tolist())
    if "Opponent" in explicit:
        return "Opponent"
    others = [p for p in explicit if p != player_label]
    return others[0] if others else player_label


def resolve_selected_player_name(player_label, d):
    explicit = sorted(d["Player"].dropna().astype(str).unique().tolist())
    if player_label in explicit:
        return player_label
    exact_ci = [p for p in explicit if p.lower() == str(player_label).lower()]
    if exact_ci:
        return exact_ci[0]
    starts = [p for p in explicit if p.lower().startswith(str(player_label).lower())]
    if starts:
        return starts[0]
    non_opp = [p for p in explicit if p != "Opponent"]
    return non_opp[0] if non_opp else (explicit[0] if explicit else player_label)


def prepare_view_context(player_label, selected_file):
    d = filter_base_df(player_label, selected_file)
    if d.empty:
        return {"df": d, "points": pd.DataFrame(), "last_shots": pd.DataFrame(), "players": []}

    shots_per_point = d.groupby("_point_id").size().rename("Shot_Count").reset_index()
    point_winner = d.groupby("_point_id")["Winner"].first().reset_index().rename(columns={"Winner": "Point_Winner"})
    points = shots_per_point.merge(point_winner, on="_point_id", how="left")

    point_meta = d.groupby("_point_id").agg(
        source_file=("source_file", "first"),
        Set=("Set", "first"),
        Game=("Game", "first"),
        Point=("Point", "first"),
    ).reset_index()
    points = points.merge(point_meta, on="_point_id", how="left")
    d = d.merge(shots_per_point, on="_point_id", how="left")
    last_shots = d.groupby("_point_id").tail(1).copy()
    return {
        "df": d,
        "points": points,
        "last_shots": last_shots,
        "players": sorted(d["Player"].dropna().astype(str).unique().tolist()),
        "selected_player": resolve_selected_player_name(player_label, d),
        "selected_file": selected_file,
        "opponent": pick_opponent(player_label, d),
    }


def pct_str(made, total):
    if total == 0:
        return "0.0%"
    return f"{100 * made / total:.1f}%"


def made_of_total_str(made, total):
    if total == 0:
        return "0/0 (0.0%)"
    return f"{int(made)}/{int(total)} ({100 * made / total:.1f}%)"


def serve_summary_table(ctx):
    d = ctx["df"].copy()
    if d.empty:
        return pd.DataFrame(columns=["Player", "Aces", "Double_Faults", "First_Serve_%", "First_Serve_Won_%", "Second_Serve_%", "Second_Serve_Won_%"])

    one_row_ids = set(d.groupby("_point_id").size().pipe(lambda s: s[s == 1]).index.tolist())
    rows = []
    for p in sorted(d["Player"].dropna().astype(str).unique()):
        dp = d[d["Player"] == p].copy()
        first = dp[dp["Type"] == "first_serve"]
        second = dp[dp["Type"] == "second_serve"]
        first_in = first[first["Result"] == "In"]
        second_in = second[second["Result"] == "In"]
        aces = dp[(dp["_point_id"].isin(one_row_ids)) & (dp["Type"] == "first_serve") & (dp["Result"] == "In")].shape[0]
        double_faults = dp[(dp["Type"] == "second_serve") & (dp["Result"] != "In")].shape[0]
        rows.append({
            "Player": p,
            "Aces": int(aces),
            "Double_Faults": int(double_faults),
            "First_Serve_%": round(100 * first["Result"].eq("In").mean(), 1) if len(first) else 0.0,
            "First_Serve_Won_%": round(100 * first_in["Winner"].eq(p).mean(), 1) if len(first_in) else 0.0,
            "Second_Serve_%": round(100 * second["Result"].eq("In").mean(), 1) if len(second) else 0.0,
            "Second_Serve_Won_%": round(100 * second_in["Winner"].eq(p).mean(), 1) if len(second_in) else 0.0,
        })
    return pd.DataFrame(rows)


def return_summary_table(ctx):
    d = ctx["df"].copy()
    if d.empty:
        return pd.DataFrame(columns=["Player", "Return_Winners", "Return_Forced_Errors", "Second_Return_Missed", "First_Return_Won_%", "Second_Return_Won_%"])

    rows = []
    for p in sorted(d["Player"].dropna().astype(str).unique()):
        dp = d[d["Player"] == p].copy()
        first_ret = dp[dp["Type"] == "first_return"]
        second_ret = dp[dp["Type"] == "second_return"]
        rows.append({
            "Player": p,
            "Return_Winners": int(((dp["Type"].isin(["first_return", "second_return"])) & (dp["Winner"] == p)).sum()),
            "Return_Forced_Errors": int(((dp["Type"].isin(["first_return", "second_return"])) & (dp["Result"].astype(str).str.lower() == "forced error")).sum()),
            "Second_Return_Missed": int(((dp["Type"] == "second_return") & (dp["Result"] != "In")).sum()),
            "First_Return_Won_%": round(100 * first_ret["Winner"].eq(p).mean(), 1) if len(first_ret) else 0.0,
            "Second_Return_Won_%": round(100 * second_ret["Winner"].eq(p).mean(), 1) if len(second_ret) else 0.0,
        })
    return pd.DataFrame(rows)


def winners_summary_table(ctx):
    d = ctx["df"].copy()
    points = ctx["points"].copy()
    if d.empty:
        return pd.DataFrame(columns=["Player", "Winners", "Unforced_Errors", "Net_Point_Won_%", "All_Points_Won"])

    rows = []
    for p in sorted(d["Player"].dropna().astype(str).unique()):
        dp = d[d["Player"] == p].copy()
        net_points = dp[dp["Stroke"].isin(["Volley", "Overhead"])]["_point_id"].dropna().unique().tolist()
        net_total = len(net_points)
        net_won = points[(points["_point_id"].isin(net_points)) & (points["Point_Winner"] == p)].shape[0]
        rows.append({
            "Player": p,
            "Winners": int((dp["Winner"] == p).sum()),
            "Unforced_Errors": int((dp["Result"].astype(str).str.lower() == "unforced error").sum()),
            "Net_Point_Won_%": pct_str(net_won, net_total),
            "All_Points_Won": made_of_total_str((points["Point_Winner"] == p).sum(), len(points)),
        })
    return pd.DataFrame(rows)


def rally_tables(ctx):
    points = ctx["points"].copy()
    if points.empty:
        return pd.DataFrame(columns=["Category", "Count"]), pd.DataFrame(columns=["Player", "Most_Won_Shots", "Most_Lost_Shots"]), pd.DataFrame(columns=["Category", "Result", "Count"])

    def rally_cat(n):
        if n <= 2:
            return "0–2"
        if n <= 4:
            return "3–4"
        if n <= 6:
            return "5–6"
        if n <= 8:
            return "7–8"
        return "9+"

    points["Category"] = points["Shot_Count"].apply(rally_cat)
    overall = points["Category"].value_counts().rename_axis("Category").reset_index(name="Count")
    order = ["0–2", "3–4", "5–6", "7–8", "9+"]
    overall["Category"] = pd.Categorical(overall["Category"], categories=order, ordered=True)
    overall = overall.sort_values("Category")

    rows = []
    bar_rows = []
    for p in sorted(ctx["df"]["Player"].dropna().astype(str).unique()):
        won = points[points["Point_Winner"] == p]["Shot_Count"]
        lost = points[points["Point_Winner"] != p]["Shot_Count"]
        rows.append({
            "Player": p,
            "Most_Won_Shots": int(won.mode().iloc[0]) if len(won) else 0,
            "Most_Lost_Shots": int(lost.mode().iloc[0]) if len(lost) else 0,
        })
        for result_name, subset in [("Won", points[points["Point_Winner"] == p]), ("Lost", points[points["Point_Winner"] != p])]:
            tmp = subset["Category"].value_counts().rename_axis("Category").reset_index(name="Count")
            tmp["Result"] = result_name
            tmp["Player"] = p
            bar_rows.append(tmp)
    by_player = pd.concat(bar_rows, ignore_index=True) if bar_rows else pd.DataFrame(columns=["Category", "Count", "Result", "Player"])
    if not by_player.empty:
        by_player["Category"] = pd.Categorical(by_player["Category"], categories=order, ordered=True)
        by_player = by_player.sort_values(["Player", "Category", "Result"])
    mode_df = pd.DataFrame(rows)
    return overall, mode_df, by_player


def speed_plot_df(ctx):
    d = ctx["df"].copy()
    player = ctx["selected_player"]
    opp = ctx["opponent"]
    compare = [player] + ([opp] if opp != player else [])
    out = []
    for p in compare:
        dp = d[d["Player"] == p].copy()
        dp["Speed (MPH)"] = pd.to_numeric(dp["Speed (MPH)"], errors="coerce")
        if dp.empty:
            continue
        serve_stats = (dp[dp["Stroke"] == "Serve"].groupby("Type")["Speed (MPH)"].agg(Mean="mean", Min="min", Max="max").reset_index().rename(columns={"Type": "StrokeType"}))
        rally_stats = (dp[dp["Stroke"] != "Serve"].groupby("Stroke")["Speed (MPH)"].agg(Mean="mean", Min="min", Max="max").reset_index().rename(columns={"Stroke": "StrokeType"}))
        comb = pd.concat([serve_stats, rally_stats], ignore_index=True)
        comb["Player"] = p
        out.append(comb)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["StrokeType", "Mean", "Min", "Max", "Player"])


def plus_one_table(ctx, role="serve"):
    d = ctx["df"].copy()
    if d.empty:
        return pd.DataFrame(columns=["Player", "Side", "Won", "Total", "Won_%"])
    if role == "serve":
        anchor_types = ["first_serve", "second_serve"]
        plus_type = "serve_plus_one"
    else:
        anchor_types = ["first_return", "second_return"]
        plus_type = "return_plus_one"

    rows = []
    for p in sorted(d["Player"].dropna().astype(str).unique()):
        dp = d[d["Player"] == p].copy()
        valid_points = dp.groupby("_point_id")["Type"].apply(lambda s: any(t in set(s) for t in anchor_types) and plus_type in set(s))
        valid_ids = valid_points[valid_points].index.tolist()
        sub = dp[(dp["_point_id"].isin(valid_ids)) & (dp["Type"] == plus_type)].copy()
        if sub.empty:
            continue
        sub["Side"] = sub["Hit Zone"].apply(normalize_side)
        for side in ["Ad", "Deuce"]:
            ss = sub[sub["Side"] == side]
            total = len(ss)
            won = (ss["Winner"] == p).sum()
            rows.append({"Player": p, "Side": side, "Won": int(won), "Total": int(total), "Won_%": round(100 * won / total, 1) if total else 0.0})
    return pd.DataFrame(rows)


def move_tables(ctx):
    d = ctx["df"].copy()
    points = ctx["points"].copy()
    if d.empty:
        empty1 = pd.DataFrame(columns=["Player", "Mean_Hit_X", "Mean_Hit_Y", "Mean_Bounce_X", "Mean_Bounce_Y"])
        empty2 = pd.DataFrame(columns=["Player", "inside_service_box", "inside_baseline", "behind_baseline", "outside_right_singles", "outside_left_singles"])
        return empty1, empty2

    move_summary = d.groupby("Player").agg(
        Mean_Hit_X=("hit_x_std", "mean"),
        Mean_Hit_Y=("hit_y_std", "mean"),
        Mean_Bounce_X=("bounce_x_std", "mean"),
        Mean_Bounce_Y=("bounce_y_std", "mean"),
    ).round(2).reset_index()

    last = d.groupby("_point_id").tail(1).copy()
    last = last.merge(points[["_point_id", "Point_Winner"]], on="_point_id", how="left", suffixes=("", "_pt"))
    rows = []
    for p in sorted(last["Player"].dropna().astype(str).unique()):
        lp = last[last["Player"] == p].copy()
        row = {"Player": p}
        for region in ["inside_service_box", "inside_baseline", "behind_baseline", "outside_right_singles", "outside_left_singles"]:
            reg = lp[lp["hit_region"] == region]
            total = len(reg)
            made = (reg["Point_Winner"] == p).sum()
            row[region] = made_of_total_str(made, total)
        rows.append(row)
    region_tbl = pd.DataFrame(rows)
    return move_summary, region_tbl


def build_basic_court(scope="half", title="Court"):
    fig = go.Figure()
    # Outer lines
    if scope == "half":
        y0, y1 = Y_NEAR, Y_NET
    else:
        y0, y1 = Y_NEAR, Y_FAR
    line_kw = dict(line=dict(color="black", width=2))
    fig.add_shape(type="rect", x0=-X_DOUBLES, x1=X_DOUBLES, y0=y0, y1=y1, **line_kw)
    fig.add_shape(type="line", x0=-X_DOUBLES, x1=X_DOUBLES, y0=Y_NET, y1=Y_NET, **line_kw)
    fig.add_shape(type="line", x0=-X_SINGLES, x1=-X_SINGLES, y0=y0, y1=y1, **line_kw)
    fig.add_shape(type="line", x0=X_SINGLES, x1=X_SINGLES, y0=y0, y1=y1, **line_kw)
    if scope == "half":
        fig.add_shape(type="line", x0=-X_SINGLES, x1=X_SINGLES, y0=Y_NEAR_SERVICE, y1=Y_NEAR_SERVICE, **line_kw)
        fig.add_shape(type="line", x0=0, x1=0, y0=Y_NEAR_SERVICE, y1=Y_NET, **line_kw)
    else:
        fig.add_shape(type="line", x0=-X_SINGLES, x1=X_SINGLES, y0=Y_NEAR_SERVICE, y1=Y_NEAR_SERVICE, **line_kw)
        fig.add_shape(type="line", x0=-X_SINGLES, x1=X_SINGLES, y0=Y_FAR_SERVICE, y1=Y_FAR_SERVICE, **line_kw)
        fig.add_shape(type="line", x0=0, x1=0, y0=Y_NEAR_SERVICE, y1=Y_FAR_SERVICE, **line_kw)
    fig.update_xaxes(range=[-6, 6], visible=False)
    fig.update_yaxes(range=[-1, y1 + 1], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(title=title, plot_bgcolor="white", paper_bgcolor="white", height=620, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def build_court_figure(df_in, view="both", scope="half", title="Court"):
    fig = build_basic_court(scope=scope, title=title)
    d = df_in.copy()
    if d.empty:
        fig.update_layout(title=f"{title}<br><sup>No data</sup>")
        return fig

    if view in ["hit", "both"]:
        hit = d.dropna(subset=["hit_x_std", "hit_y_std"]).copy()
        if scope == "half":
            hit = hit[(hit["hit_y_std"] >= Y_NEAR) & (hit["hit_y_std"] <= Y_NET)]
        fig.add_trace(go.Scatter(
            x=hit["hit_x_std"], y=hit["hit_y_std"], mode="markers",
            name="Hit", marker=dict(size=10, color="#2E86AB", opacity=0.75),
            customdata=np.stack([hit["_point_id"], hit["source_file"], hit["Shot"]], axis=-1),
            hovertemplate="Point: %{customdata[0]}<br>File: %{customdata[1]}<br>Shot: %{customdata[2]}<br>Hit (%{x:.2f}, %{y:.2f})<extra></extra>"
        ))
    if view in ["bounce", "both"]:
        bounce = d.dropna(subset=["bounce_x_std", "bounce_y_std"]).copy()
        if scope == "half":
            bounce = bounce[(bounce["bounce_y_std"] >= Y_NEAR) & (bounce["bounce_y_std"] <= Y_NET)]
        fig.add_trace(go.Scatter(
            x=bounce["bounce_x_std"], y=bounce["bounce_y_std"], mode="markers",
            name="Bounce", marker=dict(size=10, color="#E76F51", opacity=0.75, symbol="diamond"),
            customdata=np.stack([bounce["_point_id"], bounce["source_file"], bounce["Shot"]], axis=-1),
            hovertemplate="Point: %{customdata[0]}<br>File: %{customdata[1]}<br>Shot: %{customdata[2]}<br>Bounce (%{x:.2f}, %{y:.2f})<extra></extra>"
        ))
    return fig


def unique_sorted_values(series):
    vals = [v for v in pd.Series(series).dropna().astype(str).unique().tolist() if str(v).strip() != ""]
    return sorted(vals, key=lambda x: x.lower())


def point_ids_for_mode(ctx, mode):
    d = ctx["df"].copy()
    if d.empty:
        return [{"label": "All points", "value": "__ALL__"}]
    player = ctx["selected_player"]
    d = d[d["Player"] == player].copy()
    if mode == "serve":
        keep = d.groupby("_point_id")["Type"].transform(lambda s: any(t in set(s) for t in ["first_serve", "second_serve"]))
        d = d[keep & d["Type"].isin(["first_serve", "second_serve"])]
        label = "All serve points"
    elif mode == "return":
        keep = d.groupby("_point_id")["Type"].transform(lambda s: any(t in set(s) for t in ["first_return", "second_return"]))
        d = d[keep & d["Type"].isin(["first_return", "second_return"])]
        label = "All return points"
    else:
        d = d[d["Type"].isin(["in_play", "serve_plus_one", "return_plus_one"])]
        label = "All in-play points"
    pids = sorted(d["_point_id"].dropna().unique().tolist())
    return [{"label": label, "value": "__ALL__"}] + [{"label": pid, "value": pid} for pid in pids]


def filter_court_mode_df(ctx, mode, serve_phase, serve_direction, serve_spin, serve_result, serve_side, serve_winner,
                         return_phase, return_stroke, return_direction, return_spin, return_result, return_side, return_winner,
                         inplay_stroke, inplay_direction, inplay_spin, inplay_result, inplay_side, inplay_winner, point_id):
    d = ctx["df"].copy()
    player = ctx["selected_player"]
    d = d[d["Player"] == player].copy()
    if d.empty:
        return d

    if mode == "serve":
        keep_points = d.groupby("_point_id")["Type"].transform(
            lambda s: ("first_serve" in set(s)) if serve_phase == "first" else (("second_serve" in set(s)) if serve_phase == "second" else any(t in set(s) for t in ["first_serve", "second_serve"]))
        )
        d = d[keep_points & d["Type"].isin(["first_serve", "second_serve"])]
        if serve_direction:
            d = d[d["Direction"].isin(serve_direction)]
        if serve_spin:
            d = d[d["Spin"].isin(serve_spin)]
        if serve_result:
            d = d[d["Result"].isin(serve_result)]
        if serve_side:
            d = d[d["court_side"].isin(serve_side)]
        if serve_winner:
            d = d[d["Winner"].isin(serve_winner)]
    elif mode == "return":
        keep_points = d.groupby("_point_id")["Type"].transform(
            lambda s: ("first_return" in set(s)) if return_phase == "first" else (("second_return" in set(s)) if return_phase == "second" else any(t in set(s) for t in ["first_return", "second_return"]))
        )
        d = d[keep_points & d["Type"].isin(["first_return", "second_return"])]
        if return_stroke:
            d = d[d["Stroke"].isin(return_stroke)]
        if return_direction:
            d = d[d["Direction"].isin(return_direction)]
        if return_spin:
            d = d[d["Spin"].isin(return_spin)]
        if return_result:
            d = d[d["Result"].isin(return_result)]
        if return_side:
            d = d[d["court_side"].isin(return_side)]
        if return_winner:
            d = d[d["Winner"].isin(return_winner)]
    else:
        d = d[d["Type"].isin(["in_play", "serve_plus_one", "return_plus_one"])]
        if inplay_stroke:
            d = d[d["Stroke"].isin(inplay_stroke)]
        if inplay_direction:
            d = d[d["Direction"].isin(inplay_direction)]
        if inplay_spin:
            d = d[d["Spin"].isin(inplay_spin)]
        if inplay_result:
            d = d[d["Result"].isin(inplay_result)]
        if inplay_side:
            d = d[d["court_side"].isin(inplay_side)]
        if inplay_winner:
            d = d[d["Winner"].isin(inplay_winner)]

    if point_id and point_id != "__ALL__":
        d = d[d["_point_id"] == point_id]
    return d


# =========================================================
# Tab renderers
# =========================================================
def render_serve_tab(ctx):
    tbl = serve_summary_table(ctx)
    player = ctx["selected_player"]
    row = tbl[tbl["Player"] == player]
    row = row.iloc[0] if not row.empty else pd.Series({"Aces": 0, "Double_Faults": 0, "First_Serve_%": 0.0, "First_Serve_Won_%": 0.0, "Second_Serve_%": 0.0, "Second_Serve_Won_%": 0.0})
    return html.Div([
        html.H2("Serve Statistics", style={"color": colors["primary"]}),
        html.Div([
            kpi_card("Aces", str(int(row["Aces"]))),
            kpi_card("Double Faults", str(int(row["Double_Faults"]))),
            kpi_card("1st Serve %", f'{row["First_Serve_%"]:.1f}%'),
            kpi_card("1st Serve Won %", f'{row["First_Serve_Won_%"]:.1f}%'),
            kpi_card("2nd Serve %", f'{row["Second_Serve_%"]:.1f}%'),
            kpi_card("2nd Serve Won %", f'{row["Second_Serve_Won_%"]:.1f}%'),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "10px", "marginBottom": "12px"}),
        html.Div([html.H4("Serve Summary"), render_table(tbl)], style={"background": "white", "padding": "12px", "borderRadius": "12px", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"})
    ])


def render_return_tab(ctx):
    tbl = return_summary_table(ctx)
    player = ctx["selected_player"]
    row = tbl[tbl["Player"] == player]
    row = row.iloc[0] if not row.empty else pd.Series({"Return_Winners": 0, "Return_Forced_Errors": 0, "Second_Return_Missed": 0, "First_Return_Won_%": 0.0, "Second_Return_Won_%": 0.0})
    return html.Div([
        html.H2("Return Statistics", style={"color": colors["primary"]}),
        html.Div([
            kpi_card("Return Winners", str(int(row["Return_Winners"]))),
            kpi_card("Return Forced Errors", str(int(row["Return_Forced_Errors"]))),
            kpi_card("2nd Return Missed", str(int(row["Second_Return_Missed"]))),
            kpi_card("1st Return Won %", f'{row["First_Return_Won_%"]:.1f}%'),
            kpi_card("2nd Return Won %", f'{row["Second_Return_Won_%"]:.1f}%'),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "10px", "marginBottom": "12px"}),
        html.Div([html.H4("Return Summary"), render_table(tbl)], style={"background": "white", "padding": "12px", "borderRadius": "12px", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"})
    ])


def render_winners_tab(ctx):
    tbl = winners_summary_table(ctx)
    player = ctx["selected_player"]
    row = tbl[tbl["Player"] == player]
    row = row.iloc[0] if not row.empty else pd.Series({"Winners": 0, "Unforced_Errors": 0, "Net_Point_Won_%": "0.0%", "All_Points_Won": "0/0 (0.0%)"})
    return html.Div([
        html.H2("Winners / Net / Errors", style={"color": colors["primary"]}),
        html.Div([
            kpi_card("Winners", str(int(row["Winners"]))),
            kpi_card("Unforced Errors", str(int(row["Unforced_Errors"]))),
            kpi_card("Net Point Won %", row["Net_Point_Won_%"]),
            kpi_card("All Points Won", row["All_Points_Won"]),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "10px", "marginBottom": "12px"}),
        html.Div([html.H4("Summary"), render_table(tbl)], style={"background": "white", "padding": "12px", "borderRadius": "12px", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"})
    ])


def render_speed_tab(ctx):
    plot_df = speed_plot_df(ctx)
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Speed Distribution (No data)")
    else:
        strokes = plot_df["StrokeType"].dropna().unique().tolist()
        fig = go.Figure()
        for p in plot_df["Player"].unique():
            d = plot_df[plot_df["Player"] == p].set_index("StrokeType").reindex(strokes)
            fig.add_trace(go.Bar(
                x=strokes, y=d["Mean"], name=p,
                error_y=dict(type="data", symmetric=False, array=(d["Max"] - d["Mean"]).fillna(0), arrayminus=(d["Mean"] - d["Min"]).fillna(0))
            ))
        fig.update_layout(title="Speed Distribution — Player vs Opponent", barmode="group", xaxis_title="Stroke / Type", yaxis_title="Speed (MPH)", plot_bgcolor="white", paper_bgcolor="white", height=540)
    return html.Div([
        html.H2("Speed (Comparison)", style={"color": colors["primary"]}),
        html.Div([dcc.Graph(figure=fig)], style={"background": "white", "padding": "12px", "borderRadius": "12px", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"})
    ])


def render_rally_tab(ctx):
    overall, mode_df, by_player = rally_tables(ctx)
    player = ctx["selected_player"]
    row = mode_df[mode_df["Player"] == player]
    row = row.iloc[0] if not row.empty else pd.Series({"Most_Won_Shots": 0, "Most_Lost_Shots": 0})

    fig_overall = px.pie(overall, values="Count", names="Category", hole=0.35, title="Rally Length Distribution") if not overall.empty else go.Figure()
    fig_overall.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=430)

    pdat = by_player[by_player["Player"] == player] if not by_player.empty else pd.DataFrame()
    if not pdat.empty:
        fig_player = px.bar(pdat, x="Category", y="Count", color="Result", barmode="group", title=f"Rally Length — {player} (Won vs Lost)")
        fig_player.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=430)
    else:
        fig_player = go.Figure()
        fig_player.update_layout(title="No data", height=430)

    return html.Div([
        html.H2("Rally Analysis", style={"color": colors["primary"]}),
        html.Div([
            kpi_card("Most Common (Won)", str(int(row["Most_Won_Shots"]))),
            kpi_card("Most Common (Lost)", str(int(row["Most_Lost_Shots"]))),
            kpi_card("Total Points", str(len(ctx["points"]))),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "10px", "marginBottom": "12px"}),
        html.Div([
            html.Div([dcc.Graph(figure=fig_overall)], style={"background": "white", "padding": "12px", "borderRadius": "12px", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"}),
            html.Div([dcc.Graph(figure=fig_player)], style={"background": "white", "padding": "12px", "borderRadius": "12px", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"}),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}),
    ])


def render_stroke_tab(ctx):
    d = ctx["df"].copy()
    player = ctx["selected_player"]
    opp = ctx["opponent"]
    compare = [player] + ([opp] if opp != player else [])
    dd = d[d["Player"].isin(compare)].copy()
    if dd.empty:
        fig1 = go.Figure(); fig2 = go.Figure()
    else:
        fig1 = px.histogram(dd, x="Spin", color="Player", barmode="group", title="Spin Distribution")
        fig2 = px.histogram(dd, x="Direction", color="Player", barmode="group", title="Direction Distribution")
        for fig in [fig1, fig2]:
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=450)
    return html.Div([
        html.H2("Spin & Direction (Comparison)", style={"color": colors["primary"]}),
        html.Div([
            html.Div([dcc.Graph(figure=fig1)], style={"background": "white", "padding": "12px", "borderRadius": "12px", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"}),
            html.Div([dcc.Graph(figure=fig2)], style={"background": "white", "padding": "12px", "borderRadius": "12px", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"}),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}),
    ])


def render_plus1_tab(ctx):
    s_tbl = plus_one_table(ctx, "serve")
    r_tbl = plus_one_table(ctx, "return")
    fig_s = px.pie(s_tbl, values="Total", names="Side", color="Player", hole=0.4, title="Serve+1 Side Split") if not s_tbl.empty else go.Figure()
    fig_r = px.pie(r_tbl, values="Total", names="Side", color="Player", hole=0.4, title="Return+1 Side Split") if not r_tbl.empty else go.Figure()
    fig_s.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=420)
    fig_r.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=420)
    return html.Div([
        html.H2("Serve+1 / Return+1", style={"color": colors["primary"]}),
        html.Div([
            html.Div([dcc.Graph(figure=fig_s)], style={"background": "white", "padding": "10px", "borderRadius": "12px", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"}),
            html.Div([dcc.Graph(figure=fig_r)], style={"background": "white", "padding": "10px", "borderRadius": "12px", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"}),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}),
        html.Div([
            html.Div([html.H4("Serve+1 Table"), render_table(s_tbl)], style={"background": "white", "padding": "12px", "borderRadius": "12px", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"}),
            html.Div([html.H4("Return+1 Table"), render_table(r_tbl)], style={"background": "white", "padding": "12px", "borderRadius": "12px", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"}),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px", "marginTop": "12px"}),
    ])


def render_move_tab(ctx):
    move_summary, region_tbl = move_tables(ctx)
    return html.Div([
        html.H2("Movement + Regions", style={"color": colors["primary"]}),
        html.Div([
            html.Div([html.H4("Movement Summary"), render_table(move_summary)], style={"background": "white", "padding": "12px", "borderRadius": "12px", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"}),
            html.Div([html.H4("Region Table"), render_table(region_tbl)], style={"background": "white", "padding": "12px", "borderRadius": "12px", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"}),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"})
    ])


def render_court_tab(ctx):
    player = ctx["selected_player"]
    d = ctx["df"]
    winner_vals = [x for x in unique_sorted_values(d["Winner"]) if x and x.lower() != "nan"]
    serve_direction_vals = [x for x in ["down the T", "out wide"] if x in unique_sorted_values(d["Direction"])]
    serve_spin_vals = [x for x in ["Flat", "Kick", "Slice"] if x in unique_sorted_values(d["Spin"])]
    return_direction_vals = [x for x in unique_sorted_values(d["Direction"]) if x not in ["---", "down the T", "out wide"]]
    return_spin_vals = [x for x in unique_sorted_values(d["Spin"]) if x in ["Flat", "Kick", "Slice", "Topspin"]]
    return_stroke_vals = [x for x in unique_sorted_values(d["Stroke"]) if x in ["Forehand", "Backhand"]]
    inplay_direction_vals = [x for x in unique_sorted_values(d["Direction"]) if x not in ["---", "down the T", "out wide"]]
    inplay_spin_vals = [x for x in unique_sorted_values(d["Spin"]) if x in ["Flat", "Kick", "Slice", "Topspin"]]
    inplay_stroke_vals = [x for x in unique_sorted_values(d["Stroke"]) if x in ["Forehand", "Backhand", "Volley", "Overhead"]]
    init_df = filter_court_mode_df(
        ctx, "serve",
        "__ALL__", None, None, None, None, None,
        "__ALL__", None, None, None, None, None, None,
        None, None, None, None, None, None, "__ALL__"
    )
    init_fig = build_court_figure(init_df, view="both", scope="half", title=f"Serve Explorer — {player} (Half court)")
    point_opts = point_ids_for_mode(ctx, "serve")

    return html.Div([
        dcc.Store(id="court-context", data={"player": player, "selected_file": ctx["selected_file"]}),
        html.H2("Court Visualization", style={"color": colors["primary"]}),
        html.Div([
            html.Div([
                html.H4("Explorer Controls", style={"marginTop": 0}),
                html.Label("Mode", style={"fontWeight": "700"}),
                dcc.RadioItems(id="court-mode", options=[
                    {"label": "Serve", "value": "serve"},
                    {"label": "Return", "value": "return"},
                    {"label": "In Play", "value": "inplay"},
                ], value="serve", inline=True, style={"marginBottom": "8px"}),
                html.Label("View", style={"fontWeight": "700"}),
                dcc.RadioItems(id="court-view", options=[
                    {"label": "Hit", "value": "hit"},
                    {"label": "Bounce", "value": "bounce"},
                    {"label": "Hit + Bounce", "value": "both"},
                ], value="both", inline=True, style={"marginBottom": "8px"}),
                html.Label("Court Scope", style={"fontWeight": "700"}),
                dcc.RadioItems(id="court-scope", options=[
                    {"label": "Half court", "value": "half"},
                    {"label": "Full court", "value": "full"},
                ], value="half", inline=True, style={"marginBottom": "10px"}),

                html.Div(id="serve-filter-box", children=[
                    html.H4("Serve Filters", style={"marginTop": 0}),
                    html.Label("Serve phase", style={"fontWeight": "700"}),
                    dcc.Dropdown(id="serve-phase", options=[
                        {"label": "All serves", "value": "__ALL__"},
                        {"label": "First serve", "value": "first"},
                        {"label": "Second serve", "value": "second"},
                    ], value="__ALL__", clearable=False),
                    html.Label("Direction", style={"fontWeight": "700", "marginTop": "10px"}),
                    dcc.Dropdown(id="serve-direction", options=[{"label": x, "value": x} for x in serve_direction_vals], value=None, multi=True),
                    html.Label("Spin", style={"fontWeight": "700", "marginTop": "10px"}),
                    dcc.Dropdown(id="serve-spin", options=[{"label": x, "value": x} for x in serve_spin_vals], value=None, multi=True),
                    html.Label("Result", style={"fontWeight": "700", "marginTop": "10px"}),
                    dcc.Dropdown(id="serve-result", options=[{"label": x, "value": x} for x in ["In", "Out", "Net"]], value=None, multi=True),
                    html.Label("Side", style={"fontWeight": "700", "marginTop": "10px"}),
                    dcc.Dropdown(id="serve-side", options=[{"label": x, "value": x} for x in ["Ad", "Deuce"]], value=None, multi=True),
                    html.Label("Winner", style={"fontWeight": "700", "marginTop": "10px"}),
                    dcc.Dropdown(id="serve-winner", options=[{"label": x, "value": x} for x in winner_vals], value=None, multi=True),
                ]),

                html.Div(id="return-filter-box", style={"display": "none"}, children=[
                    html.H4("Return Filters", style={"marginTop": 0}),
                    html.Label("Return phase", style={"fontWeight": "700"}),
                    dcc.Dropdown(id="return-phase", options=[
                        {"label": "All returns", "value": "__ALL__"},
                        {"label": "First return", "value": "first"},
                        {"label": "Second return", "value": "second"},
                    ], value="__ALL__", clearable=False),
                    html.Label("Stroke", style={"fontWeight": "700", "marginTop": "10px"}),
                    dcc.Dropdown(id="return-stroke", options=[{"label": x, "value": x} for x in return_stroke_vals], value=None, multi=True),
                    html.Label("Direction", style={"fontWeight": "700", "marginTop": "10px"}),
                    dcc.Dropdown(id="return-direction", options=[{"label": x, "value": x} for x in return_direction_vals], value=None, multi=True),
                    html.Label("Spin", style={"fontWeight": "700", "marginTop": "10px"}),
                    dcc.Dropdown(id="return-spin", options=[{"label": x, "value": x} for x in return_spin_vals], value=None, multi=True),
                    html.Label("Result", style={"fontWeight": "700", "marginTop": "10px"}),
                    dcc.Dropdown(id="return-result", options=[{"label": x, "value": x} for x in ["In", "Out", "Net"]], value=None, multi=True),
                    html.Label("Side", style={"fontWeight": "700", "marginTop": "10px"}),
                    dcc.Dropdown(id="return-side", options=[{"label": x, "value": x} for x in ["Ad", "Deuce"]], value=None, multi=True),
                    html.Label("Winner", style={"fontWeight": "700", "marginTop": "10px"}),
                    dcc.Dropdown(id="return-winner", options=[{"label": x, "value": x} for x in winner_vals], value=None, multi=True),
                ]),

                html.Div(id="inplay-filter-box", style={"display": "none"}, children=[
                    html.H4("In Play Filters", style={"marginTop": 0}),
                    html.Label("Stroke", style={"fontWeight": "700"}),
                    dcc.Dropdown(id="inplay-stroke", options=[{"label": x, "value": x} for x in inplay_stroke_vals], value=None, multi=True),
                    html.Label("Direction", style={"fontWeight": "700", "marginTop": "10px"}),
                    dcc.Dropdown(id="inplay-direction", options=[{"label": x, "value": x} for x in inplay_direction_vals], value=None, multi=True),
                    html.Label("Spin", style={"fontWeight": "700", "marginTop": "10px"}),
                    dcc.Dropdown(id="inplay-spin", options=[{"label": x, "value": x} for x in inplay_spin_vals], value=None, multi=True),
                    html.Label("Result", style={"fontWeight": "700", "marginTop": "10px"}),
                    dcc.Dropdown(id="inplay-result", options=[{"label": x, "value": x} for x in ["In", "Out", "Net"]], value=None, multi=True),
                    html.Label("Side", style={"fontWeight": "700", "marginTop": "10px"}),
                    dcc.Dropdown(id="inplay-side", options=[{"label": x, "value": x} for x in ["Ad", "Deuce"]], value=None, multi=True),
                    html.Label("Winner", style={"fontWeight": "700", "marginTop": "10px"}),
                    dcc.Dropdown(id="inplay-winner", options=[{"label": x, "value": x} for x in winner_vals], value=None, multi=True),
                ]),
                html.Label("Point (optional) — click any dot to auto-select", style={"fontWeight": "700", "marginTop": "10px"}),
                dcc.Dropdown(id="f-point", options=point_opts, value="__ALL__", searchable=True),
                html.Div(id="court-count", style={"marginTop": "10px", "opacity": 0.85}),
            ], style={"background": "white", "padding": "12px", "borderRadius": "12px", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"}),
            html.Div([
                dcc.Graph(id="court-graph", figure=init_fig),
                html.Div([
                    html.H4("Selected Point — Shot-by-shot"),
                    dash_table.DataTable(id="point-table", page_size=20, style_table={"overflowX": "auto"}, style_cell={"fontFamily": "Arial", "fontSize": 12, "padding": "6px", "whiteSpace": "normal", "height": "auto"}, style_header={"fontWeight": "700", "backgroundColor": "#f1f3f5"}),
                ], style={"background": "white", "padding": "12px", "borderRadius": "12px", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"})
            ])
        ], style={"display": "grid", "gridTemplateColumns": "420px 1fr", "gap": "12px"}),
        html.Div(id="court-mode-note", children="Serve mode shows only first_serve and second_serve rows from the selected player's serve points.", style={"marginTop": "12px", "background": "white", "padding": "10px 12px", "borderRadius": "12px", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)", "opacity": 0.85})
    ])


# =========================================================
# Dash app
# =========================================================
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Tennis Analytics Dashboard - SwingVision"

app.layout = html.Div([
    html.Div([
        html.H1("🎾 Tennis Analytics Dashboard - SwingVision", style={"textAlign": "center", "color": colors["primary"], "margin": "8px 0"}),
        html.Div("multi-file dashboard with player + match selection and court visualization", style={"textAlign": "center", "color": colors["text"], "opacity": 0.85}),
    ], style={"background": "white", "padding": "14px", "borderRadius": "12px", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)", "margin": "12px"}),

    html.Div([
        html.Div([
            html.Label("Select Player", style={"fontWeight": "700", "marginBottom": "6px"}),
            dcc.Dropdown(id="player-selector", options=[{"label": p, "value": p} for p in PLAYER_OPTIONS], value=DEFAULT_PLAYER, clearable=False),
        ], style={"width": "300px"}),
        html.Div([
            html.Label("Select Match / File", style={"fontWeight": "700", "marginBottom": "6px"}),
            dcc.Dropdown(id="file-selector", options=get_match_options(DEFAULT_PLAYER), value="__ALL__", clearable=False),
        ], style={"width": "340px"}),
    ], style={"display": "flex", "gap": "16px", "background": "white", "padding": "12px", "borderRadius": "12px", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)", "margin": "0 12px 12px 12px"}),

    dcc.Tabs(id="tabs", value="serve", children=[
        dcc.Tab(label="Serve", value="serve"),
        dcc.Tab(label="Return", value="return"),
        dcc.Tab(label="Winners / Net / UE", value="winners"),
        dcc.Tab(label="Speed (Comparison)", value="speed"),
        dcc.Tab(label="Rally", value="rally"),
        dcc.Tab(label="Spin & Direction (Comparison)", value="stroke"),
        dcc.Tab(label="Serve+1 / Return+1", value="plus1"),
        dcc.Tab(label="Court Visualization", value="court"),
        dcc.Tab(label="Movement + Regions", value="move"),
    ], style={"margin": "0 12px"}),

    html.Div(id="tab-content", style={"margin": "12px"}),
], style={"background": colors["bg"], "minHeight": "100vh", "paddingBottom": "24px"})


@app.callback(
    Output("file-selector", "options"),
    Output("file-selector", "value"),
    Input("player-selector", "value"),
    State("file-selector", "value"),
)
def update_file_dropdown(player, current_value):
    opts = get_match_options(player)
    valid = {o["value"] for o in opts}
    value = current_value if current_value in valid else "__ALL__"
    return opts, value


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("player-selector", "value"),
    Input("file-selector", "value"),
)
def render_tab_content(tab, player, selected_file):
    ctx = prepare_view_context(player, selected_file)
    if tab == "serve":
        return render_serve_tab(ctx)
    if tab == "return":
        return render_return_tab(ctx)
    if tab == "winners":
        return render_winners_tab(ctx)
    if tab == "speed":
        return render_speed_tab(ctx)
    if tab == "rally":
        return render_rally_tab(ctx)
    if tab == "stroke":
        return render_stroke_tab(ctx)
    if tab == "plus1":
        return render_plus1_tab(ctx)
    if tab == "court":
        return render_court_tab(ctx)
    if tab == "move":
        return render_move_tab(ctx)
    return html.Div("Unknown tab")


@app.callback(
    Output("serve-filter-box", "style"),
    Output("return-filter-box", "style"),
    Output("inplay-filter-box", "style"),
    Output("court-mode-note", "children"),
    Output("f-point", "options"),
    Output("f-point", "value"),
    Input("court-mode", "value"),
    Input("player-selector", "value"),
    Input("file-selector", "value"),
    State("f-point", "value"),
)
def update_court_mode_ui(mode, player, selected_file, current_point):
    ctx = prepare_view_context(player, selected_file)
    show = {"marginTop": "0px"}
    hide = {"display": "none"}
    if mode == "serve":
        note = "Serve mode shows only first_serve and second_serve rows from the selected player's serve points."
        opts = point_ids_for_mode(ctx, "serve")
        box_styles = (show, hide, hide)
    elif mode == "return":
        note = "Return mode shows only first_return and second_return rows from the selected player's return points."
        opts = point_ids_for_mode(ctx, "return")
        box_styles = (hide, show, hide)
    else:
        note = "In Play mode shows in_play, serve_plus_one, and return_plus_one rows from the selected player. In-play direction excludes down the T and out wide."
        opts = point_ids_for_mode(ctx, "inplay")
        box_styles = (hide, hide, show)
    valid_values = {o["value"] for o in opts}
    point_value = current_point if current_point in valid_values else "__ALL__"
    return *box_styles, note, opts, point_value


@app.callback(
    Output("court-graph", "figure"),
    Output("court-count", "children"),
    Input("court-mode", "value"),
    Input("court-view", "value"),
    Input("court-scope", "value"),
    Input("serve-phase", "value"),
    Input("serve-direction", "value"),
    Input("serve-spin", "value"),
    Input("serve-result", "value"),
    Input("serve-side", "value"),
    Input("serve-winner", "value"),
    Input("return-phase", "value"),
    Input("return-stroke", "value"),
    Input("return-direction", "value"),
    Input("return-spin", "value"),
    Input("return-result", "value"),
    Input("return-side", "value"),
    Input("return-winner", "value"),
    Input("inplay-stroke", "value"),
    Input("inplay-direction", "value"),
    Input("inplay-spin", "value"),
    Input("inplay-result", "value"),
    Input("inplay-side", "value"),
    Input("inplay-winner", "value"),
    Input("f-point", "value"),
    Input("player-selector", "value"),
    Input("file-selector", "value"),
)
def update_court_graph(mode, view, scope,
                       serve_phase, serve_direction, serve_spin, serve_result, serve_side, serve_winner,
                       return_phase, return_stroke, return_direction, return_spin, return_result, return_side, return_winner,
                       inplay_stroke, inplay_direction, inplay_spin, inplay_result, inplay_side, inplay_winner,
                       point_id, player, selected_file):
    ctx = prepare_view_context(player, selected_file)
    d = filter_court_mode_df(ctx, mode, serve_phase, serve_direction, serve_spin, serve_result, serve_side, serve_winner,
                             return_phase, return_stroke, return_direction, return_spin, return_result, return_side, return_winner,
                             inplay_stroke, inplay_direction, inplay_spin, inplay_result, inplay_side, inplay_winner, point_id)
    mode_title = {"serve": "Serve Visualization", "return": "Return Visualization", "inplay": "In Play Visualization"}[mode]
    file_label = "All files" if selected_file == "__ALL__" else selected_file
    fig = build_court_figure(d, view=view, scope=scope, title=f"{mode_title} — {player} — {file_label}")
    return fig, f"Filtered rows: {len(d)} | Points: {d['_point_id'].nunique() if len(d) else 0}"


@app.callback(
    Output("f-point", "value", allow_duplicate=True),
    Input("court-graph", "clickData"),
    State("f-point", "value"),
    prevent_initial_call=True,
)
def click_select_point(clickData, current_value):
    if not clickData or "points" not in clickData or len(clickData["points"]) == 0:
        return current_value
    cd = clickData["points"][0].get("customdata")
    if cd is not None:
        if isinstance(cd, (list, tuple, np.ndarray)) and len(cd) >= 1:
            return cd[0]
        return cd
    return current_value


@app.callback(
    Output("point-table", "data"),
    Output("point-table", "columns"),
    Input("f-point", "value"),
    Input("player-selector", "value"),
    Input("file-selector", "value"),
)
def update_point_table(point_id, player, selected_file):
    if not point_id or point_id == "__ALL__":
        return [], []
    ctx = prepare_view_context(player, selected_file)
    d = ctx["df"]
    d = d[d["_point_id"] == point_id].copy().sort_values(["Shot"]).reset_index(drop=True)
    keep = [
        "source_file", "Player", "Shot", "Type", "Stroke", "Spin", "Speed (MPH)", "Result", "Winner",
        "Direction", "Hit Zone", "Bounce Zone", "hit_x_std", "hit_y_std", "bounce_x_std", "bounce_y_std", "Shot_Count"
    ]
    keep = [c for c in keep if c in d.columns]
    d = d[keep]
    return d.to_dict("records"), [{"name": c, "id": c} for c in d.columns]


if __name__ == "__main__":
    app.run(debug=True, port=8050)
