import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import elo_model as em

TABLE_DIR = os.path.join(em.OUTPUT_DIR, "tables")
CHART_DIR = os.path.join(em.OUTPUT_DIR, "charts")
TEAM_CHART_DIR = os.path.join(CHART_DIR, "teams")
BACKTEST_DIR = os.path.join(em.OUTPUT_DIR, "backtest")

BIG_SIX = ["Arsenal", "Chelsea", "Liverpool", "Man City", "Man United", "Tottenham"]

PARAM_DESCRIPTIONS = {
    "k": "How far ratings move after one match",
    "goal_diff_power": "Bigger wins count more: K x (1 + goal difference) ^ power",
    "home_adv": "Home advantage in Elo points at the start (1993-94)",
    "home_adv_rate": "How much home advantage is corrected after each season",
    "carry_over": "Share of distance from 1500 kept into the next season",
    "promoted_elo": "Rating promoted teams start on",
    "scale": "Elo points per step on the forecast curve (the update curve uses UPDATE_SCALE)",
    "draw_width": "Controls how likely draws are",
}

plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def safe_name(team):
    return re.sub(r"[^A-Za-z0-9]+", "_", team).strip("_")


def interval(low, high, fmt="{:.4f}"):
    return f"{fmt.format(low)} to {fmt.format(high)}"


def save_table(df, name, title):
    """Save a table as a CSV and as a PNG image."""
    df.to_csv(os.path.join(TABLE_DIR, name + ".csv"), index=False)

    fig, ax = plt.subplots(figsize=(1, 1))
    ax.axis("off")
    table = ax.table(cellText=df.astype(str).values, colLabels=df.columns, cellLoc="center", loc="upper left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(range(len(df.columns)))
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#2e4358")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f3f5")

    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    fig.savefig(os.path.join(TABLE_DIR, name + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)




def with_gaps(team_rows):
    """Add a blank point wherever a team was out of the league, so the line doesn't join across it."""
    rows = team_rows[["date", "elo_after"]]
    gap = rows["date"].diff().dt.days > 150
    blanks = pd.DataFrame({"date": rows.loc[gap, "date"] - pd.Timedelta(days=1), "elo_after": np.nan})
    return pd.concat([rows, blanks]).sort_values("date")


def plot_elo(history, teams, title, path, show_others=True):
    fig, ax = plt.subplots(figsize=(12, 6))

    if show_others:
        for team, rows in history.groupby("team"):
            if team not in teams:
                line = with_gaps(rows)
                ax.plot(line["date"], line["elo_after"], color="lightgrey", linewidth=0.6, zorder=1)

    for team in teams:
        line = with_gaps(history[history["team"] == team])
        ax.plot(line["date"], line["elo_after"], linewidth=1.6, label=team, zorder=3)

    ax.axhline(em.START_ELO, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlim(history["date"].min(), history["date"].max())
    ax.set_title(title)
    ax.set_ylabel("Elo rating")
    ax.set_xlabel("Date")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def peak_ratings_table(history):
    peaks = history.loc[history.groupby("team")["elo_after"].idxmax()]
    peaks = peaks.sort_values("elo_after", ascending=False).head(20).reset_index(drop=True)
    df = pd.DataFrame({
        "Rank": np.arange(1, len(peaks) + 1),
        "Team": peaks["team"],
        "Peak Elo": peaks["elo_after"].round().astype(int),
        "Date": peaks["date"].dt.strftime("%d %b %Y"),
        "Season": peaks["season"],
    })
    save_table(df, "peak_ratings", "Highest Elo rating each club has reached (top 20)")



def skill_table(wf):
    """How much of the way from the same-odds baseline to the bookmakers the Elo model gets."""
    cols = ["elo_rps", "book_rps", "base_rps"]
    elo, book, base = wf[cols].mean()
    #every season has 380 matches with odds, so each season counts equally
    boot_elo, boot_book, boot_base = em.bootstrap_means(wf[cols], wf["season"]).T

    share = (base - elo) / (base - book)
    low, high = np.percentile((boot_base - boot_elo) / (boot_base - boot_book), [2.5, 97.5])
    df = pd.DataFrame([{"Measure": "Share of the gap from the baseline to the bookmakers that Elo closes",
                        "Value": f"{share:.1%}", "95% interval": interval(low, high, "{:.1%}")}])
    save_table(df, "model_skill", "How good the Elo model is at predicting results")


def rps_table(wf, check):
    """Average RPS of each way of forecasting, on the same matches (lower is better)."""
    df = pd.DataFrame({
        "Forecast": ["Elo model tuned on all seasons (in sample)",
                     "Elo model walk-forward test (out of sample)",
                     "Bookmakers",
                     "Same odds every match (naive)"],
        "RPS": [check.loc[0, "value"], check.loc[1, "value"], wf["book_rps"].mean(), wf["base_rps"].mean()],
    })
    df["RPS"] = df["RPS"].map("{:.4f}".format)
    save_table(df, "rps_comparison", "Average RPS of each forecast (lower is better)")


def params_table(params, elo):
    rows = [{"Parameter": k, "Value": f"{v:g}", "What it does": PARAM_DESCRIPTIONS[k]}
            for k, v in params.items()]
    rows.append({"Parameter": "home_adv (2025-26)", "Value": f"{elo['home_adv'].iloc[-1]:.1f}",
                 "What it does": "Home advantage after the season-by-season corrections"})
    save_table(pd.DataFrame(rows), "model_parameters", "Model parameters")

def team_charts(history):
    start, end = history["date"].min(), history["date"].max()
    for team, rows in history[history["team"].isin(BIG_SIX)].groupby("team"):
        line = with_gaps(rows)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(line["date"], line["elo_after"], linewidth=1.4)
        ax.axhline(em.START_ELO, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlim(start, end)
        seasons = rows["season"].nunique()
        ax.set_title(f"{team} - Elo rating ({seasons} Premier League season{'s' if seasons != 1 else ''})")
        ax.set_ylabel("Elo rating")
        fig.tight_layout()
        fig.savefig(os.path.join(TEAM_CHART_DIR, safe_name(team) + ".png"))
        plt.close(fig)


if __name__ == "__main__":
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(TEAM_CHART_DIR, exist_ok=True)
    matches = em.load_matches()
    params = em.load_params()
    elo = em.run_elo(matches, params)
    history = em.elo_history(matches, elo)
    history.to_csv(os.path.join(em.OUTPUT_DIR, "elo_history.csv"), index=False)
    wf = pd.read_csv(os.path.join(BACKTEST_DIR, "walk_forward.csv"))
    check = pd.read_csv(os.path.join(BACKTEST_DIR, "overfitting_check.csv"))

    print("Making tables now")
    peak_ratings_table(history)
    skill_table(wf)
    rps_table(wf, check)
    params_table(params, elo)

    print("Making charts now")
    peaks = history.groupby("team")["elo_after"].max().sort_values(ascending=False)
    plot_elo(history, list(peaks.index[:8]), "Premier League Elo ratings, 1993-94 to 2025-26 (8 highest peaks)",os.path.join(CHART_DIR, "elo_top_teams.png"))
    plot_elo(history, BIG_SIX, "Big six Elo ratings, 1993-94 to 2025-26", os.path.join(CHART_DIR, "elo_big_six.png"))
    team_charts(history)

    print(f"Done: see {em.OUTPUT_DIR}")
