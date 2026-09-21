import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import elo_model as em

CLEAR_LEAD = 25  #Elo points ahead of second place


def days_as_best(history):
    """Days each team has spent on top of the ratings, from the second season's first match to the last."""
    #the first season is left out
    history = history[history["season"] != history["season"].min()]
    leaders = []
    margins = []
    for season, rows in history.groupby("season"):
        #every team's rating after each match day, held until it plays again
        ratings = rows.pivot_table(index="date", columns="team", values="elo_after", aggfunc="last").ffill()
        #before its first match a team is still on the rating it started the season with
        ratings = ratings.fillna(rows.groupby("team")["elo_before"].first())
        leaders.append(ratings.idxmax(axis=1))
        #how far the leader is ahead of second place
        top_two = np.sort(ratings.to_numpy(), axis=1)[:, -2:]
        margins.append(pd.Series(top_two[:, 1] - top_two[:, 0], index=ratings.index))
    leaders = pd.concat(leaders)
    margins = pd.concat(margins)

    #the leader stays on top until the next match day
    days = leaders.index.to_series().diff().shift(-1).dt.days
    clear_days = days.where(margins >= CLEAR_LEAD, 0)
    result = pd.DataFrame({"days": days.groupby(leaders).sum(), "clear": clear_days.groupby(leaders).sum()})
    return result.sort_values("days", ascending=False)


def chart_days_as_best(days, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    narrow = days["days"] - days["clear"]
    ax.barh(days.index, days["clear"], color="tab:blue", label=f"Clear lead ({CLEAR_LEAD}+ Elo ahead of second place)")
    ax.barh(days.index, narrow, left=days["clear"], color="lightsteelblue",
            label=f"Narrow lead (under {CLEAR_LEAD} Elo ahead)")
    ax.invert_yaxis()  #most days at the top
    for y, count in enumerate(days["days"]):
        ax.text(count, y, f"  {count:,.0f} ({count / days['days'].sum():.1%})", va="center")
    ax.set_xlim(0, days["days"].max() * 1.18)  #room for the labels
    ax.set_xlabel("Days as the highest-rated team")
    ax.set_title("Days spent as the best team in the Premier League, 1994-95 to 2025-26")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    matches = em.load_matches()
    elo = em.run_elo(matches, em.load_params())
    history = em.elo_history(matches, elo)

    days = days_as_best(history)
    table = pd.DataFrame({
        "Team": days.index,
        "Days as best team": days["days"].to_numpy().astype(int),
        "Share of all days %": (days["days"] / days["days"].sum() * 100).round(1).to_numpy(),
        f"Days with a clear lead ({CLEAR_LEAD}+ Elo)": days["clear"].to_numpy().astype(int),
    })
    print(table.to_string(index=False))

    table_dir = os.path.join(em.OUTPUT_DIR, "tables")
    os.makedirs(table_dir, exist_ok=True)
    table.to_csv(os.path.join(table_dir, "days_as_best.csv"), index=False)

    chart_dir = os.path.join(em.OUTPUT_DIR, "charts")
    os.makedirs(chart_dir, exist_ok=True)
    chart_days_as_best(days, os.path.join(chart_dir, "days_as_best.png"))
