import csv
import glob
import json
import os
import numpy as np
import pandas as pd

FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(FOLDER, "Data")
OUTPUT_DIR = os.path.join(FOLDER, "output")
START_ELO = 1500
#400 / ln(10) is the classic chess 400 divisor (base 10) written in base e, so the curve is the same as that one
UPDATE_SCALE = 400 / np.log(10)
BOOKMAKERS = ["B365", "BW", "IW", "PS", "WH", "VC", "GB", "SB", "LB", "SJ", "BFD", "BMGM", "BV", "CL"]
#starting point for the parameter search in tune_params.py
DEFAULT_PARAMS = {
    "k": 20,                 #how far ratings move after one match
    "goal_diff_power": 0.5,  #bigger wins count for more: K * (1 + goal difference) ** power
    "home_adv": 65,          #Elo points added to the home team's rating
    "home_adv_rate": 0,      #how much home advantage gets corrected after each season (0 = never)
    "carry_over": 0.85,      #share of a team's distance from 1500 kept into the next season
    "promoted_elo": 1400,    #rating a promoted team starts the season on
}
# values tried when fitting the probability curve (see fit_prediction)
SCALES = np.arange(90, 351, 5)
DRAW_WIDTHS = np.arange(0.30, 1.001, 0.02)

def season_label(filename):
    """eg 93-94.csv-> 1993-94"""
    first = int(filename[:2])
    year = 1900 + first if first >= 90 else 2000 + first
    return f"{year}-{filename[3:5]}"

def load_season(path):
    with open(path, encoding="latin-1", newline="") as f:  #data has some non utf-8
        rows = list(csv.reader(f))   #some names have commas in
    header = rows[0]
    width = len(header)
    games = [(row + [""] * width)[:width] for row in rows[1:] if row and row[0].strip() == "E0"] #some rows have extra or missing fields
    raw = pd.DataFrame(games, columns=header)

    #early seasons write dates as dd/mm/yy but later ones do dd/mm/yyyy
    dates = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
    dates = dates.fillna(pd.to_datetime(raw["Date"], format="%d/%m/%y", errors="coerce"))

    season = pd.DataFrame({
        "season": season_label(os.path.basename(path)),
        "date": dates,
        "home": raw["HomeTeam"],
        "away": raw["AwayTeam"],
        "home_goals": pd.to_numeric(raw["FTHG"], errors="coerce"),
        "away_goals": pd.to_numeric(raw["FTAG"], errors="coerce"),
    })
    season[["book_home", "book_draw", "book_away"]] = bookmaker_probabilities(raw)
    return season.dropna(subset=["date", "home_goals", "away_goals"]) #no book for early seasons


def bookmaker_probabilities(raw):
    """average of the bookmakers' implied probabilities with each bookmaker's margin removed"""
    prices = []
    for book in BOOKMAKERS:
        cols = [book + "H", book + "D", book + "A"]
        if not all(c in raw.columns for c in cols):
            continue
        odds = raw[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float, copy=True)
        odds[odds <= 1] = np.nan #get rid of any impossible odds (implied = 1/odds)
        implied = 1/odds
        prices.append(implied/implied.sum(axis=1, keepdims=True)) #1/NaN=NaN

    if not prices:
        return np.full((len(raw), 3), np.nan)

    
    prices = np.stack(prices)
    quoted = np.isfinite(prices)
    #basically nanmean but with less warning messages
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(quoted, prices, 0).sum(axis=0) / quoted.sum(axis=0)


def load_matches():
    files = glob.glob(os.path.join(DATA_DIR, "[0-9][0-9]-[0-9][0-9].csv"))
    files.sort(key=lambda p: season_label(os.path.basename(p)))
    matches = pd.concat([load_season(f) for f in files], ignore_index=True)

    matches["home_goals"] = matches["home_goals"].astype(int)
    matches["away_goals"] = matches["away_goals"].astype(int)
    matches["result"] = np.select(
        [matches["home_goals"] > matches["away_goals"], matches["home_goals"] < matches["away_goals"]],
        ["H", "A"],
        default="D",
    )
    matches["outcome"] = matches["result"].map({"H": 0, "D": 1, "A": 2})
    matches["season_num"] = pd.factorize(matches["season"])[0]

    #every season should have 380 matches or 462 in the 22-team seasons before 1995-96
    counts = matches["season"].value_counts()
    odd = counts[~counts.isin([380, 462])]
    if len(odd):
        print("Warning - unexpected number of matches:", odd.to_dict())
    return matches


def run_elo(matches, params):
    """Go through every match in date order and return both teams' ratings before and after"""
    k = params["k"]
    power = params["goal_diff_power"]
    carry_over = params["carry_over"]
    home_adv = params["home_adv"]

    ratings = {}
    rated_in = {}  #last season each team's rating was used
    season_error = 0.0  #home teams' actual score minus expected score, summed over the season
    season_games = 0

    #quicker to loop over python list
    seasons = matches["season_num"].tolist()
    homes = matches["home"].tolist()
    aways = matches["away"].tolist()
    home_goals = matches["home_goals"].tolist()
    away_goals = matches["away_goals"].tolist()

    n = len(matches)
    home_before = np.empty(n)
    away_before = np.empty(n)
    home_after= np.empty(n)
    away_after = np.empty(n)
    home_adv_used = np.empty(n)

    current_season = 0
    for i in range(n):
        season = seasons[i]
        if season != current_season:
            #new season: move home advantage towards what home teams actually managed last season
            home_adv += params["home_adv_rate"] * season_error / season_games 
            season_error = 0.0
            season_games = 0
            current_season = season
        for team in (homes[i], aways[i]):
            if rated_in.get(team) == season:
                continue
            if season == 0:
                ratings[team] = START_ELO
            elif rated_in.get(team) == season - 1:
                #stayed up: pull part of the way back towards 1500
                ratings[team] = START_ELO + carry_over * (ratings[team] - START_ELO)
            else:
                #promoted (whether or not they've been in the league before)
                ratings[team] = params["promoted_elo"]
            rated_in[team] = season
        h = ratings[homes[i]]
        a = ratings[aways[i]]
        expected = 1 / (1 + np.exp((a - h - home_adv) / UPDATE_SCALE))
        if home_goals[i] > away_goals[i]:
            actual = 1.0
        elif home_goals[i] < away_goals[i]:
            actual = 0.0
        else:
            actual = 0.5

        goal_diff = abs(home_goals[i] - away_goals[i])
        change = k*(1+goal_diff)**power*(actual-expected)

        home_before[i], away_before[i], home_adv_used[i] = h, a, home_adv
        ratings[homes[i]] = home_after[i] = h + change
        ratings[aways[i]] = away_after[i] = a - change

        season_error += actual - expected
        season_games += 1

    return pd.DataFrame({
        "home_before": home_before,
        "away_before": away_before,
        "home_after": home_after,
        "away_after": away_after,
        "home_adv": home_adv_used,
        "elo_diff": home_before + home_adv_used - away_before,
    }, index=matches.index)


def predict_probabilities(elo_diff, scale, draw_width):
    """Home win/draw/away win probabilities from the Elo gap (home rating + home advantage - away)"""
    elo_diff = np.asarray(elo_diff, dtype=float)
    p_home = 1/(1+np.exp(-(elo_diff/scale-draw_width)))
    p_away = 1/(1+np.exp(elo_diff/scale+draw_width))
    return np.column_stack([p_home, 1 - p_home - p_away, p_away])


def rps(probs, outcome):
    """Normalised Ranked probability score for each match (0 = perfect). Outcome: 0 home, 1 draw, 2 away"""
    outcome = np.asarray(outcome)
    home_or_less = probs[:, 0]
    draw_or_less = probs[:, 0] + probs[:, 1]
    return ((home_or_less - (outcome == 0)) ** 2 + (draw_or_less - (outcome <= 1)) ** 2) / 2


def score(probs, outcome):
    outcome = np.asarray(outcome)
    return {
        "rps": float(rps(probs, outcome).mean()),
        "accuracy": float((probs.argmax(axis=1) == outcome).mean()),
        "matches": int(len(outcome)),
    }


def bootstrap_means(values, groups, draws=5000, seed=1):
    """Average of each column of values over many resamples of the groups (drawn with replacement).
    Resample whole seasons when matches in a season share the same errors, a home advantage
    that's off for a season affects every match in it. Returns one row per resample.
    """
    values = np.asarray(values, dtype=float).reshape(len(groups), -1)
    codes, labels = pd.factorize(np.asarray(groups))
    sums = np.column_stack([np.bincount(codes, weights=col, minlength=len(labels)) for col in values.T])
    sizes = np.bincount(codes, minlength=len(labels))

    rng = np.random.default_rng(seed)
    means = np.empty((draws, values.shape[1]))
    for i in range(draws):
        pick = rng.integers(0, len(labels), len(labels))
        means[i] = sums[pick].sum(axis=0) / sizes[pick].sum()
    return means


def fit_prediction(elo_diff, outcome):
    """Find the scale and draw width that give the lowest average RPS.
    The score only depends on the Elo gap, so matches are grouped into 5-point buckets and
    scale/width combinations are checked all at once. Every 4th value is tried first, then
    every value around the best of those.
    """
    steps = np.round(np.asarray(elo_diff) / 5).astype(int)
    first = steps.min()
    counts = np.bincount((steps - first) * 3 + np.asarray(outcome), minlength=(steps.max() - first + 1) * 3)
    n_home, n_draw, n_away = counts.reshape(-1, 3).T.astype(float)
    diffs = (np.arange(len(n_home)) + first) * 5.0

    def average_rps(scales, widths):
        scale = scales[:, None, None]
        width = widths[None, :, None]
        p_home = 1 / (1 + np.exp(-(diffs / scale - width)))
        p_home_or_draw = 1 - 1 / (1 + np.exp(diffs / scale + width))
        total = (n_home * ((p_home - 1) ** 2 + (p_home_or_draw - 1) ** 2)+ n_draw * (p_home ** 2 + (p_home_or_draw - 1) ** 2)+ n_away * (p_home ** 2 + p_home_or_draw ** 2)).sum(axis=2)
        return total / 2 / len(steps)

    coarse = average_rps(SCALES[::4], DRAW_WIDTHS[::4]) 
    i, j = np.unravel_index(coarse.argmin(), coarse.shape)
    scales = SCALES[max(4 * i - 4, 0):4 * i + 5]
    widths = DRAW_WIDTHS[max(4 * j - 4, 0):4 * j + 5]
    loss = average_rps(scales, widths)
    i, j = np.unravel_index(loss.argmin(), loss.shape)
    return float(scales[i]), float(widths[j]), float(loss[i, j])


def elo_history(matches, elo):
    """One row per team per match, with the rating before and after"""
    home = pd.DataFrame({
        "date": matches["date"], 
        "season": matches["season"],
        "team": matches["home"], 
        "opponent": matches["away"], "venue": "H",
        "elo_before": elo["home_before"], 
        "elo_after": elo["home_after"],
    })
    away = pd.DataFrame({
        "date": matches["date"], 
        "season": matches["season"],
        "team": matches["away"], 
        "opponent": matches["home"], 
        "venue": "A",
        "elo_before": elo["away_before"], 
        "elo_after": elo["away_after"],
    })
    history = pd.concat([home, away]).sort_values(["team", "date"], kind="stable")
    return history.reset_index(drop=True)

def load_params():
    path = os.path.join(OUTPUT_DIR, "best_params.json")
    if not os.path.exists(path):
        raise SystemExit("output/best_params.json not found - run tune_params.py first")
    with open(path) as f:
        return json.load(f)
