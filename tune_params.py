import json
import os

import numpy as np
import pandas as pd

import elo_model as em

BACKTEST_DIR = os.path.join(em.OUTPUT_DIR, "backtest")

#values tried for each parameter
SEARCH_GRID = {
    "k": [3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 22, 26, 30, 36],
    "goal_diff_power": [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    "home_adv": [30, 45, 60, 75, 90, 105, 120, 135, 150, 165],
    "home_adv_rate": [0, 100, 200, 400, 700],
    "carry_over": [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
    "promoted_elo": [1320, 1360, 1400, 1440, 1480],
}

SKIP_SEASONS = 2  # everyone starts on 1500, so don't score the first two seasons
FIRST_WALK_FORWARD_SEASON = "2001-02"
RETUNE_EVERY = 3  # seasons


def evaluate(matches, params, use):
    """RPS on the chosen matches, after fitting the probability curve to those same matches."""
    elo = em.run_elo(matches, params)
    scale, width, loss = em.fit_prediction(
        elo["elo_diff"].to_numpy()[use], matches["outcome"].to_numpy()[use]
    )
    return loss, scale, width


def search(matches, use, start=None, max_rounds=10, restarts=5):
    """Climb from the starting parameters and from a few random ones, and keep the best.

    Changing one parameter at a time can get stuck, because some parameters trade off against each other 
    """
    start = dict(start or em.DEFAULT_PARAMS)
    rng = np.random.default_rng(0)
    starts = [start]
    for _ in range(restarts):
        starts.append({**start, **{name: rng.choice(values).item()
                                   for name, values in SEARCH_GRID.items()}})

    best, best_loss = min((climb(matches, use, s, max_rounds) for s in starts), key=lambda r: r[1])
    _, scale, width = evaluate(matches, best, use)
    best["scale"] = scale
    best["draw_width"] = round(width, 2)
    return best, best_loss


def climb(matches, use, start, max_rounds):
    """Try every value of one parameter at a time and keep any that improve the RPS.
    Repeat until a whole round makes no difference
    greedy"""
    best = dict(start)
    best_loss = evaluate(matches, best, use)[0]

    for _ in range(max_rounds):
        improved = False
        for name, values in SEARCH_GRID.items():
            for value in values:
                if value == best[name]:
                    continue
                trial = {**best, name: value}
                loss = evaluate(matches, trial, use)[0]
                if loss < best_loss - 1e-7:
                    best, best_loss, improved = trial, loss, True
        if not improved:
            break
    return best, best_loss


def predict(matches, params, rows):
    elo = em.run_elo(matches, params)
    return em.predict_probabilities(elo["elo_diff"].to_numpy()[rows], params["scale"], params["draw_width"])


def walk_forward(matches):
    seasons = matches["season"].unique().tolist()
    season_num = matches["season_num"].to_numpy()
    outcome = matches["outcome"].to_numpy()
    books = matches[["book_home", "book_draw", "book_away"]].to_numpy()
    first = seasons.index(FIRST_WALK_FORWARD_SEASON)

    params = dict(em.DEFAULT_PARAMS)
    rows = []
    predictions = []

    for s in range(first, len(seasons)):
        if (s - first) % RETUNE_EVERY == 0:
            train = (season_num >= SKIP_SEASONS) & (season_num < s)
            params, _ = search(matches, train, start=params)
            print(f"  tuned on 1995-96 to {seasons[s - 1]}: k={params['k']}, "
                  f"goal_diff_power={params['goal_diff_power']}, home_adv={params['home_adv']}")

        test = season_num == s
        probs = predict(matches, params, test)
        has_odds = np.isfinite(books[test]).all(axis=1)
        elo_score = em.score(probs[has_odds], outcome[test][has_odds])
        book_score = em.score(books[test][has_odds], outcome[test][has_odds])

        # no-skill baseline: the same home/draw/away rates for every match, taken from earlier seasons
        past = (season_num >= SKIP_SEASONS) & (season_num < s)
        rates = np.bincount(outcome[past], minlength=3) / past.sum()
        base_score = em.score(np.tile(rates, (has_odds.sum(), 1)), outcome[test][has_odds])

        #season summar
        rows.append({
            "season": seasons[s],
            "matches": elo_score["matches"],
            "elo_rps": elo_score["rps"],
            "book_rps": book_score["rps"],
            "base_rps": base_score["rps"],
            "elo_accuracy": elo_score["accuracy"],
            "book_accuracy": book_score["accuracy"],
        })
        #individual forecast along with what happened
        season_preds = matches.loc[test, ["season", "date", "home", "away", "result"]].copy()
        season_preds["elo_home"], season_preds["elo_draw"], season_preds["elo_away"] = probs.T
        season_preds["book_home"], season_preds["book_draw"], season_preds["book_away"] = books[test].T
        predictions.append(season_preds)

    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def walk_forward_summary(preds):
    """Average RPS over all the walk-forward seasons, with 95% intervals from resampling whole seasons."""
    preds = preds.dropna(subset=["book_home", "book_draw", "book_away"])
    outcome = preds["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    elo = em.rps(preds[["elo_home", "elo_draw", "elo_away"]].to_numpy(), outcome)
    book = em.rps(preds[["book_home", "book_draw", "book_away"]].to_numpy(), outcome)

    per_match = np.column_stack([elo, book, elo - book])
    low, high = np.percentile(em.bootstrap_means(per_match, preds["season"]), [2.5, 97.5], axis=0)
    return pd.DataFrame({
        "measure": ["Elo model RPS", "Bookmaker RPS", "Elo minus bookmakers"],
        "value": per_match.mean(axis=0),
        "low": low,
        "high": high,
    })


if __name__ == "__main__":
    os.makedirs(BACKTEST_DIR, exist_ok=True)
    matches = em.load_matches()
    season_num = matches["season_num"].to_numpy()
    print(f"Loaded {len(matches)} matches from {matches['season'].nunique()} seasons")

    print("\nTuning on every season (these parameters are used for the ratings)")
    best, loss = search(matches, season_num >= SKIP_SEASONS)
    with open(os.path.join(em.OUTPUT_DIR, "best_params.json"), "w") as f:
        json.dump(best, f, indent=2)
    print("  best:", best, f"RPS {loss:.4f}")

    print("\nWalk-forward test (each season predicted from earlier seasons only)")
    wf, preds = walk_forward(matches)
    wf.to_csv(os.path.join(BACKTEST_DIR, "walk_forward.csv"), index=False)
    preds.to_csv(os.path.join(BACKTEST_DIR, "walk_forward_predictions.csv"), index=False)
    print(f"  average RPS - Elo {wf['elo_rps'].mean():.4f}, bookmakers {wf['book_rps'].mean():.4f}, "
          f"same odds every match {wf['base_rps'].mean():.4f}")
    wf_summary = walk_forward_summary(preds)
    wf_summary.to_csv(os.path.join(BACKTEST_DIR, "walk_forward_summary.csv"), index=False)
    gap = wf_summary.iloc[2]
    print(f"  Elo minus bookmakers {gap['value']:+.4f} (95% interval {gap['low']:+.4f} to {gap['high']:+.4f}), "
          f"Elo better in {(wf['elo_rps'] < wf['book_rps']).sum()} of {len(wf)} seasons")
    # overfitting check
    first = matches["season"].unique().tolist().index(FIRST_WALK_FORWARD_SEASON)
    has_odds = np.isfinite(matches[["book_home", "book_draw", "book_away"]].to_numpy()).all(axis=1)
    same_matches = (season_num >= first) & has_odds
    in_sample = em.score(predict(matches, best, same_matches), matches["outcome"].to_numpy()[same_matches])
    walk_forward_rps = wf_summary.iloc[0]["value"]
    print(f"\nOverfitting check on the {in_sample['matches']} matches with odds from {FIRST_WALK_FORWARD_SEASON} on")
    print(f"  tuned on all seasons (in sample) RPS {in_sample['rps']:.4f}, "f"walk-forward (out of sample) RPS {walk_forward_rps:.4f}, " f"difference {walk_forward_rps - in_sample['rps']:+.4f}")
    pd.DataFrame({
        "measure": ["In-sample Elo RPS", "Walk-forward Elo RPS", "Difference"],
        "value": [in_sample["rps"], walk_forward_rps, walk_forward_rps - in_sample["rps"]],
    }).to_csv(os.path.join(BACKTEST_DIR, "overfitting_check.csv"), index=False)

    print("\nDone - now run make_outputs.py")
