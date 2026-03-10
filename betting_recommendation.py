import pandas as pd
import numpy as np

def poisson_btts(home_xg, away_xg):

    prob_home = 1 - np.exp(-home_xg)
    prob_away = 1 - np.exp(-away_xg)

    return prob_home * prob_away


def predict_match(home_team, away_team, home_xg, away_xg):

    total_xg = home_xg + away_xg

    btts_prob = poisson_btts(home_xg, away_xg)

    prediction = {
        "match": f"{home_team} vs {away_team}",
        "expected_goals": round(total_xg,2),
        "btts_probability": round(btts_prob*100,2)
    }

    return prediction


print(
    predict_match(
        "TeamA",
        "TeamB",
        1.6,
        1.4
    )
)
