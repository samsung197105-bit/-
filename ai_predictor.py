import pandas as pd
import numpy as np
from scipy.stats import poisson

# загрузка данных матчей
data = pd.read_csv("matches.csv")

def poisson_prob(home_xg, away_xg):

    max_goals = 6
    matrix = []

    for i in range(max_goals):
        row = []
        for j in range(max_goals):
            prob = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
            row.append(prob)
        matrix.append(row)

    return np.array(matrix)


def analyze_match(row):

    home = row["home_team"]
    away = row["away_team"]

    home_xg = row["home_xg"]
    away_xg = row["away_xg"]

    matrix = poisson_prob(home_xg, away_xg)

    total_xg = home_xg + away_xg

    # вероятность ОЗ
    btts = 1 - (matrix[0,:].sum() + matrix[:,0].sum() - matrix[0,0])

    # вероятность ТБ 2.5
    over25 = 0
    for i in range(6):
        for j in range(6):
            if i + j > 2:
                over25 += matrix[i,j]

    # наиболее вероятный счёт
    idx = np.unravel_index(matrix.argmax(), matrix.shape)
    score = f"{idx[0]}-{idx[1]}"

    # безопасный экспресс
    express = None
    if total_xg >= 2.6:
        express = "ТБ 1.5"
    else:
        express = "ТМ 3.5"

    return {
        "match": f"{home} vs {away}",
        "expected_goals": round(total_xg,2),
        "btts_probability": round(btts*100,2),
        "over25_probability": round(over25*100,2),
        "correct_score": score,
        "express_pick": express
    }


for _, row in data.iterrows():

    result = analyze_match(row)

    print("Матч:", result["match"])
    print("Ожидаемые голы:", result["expected_goals"])
    print("Вероятность ОЗ:", result["btts_probability"], "%")
    print("Вероятность ТБ2.5:", result["over25_probability"], "%")
    print("Прогноз счёта:", result["correct_score"])
    print("Экспресс ставка:", result["express_pick"])
    print("--------------")
