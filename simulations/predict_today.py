import requests
import pandas as pd
import numpy as np
from io import StringIO
from scipy.stats import poisson

LEAGUES = {
    # TIER 1
    "EPL":           ("E0",  "2526"),
    "Bundesliga":    ("D1",  "2526"),
    "Serie A":       ("I1",  "2526"),
    "La Liga":       ("SP1", "2526"),
    "Ligue 1":       ("F1",  "2526"),
    # TIER 2
    "Championship":  ("E1",  "2526"),
    "League One":    ("E2",  "2526"),
    "League Two":    ("E3",  "2526"),
    "2.Bundesliga":  ("D2",  "2526"),
    "Serie B":       ("I2",  "2526"),
    "La Liga 2":     ("SP2", "2526"),
    "Ligue 2":       ("F2",  "2526"),
    "Eredivisie":    ("N1",  "2526"),
    "Eerste Div":    ("N2",  "2526"),
    "Primeira Liga": ("P1",  "2526"),
    "Liga 2 Port":   ("P2",  "2526"),
    "Jupiler Pro":   ("B1",  "2526"),
    "Ekstraklasa":   ("PL1", "2526"),
    "Scottish Prem": ("SC0", "2526"),
    "Scottish Champ":("SC1", "2526"),
    "Greece Super":  ("G1",  "2526"),
    "Turkey Super":  ("T1",  "2526"),
}

def load_league(code, season):
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
    r = requests.get(url, timeout=10)
    df = pd.read_csv(StringIO(r.text))
    df = df.dropna(subset=["HomeTeam", "FTHG", "FTAG"])
    return df

def team_stats(df, team, n=6):
    home = df[df["HomeTeam"] == team].tail(n)
    away = df[df["AwayTeam"] == team].tail(n)
    if len(home) == 0 and len(away) == 0:
        return None
    gf = list(home["FTHG"]) + list(away["FTAG"])
    ga = list(home["FTAG"]) + list(away["FTHG"])
    sf = list(home.get("HST", pd.Series())) + list(away.get("AST", pd.Series()))
    sa = list(home.get("AST", pd.Series())) + list(away.get("HST", pd.Series()))
    xg_for = round((np.mean(gf)*0.6 + (np.mean(sf)*0.1 if sf else 0)), 2) if gf else 1.1
    xg_ag  = round((np.mean(ga)*0.6 + (np.mean(sa)*0.1 if sa else 0)), 2) if ga else 1.0
    return xg_for, xg_ag

def predict(home_xg, away_xg):
    max_g = 6
    prob = np.zeros((max_g+1, max_g+1))
    for i in range(max_g+1):
        for j in range(max_g+1):
            prob[i][j] = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
    p1   = float(np.sum(np.tril(prob, -1)))
    draw = float(np.sum(np.diag(prob)))
    p2   = float(np.sum(np.triu(prob, 1)))
    btts = float(np.sum(prob[1:, 1:]))
    tb15 = float(1 - prob[0,0] - prob[1,0] - prob[0,1])
    tb25 = float(1 - sum(prob[i][j] for i in range(max_g+1)
                         for j in range(max_g+1) if i+j <= 2))
    tm35 = float(sum(prob[i][j] for i in range(max_g+1)
                     for j in range(max_g+1) if i+j <= 3))
    scores = sorted([(i,j,prob[i][j]) for i in range(max_g+1)
                     for j in range(max_g+1)], key=lambda x: -x[2])
    return {"p1":p1,"draw":draw,"p2":p2,"btts":btts,
            "tb15":tb15,"tb25":tb25,"tm35":tm35,"top3":scores[:3]}

def verdict(r, home, away):
    lines = []
    if r["p1"] > 0.50:
        outcome = f"П1 {home}"
    elif r["p2"] > 0.50:
        outcome = f"П2 {away}"
    elif r["p1"]+r["draw"] > 0.65:
        outcome = "1Х (П1 или Ничья)"
    elif r["p2"]+r["draw"] > 0.65:
        outcome = "Х2 (П2 или Ничья)"
    else:
        outcome = "1Х (П1 или Ничья)"
    lines.append(f"  Исход:     {outcome}")
    lines.append(f"  1Х/Х2:     1Х {r['p1']+r['draw']:.1%}  |  Х2 {r['p2']+r['draw']:.1%}")
    if r["tb25"] >= 0.55:
        lines.append(f"  Тотал 2.5: ТБ 2.5 ({r['tb25']:.1%})")
    else:
        lines.append(f"  Тотал 2.5: ТМ 2.5 ({1-r['tb25']:.1%})")
    lines.append(f"  Счёт (топ-3):")
    for i,j,p in r["top3"]:
        lines.append(f"    {i}:{j} — {p:.1%}")
    express = []
    if r["tb15"] >= 0.72:
        express.append(f"ТБ 1.5  КФ ~1.17  ({r['tb15']:.1%})")
    if r["tm35"] >= 0.72:
        express.append(f"ТМ 3.5  КФ ~1.17  ({r['tm35']:.1%})")
    if r["btts"] >= 0.60:
        express.append(f"BTTS    КФ ~1.85  ({r['btts']:.1%})")
    if express:
        lines.append(f"\n  ЭКСПРЕСС:")
        for e in express:
            lines.append(f"  -> {e}")
    return "\n".join(lines)

def analyze(matches):
    print("\n" + "="*60)
    print("   ПРОГНОЗ — ДАННЫЕ 2025-26")
    print("="*60)
    data = {}
    needed = set(m[2] for m in matches)
    for league in needed:
        if league in LEAGUES:
            code, season = LEAGUES[league]
            print(f"  Загружаем {league}...", end=" ", flush=True)
            try:
                data[league] = load_league(code, season)
                print(f"{len(data[league])} матчей ✓")
            except:
                print("ОШИБКА — нет данных")
    print()
    for home, away, league in matches:
        print("="*60)
        print(f"  {home} vs {away}  [{league}]")
        if league not in data:
            print(f"  ⛔ Лига '{league}' не в базе\n")
            continue
        df = data[league]
        hs = team_stats(df, home)
        as_ = team_stats(df, away)
        if hs is None:
            print(f"  ⚠ '{home}' не найден!")
            print(f"  Команды: {sorted(df['HomeTeam'].unique())}\n")
            continue
        if as_ is None:
            print(f"  ⚠ '{away}' не найден!")
            print(f"  Команды: {sorted(df['HomeTeam'].unique())}\n")
            continue
        home_xg = round((hs[0] + as_[1]) / 2, 2)
        away_xg = round((as_[0] + hs[1]) / 2, 2)
        print(f"  xG: {home_xg} — {away_xg}")
        r = predict(home_xg, away_xg)
        print(f"  П1: {r['p1']:.1%}  Х: {r['draw']:.1%}  П2: {r['p2']:.1%}")
        print(f"  BTTS: {r['btts']:.1%}  ТБ1.5: {r['tb15']:.1%}  ТБ2.5: {r['tb25']:.1%}  ТМ3.5: {r['tm35']:.1%}")
        print(f"\n  РЕКОМЕНДАЦИЯ:")
        print(verdict(r, home, away))
        print()
    print("="*60)
    print("  Анализ завершён")
    print("="*60)

# ══════════════════════════════════════════════
# МАТЧИ НА СЕГОДНЯ — меняй каждый день
# ══════════════════════════════════════════════
matches = [
    ("Birmingham",       "QPR",              "Championship"),
    ("Norwich",          "Sheffield United",  "Championship"),
    ("Oxford",           "Blackburn",         "Championship"),
]

analyze(matches)