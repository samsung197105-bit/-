import requests
import pandas as pd
import numpy as np
from io import StringIO
from scipy.stats import poisson

LEAGUES = {
    "EPL":           ("E0",  "2526"),
    "Championship":  ("E1",  "2526"),
    "League One":    ("E2",  "2526"),
    "League Two":    ("E3",  "2526"),
    "Bundesliga":    ("D1",  "2526"),
    "2.Bundesliga":  ("D2",  "2526"),
    "Serie A":       ("I1",  "2526"),
    "Serie B":       ("I2",  "2526"),
    "La Liga":       ("SP1", "2526"),
    "La Liga 2":     ("SP2", "2526"),
    "Ligue 1":       ("F1",  "2526"),
    "Ligue 2":       ("F2",  "2526"),
    "Eredivisie":    ("N1",  "2526"),
    "Eerste Div":    ("N2",  "2526"),
    "Primeira Liga": ("P1",  "2526"),
    "Liga 2 Port":   ("P2",  "2526"),
    "Jupiler Pro":   ("B1",  "2526"),
    "Ekstraklasa":   ("PL1", "2526"),
    "Scottish Prem": ("SC0", "2526"),
    "Turkey Super":  ("T1",  "2526"),
    "Greece Super":  ("G1",  "2526"),
}

# Исторические данные для H2H (прошлые сезоны)
H2H_LEAGUES = {
    "EPL":           [("E0","2425"),("E0","2324"),("E0","2223")],
    "Championship":  [("E1","2425"),("E1","2324"),("E1","2223")],
    "Bundesliga":    [("D1","2425"),("D1","2324"),("D1","2223")],
    "Serie A":       [("I1","2425"),("I1","2324"),("I1","2223")],
    "La Liga":       [("SP1","2425"),("SP1","2324"),("SP1","2223")],
    "Ligue 1":       [("F1","2425"),("F1","2324"),("F1","2223")],
    "Eredivisie":    [("N1","2425"),("N1","2324"),("N1","2223")],
    "Primeira Liga": [("P1","2425"),("P1","2324"),("P1","2223")],
    "Jupiler Pro":   [("B1","2425"),("B1","2324"),("B1","2223")],
}

def load_league(code, season):
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
    r = requests.get(url, timeout=10)
    df = pd.read_csv(StringIO(r.text))
    df = df.dropna(subset=["HomeTeam", "FTHG", "FTAG"])
    return df

# ══════════════════════════════════════════════
#  H2H СТАТИСТИКА
# ══════════════════════════════════════════════
def get_h2h(home, away, league, current_df, n=5):
    frames = [current_df]

    # Добавляем прошлые сезоны
    if league in H2H_LEAGUES:
        for code, season in H2H_LEAGUES[league]:
            try:
                df_old = load_league(code, season)
                frames.append(df_old)
            except:
                pass

    all_data = pd.concat(frames, ignore_index=True)

    # Все встречи между командами
    h2h = all_data[
        ((all_data["HomeTeam"] == home) & (all_data["AwayTeam"] == away)) |
        ((all_data["HomeTeam"] == away) & (all_data["AwayTeam"] == home))
    ].tail(n)

    if len(h2h) == 0:
        return None

    results = {"home_wins": 0, "draws": 0, "away_wins": 0,
               "avg_goals": 0, "btts_count": 0, "over25": 0, "matches": []}

    for _, row in h2h.iterrows():
        hg = int(row["FTHG"])
        ag = int(row["FTAG"])
        total = hg + ag

        # Определяем победителя относительно нашего home
        if row["HomeTeam"] == home:
            if hg > ag: results["home_wins"] += 1
            elif hg == ag: results["draws"] += 1
            else: results["away_wins"] += 1
            results["matches"].append(f"{hg}:{ag}")
        else:
            if ag > hg: results["home_wins"] += 1
            elif hg == ag: results["draws"] += 1
            else: results["away_wins"] += 1
            results["matches"].append(f"{ag}:{hg}")

        results["avg_goals"] += total
        if hg > 0 and ag > 0: results["btts_count"] += 1
        if total > 2: results["over25"] += 1

    cnt = len(h2h)
    results["avg_goals"] = round(results["avg_goals"] / cnt, 2)
    results["btts_pct"] = round(results["btts_count"] / cnt * 100)
    results["over25_pct"] = round(results["over25"] / cnt * 100)
    results["total"] = cnt

    # H2H корректировка xG
    h2h_xg_adj = round((results["avg_goals"] / 2.5 - 1) * 0.15, 3)

    # Психологический H2H блок — 0 побед в 5 встречах
    psych_lock = results["home_wins"] == 0 and cnt >= 4

    results["xg_adj"] = h2h_xg_adj
    results["psych_lock"] = psych_lock
    return results

def team_stats(df, team, n=6):
    home = df[df["HomeTeam"] == team].tail(n)
    away = df[df["AwayTeam"] == team].tail(n)
    if len(home) == 0 and len(away) == 0:
        return None

    gf = list(home["FTHG"]) + list(away["FTAG"])
    ga = list(home["FTAG"]) + list(away["FTHG"])

    try:
        sf = list(home["HST"]) + list(away["AST"])
        sa = list(home["AST"]) + list(away["HST"])
        sf = [x for x in sf if pd.notna(x)]
        sa = [x for x in sa if pd.notna(x)]
    except:
        sf, sa = [], []

    xg_for = round((np.mean(gf)*0.6 + (np.mean(sf)*0.1 if sf else 0)), 2) if gf else 1.1
    xg_ag  = round((np.mean(ga)*0.6 + (np.mean(sa)*0.1 if sa else 0)), 2) if ga else 1.0
    ga_avg = round(np.mean(ga), 2) if ga else 0

    all_matches = []
    for _, row in home.iterrows():
        if row["FTHG"] > row["FTAG"]: all_matches.append(3)
        elif row["FTHG"] == row["FTAG"]: all_matches.append(1)
        else: all_matches.append(0)
    for _, row in away.iterrows():
        if row["FTAG"] > row["FTHG"]: all_matches.append(3)
        elif row["FTHG"] == row["FTAG"]: all_matches.append(1)
        else: all_matches.append(0)

    last5 = all_matches[-5:] if len(all_matches) >= 5 else all_matches
    momentum = round(np.mean(last5), 2) if last5 else 1.0

    all_goals = list(home["FTHG"]) + list(away["FTAG"])
    no_goal_streak = len(all_goals) >= 2 and all_goals[-1] == 0 and all_goals[-2] == 0

    return {
        "xg_for": xg_for, "xg_ag": xg_ag, "ga_avg": ga_avg,
        "momentum": momentum, "no_goal_streak": no_goal_streak,
        "last5_goals": all_goals[-5:]
    }

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

def apply_filters(home, away, pos_h, pos_a, odds_p1, odds_p2,
                  total_teams, hs, as_, r, h2h):
    blocks = []

    if pos_h <= 2:
        blocks.append(f"⛔ TABLE LOCK: {home} = {pos_h}-е место (топ-2)")
    if pos_a <= 2:
        blocks.append(f"⛔ TABLE LOCK: {away} = {pos_a}-е место (топ-2)")
    if pos_h >= total_teams - 2:
        blocks.append(f"⛔ TABLE LOCK: {home} = {pos_h}-е место (зона вылета)")
    if pos_a >= total_teams - 2:
        blocks.append(f"⛔ TABLE LOCK: {away} = {pos_a}-е место (зона вылета)")

    zone = None
    if odds_p1 > 2.00 and odds_p2 > 2.00:
        zone = "РАВНОВЕСИЕ"
        if r["tb25"] >= 0.55:
            blocks.append(f"⛔ ODDS SYNC: Зона равновесия — только ТМ 3.5, модель даёт ТБ")
    elif odds_p1 < 2.00 or odds_p2 < 2.00:
        zone = "ДОМИНИРОВАНИЕ"
        if r["tm35"] >= 0.72 and r["tb15"] < 0.72:
            blocks.append(f"⛔ ODDS SYNC: Зона доминирования — только ТБ 1.5, модель даёт ТМ")

    underdog_goals = None
    if odds_p1 > 3.0 and hs:
        last3 = hs["last5_goals"][-3:]
        if len(last3) == 3 and all(g > 0 for g in last3):
            underdog_goals = home
    if odds_p2 > 3.0 and as_:
        last3 = as_["last5_goals"][-3:]
        if len(last3) == 3 and all(g > 0 for g in last3):
            underdog_goals = away
    if underdog_goals:
        blocks.append(f"⛔ MOMENTUM: {underdog_goals} (андердог) забивал 3 матча подряд — ТМ запрещён")

    if hs and as_:
        ga_total = hs["ga_avg"] + as_["ga_avg"]
        if ga_total < 2.5:
            blocks.append(f"⚠ ШЛЮЗ GA: суммарные пропуски {ga_total:.2f} < 2.5 — ТБ/ОЗ ослаблены")

    if hs and hs["no_goal_streak"]:
        blocks.append(f"⚠ СУХАЯ СЕРИЯ: {home} не забивал 2 матча подряд — ТБ заблокирован")
    if as_ and as_["no_goal_streak"]:
        blocks.append(f"⚠ СУХАЯ СЕРИЯ: {away} не забивал 2 матча подряд — ТБ заблокирован")

    # H2H блоки
    if h2h and h2h["psych_lock"]:
        blocks.append(f"⛔ H2H ПСИХО-БЛОК: {home} — 0 побед в последних {h2h['total']} встречах")

    return blocks, zone

def verdict(r, home, away, zone, blocks):
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

    if zone == "РАВНОВЕСИЕ":
        lines.append(f"  Тотал:     ТМ 3.5 ({r['tm35']:.1%})  [Зона равновесия]")
    elif zone == "ДОМИНИРОВАНИЕ":
        lines.append(f"  Тотал:     ТБ 1.5 ({r['tb15']:.1%})  [Зона доминирования]")
    else:
        if r["tb25"] >= 0.55:
            lines.append(f"  Тотал 2.5: ТБ 2.5 ({r['tb25']:.1%})")
        else:
            lines.append(f"  Тотал 2.5: ТМ 2.5 ({1-r['tb25']:.1%})")

    lines.append(f"  Счёт (топ-3):")
    for i,j,p in r["top3"]:
        lines.append(f"    {i}:{j} — {p:.1%}")

    hard_blocks = [b for b in blocks if b.startswith("⛔")]
    if not hard_blocks:
        express = []
        if r["tb15"] >= 0.72:
            express.append(f"ТБ 1.5  КФ ~1.17  ({r['tb15']:.1%})")
        if r["tm35"] >= 0.72:
            express.append(f"ТМ 3.5  КФ ~1.17  ({r['tm35']:.1%})")
        if r["btts"] >= 0.60:
            express.append(f"BTTS    КФ ~1.85  ({r['btts']:.1%})")
        if express:
            lines.append(f"\n  ✅ ЭКСПРЕСС:")
            for e in express:
                lines.append(f"  -> {e}")
    return "\n".join(lines)

def analyze(matches, total_teams=24):
    print("\n" + "="*62)
    print("   ПРОГНОЗ v25.3 + H2H — ДАННЫЕ 2025-26")
    print("="*62)

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
                print("ОШИБКА")

    print()
    for match in matches:
        home, away, league = match[0], match[1], match[2]
        pos_h   = match[3] if len(match) > 3 else 10
        pos_a   = match[4] if len(match) > 4 else 10
        odds_p1 = match[5] if len(match) > 5 else 2.50
        odds_x  = match[6] if len(match) > 6 else 3.20
        odds_p2 = match[7] if len(match) > 7 else 2.50

        print("="*62)
        print(f"  {home} vs {away}  [{league}]")
        print(f"  Позиции: {pos_h} vs {pos_a}  |  КФ: {odds_p1} / {odds_x} / {odds_p2}")

        if league not in data:
            print(f"  ⛔ Лига не загружена\n")
            continue

        df = data[league]
        hs  = team_stats(df, home)
        as_ = team_stats(df, away)

        if hs is None:
            print(f"  ⚠ '{home}' не найден! Команды: {sorted(df['HomeTeam'].unique())}\n")
            continue
        if as_ is None:
            print(f"  ⚠ '{away}' не найден! Команды: {sorted(df['HomeTeam'].unique())}\n")
            continue

        # H2H
        print(f"  Загружаем H2H...", end=" ", flush=True)
        h2h = get_h2h(home, away, league, df)
        if h2h:
            print(f"{h2h['total']} встреч найдено")
            print(f"  H2H: {home} {h2h['home_wins']}П / {h2h['draws']}Н / {h2h['away_wins']}П {away}")
            print(f"  H2H счета: {' | '.join(h2h['matches'])}")
            print(f"  H2H avg голов: {h2h['avg_goals']}  BTTS: {h2h['btts_pct']}%  ТБ2.5: {h2h['over25_pct']}%")
        else:
            print(f"нет данных")

        # xG с H2H корректировкой
        home_xg = round((hs["xg_for"] + as_["xg_ag"]) / 2, 2)
        away_xg = round((as_["xg_for"] + hs["xg_ag"]) / 2, 2)
        if h2h:
            home_xg = round(home_xg + h2h["xg_adj"], 2)
            away_xg = round(away_xg + h2h["xg_adj"], 2)
        print(f"  xG (с H2H корр.): {home_xg} — {away_xg}")

        r = predict(home_xg, away_xg)
        print(f"  П1: {r['p1']:.1%}  Х: {r['draw']:.1%}  П2: {r['p2']:.1%}")
        print(f"  BTTS: {r['btts']:.1%}  ТБ1.5: {r['tb15']:.1%}  ТБ2.5: {r['tb25']:.1%}  ТМ3.5: {r['tm35']:.1%}")
        print(f"  Momentum: {home}={hs['momentum']}  {away}={as_['momentum']}")

        blocks, zone = apply_filters(
            home, away, pos_h, pos_a, odds_p1, odds_p2,
            total_teams, hs, as_, r, h2h
        )

        if blocks:
            print(f"\n  ФИЛЬТРЫ ПРОТОКОЛА:")
            for b in blocks:
                print(f"  {b}")

        hard = [b for b in blocks if b.startswith("⛔")]
        if hard:
            print(f"\n  🚫 МАТЧ ЗАБЛОКИРОВАН ПРОТОКОЛОМ\n")
        else:
            print(f"\n  РЕКОМЕНДАЦИЯ:")
            print(verdict(r, home, away, zone, blocks))
            print()

    print("="*62)
    print("  Анализ завершён  |  Протокол v25.3 + H2H")
    print("="*62)

# ══════════════════════════════════════════════
# МАТЧИ НА СЕГОДНЯ
# Формат: ("Хозяин", "Гость", "Лига", Поз_х, Поз_г, КФ_П1, КФ_Х, КФ_П2)
# ══════════════════════════════════════════════
matches = [
    ("Birmingham",      "QPR",             "Championship", 13, 14, 1.95, 3.40, 3.80),
    ("Norwich",         "Sheffield United", "Championship", 12, 11, 2.10, 3.20, 3.50),
    ("Oxford",          "Blackburn",        "Championship", 15, 16, 2.70, 3.10, 2.70),
]

analyze(matches, total_teams=24)