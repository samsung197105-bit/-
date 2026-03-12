# -*- coding: utf-8 -*-
import requests
import pandas as pd
import numpy as np
from io import StringIO
from scipy.stats import poisson
import json
import os

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

PENDING_FILE = "C:/football_ai/data/pending_bets.json"

def load_league(code, season):
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
    r = requests.get(url, timeout=10)
    df = pd.read_csv(StringIO(r.text))
    df = df.dropna(subset=["HomeTeam", "FTHG", "FTAG"])
    return df

def get_h2h(home, away, league, current_df, n=5):
    frames = [current_df]
    if league in H2H_LEAGUES:
        for code, season in H2H_LEAGUES[league]:
            try:
                frames.append(load_league(code, season))
            except:
                pass
    all_data = pd.concat(frames, ignore_index=True)
    h2h = all_data[
        ((all_data["HomeTeam"] == home) & (all_data["AwayTeam"] == away)) |
        ((all_data["HomeTeam"] == away) & (all_data["AwayTeam"] == home))
    ].tail(n)
    if len(h2h) == 0:
        return None
    res = {"home_wins":0,"draws":0,"away_wins":0,
           "avg_goals":0,"btts_count":0,"over25":0,"matches":[]}
    for _, row in h2h.iterrows():
        hg, ag = int(row["FTHG"]), int(row["FTAG"])
        total  = hg + ag
        if row["HomeTeam"] == home:
            if hg > ag: res["home_wins"] += 1
            elif hg == ag: res["draws"] += 1
            else: res["away_wins"] += 1
            res["matches"].append(f"{hg}:{ag}")
        else:
            if ag > hg: res["home_wins"] += 1
            elif hg == ag: res["draws"] += 1
            else: res["away_wins"] += 1
            res["matches"].append(f"{ag}:{hg}")
        res["avg_goals"] += total
        if hg > 0 and ag > 0: res["btts_count"] += 1
        if total > 2: res["over25"] += 1
    cnt = len(h2h)
    res["avg_goals"]  = round(res["avg_goals"] / cnt, 2)
    res["btts_pct"]   = round(res["btts_count"] / cnt * 100)
    res["over25_pct"] = round(res["over25"] / cnt * 100)
    res["total"]      = cnt
    res["xg_adj"]     = round((res["avg_goals"] / 2.5 - 1) * 0.15, 3)
    res["psych_lock"] = res["home_wins"] == 0 and cnt >= 4
    return res

def team_stats_split(df, team, n=6):
    hm = df[df["HomeTeam"] == team].tail(n)
    am = df[df["AwayTeam"] == team].tail(n)
    if len(hm) == 0 and len(am) == 0:
        return None

    def calc(matches, gf_col, ga_col, sf_col, sa_col, is_home):
        gf = list(matches[gf_col]) if len(matches) > 0 else []
        ga = list(matches[ga_col]) if len(matches) > 0 else []
        try:
            sf = [x for x in list(matches[sf_col]) if pd.notna(x)]
            sa = [x for x in list(matches[sa_col]) if pd.notna(x)]
        except:
            sf, sa = [], []
        xg_for = round((np.mean(gf)*0.6 + (np.mean(sf)*0.1 if sf else 0)), 2) if gf else None
        xg_ag  = round((np.mean(ga)*0.6 + (np.mean(sa)*0.1 if sa else 0)), 2) if ga else None
        ga_avg = round(np.mean(ga), 2) if ga else 0
        pts = []
        for _, row in matches.iterrows():
            hg, ag = row["FTHG"], row["FTAG"]
            if is_home:
                if hg > ag: pts.append(3)
                elif hg == ag: pts.append(1)
                else: pts.append(0)
            else:
                if ag > hg: pts.append(3)
                elif hg == ag: pts.append(1)
                else: pts.append(0)
        momentum = round(np.mean(pts[-5:]), 2) if pts else 1.0
        no_goal  = len(gf) >= 2 and gf[-1] == 0 and gf[-2] == 0
        return {"xg_for":xg_for,"xg_ag":xg_ag,"ga_avg":ga_avg,
                "momentum":momentum,"no_goal_streak":no_goal,
                "last5_goals":gf[-5:],"n_matches":len(matches)}

    hs  = calc(hm, "FTHG","FTAG","HST","AST", True)
    as_ = calc(am, "FTAG","FTHG","AST","HST", False)
    all_pts = []
    for _, row in hm.iterrows():
        hg, ag = row["FTHG"], row["FTAG"]
        if hg > ag: all_pts.append(3)
        elif hg == ag: all_pts.append(1)
        else: all_pts.append(0)
    for _, row in am.iterrows():
        hg, ag = row["FTHG"], row["FTAG"]
        if ag > hg: all_pts.append(3)
        elif hg == ag: all_pts.append(1)
        else: all_pts.append(0)
    momentum_all = round(np.mean(all_pts[-5:]), 2) if all_pts else 1.0
    all_goals    = list(hm["FTHG"]) + list(am["FTAG"])
    no_goal_all  = len(all_goals) >= 2 and all_goals[-1] == 0 and all_goals[-2] == 0
    return {
        "home": hs, "away": as_,
        "xg_for": hs["xg_for"] or as_["xg_for"] or 1.1,
        "xg_ag":  hs["xg_ag"]  or as_["xg_ag"]  or 1.0,
        "ga_avg": hs["ga_avg"],
        "momentum": momentum_all,
        "no_goal_streak": no_goal_all,
        "last5_goals": all_goals[-5:]
    }

def calc_xg_split(hs, as_):
    h = hs["home"]
    a = as_["away"]
    h_att = h["xg_for"] if h and h["xg_for"] else hs["xg_for"]
    a_def = a["xg_ag"]  if a and a["xg_ag"]  else as_["xg_ag"]
    a_att = a["xg_for"] if a and a["xg_for"] else as_["xg_for"]
    h_def = h["xg_ag"]  if h and h["xg_ag"]  else hs["xg_ag"]
    return (max(round((h_att + a_def) / 2, 2), 0.3),
            max(round((a_att + h_def) / 2, 2), 0.3))

def predict(hxg, axg):
    max_g = 6
    prob  = np.zeros((max_g+1, max_g+1))
    for i in range(max_g+1):
        for j in range(max_g+1):
            prob[i][j] = poisson.pmf(i, hxg) * poisson.pmf(j, axg)
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

# ============================================================
#  МЯГКИЕ ФИЛЬТРЫ - уровни уверенности
# ============================================================
RISKS = {
    "TABLE_LOCK_TOP": {
        "label": "TABLE LOCK (топ-2)",
        "danger": "Лидеры часто играют нестандартно: ротация состава, осторожная тактика перед кубком/еврокубком. Статистика xG может не отражать реальную мотивацию.",
        "impact": "HIGH"
    },
    "TABLE_LOCK_BOT": {
        "label": "TABLE LOCK (зона вылета)",
        "danger": "Команды в зоне вылета играют с максимальной мотивацией и часто нарушают прогнозы. Возможен навал, стандарты, удаления.",
        "impact": "HIGH"
    },
    "ODDS_SYNC_DOM": {
        "label": "ODDS SYNC (доминирование)",
        "danger": "Букмекер ставит явного фаворита (КФ < 2.0), но модель даёт ТМ. Противоречие: фаворит атакует много, что увеличивает голы. Риск ТБ вырастает.",
        "impact": "HIGH"
    },
    "ODDS_SYNC_BAL": {
        "label": "ODDS SYNC (равновесие)",
        "danger": "Оба КФ > 2.0 — равная игра. Модель ожидает мало голов, но равные команды часто играют открыто. Возможен неожиданный результат.",
        "impact": "MEDIUM"
    },
    "MOMENTUM": {
        "label": "MOMENTUM андердога",
        "danger": "Андердог забивал 3 матча подряд. Серия атак может продолжиться — ТМ под угрозой. Вероятность гола от аутсайдера выше нормы.",
        "impact": "HIGH"
    },
    "GA_GATE": {
        "label": "ШЛЮЗ GA (мало пропускают)",
        "danger": "Обе команды пропускают мало (< 2.5 суммарно). Матч может быть закрытым и тактическим. ТБ и BTTS под вопросом.",
        "impact": "MEDIUM"
    },
    "DRY_RUN": {
        "label": "СУХАЯ СЕРИЯ",
        "danger": "Команда не забивала 2 матча подряд. Атака в кризисе или соперник нейтрализует. Вероятность 0 голов от этой команды повышена.",
        "impact": "MEDIUM"
    },
    "H2H_PSYCH": {
        "label": "H2H ПСИХО-БЛОК",
        "danger": "Хозяин не выиграл ни разу в последних встречах с этим соперником. Психологическое давление реально влияет на игру — возможна ничья или поражение.",
        "impact": "HIGH"
    },
}

def apply_filters_soft(home, away, pos_h, pos_a, odds_p1, odds_p2,
                       total_teams, hs, as_, r, h2h):
    warnings = []   # MEDIUM риски
    dangers  = []   # HIGH риски

    def add(key, extra=""):
        info = RISKS[key]
        msg  = f"[{info['impact']}] {info['label']}"
        if extra:
            msg += f" ({extra})"
        msg += f"\n         ОПАСНОСТЬ: {info['danger']}"
        if info["impact"] == "HIGH":
            dangers.append(msg)
        else:
            warnings.append(msg)

    # TABLE LOCK
    if pos_h <= 2:
        add("TABLE_LOCK_TOP", f"{home} = {pos_h} место")
    if pos_a <= 2:
        add("TABLE_LOCK_TOP", f"{away} = {pos_a} место")
    if pos_h >= total_teams - 2:
        add("TABLE_LOCK_BOT", f"{home} = {pos_h} место")
    if pos_a >= total_teams - 2:
        add("TABLE_LOCK_BOT", f"{away} = {pos_a} место")

    # ODDS SYNC
    zone = None
    if odds_p1 > 2.00 and odds_p2 > 2.00:
        zone = "BALANCE"
        if r["tb25"] >= 0.55:
            add("ODDS_SYNC_BAL")
    elif odds_p1 < 2.00 or odds_p2 < 2.00:
        zone = "DOMINATION"
        if r["tm35"] >= 0.72 and r["tb15"] < 0.72:
            add("ODDS_SYNC_DOM")

    # MOMENTUM
    if odds_p1 > 3.0 and hs:
        last3 = hs["last5_goals"][-3:]
        if len(last3) == 3 and all(g > 0 for g in last3):
            add("MOMENTUM", f"{home} (андердог)")
    if odds_p2 > 3.0 and as_:
        last3 = as_["last5_goals"][-3:]
        if len(last3) == 3 and all(g > 0 for g in last3):
            add("MOMENTUM", f"{away} (андердог)")

    # GA GATE
    if hs and as_:
        ga_total = hs["ga_avg"] + as_["ga_avg"]
        if ga_total < 2.5:
            add("GA_GATE", f"GA={ga_total:.2f}")

    # DRY RUN
    if hs and hs["no_goal_streak"]:
        add("DRY_RUN", f"{home}")
    if as_ and as_["no_goal_streak"]:
        add("DRY_RUN", f"{away}")

    # H2H PSYCH
    if h2h and h2h["psych_lock"]:
        add("H2H_PSYCH", f"{home} 0W в {h2h['total']} встречах")

    # Уровень уверенности
    if len(dangers) == 0 and len(warnings) == 0:
        confidence = "HIGH"
        conf_label = "HIGH  [====]  Все фильтры OK. Ставь смело."
    elif len(dangers) == 0 and len(warnings) > 0:
        confidence = "MEDIUM"
        conf_label = "MEDIUM [===]  Есть предупреждения. Ставь осторожно."
    elif len(dangers) == 1:
        confidence = "LOW"
        conf_label = "LOW   [==]   Есть серьёзный риск. Уменьши ставку."
    else:
        confidence = "RISKY"
        conf_label = "RISKY  [=]   Несколько рисков. Рассмотри как информацию."

    return warnings, dangers, confidence, conf_label, zone

def auto_save_bet(home, away, league, r, odds_x, confidence):
    if r["tm35"] >= 0.72:
        bet_type, prob, odds = "TM3.5", r["tm35"], 1.17
    elif r["tb15"] >= 0.72:
        bet_type, prob, odds = "TB1.5", r["tb15"], 1.17
    elif r["btts"] >= 0.60:
        bet_type, prob, odds = "BTTS",  r["btts"], 1.85
    elif r["tb25"] >= 0.55:
        bet_type, prob, odds = "TB2.5", r["tb25"], 1.85
    else:
        bet_type, prob, odds = "1X",    r["p1"]+r["draw"], odds_x
    pending = []
    if os.path.exists(PENDING_FILE):
        with open(PENDING_FILE, "r", encoding="utf-8") as f:
            pending = json.load(f)
    if not any(e["match"] == f"{home} vs {away}" for e in pending):
        pending.append({
            "match":       f"{home} vs {away}",
            "league":      league,
            "bet_type":    bet_type,
            "probability": round(prob, 3),
            "odds":        odds,
            "confidence":  confidence,
            "result":      None
        })
        os.makedirs(os.path.dirname(PENDING_FILE), exist_ok=True)
        with open(PENDING_FILE, "w", encoding="utf-8") as f:
            json.dump(pending, f, ensure_ascii=False, indent=2)
        print(f"  [SAVED] {bet_type}  {prob*100:.1f}%  confidence={confidence}")

def btts_signal(btts_prob, h2h):
    h2h_btts = h2h["btts_pct"] if h2h else 50
    combined = round((btts_prob * 100 + h2h_btts) / 2, 1)
    if combined >= 60:   level = "HIGH  ++++"
    elif combined >= 45: level = "MED   +++"
    else:                level = "LOW   ++"
    return (f"  BTTS: model {btts_prob*100:.1f}%"
            f"  H2H {h2h_btts}%"
            f"  total {combined}%  [{level}]")

def show_recommendation(r, home, away, zone, h2h,
                        warnings, dangers, confidence, conf_label):
    lines = []

    # Уровень уверенности
    lines.append(f"\n  {'='*56}")
    lines.append(f"  CONFIDENCE: {conf_label}")
    lines.append(f"  {'='*56}")

    # Риски
    if dangers:
        lines.append(f"\n  РИСКИ HIGH:")
        for d in dangers:
            lines.append(f"    ! {d}")
    if warnings:
        lines.append(f"\n  РИСКИ MEDIUM:")
        for w in warnings:
            lines.append(f"    ~ {w}")

    lines.append(f"\n  ПРОГНОЗ:")

    # Исход
    if r["p1"] > 0.50:             outcome = f"P1 {home}"
    elif r["p2"] > 0.50:           outcome = f"P2 {away}"
    elif r["p1"]+r["draw"] > 0.65: outcome = "1X (P1 or Draw)"
    elif r["p2"]+r["draw"] > 0.65: outcome = "X2 (P2 or Draw)"
    else:                           outcome = "1X (P1 or Draw)"

    lines.append(f"  Outcome:    {outcome}")
    lines.append(f"  1X / X2:    1X {r['p1']+r['draw']:.1%}  |  X2 {r['p2']+r['draw']:.1%}")

    if zone == "BALANCE":
        lines.append(f"  Total:      TM 3.5 ({r['tm35']:.1%})  [Balance zone]")
    elif zone == "DOMINATION":
        lines.append(f"  Total:      TB 1.5 ({r['tb15']:.1%})  [Domination zone]")
    else:
        if r["tb25"] >= 0.55:
            lines.append(f"  Total 2.5:  TB 2.5 ({r['tb25']:.1%})")
        else:
            lines.append(f"  Total 2.5:  TM 2.5 ({1-r['tb25']:.1%})")

    lines.append(btts_signal(r["btts"], h2h))

    lines.append(f"  Top scores:")
    for i,j,p in r["top3"]:
        lines.append(f"    {i}:{j}  {p:.1%}")

    # Экспресс только для HIGH и MEDIUM
    if confidence in ["HIGH", "MEDIUM"]:
        express = []
        if r["tb15"] >= 0.72: express.append(f"TB 1.5  odds ~1.17  ({r['tb15']:.1%})")
        if r["tm35"] >= 0.72: express.append(f"TM 3.5  odds ~1.17  ({r['tm35']:.1%})")
        if r["btts"] >= 0.60: express.append(f"BTTS    odds ~1.85  ({r['btts']:.1%})")
        if express:
            lines.append(f"\n  ===== EXPRESS ({confidence}) =====")
            for e in express:
                lines.append(f"    -> {e}")
    else:
        lines.append(f"\n  [Express отключен для {confidence} — слишком высокий риск]")

    return "\n".join(lines)

def analyze(matches, total_teams=24):
    print("\n" + "="*60)
    print("  FORECAST  v25.3 + SOFT FILTERS + SPLIT xG  |  2025-26")
    print("="*60)

    if os.path.exists(PENDING_FILE):
        os.remove(PENDING_FILE)

    data   = {}
    needed = set(m[2] for m in matches)
    for league in needed:
        if league in LEAGUES:
            code, season = LEAGUES[league]
            print(f"  Loading {league}...", end=" ", flush=True)
            try:
                data[league] = load_league(code, season)
                print(f"{len(data[league])} matches OK")
            except:
                print("ERROR")

    print()
    summary = []  # итоговая сводка

    for match in matches:
        home, away, league = match[0], match[1], match[2]
        pos_h   = match[3] if len(match) > 3 else 10
        pos_a   = match[4] if len(match) > 4 else 10
        odds_p1 = match[5] if len(match) > 5 else 2.50
        odds_x  = match[6] if len(match) > 6 else 3.20
        odds_p2 = match[7] if len(match) > 7 else 2.50

        print("-"*60)
        print(f"  MATCH:  {home}  vs  {away}  [{league}]")
        print(f"  POS: {pos_h} vs {pos_a}   ODDS: {odds_p1} / {odds_x} / {odds_p2}")

        if league not in data:
            print(f"  ERROR: League not loaded\n")
            continue

        df  = data[league]
        hs  = team_stats_split(df, home)
        as_ = team_stats_split(df, away)

        if hs is None:
            print(f"  '{home}' not found! Teams: {sorted(df['HomeTeam'].unique())}\n")
            continue
        if as_ is None:
            print(f"  '{away}' not found! Teams: {sorted(df['HomeTeam'].unique())}\n")
            continue

        hh = hs["home"]
        aa = as_["away"]
        print(f"  HOME {home}:  att {hh['xg_for']}  def {hh['xg_ag']}  "
              f"mom {hh['momentum']}  ({hh['n_matches']}g)")
        print(f"  AWAY {away}:  att {aa['xg_for']}  def {aa['xg_ag']}  "
              f"mom {aa['momentum']}  ({aa['n_matches']}g)")

        print(f"  H2H...", end=" ", flush=True)
        h2h = get_h2h(home, away, league, df)
        if h2h:
            print(f"{h2h['total']} matches")
            print(f"  H2H: {home} {h2h['home_wins']}W/{h2h['draws']}D/{h2h['away_wins']}W {away}  "
                  f"scores: {' | '.join(h2h['matches'])}")
            print(f"  H2H: avg={h2h['avg_goals']}  BTTS={h2h['btts_pct']}%  Over2.5={h2h['over25_pct']}%")
        else:
            print("no data")

        hxg, axg = calc_xg_split(hs, as_)
        if h2h:
            hxg = max(round(hxg + h2h["xg_adj"], 2), 0.3)
            axg = max(round(axg + h2h["xg_adj"], 2), 0.3)

        print(f"  xG SPLIT: {hxg} vs {axg}")
        r = predict(hxg, axg)
        print(f"  P1:{r['p1']:.1%}  X:{r['draw']:.1%}  P2:{r['p2']:.1%}  "
              f"U3.5:{r['tm35']:.1%}  O1.5:{r['tb15']:.1%}  O2.5:{r['tb25']:.1%}")

        warnings, dangers, confidence, conf_label, zone = apply_filters_soft(
            home, away, pos_h, pos_a, odds_p1, odds_p2,
            total_teams, hs, as_, r, h2h
        )

        print(show_recommendation(r, home, away, zone, h2h,
                                  warnings, dangers, confidence, conf_label))

        auto_save_bet(home, away, league, r, odds_x, confidence)

        # Добавляем в сводку
        if r["tm35"] >= 0.72:
            best = f"TM3.5 {r['tm35']:.1%}"
        elif r["tb15"] >= 0.72:
            best = f"TB1.5 {r['tb15']:.1%}"
        elif r["btts"] >= 0.60:
            best = f"BTTS {r['btts']:.1%}"
        else:
            best = f"TM2.5 {1-r['tb25']:.1%}"
        summary.append((confidence, home, away, best))
        print()

    # Итоговая сводка дня
    print("="*60)
    print("  SUMMARY OF THE DAY")
    print("="*60)
    order = {"HIGH":0, "MEDIUM":1, "LOW":2, "RISKY":3}
    summary.sort(key=lambda x: order.get(x[0], 4))
    for conf, h, a, bet in summary:
        print(f"  [{conf:6s}]  {h} vs {a}  ->  {bet}")
    print("="*60)
    print("  Analysis complete  |  v25.3 + SOFT FILTERS + SPLIT xG")
    print("="*60)

# ============================================================
# TODAY'S MATCHES
# Format: ("Home","Away","League", Pos_H,Pos_A, Odds_1,Odds_X,Odds_2)
# ============================================================
matches = [
    ("Birmingham",      "QPR",             "Championship", 13, 14, 1.95, 3.40, 3.80),
    ("Norwich",         "Sheffield United", "Championship", 12, 11, 2.10, 3.20, 3.50),
    ("Oxford",          "Blackburn",        "Championship", 15, 16, 2.70, 3.10, 2.70),
]

analyze(matches, total_teams=24)