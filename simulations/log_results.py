# -*- coding: utf-8 -*-
import json
import os
from datetime import datetime

LOG_FILE     = "C:/football_ai/data/results_log.json"
PENDING_FILE = "C:/football_ai/data/pending_bets.json"

def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_log(log):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

def load_pending():
    if os.path.exists(PENDING_FILE):
        with open(PENDING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def enter_results():
    pending = load_pending()
    if not pending:
        print("\n  Нет сохраненных ставок в pending_bets.json")
        print("  Сначала запусти predict_today.py")
        return

    log  = load_log()
    date = input("\n  Дата матчей (Enter = сегодня, или DD.MM.YYYY): ").strip()
    if not date:
        date = datetime.now().strftime("%d.%m.%Y")

    print(f"\n  Вводим результаты за {date}")
    print("  Формат счета: 1:0  или  2:2  или  0:3\n")

    for bet in pending:
        match      = bet["match"]
        bet_type   = bet["bet_type"]
        prob       = bet["probability"]
        confidence = bet.get("confidence", "?")
        league     = bet.get("league", "?")

        print(f"  {match}  [{league}]")
        print(f"  Ставка: {bet_type}  вер. {prob*100:.1f}%  [{confidence}]")

        score = input(f"  Счет (или Enter = пропустить): ").strip()
        if not score:
            print(f"  Пропущено\n")
            continue

        try:
            hg, ag = map(int, score.split(":"))
        except:
            print(f"  Неверный формат, пропускаем\n")
            continue

        total  = hg + ag
        btts   = hg > 0 and ag > 0

        # Проверяем ставку
        if bet_type == "TM3.5":
            win = total <= 3
        elif bet_type == "TB1.5":
            win = total >= 2
        elif bet_type == "TB2.5":
            win = total >= 3
        elif bet_type == "BTTS":
            win = btts
        elif bet_type == "1X":
            win = hg >= ag
        elif bet_type == "X2":
            win = ag >= hg
        else:
            win = False

        odds = bet.get("odds", 1.17)
        roi  = round((odds - 1) * 100 if win else -100, 1)

        status = "WIN" if win else "LOSS"
        print(f"  {score}  ->  {status}  ROI: {roi:+.1f}%\n")

        log.append({
            "date":       date,
            "match":      match,
            "league":     league,
            "bet_type":   bet_type,
            "probability": prob,
            "confidence": confidence,
            "score":      score,
            "win":        win,
            "odds":       odds,
            "roi":        roi
        })

    save_log(log)
    print(f"  Сохранено в {LOG_FILE}")

def show_dashboard():
    log = load_log()
    if not log:
        print("\n  Нет данных. Сначала введи результаты.")
        return

    print("\n" + "="*58)
    print("   DASHBOARD  |  Football AI  v25.3")
    print("="*58)

    total  = len(log)
    wins   = sum(1 for e in log if e["win"])
    acc    = wins / total * 100
    avg_roi = sum(e["roi"] for e in log) / total

    print(f"  Всего ставок:   {total}")
    print(f"  Побед:          {wins}")
    print(f"  Точность:       {acc:.1f}%")
    print(f"  Средний ROI:    {avg_roi:+.1f}%")

    # По типу ставки
    print(f"\n  --- По типу ставки ---")
    types = {}
    for e in log:
        t = e["bet_type"]
        if t not in types:
            types[t] = {"w":0,"t":0}
        types[t]["t"] += 1
        if e["win"]:
            types[t]["w"] += 1
    for t, v in sorted(types.items()):
        pct = v["w"]/v["t"]*100
        print(f"  {t:8s}: {v['w']}/{v['t']} = {pct:.0f}%")

    # По confidence
    print(f"\n  --- По уровню уверенности ---")
    confs = {}
    for e in log:
        c = e.get("confidence", "?")
        if c not in confs:
            confs[c] = {"w":0,"t":0,"roi":0}
        confs[c]["t"] += 1
        confs[c]["roi"] += e["roi"]
        if e["win"]:
            confs[c]["w"] += 1
    order = ["HIGH","MEDIUM","LOW","RISKY","?"]
    for c in order:
        if c in confs:
            v   = confs[c]
            pct = v["w"]/v["t"]*100
            avg = v["roi"]/v["t"]
            print(f"  {c:8s}: {v['w']}/{v['t']} = {pct:.0f}%  ROI {avg:+.1f}%")

    # По лиге
    print(f"\n  --- По лиге ---")
    leagues = {}
    for e in log:
        lg = e.get("league","?")
        if lg not in leagues:
            leagues[lg] = {"w":0,"t":0}
        leagues[lg]["t"] += 1
        if e["win"]:
            leagues[lg]["w"] += 1
    for lg, v in sorted(leagues.items()):
        pct = v["w"]/v["t"]*100
        print(f"  {lg:15s}: {v['w']}/{v['t']} = {pct:.0f}%")

    # Последние 5
    print(f"\n  --- Последние 5 ставок ---")
    for e in log[-5:]:
        icon = "OK" if e["win"] else "XX"
        print(f"  [{icon}] {e['date']} | {e['match'][:28]:28s} | "
              f"{e['bet_type']:6s} | {e['score']:4s} | "
              f"[{e.get('confidence','?'):6s}] | ROI {e['roi']:+.0f}%")

    print("="*58)

    # Рекомендация по модели
    if total >= 5:
        print(f"\n  --- Анализ модели ---")
        if acc >= 70:
            print(f"  Model OK: точность {acc:.0f}% - продолжай")
        elif acc >= 55:
            print(f"  ВНИМАНИЕ: точность {acc:.0f}% - анализируй ошибки")
        else:
            print(f"  СТОП-СИГНАЛ: точность {acc:.0f}% - пауза 24ч")

        # Лучший confidence
        best_c = max(confs.items(),
                     key=lambda x: x[1]["w"]/x[1]["t"] if x[1]["t"]>=2 else 0)
        print(f"  Лучший уровень: [{best_c[0]}] "
              f"{best_c[1]['w']}/{best_c[1]['t']} = "
              f"{best_c[1]['w']/best_c[1]['t']*100:.0f}%")
        print("="*58)

def main():
    print("\n  Football AI  |  Журнал результатов")
    print("  1 - Ввести результаты матчей")
    print("  2 - Показать dashboard")
    print("  3 - Выход")

    choice = input("\n  Выбор: ").strip()
    if choice == "1":
        enter_results()
    elif choice == "2":
        show_dashboard()
    else:
        print("  Выход")

if __name__ == "__main__":
    main()