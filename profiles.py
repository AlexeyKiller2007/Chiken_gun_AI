"""Во что играет муха: BlueStacks (эмулятор) или десктопное приложение.

Профиль меняет значения в config.py при запуске, поэтому всё, что в нём не
упомянуто, берётся из config.py как обычно.
"""
import json
import re
import sys

import config
import winutil

GAMES = {
    "bluestacks": "BlueStacks (эмулятор)",
    "desktop": "Десктопное приложение",
}
# старые названия из прошлых версий
ALIASES = {"chicken_gun": "bluestacks", "roblox": "desktop"}
NAMES = list(GAMES) + list(ALIASES)

LAST_WINDOW_FILE = config.DATA / "last_window.json"

# Что уже известно о некоторых программах: красные зоны и подсказка про мышь.
# Для остальных зон нет — нарисуй их мышкой в окне Fly view и нажми F6.
KNOWN_APPS = {
    "robloxplayerbeta.exe": {
        "masks": [
            (0.00, 0.00, 0.30, 0.10),   # кнопка меню Roblox и чат
            (0.78, 0.00, 1.00, 0.35),   # список игроков
            (0.25, 0.86, 0.75, 1.00),   # панель предметов и здоровье
        ],
        "hint": "Включи в Roblox Shift Lock (Настройки -> Shift Lock Switch, "
                "потом клавиша Shift в игре), чтобы камера крутилась мышью.",
    },
}


def pick_window(title_part, ask):
    """Какое окно отдать мухе: (hwnd, заголовок, процесс) или None.
    title_part — часть заголовка из флага --window; иначе спрашиваем,
    а Enter выбирает то же окно, что в прошлый раз."""
    windows = winutil.app_windows()
    if title_part:
        for w in windows:
            if title_part.lower() in w[1].lower():
                return w
        print(f"Не нашёл окно, в заголовке которого есть «{title_part}».")
        return None

    last = None
    if LAST_WINDOW_FILE.exists():
        saved = json.loads(LAST_WINDOW_FILE.read_text(encoding="utf-8"))
        same_app = [w for w in windows if w[2].lower() == saved["process"].lower()]
        exact = [w for w in same_app if w[1] == saved["title"]]
        last = (exact or same_app or [None])[0]
    if not ask:
        if last is None:
            print("Не знаю, какое окно брать: запусти с --window \"часть заголовка\".")
        return last
    if not windows:
        print("Не вижу ни одного окна программ.")
        return None

    print("Какое окно отдать мухе? Сначала открой игру и зайди в неё.")
    for i, (hwnd, title, process) in enumerate(windows, 1):
        note = "  (свёрнуто)" if winutil.user32.IsIconic(hwnd) else ""
        print(f"  {i} — {title[:60]}  [{process}]{note}")
    hint = f"Enter = «{last[1][:40]}»" if last else "номер"
    while True:
        try:
            answer = input(f"Выбор ({hint}): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not answer and last:
            return last
        if answer.isdigit() and 1 <= int(answer) <= len(windows):
            return windows[int(answer) - 1]
        print(f"Напиши число от 1 до {len(windows)}.")


def apply_desktop(window):
    """Обычная программа Windows: клавиши как в играх на ПК, взгляд — мышью."""
    c = config
    hwnd, title, process = window
    c.WINDOW_HWND = hwnd
    c.WINDOW_TITLE = title
    c.WINDOW_PROCESS = process
    c.WINDOW_MISSING = "Открой программу снова и перезапусти муху."
    c.CROP = {"top": 0, "bottom": 0, "left": 0, "right": 0}   # берём внутреннюю часть окна целиком
    LAST_WINDOW_FILE.write_text(json.dumps({"title": title, "process": process}, ensure_ascii=False),
                                encoding="utf-8")

    # Красные зоны — свои для каждой программы (подвинь мышкой в окне Fly view)
    known = KNOWN_APPS.get(process.lower(), {})
    c.HUD_MASKS = list(known.get("masks", []))
    c.HUD_MASKS_DEFAULT = list(c.HUD_MASKS)
    app = re.sub(r"[^\w-]+", "_", process.lower().removesuffix(".exe")) or "app"
    c.MASKS_FILE = c.DATA / f"hud_masks_{app}.json"
    if c.MASKS_FILE.exists():
        c.HUD_MASKS = [tuple(m) for m in json.loads(c.MASKS_FILE.read_text(encoding="utf-8"))]

    # Камера в играх на ПК крутится мышью, кнопки — стандартные
    c.MOUSE_LOOK = True
    c.MOUSE_HINT = known.get("hint", "Если камера в игре крутится только с зажатой кнопкой "
                                     "или в режиме прицела — включи этот режим.")
    c.LOOK_AFTER_MOVE = False          # на ПК можно смотреть по сторонам на ходу
    controls = c.CONTROLS
    c.CONTROLS = {key: controls[key] for key in ("w", "s", "a", "d", "v", "n", "c", "b", "space")
                  if key in controls}
    # детектор надвигания -> левая кнопка мыши (удар, выстрел, предмет в руке)
    c.CONTROLS["lmb"] = controls.get("y", [("DNp70", None)])
    c.THRESHOLD_HZ["lmb"] = c.THRESHOLD_HZ.get("y", 10)
    c.TAP_KEYS = {"space"}
    c.COOLDOWN_S = {"space": 1}
    c.TOGGLE_BACK_S = {}

    # Предметы — на цифрах 1–5 сверху, без отдельного меню
    c.WEAPON_MENU_KEY = None
    c.WEAPON_VOTERS = {str(i + 1): spec for i, spec in enumerate(c.WEAPON_VOTERS.values())}
    c.WEAPON_NAMES = {str(i + 1): f"предмет {i + 1}" for i in range(len(c.WEAPON_VOTERS))}
    c.WEAPON_REASONS = {str(i + 1): reason for i, reason in enumerate(c.WEAPON_REASONS.values())}

    # Экран смерти и меню с пропами умеем узнавать только в Chicken Gun
    c.RESPAWN_ENABLED = False
    c.KEYTEST_HERO = "персонаж"
    c.KEYTEST_SEQUENCE = [("d", "идёт вправо"), ("a", "идёт влево"), ("w", "идёт вперёд"),
                          ("s", "идёт назад"), ("space", "прыгает"), ("1", "берёт предмет 1")]


def apply(game, window=None):
    game = ALIASES.get(game, game)
    if game == "desktop":
        apply_desktop(window)
    config.GAME = game


def choose(chosen, ask):
    """Во что играем. Спрашивает при запуске, если не задано флагом."""
    chosen = ALIASES.get(chosen, chosen)
    default = ALIASES.get(config.GAME, config.GAME)
    if not chosen and ask:
        print("Во что играем?")
        for i, (key, title) in enumerate(GAMES.items(), 1):
            print(f"  {i} — {title}")
        try:
            answer = input(f"Выбор (Enter = {GAMES[default]}): ").strip()
        except (EOFError, KeyboardInterrupt):
            answer = ""
            print()
        keys = list(GAMES)
        if answer.isdigit() and 1 <= int(answer) <= len(keys):
            chosen = keys[int(answer) - 1]
    return chosen or default


def setup(chosen, window_title, ask):
    """Выбрать игру (и окно для десктопа) и применить профиль. None — выбора нет."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # в заголовках окон бывают эмодзи
    game = choose(chosen, ask)
    window = None
    if game == "desktop":
        window = pick_window(window_title, ask)
        if window is None:
            return None
    apply(game, window)
    where = f": «{window[1]}» [{window[2]}]" if window else ""
    print(f"Играем: {GAMES[game]}{where}.")
    return game


def open_window():
    return winutil.GameWindow(config.WINDOW_TITLE, config.WINDOW_PROCESS, config.CROP,
                              hwnd=config.WINDOW_HWND)
