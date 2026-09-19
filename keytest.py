"""Проверка клавиш без мухи: персонаж шагает вправо-влево и оглядывается.

Запусти keytest.bat и кликни в окно игры.
    keytest.bat                      # спросит, во что играем
    keytest.bat bluestacks
    keytest.bat desktop "Roblox"     # окно, в заголовке которого есть «Roblox»
"""
import time

import config
import winutil


def main():
    import sys
    import profiles
    winutil.make_dpi_aware()
    game = profiles.setup(sys.argv[1] if len(sys.argv) > 1 else None,
                          sys.argv[2] if len(sys.argv) > 2 else None, sys.stdin.isatty())
    if game is None:
        return
    window = profiles.open_window()
    if not window.found():
        print(f"Не нашёл окно «{config.WINDOW_TITLE}». {config.WINDOW_MISSING}")
        return

    print("Кликни в окно игры (жду 15 секунд)...")
    deadline = time.time() + 15
    while not window.focused():
        if time.time() > deadline:
            print("Не дождался окна игры.")
            return
        time.sleep(0.1)
    time.sleep(0.5)

    for key, where in config.KEYTEST_SEQUENCE:
        if not window.focused():
            print("Окно игры больше не активно, останавливаюсь.")
            return
        print(f"Жму {key.upper()}: {config.KEYTEST_HERO} {where}")
        winutil.send_key(key, True)
        time.sleep(0.15 if key in config.TAP_KEYS else 0.7)
        winutil.send_key(key, False)
        time.sleep(0.8)
    print(f"Готово. Если {config.KEYTEST_HERO} стоял(а), проверь, что W A S D работают "
          f"с обычной клавиатуры.")


if __name__ == "__main__":
    main()
