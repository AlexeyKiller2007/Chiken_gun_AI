"""Проверка клавиш без мухи: курица шагает вправо-влево и оглядывается.

Запусти keytest.bat и кликни в окно игры.
"""
import time

import config
import winutil


def main():
    winutil.make_dpi_aware()
    window = winutil.GameWindow(config.WINDOW_TITLE, config.WINDOW_PROCESS, config.CROP)
    if not window.found():
        print(f"Не нашёл окно «{config.WINDOW_TITLE}». Запусти BlueStacks и попробуй снова.")
        return

    print("Кликни в окно игры (жду 15 секунд)...")
    deadline = time.time() + 15
    while not window.focused():
        if time.time() > deadline:
            print("Не дождался окна игры.")
            return
        time.sleep(0.1)
    time.sleep(0.5)

    for key, where in (("d", "шагает вправо"), ("a", "шагает влево"),
                       ("n", "смотрит вправо"), ("v", "смотрит влево"),
                       ("w", "идёт вперёд"), ("s", "идёт назад"),
                       ("space", "прыгает"), ("x", "открывает меню"), ("x", "закрывает меню"),
                       ("z", "включает вторую камеру"), ("z", "возвращает камеру"),
                       ("tab", "открывает меню оружия"), ("num3", "берёт автомат")):
        if not window.focused():
            print("Окно игры больше не активно, останавливаюсь.")
            return
        print(f"Жму {key.upper()}: курица {where}")
        winutil.send_key(key, True)
        time.sleep(0.15 if key in config.TAP_KEYS else 0.7)
        winutil.send_key(key, False)
        time.sleep(0.8)
    print("Готово. Если курица стояла, проверь, что W A S D работают с обычной клавиатуры.")


if __name__ == "__main__":
    main()
