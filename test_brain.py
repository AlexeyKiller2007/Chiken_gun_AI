"""Проверка мозга без игры: показываем мухе искусственные картинки.

    .venv\\Scripts\\python test_brain.py

Печатает частоты нейронов каждой клавиши и самые активные нисходящие нейроны.
Помогает подобрать пороги в config.py.
"""
import time

import numpy as np

import config
import winutil
from brain import FlyBrain
from eye import FlyEye
from motor import Motor

W, H = config.EYE_GRID
FRAMES = 10


def gray(t):
    return np.full((H, W), 0.5, np.float32)


def flash(t):
    return np.full((H, W), 0.8 if t % 2 else 0.2, np.float32)


def looming(t):
    img = gray(t)
    yy, xx = np.mgrid[0:H, 0:W]
    r = 1 + (t / (FRAMES - 1)) ** 3 * W * 0.6
    img[(xx - W / 2) ** 2 + (yy - H / 2) ** 2 < r * r] = 0.0
    return img


def bar(x0, x1):
    def stim(t):
        img = gray(t)
        x = int(x0 + (x1 - x0) * t / (FRAMES - 1))
        img[:, max(0, x - 2):x + 2] = 0.0
        return img
    return stim


def flicker(x0, x1):
    def stim(t):
        img = gray(t)
        img[:, x0:x1] = 0.9 if t % 2 else 0.1
        return img
    return stim


STIMULI = {
    "серый экран": gray,
    "вспышки на всём экране": flash,
    "надвигается тёмный объект": looming,
    "полоса едет слева": bar(0, W // 2),
    "полоса едет справа": bar(W - 1, W // 2),
    "мерцание слева": flicker(0, W // 2),
    "мерцание справа": flicker(W // 2, W),
}


def main():
    winutil.boost_process()
    t0 = time.time()
    brain = FlyBrain()
    eye = FlyEye(brain)
    motor = Motor(brain)
    print(f"мозг загружен за {time.time() - t0:.1f} с, нейронов: {brain.n}\n")

    dn = brain.super_class == "descending"
    keys = list(config.CONTROLS)
    print(f"{'стимул':28s}" + "".join(f"{k:>7s}" for k in keys) + "   активных нейронов  скорость")
    for name, stim in STIMULI.items():
        brain.reset()
        eye.prev = None
        counts = np.zeros(brain.n, np.int32)
        t0 = time.time()
        for t in range(FRAMES):
            brain.set_eye_rates(eye.rates(stim(t)))
            counts += brain.run(config.BRAIN_MS_PER_FRAME)
        wall = time.time() - t0
        ms = FRAMES * config.BRAIN_MS_PER_FRAME
        hz = motor.rates(counts, ms)
        speed = ms / 1000 / wall
        print(f"{name:28s}" + "".join(f"{hz[k]:7.1f}" for k in keys)
              + f"   {np.count_nonzero(counts):>8d}         x{speed:.2f}")

        # самые активные нисходящие типы
        types = {}
        for i in np.flatnonzero(dn & (counts > 0)):
            types.setdefault(brain.cell_type[i] or "?", []).append(counts[i])
        top = sorted(types.items(), key=lambda kv: -np.mean(kv[1]))[:6]
        if top:
            print("    нисходящие: " + ", ".join(
                f"{t} {np.mean(c) * 1000 / ms:.0f}Гц" for t, c in top))
    print("\nскорость x1.00 = мозг считается в реальном времени")


if __name__ == "__main__":
    main()
