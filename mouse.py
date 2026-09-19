"""Мышь мухи: плавный поворот камеры по сигналу нейронов поворота."""
import ctypes
import threading
import time
from ctypes import wintypes

import config
import winutil


class MouseLook(threading.Thread):
    """Двигает мышь с заданной скоростью маленькими шагами ~100 раз в секунду.
    Работает, только пока включена и курсор внутри окна игры."""

    STEP_S = 0.01

    def __init__(self):
        super().__init__(daemon=True)
        self.vx = self.vy = 0.0        # пикселей в секунду
        self.enabled = False
        self.box = None                # где игра на экране
        self.running = True
        self.acc_x = self.acc_y = 0.0

    def set_velocity(self, drive_x, drive_y, box, enabled):
        """drive_x > 0 — повернуть вправо, drive_y > 0 — посмотреть вниз (Гц сигнала)."""
        def speed(drive):
            if abs(drive) < config.MOUSE_DEADZONE_HZ:
                return 0.0
            v = drive * config.MOUSE_SPEED
            return max(-config.MOUSE_MAX_SPEED, min(config.MOUSE_MAX_SPEED, v))
        self.vx = speed(drive_x)
        self.vy = speed(drive_y) * (-1 if config.MOUSE_INVERT_Y else 1)
        self.box = box
        self.enabled = enabled

    def cursor_in_game(self):
        if self.box is None:
            return False
        p = wintypes.POINT()
        winutil.user32.GetCursorPos(ctypes.byref(p))
        b = self.box
        return b["left"] <= p.x < b["left"] + b["width"] and b["top"] <= p.y < b["top"] + b["height"]

    def run(self):
        last = time.perf_counter()
        while self.running:
            time.sleep(self.STEP_S)
            now = time.perf_counter()
            dt, last = now - last, now
            if not self.enabled or not self.cursor_in_game():
                self.acc_x = self.acc_y = 0.0
                continue
            self.acc_x += self.vx * dt
            self.acc_y += self.vy * dt
            dx, dy = int(self.acc_x), int(self.acc_y)
            if dx or dy:
                winutil.move_mouse(dx, dy)
                self.acc_x -= dx
                self.acc_y -= dy

    def stop(self):
        self.enabled = False
        self.running = False
