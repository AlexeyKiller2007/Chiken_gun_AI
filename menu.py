"""Меню с пропами (X): муха водит указателем по предметам и выбирает один."""
import time

import numpy as np

import config
import winutil
from respawn import box_pixels


def menu_visible(img):
    """Открыто ли меню: красная кнопка закрытия и белое поле «Поиск»."""
    b, g, r = box_pixels(img, config.MENU_CLOSE_BOX).T   # кадр в BGR
    red = np.mean((r > 170) & (g < 100) & (b < 100))
    white = np.mean(box_pixels(img, config.MENU_SEARCH_BOX).min(axis=1) > 200)
    return red > 0.3 and white > 0.4


def close_menu():
    winutil.send_key("x", True)
    time.sleep(0.1)
    winutil.send_key("x", False)


class MenuPicker:
    OPEN_TIMEOUT_S = 1.5   # меню должно появиться за это время после X
    CLOSE_CHECK_S = 2.0    # через столько после выбора проверяем, закрылось ли меню

    def __init__(self):
        self.started = None
        self.seen = None
        self.picked = None
        self.col = self.row = 0
        self.prev = set()

    @property
    def active(self):
        return self.started is not None

    def start(self, now):
        grid = config.MENU_GRID
        self.started = now
        self.seen = None
        self.picked = None
        self.col, self.row = grid["cols"] // 2, grid["rows"] // 2
        self.prev = set()

    def stop(self, close=False):
        """close: меню, возможно, открыто — закрыть его (только если игра активна)."""
        if close and self.active and self.seen is not None and self.picked is None:
            close_menu()
        self.started = None

    def cell_rect(self):
        """Клетка под указателем в долях кадра: (x0, y0, x1, y1)."""
        grid = config.MENU_GRID
        x0 = grid["x0"] + self.col * grid["cell_w"]
        y0 = grid["y0"] + self.row * grid["cell_h"]
        return x0, y0, x0 + grid["cell_w"], y0 + grid["cell_h"]

    def update(self, small, want, box, now, can_act):
        """Двигает указатель и выбирает предмет.
        Возвращает клавиши, которые можно отправить в игру (без X и R)."""
        visible = menu_visible(small)
        pressed = want - self.prev
        self.prev = set(want)
        grid = config.MENU_GRID

        if self.seen is None:
            if visible:
                self.seen = now
            elif now - self.started > self.OPEN_TIMEOUT_S:
                print("Меню не открылось.")
                self.stop()
        elif self.picked is None:
            # каждое новое нажатие клавиши взгляда/шага сдвигает указатель на клетку
            for key in pressed:
                dx, dy = config.MENU_MOVES.get(key, (0, 0))
                self.col = min(max(self.col + dx, 0), grid["cols"] - 1)
                self.row = min(max(self.row + dy, 0), grid["rows"] - 1)
            if can_act and ("r" in want or now - self.seen >= config.MENU_PICK_S):
                x0, y0, x1, y1 = self.cell_rect()
                winutil.click(box["left"] + round((x0 + x1) / 2 * box["width"]),
                              box["top"] + round((y0 + y1) / 2 * box["height"]))
                self.picked = now
                how = "нейрон «взять»" if "r" in want else "время вышло"
                print(f"Муха выбрала предмет №{self.row * grid['cols'] + self.col + 1} "
                      f"(ряд {self.row + 1}, клетка {self.col + 1}; {how}).")
        elif now - self.picked >= self.CLOSE_CHECK_S:
            if visible and can_act:
                close_menu()
            self.stop()
        return want - {"r", "x"}
