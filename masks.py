"""Красные зоны: то, что скрыто от глаз мухи. Редактируются мышкой в окне Fly view."""
import json

import cv2

import config

MIN_SIZE = 0.02   # совсем маленькие прямоугольники не считаем


def inside(mask, fx, fy):
    x0, y0, x1, y1 = mask
    return x0 <= fx <= x1 and y0 <= fy <= y1


class MaskEditor:
    """Мышкой: потянуть по пустому месту — новая зона, потянуть внутри зоны —
    подвинуть её, правая кнопка — удалить. Клавиша сохраняет зоны в файл."""

    def __init__(self, width, height):
        self.width, self.height = width, height   # размер кадра игры в окне, пиксели
        self.masks = [tuple(m) for m in config.HUD_MASKS]
        self.preview = None      # прямоугольник, который сейчас рисуют
        self.drag = None         # ("new", fx, fy) или ("move", индекс, dx, dy)
        self.changed = False

    def on_mouse(self, event, x, y, flags=None, param=None):
        if y >= self.height or x >= self.width:
            return   # клик не по кадру игры, а по другим панелям окна
        fx, fy = x / self.width, y / self.height
        if event == cv2.EVENT_RBUTTONDOWN:
            for i, mask in enumerate(self.masks):
                if inside(mask, fx, fy):
                    self.masks.pop(i)
                    self.changed = True
                    print(f"Зона удалена, осталось {len(self.masks)}.")
                    return
        elif event == cv2.EVENT_LBUTTONDOWN:
            for i, mask in enumerate(self.masks):
                if inside(mask, fx, fy):
                    self.drag = ("move", i, fx - mask[0], fy - mask[1])
                    return
            self.drag = ("new", fx, fy)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            if self.drag[0] == "new":
                _, x0, y0 = self.drag
                self.preview = (min(x0, fx), min(y0, fy), max(x0, fx), max(y0, fy))
            else:
                _, i, dx, dy = self.drag
                x0, y0, x1, y1 = self.masks[i]
                w, h = x1 - x0, y1 - y0
                nx = min(max(fx - dx, 0.0), 1.0 - w)
                ny = min(max(fy - dy, 0.0), 1.0 - h)
                self.masks[i] = (nx, ny, nx + w, ny + h)
                self.changed = True
        elif event == cv2.EVENT_LBUTTONUP and self.drag:
            if self.drag[0] == "new" and self.preview:
                x0, y0, x1, y1 = self.preview
                if x1 - x0 >= MIN_SIZE and y1 - y0 >= MIN_SIZE:
                    self.masks.append((round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)))
                    self.changed = True
                    print(f"Новая зона, всего {len(self.masks)}.")
            self.drag = self.preview = None

    def apply(self):
        """Отдать зоны мухе (глаз читает config.HUD_MASKS)."""
        config.HUD_MASKS = list(self.masks)

    def save(self):
        config.MASKS_FILE.write_text(json.dumps([list(m) for m in self.masks], indent=1), encoding="utf-8")
        self.changed = False
        print(f"Зоны сохранены ({len(self.masks)} шт.): {config.MASKS_FILE.name}")

    def reset(self):
        self.masks = [tuple(m) for m in config.HUD_MASKS_DEFAULT]
        self.changed = True
        print(f"Зоны сброшены к обычным ({len(self.masks)} шт.).")
