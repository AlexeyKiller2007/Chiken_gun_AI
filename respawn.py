"""Экран смерти: муха жмёт «Продолжить», пока не возродится."""
import numpy as np

import config
import winutil


def box_pixels(img, box):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w), :3].reshape(-1, 3).astype(np.int16)


def looks_like_button(img, box):
    """Белая кнопка с тёмным текстом."""
    px = box_pixels(img, box)
    white = np.mean(px.min(axis=1) > 215)
    dark = np.mean(px.max(axis=1) < 100)
    return white > 0.5 and dark > 0.02


def health_bar_visible(img):
    b, g, r = box_pixels(img, config.HEALTH_BAR_BOX).T   # кадр в BGR
    return np.mean((g > 150) & (r < 120) & (b < 120)) > 0.15


def is_death_screen(img):
    return (looks_like_button(img, config.DEATH_CONTINUE_BOX)
            and looks_like_button(img, config.DEATH_MENU_BOX)
            and not health_bar_visible(img))


class Respawner:
    CONFIRM_FRAMES = 2   # столько кадров подряд экран смерти должен быть виден
    ALIVE_FRAMES = 3     # и столько кадров подряд не виден, чтобы считать муху живой

    def __init__(self):
        self.seen = 0
        self.gone = 0
        self.dead_since = None
        self.next_click = 0.0
        self.deaths = 0

    @property
    def dead(self):
        return self.dead_since is not None

    def update(self, small, now):
        """small — уменьшенный кадр игры. Возвращает True, пока муха мертва."""
        if is_death_screen(small):
            self.seen += 1
            self.gone = 0
        else:
            self.seen = 0
            self.gone += 1

        if not self.dead:
            if self.seen < self.CONFIRM_FRAMES:
                return False
            self.deaths += 1
            self.dead_since = now
            self.next_click = now
            print(f"Муха погибла ({self.deaths}-й раз). Жму «Продолжить»...")
        elif self.gone >= self.ALIVE_FRAMES:
            print(f"Муха снова в игре (через {now - self.dead_since:.0f} с).")
            self.dead_since = None
            return False
        return True

    def press_continue(self, box, now):
        """Кликает «Продолжить», если пора. box — где игра на экране."""
        if not self.dead or not self.seen:
            return
        # кликаем RESPAWN_CLICK_FOR_S секунд, потом пауза RESPAWN_PAUSE_S, и снова
        cycle = config.RESPAWN_CLICK_FOR_S + config.RESPAWN_PAUSE_S
        clicking = (now - self.dead_since) % cycle < config.RESPAWN_CLICK_FOR_S
        if clicking and now >= self.next_click:
            x = box["left"] + round(config.DEATH_CLICK[0] * box["width"])
            y = box["top"] + round(config.DEATH_CLICK[1] * box["height"])
            winutil.click(x, y)
            self.next_click = now + config.RESPAWN_CLICK_EVERY_S
