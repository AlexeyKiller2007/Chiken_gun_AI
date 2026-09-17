"""Выбор оружия: группы нейронов «голосуют», муха жмёт Tab и цифру на нумпаде."""
import math
import time

import numpy as np

import config
import winutil


def tap(key):
    winutil.send_key(key, True)
    time.sleep(0.08)
    winutil.send_key(key, False)


class WeaponPicker:
    WARMUP_S = 5.0   # сначала копим «норму» каждой группы

    def __init__(self, brain):
        frame_ms = config.BRAIN_MS_PER_FRAME
        self.groups = {}
        self.floor = {}
        for key, spec in config.WEAPON_VOTERS.items():
            idx = np.unique(np.concatenate([brain.neurons(t, s) for t, s in spec]))
            if idx.size == 0:
                print(f"внимание: для оружия {key} не найдено нейронов {spec}")
            self.groups[key] = idx
        # голосуем по окну ~1 с: одиночные спайки маленьких групп усредняются
        self.k_smooth = 1.0 - math.exp(-frame_ms / config.WEAPON_SMOOTH_MS)
        for key, idx in self.groups.items():
            # два лишних спайка группы за кадр (после сглаживания) — ещё не всплеск, а шум
            self.floor[key] = 2 * (1000.0 / frame_ms) / max(1, idx.size) * self.k_smooth
        self.k_base = 1.0 - math.exp(-frame_ms / 1000.0 / config.BASELINE_S)
        self.smooth = dict.fromkeys(self.groups, 0.0)
        self.mean = dict.fromkeys(self.groups, 0.0)
        self.var = dict.fromkeys(self.groups, 0.0)
        self.scores = dict.fromkeys(self.groups, 0.0)
        self.current = None       # какое оружие муха взяла последним
        self.started = None
        self.next_allowed = 0.0
        self.pending = None       # (клавиша, когда нажать) — меню уже открыто

    def cancel(self):
        self.pending = None

    def update(self, counts, ms, now, can_act):
        """Считает голоса и, если пора, меняет оружие. can_act — клавиши можно слать."""
        if self.started is None:
            self.started = now
            self.next_allowed = now + self.WARMUP_S
        for key, idx in self.groups.items():
            hz = float(counts[idx].mean()) * 1000.0 / ms if idx.size else 0.0
            s = self.smooth[key] = self.smooth[key] + self.k_smooth * (hz - self.smooth[key])
            spread = max(math.sqrt(self.var[key]), self.floor[key])
            self.scores[key] = (s - self.mean[key]) / spread   # во сколько «разбросов» выше нормы
            self.mean[key] += self.k_base * (s - self.mean[key])
            self.var[key] += self.k_base * ((s - self.mean[key]) ** 2 - self.var[key])

        if self.pending is not None:
            key, at = self.pending
            if now >= at:
                self.pending = None
                if can_act:
                    tap(key)
                    self.current = key
                    print(f"Муха взяла {config.WEAPON_NAMES.get(key, key)} ({key}): "
                          f"{config.WEAPON_REASONS.get(key, '')}.")
            return

        best = max(self.scores, key=self.scores.get)
        if (can_act and now >= self.next_allowed and best != self.current
                and self.scores[best] >= config.WEAPON_Z):
            tap(config.WEAPON_MENU_KEY)
            self.pending = (best, now + config.WEAPON_MENU_DELAY_S)
            self.next_allowed = now + config.WEAPON_EVERY_S
