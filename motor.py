"""Нисходящие нейроны -> клавиши игры."""
import math

import numpy as np

import config


class Motor:
    def __init__(self, brain):
        self.brain = brain
        self.groups = {}
        for key, spec in config.CONTROLS.items():
            idx = np.unique(np.concatenate([brain.neurons(t, s) for t, s in spec]))
            if idx.size == 0:
                print(f"внимание: для клавиши {key} не найдено ни одного нейрона {spec}")
            self.groups[key] = idx
        frame_ms = config.BRAIN_MS_PER_FRAME
        self.k_smooth = 1.0 - math.exp(-frame_ms / config.SMOOTH_MS)
        self.k_base = 1.0 - math.exp(-frame_ms / 1000.0 / config.BASELINE_S)
        self.smooth = None
        self.baseline = None
        self.hold_frames = max(1, math.ceil(config.KEY_HOLD_MS / frame_ms))
        self.hold = {}
        self.frame = 0
        self.queued = {"h": 0, "v": 0}   # отложенный поворот камеры, в кадрах (+ вправо/вниз)
        self.ready_at = {}    # кадр, с которого клавишу снова можно жать
        self.toggle_at = {}   # кадр, когда переключатель надо нажать обратно

    @staticmethod
    def frames(seconds):
        return max(1, round(seconds * 1000 / config.BRAIN_MS_PER_FRAME))

    def set_group(self, key, spec):
        """Переназначить клавишу на другие нейроны (во время игры)."""
        idx = np.unique(np.concatenate([self.brain.neurons(t, s) for t, s in spec]))
        if idx.size == 0:
            return False
        self.groups[key] = idx
        if self.smooth is not None:
            self.smooth[key] = self.baseline[key] = 0.0
        self.hold.pop(key, None)
        return True

    def rates(self, counts, ms):
        """Средняя частота нейронов каждой клавиши, Гц."""
        return {key: float(counts[idx].mean()) * 1000.0 / ms if idx.size else 0.0
                for key, idx in self.groups.items()}

    def drive(self, hz):
        """Насколько нейроны каждой клавиши активнее обычного, Гц."""
        if self.smooth is None:
            self.smooth, self.baseline = dict(hz), dict(hz)
        excess = {}
        for key, r in hz.items():
            self.smooth[key] += self.k_smooth * (r - self.smooth[key])
            # провалы ниже нормы не считаем, иначе после движения вниз
            # муха «видит» движение вверх (как иллюзия водопада)
            excess[key] = max(0.0, self.smooth[key] - self.baseline[key])
            self.baseline[key] += self.k_base * (self.smooth[key] - self.baseline[key])
        for a, b in config.OPPOSITES:
            diff = excess[a] - excess[b]
            excess[a], excess[b] = diff, -diff
        return excess

    def decide(self, drive, active=True):
        """Какие клавиши должны быть нажаты в этом кадре.
        active: клавиши правда уходят в игру (иначе таймеры переключателей стоят)."""
        self.frame += 1
        fresh = {key for key, x in drive.items()
                 if x >= config.THRESHOLD_HZ.get(key, 5)
                 and self.frame >= self.ready_at.get(key, 0) and key not in self.toggle_at}
        for key in fresh - config.TAP_KEYS:
            self.hold[key] = self.hold_frames
            for a, b in config.OPPOSITES:
                if key in (a, b):
                    self.hold.pop(b if key == a else a, None)
        for key, suppressed in config.OVERRIDES.items():
            if key in self.hold or key in fresh:
                for other in suppressed:
                    self.hold.pop(other, None)

        taps = fresh & config.TAP_KEYS
        if active:
            for key in taps:
                self.ready_at[key] = self.frame + self.frames(config.COOLDOWN_S.get(key, 0))
                if key in config.TOGGLE_BACK_S:
                    self.toggle_at[key] = self.frame + self.frames(config.TOGGLE_BACK_S[key])
            due = {key for key, at in self.toggle_at.items() if at <= self.frame}
            for key in due:
                del self.toggle_at[key]
            taps |= due

        want = set(self.hold) | taps
        self.hold = {key: left - 1 for key, left in self.hold.items() if left > 1}
        return self.look_after_move(want, active)

    def look_after_move(self, want, active):
        """Пока курица идёт, камера ждёт; накопленный поворот — после ходьбы."""
        if not config.LOOK_AFTER_MOVE:
            return want
        if not active:
            self.queued = {"h": 0, "v": 0}
            return want
        looks = {key for key in want if key in config.LOOK_KEYS}
        if want & config.MOVE_KEYS:
            cap = self.frames(config.LOOK_WAIT_MAX_MS / 1000.0)
            for key in looks:
                axis, sign = config.LOOK_KEYS[key]
                self.queued[axis] = max(-cap, min(cap, self.queued[axis] + sign))
            return want - looks
        out = set(want)
        for axis in self.queued:
            if self.queued[axis] == 0:
                continue
            sign = 1 if self.queued[axis] > 0 else -1
            # сначала доделываем отложенный поворот по этой оси
            out -= {key for key, (a, _) in config.LOOK_KEYS.items() if a == axis}
            out.add(next(key for key, (a, s) in config.LOOK_KEYS.items() if a == axis and s == sign))
            self.queued[axis] -= sign
        return out

    def pending_toggles(self):
        """Переключатели, которые ещё надо вернуть (меню открыто и т.п.)."""
        keys = set(self.toggle_at)
        self.toggle_at.clear()
        return keys
