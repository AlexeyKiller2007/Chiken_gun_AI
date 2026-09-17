"""Симуляция всего мозга мухи: 139 тыс. нейронов leaky integrate-and-fire.

Модель повторяет Shiu et al., 2024 (Nature): у каждого нейрона есть
напряжение v и синаптический ток g. Спайк нейрона через задержку добавляет
к g всех его партнёров (число синапсов) x 0.275 мВ, со знаком нейромедиатора.
Сверх оригинала добавлены:
- адаптация a: каждый спайк её увеличивает, и она тянет напряжение вниз;
- синаптическая депрессия (Tsodyks-Markram): при частой стрельбе у нейрона
  кончается запас медиатора x, и каждый следующий спайк действует слабее.
  Без неё пары нейронов с сотнями общих синапсов перекидываются спайками
  на 240 Гц вечно и глушат всё, что приходит от глаз.

Считают модель движки из engines.py: видеокарта, процессор (numba) или numpy.
"""
import numpy as np
import scipy.sparse as sp

import config
from engines import make_engine


class FlyBrain:
    def __init__(self, path=config.BRAIN_FILE, seed=0, backend=config.BRAIN_BACKEND):
        d = np.load(path)
        self.root_ids = d["root_ids"]
        self.cell_type = d["cell_type"]
        self.side = d["side"]
        self.super_class = d["super_class"]
        self.cell_class = d["cell_class"]
        self.pos = d["pos"]
        self.n = len(self.root_ids)
        weights = d["w_data"].astype(np.float32) * np.float32(config.W_SYN_MV)
        # строки = кто отправляет, столбцы = кто получает
        self.w = sp.csr_matrix((weights, d["w_indices"], d["w_indptr"]), shape=(self.n, self.n))

        # зрительные нейроны, которые получают сигнал прямо из картинки
        self.eye_idx = d["eye_idx"]
        self.eye_type = d["eye_type"]
        self.eye_kind = d["eye_kind"]
        self.eye_side = d["eye_side"]
        self.eye_theta = d["eye_theta"]
        self.eye_elev = d["eye_elev"]
        self.eye_rates = np.zeros(len(self.eye_idx), np.float32)

        self.dt = config.DT_MS
        self.engine = make_engine(self, backend, seed)

    @property
    def backend(self):
        return self.engine.name

    def reset(self):
        self.engine.reset()

    def neurons(self, cell_type, side=None):
        if cell_type.startswith("class:"):
            mask = self.super_class == cell_type[len("class:"):]
        else:
            mask = self.cell_type == cell_type
        if side is not None:
            mask &= self.side == side
        return np.flatnonzero(mask)

    def set_eye_rates(self, rates_hz):
        self.eye_rates = np.asarray(rates_hz, np.float32)

    def run(self, ms):
        """Считает ms миллисекунд. Возвращает число спайков каждого нейрона."""
        steps = max(1, round(ms / self.dt))
        eye_p = self.eye_rates * np.float32(self.dt / 1000.0)   # вероятность спайка за шаг
        return self.engine.run(steps, eye_p)
