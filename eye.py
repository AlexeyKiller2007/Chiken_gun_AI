"""Глаз мухи: превращает кадр игры в частоты спайков зрительных нейронов."""
import cv2
import numpy as np

import config

LIGHT, ON, OFF, FRONT_TO_BACK, BACK_TO_FRONT, UP, DOWN, EDGE = range(8)


class FlyEye:
    def __init__(self, brain):
        self.grid_w, self.grid_h = config.EYE_GRID
        self.kind = brain.eye_kind
        self.left = brain.eye_side == "left"
        theta, elev = brain.eye_theta, brain.eye_elev

        if config.EYE_STRETCH:
            # каждый тип нейронов каждого глаза растянут на свою половину экрана
            du = np.zeros_like(theta)
            e = np.zeros_like(elev)
            for t in np.unique(brain.eye_type):
                for m in (self.left, ~self.left):
                    m = m & (brain.eye_type == t)
                    du[m] = 0.5 * percentile_rank(theta[m])
                    e[m] = percentile_rank(elev[m])
            v = 1.0 - e
        else:
            du = theta / config.GAME_HFOV_DEG
            v = 0.5 - elev / config.GAME_VFOV_DEG
        # вперёд смотрит центр экрана, левый глаз видит левую половину
        u = np.where(self.left, 0.5 - du, 0.5 + du)

        self.u, self.v = u, v
        self.visible = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
        self.px = np.clip((u * self.grid_w).astype(int), 0, self.grid_w - 1)
        self.py = np.clip((v * self.grid_h).astype(int), 0, self.grid_h - 1)
        # «спереди назад» для левого глаза — движение влево по экрану, для правого — вправо
        self.outward = np.where(self.left, -1.0, 1.0).astype(np.float32)
        self.max_rate = np.where(self.kind == EDGE, config.EDGE_MAX_HZ, config.MAX_RATE_HZ).astype(np.float32)
        self.prev = None
        self.last_edges = None   # контуры последнего кадра в полном размере (для окна)
        self.rng = np.random.default_rng()

    def preprocess(self, bgra, extra_masks=()):
        """Скриншот -> (маленькая серая картинка 0..1, плотность контуров 0..1),
        без интерфейса игры. extra_masks: ещё области (доли кадра), которые муха не видит."""
        gray = cv2.cvtColor(bgra, cv2.COLOR_BGRA2GRAY)
        # контуры ищем до масок, иначе края самих масок тоже стали бы «предметами»
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 130)
        h, w = gray.shape
        for x0, y0, x1, y1 in list(config.HUD_MASKS) + list(extra_masks):
            gray[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 128
            edges[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 0
        self.last_edges = edges
        size = (self.grid_w, self.grid_h)
        small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        density = cv2.resize(edges, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        return small, density

    def rates(self, frame, edges=None):
        """Маленькая картинка (и плотность контуров) -> частота каждого входного нейрона, Гц."""
        if edges is None:
            edges = edges_of_small(frame)
        if config.MICROSACCADES:
            edges = np.roll(edges, tuple(self.rng.integers(-1, 2, 2)), axis=(0, 1))
        if self.prev is None:
            self.prev = frame
        change = frame - self.prev
        flow = cv2.calcOpticalFlowFarneback(
            (self.prev * 255).astype(np.uint8), (frame * 255).astype(np.uint8),
            None, 0.5, 2, 7, 3, 5, 1.1, 0)
        self.prev = frame

        at = (self.py, self.px)
        d = change[at]
        fx = flow[..., 0][at] * self.outward
        fy = flow[..., 1][at]
        k = self.kind
        signal = np.select(
            [k == LIGHT, k == ON, k == OFF,
             k == FRONT_TO_BACK, k == BACK_TO_FRONT, k == UP, k == DOWN, k == EDGE],
            [config.LUM_WEIGHT * frame[at] + config.MOTION_WEIGHT * np.abs(d),
             config.CONTRAST_GAIN * d, -config.CONTRAST_GAIN * d,
             fx / config.FLOW_FULL_PX, -fx / config.FLOW_FULL_PX,
             -fy / config.FLOW_FULL_PX, fy / config.FLOW_FULL_PX,
             config.EDGE_GAIN * edges[at]])
        # за краем экрана муха видит ровный серый фон
        signal = np.where(self.visible, signal, np.where(k == LIGHT, config.LUM_WEIGHT * 0.5, 0.0))
        return (self.max_rate * np.clip(signal, 0.0, 1.0)).astype(np.float32)


def edges_of_small(frame):
    """Контуры прямо по маленькой картинке (для тестов без скриншота)."""
    img = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    big = cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4), interpolation=cv2.INTER_LINEAR)
    edges = cv2.Canny(big, 50, 130)
    return cv2.resize(edges, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def percentile_rank(x):
    return np.argsort(np.argsort(x)) / max(1, len(x) - 1)
