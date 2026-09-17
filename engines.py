"""Движки симуляции мозга: numpy (запасной), numba (процессор) и cuda (видеокарта).

Все движки считают одну и ту же модель (см. brain.py) и отдают число спайков
каждого нейрона за кусок времени. Numba и cuda делают весь шаг нейрона одной
функцией: приём спайков, напряжение, порог, депрессию синапсов и случайные
спайки глаза и шума (генератор случайных чисел — хэш от шага и номера нейрона).
"""
from collections import deque

import numpy as np

import config


class Params:
    """Числа модели, пересчитанные на один шаг."""

    def __init__(self, dt):
        self.dt = dt
        self.alpha = np.float32(dt / config.TAU_M_MS)
        self.decay = np.float32(np.exp(-dt / config.TAU_SYN_MS))
        self.adapt_inc = np.float32(config.ADAPT_MV)
        self.adapt_decay = np.float32(np.exp(-dt / config.TAU_ADAPT_MS))
        self.std_use = np.float32(config.STD_USE)
        self.std_tau_steps = np.float32(config.STD_TAU_MS / dt)
        self.delay = max(1, round(config.DELAY_MS / dt))
        self.refractory = max(1, round(config.REFRACTORY_MS / dt))
        self.v_rest = np.float32(config.V_REST)
        self.v_reset = np.float32(config.V_RESET)
        self.v_thresh = np.float32(config.V_THRESH)


# ------------------------------------------------------------------ numpy ----
class NumpyEngine:
    """Исходный вариант на numpy: медленный, но без лишних библиотек."""
    name = "numpy"

    def __init__(self, brain, seed):
        self.w = brain.w
        self.n = brain.n
        self.eye_idx = brain.eye_idx
        self.p = Params(brain.dt)
        self.spontaneous = self.n * config.SPONTANEOUS_HZ * brain.dt / 1000.0
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        empty = np.empty(0, np.intp)
        n = self.n
        self.v = np.full(n, config.V_REST, np.float32)
        self.g = np.zeros(n, np.float32)
        self.a = np.zeros(n, np.float32)
        self.buf = np.zeros(n, np.float32)
        self.x = np.ones(n, np.float32)
        self.last_spike = np.zeros(n, np.int64)
        self.t = 0
        self.in_transit = deque([empty] * self.p.delay)
        self.refractory = deque([empty] * self.p.refractory, maxlen=self.p.refractory)

    def step(self, forced):
        p = self.p
        self.t += 1
        arrived = self.in_transit.popleft()
        if arrived.size:
            rows = self.w[arrived]
            weights = rows.data
            if p.std_use:
                rest = np.exp((self.last_spike[arrived] - self.t) / p.std_tau_steps).astype(np.float32)
                x = 1 - (1 - self.x[arrived]) * rest
                self.x[arrived] = x * (1 - p.std_use)
                self.last_spike[arrived] = self.t
                weights = weights * np.repeat(x, np.diff(rows.indptr))
            if rows.nnz > self.n // 4:
                self.g += np.bincount(rows.indices, weights, self.n).astype(np.float32)
            else:
                np.add.at(self.g, rows.indices, weights)

        dv = self.buf
        np.subtract(self.g, self.a, out=dv)
        dv -= self.v
        dv += p.v_rest
        dv *= p.alpha
        self.v += dv
        for idx in self.refractory:
            self.v[idx] = p.v_reset
        self.g *= p.decay
        self.a *= p.adapt_decay
        if self.t % 32 == 0:
            self.g[np.abs(self.g) <= TINY] = 0
            self.a[self.a <= TINY] = 0

        fired = np.flatnonzero(self.v > p.v_thresh)
        if forced.size:
            fired = np.concatenate((fired, forced))
        self.v[fired] = p.v_reset
        self.a[fired] += p.adapt_inc
        self.refractory.append(fired)
        self.in_transit.append(fired)
        return fired

    def run(self, steps, eye_p):
        # молчащие входы пропускаем — на статичной картинке их большинство
        lit = np.flatnonzero(eye_p > 0)
        step_of, which = np.nonzero(self.rng.random((steps, len(lit)), dtype=np.float32) < eye_p[lit])
        forced = self.eye_idx[lit[which]]
        if self.spontaneous:
            k = self.rng.poisson(self.spontaneous * steps)
            step_of = np.concatenate((step_of, self.rng.integers(0, steps, k)))
            forced = np.concatenate((forced, self.rng.integers(0, self.n, k)))
            order = np.argsort(step_of, kind="stable")
            step_of, forced = step_of[order], forced[order]
        bounds = np.searchsorted(step_of, np.arange(steps + 1))

        counts = np.zeros(self.n, np.int32)
        for s in range(steps):
            fired = self.step(forced[bounds[s]:bounds[s + 1]])
            counts[fired] += 1
        return counts


# ------------------------------------------------------------------ numba ----
# Числа меньше этого обнуляем: иначе ток молчащих нейронов, затухая,
# становится «денормализованным», и процессор считает его в десятки раз медленнее
TINY = 1e-6
_numba = None


def _build_numba():
    import numba as nb

    @nb.njit(cache=True, inline="always")
    def geom(p):
        # через сколько шагов случится следующий случайный спайк (p — вероятность за шаг)
        if p <= 0.0:
            return 1 << 40
        return 1 + int(np.log(1.0 - np.random.random()) / np.log1p(-p))

    def make_update(parallel):
        loop = nb.prange if parallel else range

        @nb.njit(cache=True, fastmath=True, parallel=parallel)
        def update(t, v, g, a, refr, fired, next_forced, alpha, decay, adecay, v_rest, v_reset, v_th):
            # без ветвлений, чтобы компилятор считал по 8 нейронов за раз
            for j in loop(v.shape[0]):
                was = refr[j] > 0
                nv = v[j] + alpha * (g[j] - a[j] - (v[j] - v_rest))
                v[j] = v_reset if was else nv
                refr[j] = refr[j] - 1 if was else 0
                gj = g[j] * decay
                g[j] = gj if abs(gj) > TINY else 0.0
                aj = a[j] * adecay
                a[j] = aj if aj > TINY else 0.0
                fired[j] = ((not was) and nv > v_th) or next_forced[j] <= t
        return update

    update_serial = make_update(False)
    update_parallel = make_update(True)

    @nb.njit(cache=True, fastmath=True)
    def handle(t, slot, v, a, x, last, refr, counts, prob, next_forced, fired,
               hist_idx, hist_eff, ainc, use, tau, v_reset, n_refr):
        cnt = 0
        for j in range(fired.shape[0]):
            if fired[j]:
                if next_forced[j] <= t:
                    next_forced[j] = t + geom(prob[j])
                v[j] = v_reset
                a[j] += ainc
                refr[j] = n_refr
                counts[j] += 1
                # запас медиатора восстановился с прошлого спайка, этот спайк тратит его долю
                xe = 1.0 - (1.0 - x[j]) * np.exp((last[j] - t) / tau)
                x[j] = xe * (1.0 - use)
                last[j] = t
                hist_idx[slot, cnt] = j
                hist_eff[slot, cnt] = xe
                cnt += 1
        return cnt

    @nb.njit(cache=True)
    def run(steps, t, parallel, v, g, a, x, last, refr, counts, prob, next_forced, fired,
            eye_idx, hist_idx, hist_eff, hist_cnt, indptr, indices, data,
            alpha, decay, adecay, ainc, use, tau, v_rest, v_reset, v_th, n_refr, delay):
        # у глаза новые вероятности — перезаводим таймеры (у Пуассона нет памяти)
        for k in range(eye_idx.shape[0]):
            j = eye_idx[k]
            next_forced[j] = t + geom(prob[j])
        slots = delay + 1
        for _ in range(steps):
            t += 1
            # спайки, выпущенные delay шагов назад, доходят до получателей
            r = (t - delay) % slots
            for k in range(hist_cnt[r]):
                i = hist_idx[r, k]
                e = hist_eff[r, k]
                for s in range(indptr[i], indptr[i + 1]):
                    g[indices[s]] += data[s] * e
            if parallel:
                update_parallel(t, v, g, a, refr, fired, next_forced, alpha, decay, adecay, v_rest, v_reset, v_th)
            else:
                update_serial(t, v, g, a, refr, fired, next_forced, alpha, decay, adecay, v_rest, v_reset, v_th)
            w = t % slots
            hist_cnt[w] = handle(t, w, v, a, x, last, refr, counts, prob, next_forced, fired,
                                 hist_idx, hist_eff, ainc, use, tau, v_reset, n_refr)
        return t

    @nb.njit(cache=True)
    def seed_rng(seed, next_forced, prob):
        np.random.seed(seed)
        for j in range(prob.shape[0]):
            next_forced[j] = geom(prob[j])

    return run, seed_rng


class NumbaEngine:
    """Скомпилированный цикл на процессоре."""
    name = "numba"

    def __init__(self, brain, seed):
        global _numba
        if _numba is None:
            _numba = _build_numba()
        self.p = Params(brain.dt)
        self.n = brain.n
        self.eye_idx = brain.eye_idx.astype(np.int64)
        self.indptr = brain.w.indptr.astype(np.int64)
        self.indices = brain.w.indices.astype(np.int32)
        self.data = brain.w.data.astype(np.float32)
        self.spont_p = np.float32(config.SPONTANEOUS_HZ * brain.dt / 1000.0)
        self.seed = seed
        self.parallel = config.NUMBA_PARALLEL
        if self.parallel:
            import numba
            numba.set_num_threads(min(config.NUMBA_THREADS, numba.config.NUMBA_NUM_THREADS))
        self.reset()

    def reset(self):
        n, slots = self.n, self.p.delay + 1
        self.v = np.full(n, config.V_REST, np.float32)
        self.g = np.zeros(n, np.float32)
        self.a = np.zeros(n, np.float32)
        self.x = np.ones(n, np.float32)
        self.last = np.zeros(n, np.int64)
        self.refr = np.zeros(n, np.int32)
        self.fired = np.zeros(n, np.bool_)
        self.hist_idx = np.zeros((slots, n), np.int32)
        self.hist_eff = np.zeros((slots, n), np.float32)
        self.hist_cnt = np.zeros(slots, np.int64)
        self.prob = np.full(n, self.spont_p, np.float32)
        self.next_forced = np.zeros(n, np.int64)
        _numba[1](self.seed, self.next_forced, self.prob)
        self.t = 0

    def run(self, steps, eye_p):
        p = self.p
        self.prob[self.eye_idx] = self.spont_p + eye_p
        counts = np.zeros(self.n, np.int32)
        self.t = _numba[0](
            steps, self.t, self.parallel, self.v, self.g, self.a, self.x, self.last, self.refr,
            counts, self.prob, self.next_forced, self.fired, self.eye_idx,
            self.hist_idx, self.hist_eff, self.hist_cnt, self.indptr, self.indices, self.data,
            p.alpha, p.decay, p.adapt_decay, p.adapt_inc, p.std_use, p.std_tau_steps,
            p.v_rest, p.v_reset, p.v_thresh, p.refractory, p.delay)
        return counts


# ------------------------------------------------------------------- cuda ----
CUDA_SOURCE = r"""
#define TINY 1e-6f

static inline __device__ float urand(unsigned int seed, unsigned int t, unsigned int j) {
    unsigned int h = seed ^ (t * 0x9E3779B1u) ^ (j * 0x85EBCA77u);
    h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
    return (h >> 8) * (1.0f / 16777216.0f);
}

// 1) выстрелившие нейроны раскладывают спайки по ячейкам своих синапсов.
//    У каждого синапса своя ячейка, поэтому потоки не мешают друг другу.
//    Синапсы нарезаны кусками по CHUNK: длинный цикл в одном потоке заставил бы
//    ждать всю пачку потоков видеокарты.
extern "C" __global__ void push_spikes(
        const int n_chunks, const int* chunk_start, const unsigned char* chunk_len,
        const int* chunk_owner, const int* post, const float* w, const int* slot_of,
        const float* spikes_in, float* arrived, unsigned char* touched) {
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    if (c >= n_chunks) return;
    float e = spikes_in[chunk_owner[c]];
    if (e == 0.0f) return;
    int s0 = chunk_start[c];
    for (int k = 0; k < chunk_len[c]; k++) {
        int s = s0 + k;
        arrived[slot_of[s]] = w[s] * e;
        touched[post[s]] = 1;
    }
}

// 2) частичные суммы пришедшего по кускам входов (у каждого куска своя ячейка)
extern "C" __global__ void gather_chunks(
        const int n_chunks, const int* chunk_start, const unsigned char* chunk_len,
        const int* chunk_owner, const unsigned char* touched, float* arrived, float* chunk_sum) {
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    if (c >= n_chunks || !touched[chunk_owner[c]]) return;
    float sum = 0.0f;
    int s0 = chunk_start[c];
    for (int k = 0; k < chunk_len[c]; k++) { sum += arrived[s0 + k]; arrived[s0 + k] = 0.0f; }
    chunk_sum[c] = sum;
}

// 3) шаг каждого нейрона; входы складывают только те, кому что-то пришло
extern "C" __global__ void brain_step(
        const int n, const int t, const unsigned int seed, const int* in_chunk_ptr,
        float* chunk_sum, unsigned char* touched,
        float* v, float* g, float* a, float* x, int* last, int* refr, int* counts,
        const float* prob, float* spikes_out, const float* par, const int n_refr) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j >= n) return;
    const float alpha = par[0], decay = par[1], adecay = par[2], ainc = par[3], use = par[4],
                tau = par[5], v_rest = par[6], v_reset = par[7], v_th = par[8];
    if (touched[j]) {
        float in = 0.0f;
        for (int c = in_chunk_ptr[j]; c < in_chunk_ptr[j + 1]; c++) in += chunk_sum[c];
        touched[j] = 0;
        g[j] += in;
    }
    bool was_refr = refr[j] > 0;
    if (was_refr) { v[j] = v_reset; refr[j] -= 1; }
    else v[j] += alpha * (g[j] - a[j] - (v[j] - v_rest));
    g[j] *= decay;
    if (fabs(g[j]) <= TINY) g[j] = 0.0f;
    a[j] *= adecay;
    if (a[j] <= TINY) a[j] = 0.0f;
    bool fired = !was_refr && v[j] > v_th;
    if (!fired && prob[j] > 0.0f) fired = urand(seed, (unsigned int)t, (unsigned int)j) < prob[j];
    if (fired) {
        v[j] = v_reset;
        a[j] += ainc;
        refr[j] = n_refr;
        counts[j] += 1;
        float xe = 1.0f - (1.0f - x[j]) * exp((float)(last[j] - t) / tau);
        x[j] = xe * (1.0f - use);
        last[j] = t;
        spikes_out[j] = xe;
    } else {
        spikes_out[j] = 0.0f;
    }
}
"""


class CudaEngine:
    """Все нейроны параллельно на видеокарте, две функции на шаг."""
    name = "cuda"
    BLOCK = 256
    CHUNK = 16

    def __init__(self, brain, seed):
        import warnings
        import scipy.sparse as sp
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import cupy as cp
        self.cp = cp
        module = cp.RawModule(code=CUDA_SOURCE)
        self.push = module.get_function("push_spikes")
        self.gather = module.get_function("gather_chunks")
        self.step = module.get_function("brain_step")
        self.p = Params(brain.dt)
        self.n = brain.n
        csr = brain.w
        # где в таблице «по получателям» лежит каждый синапс таблицы «по отправителям»
        order = sp.csr_matrix((np.arange(1, csr.nnz + 1, dtype=np.int64), csr.indices, csr.indptr),
                              shape=csr.shape).tocsc()
        slot_of = np.empty(csr.nnz, np.int32)
        slot_of[order.data - 1] = np.arange(csr.nnz, dtype=np.int32)
        # списки синапсов нарезаны кусками по CHUNK: по отправителям для рассылки,
        # по получателям для сбора
        out_chunks, _ = self.chunks(csr.indptr)
        self.n_out = len(out_chunks[0])
        self.out_start, self.out_len, self.out_owner = (cp.asarray(a) for a in out_chunks)
        in_chunks, in_ptr = self.chunks(order.indptr)
        self.n_in = len(in_chunks[0])
        self.in_start, self.in_len, self.in_owner = (cp.asarray(a) for a in in_chunks)
        self.in_chunk_ptr = cp.asarray(in_ptr)
        self.post = cp.asarray(csr.indices.astype(np.int32))
        self.w = cp.asarray(csr.data.astype(np.float32))
        self.slot_of = cp.asarray(slot_of)
        self.eye_idx = cp.asarray(brain.eye_idx.astype(np.int32))
        self.spont_p = np.float32(config.SPONTANEOUS_HZ * brain.dt / 1000.0)
        self.seed = np.uint32(seed & 0xFFFFFFFF)
        p = self.p
        self.par = cp.asarray(np.array([p.alpha, p.decay, p.adapt_decay, p.adapt_inc, p.std_use,
                                        p.std_tau_steps, p.v_rest, p.v_reset, p.v_thresh], np.float32))
        self.nnz = csr.nnz
        self.reset()

    def chunks(self, indptr):
        """Режет строки разреженной матрицы на куски: (начало, длина, чья строка), указатели."""
        lens = np.diff(indptr)
        per_row = (lens + self.CHUNK - 1) // self.CHUNK
        owner = np.repeat(np.arange(len(lens), dtype=np.int32), per_row)
        first = np.repeat(np.cumsum(per_row) - per_row, per_row)
        start = indptr[owner] + (np.arange(len(owner)) - first) * self.CHUNK
        length = np.minimum(self.CHUNK, indptr[owner + 1] - start)
        ptr = np.concatenate(([0], np.cumsum(per_row))).astype(np.int32)
        return (start.astype(np.int32), length.astype(np.uint8), owner), ptr

    def reset(self):
        cp, n = self.cp, self.n
        self.v = cp.full(n, config.V_REST, cp.float32)
        self.g = cp.zeros(n, cp.float32)
        self.a = cp.zeros(n, cp.float32)
        self.x = cp.ones(n, cp.float32)
        self.last = cp.zeros(n, cp.int32)
        self.refr = cp.zeros(n, cp.int32)
        self.counts = cp.zeros(n, cp.int32)
        self.prob = cp.full(n, self.spont_p, cp.float32)
        self.hist = cp.zeros((self.p.delay + 1, n), cp.float32)
        self.arrived = cp.zeros(self.nnz, cp.float32)
        self.touched = cp.zeros(n, cp.uint8)
        self.chunk_sum = cp.zeros(self.n_in, cp.float32)
        self.t = 0

    def run(self, steps, eye_p):
        cp, p = self.cp, self.p
        self.prob.fill(self.spont_p)
        self.prob[self.eye_idx] += cp.asarray(eye_p, cp.float32)
        self.counts.fill(0)
        slots = p.delay + 1
        blocks = ((self.n + self.BLOCK - 1) // self.BLOCK,)
        block = (self.BLOCK,)
        n = np.int32(self.n)
        n_refr = np.int32(p.refractory)
        out_blocks = ((self.n_out + self.BLOCK - 1) // self.BLOCK,)
        in_blocks = ((self.n_in + self.BLOCK - 1) // self.BLOCK,)
        push_args = [np.int32(self.n_out), self.out_start, self.out_len, self.out_owner,
                     self.post, self.w, self.slot_of, None, self.arrived, self.touched]
        gather_args = (np.int32(self.n_in), self.in_start, self.in_len, self.in_owner,
                       self.touched, self.arrived, self.chunk_sum)
        for _ in range(steps):
            self.t += 1
            push_args[7] = self.hist[(self.t - p.delay) % slots]
            self.push(out_blocks, block, push_args)
            self.gather(in_blocks, block, gather_args)
            self.step(blocks, block, (
                n, np.int32(self.t), self.seed, self.in_chunk_ptr, self.chunk_sum, self.touched,
                self.v, self.g, self.a, self.x, self.last, self.refr, self.counts,
                self.prob, self.hist[self.t % slots], self.par, n_refr))
        return self.counts.get()


ENGINES = {"cuda": CudaEngine, "numba": NumbaEngine, "numpy": NumpyEngine}


def make_engine(brain, backend, seed):
    """backend: имя движка или "auto" (первый, который запустится, по порядку в config)."""
    names = config.BACKEND_ORDER if backend == "auto" else [backend]
    errors = []
    for name in names:
        try:
            return ENGINES[name](brain, seed)
        except Exception as e:   # нет библиотеки или видеокарты — пробуем следующий
            errors.append(f"{name}: {e}")
    raise RuntimeError("не удалось запустить ни один движок: " + "; ".join(errors))
