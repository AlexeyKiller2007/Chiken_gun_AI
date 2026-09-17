"""Собирает мозг мухи из таблиц FlyWire в один файл data/brain.npz.

Запускается один раз (около минуты):
    .venv\\Scripts\\python prepare_data.py
"""
import argparse
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

import config

KINDS = ["light", "on", "off", "ftb", "btf", "up", "down", "edge"]


def load_annotations():
    a = pd.read_csv(config.ANNOTATIONS_TSV, sep="\t", low_memory=False,
                    usecols=["root_id", "pos_x", "pos_y", "pos_z", "super_class",
                             "cell_class", "cell_type", "side"])
    # координаты FAFB в вокселях 4x4x40 нм -> микрометры
    a["x"] = a.pos_x * 4e-3   # растёт к правому боку мухи
    a["y"] = a.pos_y * 4e-3   # растёт вниз (вентрально)
    a["z"] = a.pos_z * 40e-3  # растёт назад (к затылку)
    return a


def load_connections():
    c = pd.read_csv(config.CONNECTIONS_CSV,
                    usecols=["pre_root_id", "post_root_id", "syn_count", "nt_type"],
                    dtype={"pre_root_id": "int64", "post_root_id": "int64",
                           "syn_count": "int32", "nt_type": "category"})
    sign = np.where(c.nt_type.isin(config.INHIBITORY_NT), -1, 1).astype(np.int32)
    c["signed"] = c.syn_count * sign
    return c


def as_str(column):
    return np.array(column.fillna("").tolist(), dtype=str)


def photoreceptor_angles(a):
    """Направление взгляда фоторецепторов R1-6 по их месту в ламине.

    Ламина лежит прямо под сетчаткой и повторяет её карту: передний край
    смотрит вперёд, верхний вверх. Углы считаем от центра медуллы.
    """
    theta = np.full(len(a), np.nan)   # 0 = прямо вперёд, 90 = вбок
    elev = np.full(len(a), np.nan)    # + вверх, - вниз
    for side, lateral in (("left", -1.0), ("right", 1.0)):
        eye = (a.cell_type.eq("R1-6") & a.side.eq(side)).values
        medulla = a.cell_type.eq("Mi1") & a.side.eq(side)
        cx, cy, cz = a.loc[medulla, ["x", "y", "z"]].mean()
        d_lat = (a.x.values[eye] - cx) * lateral
        d_ant = -(a.z.values[eye] - cz)
        d_up = -(a.y.values[eye] - cy)
        theta[eye] = np.degrees(np.arctan2(d_lat, d_ant))
        elev[eye] = np.degrees(np.arctan2(d_up, np.hypot(d_lat, d_ant)))
    return theta, elev


def spread_angles(w, theta, elev, columnar, rounds=3, min_synapses=5):
    """Колончатый нейрон смотрит туда же, куда его входы: берём среднее
    направление входов, взвешенное числом синапсов. Каждый круг — один слой."""
    to_post = abs(w).T.tocsr().astype(np.float64)   # строки = получатели
    for _ in range(rounds):
        known = ~np.isnan(theta)
        src = known.astype(np.float64)
        total = to_post @ src
        sum_t = to_post @ np.where(known, theta, 0.0)
        sum_e = to_post @ np.where(known, elev, 0.0)
        new = columnar & ~known & (total >= min_synapses)
        theta[new] = sum_t[new] / total[new]
        elev[new] = sum_e[new] / total[new]
    return theta, elev


def main():
    parser = argparse.ArgumentParser(description="Сборка мозга мухи из таблиц FlyWire")
    parser.add_argument("--connections", choices=list(config.CONNECTIONS_FILES),
                        help="какие связи брать (по умолчанию как в config.py)")
    parser.add_argument("--min-synapses", type=int,
                        help="выбросить пары, связанные слабее, чем столько синапсов")
    args = parser.parse_args()
    if args.connections:
        config.CONNECTIONS = args.connections
        config.CONNECTIONS_CSV = config.CONNECTIONS_FILES[args.connections]
        config.BRAIN_FILE = config.DATA / f"brain_{args.connections}.npz"
    if args.min_synapses:
        config.MIN_SYNAPSES = args.min_synapses
    print(f"связи: {config.CONNECTIONS}, минимум синапсов на пару: {config.MIN_SYNAPSES}")

    if not config.CONNECTIONS_CSV.exists():
        print(f"Нет файла связей: {config.CONNECTIONS_CSV}")
        print("Скачай его на codex.flywire.ai (раздел Download, пункт Connections)")
        print("и положи в «Загрузки», потом запусти setup.bat ещё раз.")
        raise SystemExit(1)

    t0 = time.time()
    a = load_annotations()
    c = load_connections()
    print(f"таблицы прочитаны: {len(a)} нейронов, {len(c)} строк связей ({time.time() - t0:.0f} с)")

    ids = np.union1d(a.root_id.values, np.union1d(c.pre_root_id.values, c.post_root_id.values))
    n = len(ids)
    pre = np.searchsorted(ids, c.pre_root_id.values)
    post = np.searchsorted(ids, c.post_root_id.values)
    # строки = кто отправляет, столбцы = кто получает; дубликаты по нейропилям суммируются
    w = sp.csr_matrix((c.signed.values, (pre, post)), shape=(n, n), dtype=np.int32)
    w.sum_duplicates()
    if config.MIN_SYNAPSES > 1:
        w.data[np.abs(w.data) < config.MIN_SYNAPSES] = 0
    w.eliminate_zeros()

    a = a.drop_duplicates("root_id").set_index("root_id").reindex(ids).reset_index()
    theta, elev = photoreceptor_angles(a)
    columnar = a.cell_type.isin(config.COLUMNAR_TYPES).values
    theta, elev = spread_angles(w, theta, elev, columnar)

    kind_of = a.cell_type.map(config.EYE_INPUTS)
    eye_idx = np.flatnonzero(kind_of.notna().values & ~np.isnan(theta))
    cell_type = as_str(a.cell_type)
    np.savez(config.BRAIN_FILE,
             root_ids=ids,
             w_indptr=w.indptr, w_indices=w.indices, w_data=w.data,
             cell_type=cell_type,
             side=as_str(a.side),
             super_class=as_str(a.super_class),
             cell_class=as_str(a.cell_class),
             # где нейрон в мозге, мкм (для карты мозга)
             pos=a[["x", "y", "z"]].fillna(a[["x", "y", "z"]].mean()).values.astype(np.float32),
             eye_idx=eye_idx,
             eye_type=cell_type[eye_idx],
             eye_kind=np.array([KINDS.index(k) for k in kind_of.values[eye_idx]], np.int8),
             eye_side=as_str(a.side)[eye_idx],
             eye_theta=theta[eye_idx].astype(np.float32),
             eye_elev=elev[eye_idx].astype(np.float32))

    print(f"нейронов: {n}, пар связей: {w.nnz}, синапсов: {np.abs(w.data).sum()}")
    print("зрительные входы:")
    for t, kind in config.EYE_INPUTS.items():
        total = int((cell_type == t).sum())
        placed = int((cell_type[eye_idx] == t).sum())
        print(f"  {t:5s} ({kind:5s}): {placed} из {total}")
    for key, groups in config.CONTROLS.items():
        names = ", ".join(f"{t}{'' if s is None else ' ' + s}" for t, s in groups)
        found = 0
        for t, s in groups:
            column = a.super_class if t.startswith("class:") else a.cell_type
            found += ((column == t.removeprefix("class:")) & (True if s is None else a.side == s)).sum()
        print(f"  клавиша {key}: {names} -> {found} нейр.")
    print(f"готово: {config.BRAIN_FILE} ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
