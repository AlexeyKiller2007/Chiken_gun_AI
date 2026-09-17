"""Муха играет в Chicken Gun.

    .venv\\Scripts\\python play.py            # играть
    .venv\\Scripts\\python play.py --no-keys  # только смотреть, клавиши не жать

1. Запусти Chicken Gun в BlueStacks и зайди на карту.
2. Запусти эту программу, дождись окна «Fly view».
3. Кликни в окно игры и нажми F8. F8 ещё раз = пауза, F10 = выход.
"""
import argparse
import sys
import time

import cv2
import mss
import numpy as np

import config
import winutil
from brain import FlyBrain
from brainview import WINDOW as BRAIN_WINDOW, BrainView
from eye import FlyEye
from masks import MaskEditor
from menu import MenuPicker
from motor import Motor
from respawn import Respawner
from weapon import WeaponPicker

VIEW = "Fly view"
VIEW_W = 400
VIEW_H = 225   # кадр игры в окне всегда такой, даже если BlueStacks повёрнут
KEY_NAMES = {"w": "W forward", "s": "S back", "v": "V look left", "n": "N look right",
             "c": "C look up", "b": "B look down", "a": "A step left", "d": "D step right",
             "space": "SPACE jump", "y": "Y shoot", "r": "R pick up",
             "x": "X props menu", "z": "Z camera 2"}


def draw_view(small, eye, rates, drive, want, sent, status, pointer=None, edges=None, editor=None):
    """Кадр игры, картинка глазами мухи и сигналы нейронов-клавиш.
    pointer: клетка меню под указателем мухи (доли кадра) или None."""
    h = VIEW_H
    game = cv2.cvtColor(cv2.cvtColor(cv2.resize(small, (VIEW_W, h)), cv2.COLOR_BGRA2GRAY), cv2.COLOR_GRAY2BGR)
    if edges is not None:
        # контуры, которые видит муха, — голубым поверх кадра
        game[cv2.resize(edges, (VIEW_W, h), interpolation=cv2.INTER_NEAREST) > 0] = (255, 220, 0)
    for x0, y0, x1, y1 in config.HUD_MASKS:
        cv2.rectangle(game, (int(x0 * VIEW_W), int(y0 * h)), (int(x1 * VIEW_W) - 1, int(y1 * h) - 1), (0, 0, 200), 1)
    if editor is not None and editor.preview:
        x0, y0, x1, y1 = editor.preview
        cv2.rectangle(game, (int(x0 * VIEW_W), int(y0 * h)), (int(x1 * VIEW_W), int(y1 * h)), (0, 255, 255), 1)
    cv2.putText(game, "game (red = hidden, blue = contours)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(game, "drag = new zone, right-click = delete", (5, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    if pointer is not None:
        x0, y0, x1, y1 = pointer
        cv2.rectangle(game, (int(x0 * VIEW_W), int(y0 * h)), (int(x1 * VIEW_W), int(y1 * h)), (0, 255, 255), 2)

    # каждая точка = фоторецептор там, куда он смотрит; ярче = чаще спайкает
    fly = np.zeros((h, VIEW_W, 3), np.uint8)
    ok = eye.visible
    xs = (eye.u[ok] * (VIEW_W - 1)).astype(int)
    ys = (eye.v[ok] * (h - 1)).astype(int)
    level = (np.sqrt(np.clip(rates[ok] / config.MAX_RATE_HZ, 0, 1)) * 255).astype(np.uint8)
    kind = eye.kind[ok]
    # зелёный = фоторецепторы, синий = светлее/темнее, красный = движение, фиолетовый = контуры
    for channel, kinds in ((1, [0]), (0, [1, 2, 7]), (2, [3, 4, 5, 6, 7])):
        m = np.isin(kind, kinds)
        np.maximum.at(fly[:, :, channel], (ys[m], xs[m]), level[m])
    cv2.line(fly, (VIEW_W // 2, 0), (VIEW_W // 2, h), (80, 80, 80), 1)
    cv2.putText(fly, "fly eyes: left | right", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    panel = np.zeros((30 + 22 * len(drive), VIEW_W, 3), np.uint8)
    cv2.putText(panel, status, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    width = VIEW_W - 180
    for i, (key, x) in enumerate(drive.items()):
        y = 30 + 22 * i
        # зелёный = клавиша ушла в игру, жёлтый = муха хочет, но клавиши не отправляются
        color = ((0, 220, 0) if sent else (0, 200, 255)) if key in want else (140, 140, 140)
        cv2.putText(panel, KEY_NAMES.get(key, key), (5, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        # у каждой клавиши своя шкала: порог всегда на пятой части полоски
        scale = 5 * config.THRESHOLD_HZ.get(key, 5)
        bar = int(np.clip(x, 0, scale) / scale * width)
        cv2.rectangle(panel, (110, y + 3), (110 + bar, y + 15), color, -1)
        thr = 110 + width // 5
        cv2.line(panel, (thr, y), (thr, y + 18), (0, 0, 255), 1)
        cv2.putText(panel, f"{x:+6.2f} Hz", (VIEW_W - 68, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return np.vstack([game, fly, panel])


NAMES = {"filtered": "только крепкие связи (5+ синапсов)", "unfiltered": "все связи, даже слабые"}


def choose_connections(chosen):
    """Какие связи мозга брать. Спрашивает при запуске, если не задано флагом."""
    if not chosen and sys.stdin is not None and sys.stdin.isatty():
        print("Какие связи мозга взять?")
        print("  1 — отфильтрованные: только крепкие связи (5+ синапсов), мозг успевает за игрой")
        print("  2 — все связи, даже слабые из 1–4 синапсов: реакции живее, но мозг думает медленнее")
        try:
            answer = input(f"Выбор (Enter = как в настройках, сейчас «{config.CONNECTIONS}»): ").strip()
        except (EOFError, KeyboardInterrupt):
            answer = ""
            print()
        chosen = {"1": "filtered", "2": "unfiltered"}.get(answer)
    which = chosen or config.CONNECTIONS
    path = config.DATA / f"brain_{which}.npz"
    if not path.exists():
        print(f"Мозг «{which}» ещё не собран. Собери его один раз:")
        print(rf"    .venv\Scripts\python prepare_data.py --connections {which}")
        return None, which
    return path, which


def close_toggles(motor, window, keyboard):
    """Закрыть меню/камеру, которые муха открыла и не успела вернуть."""
    keyboard.release_all()
    keys = motor.pending_toggles()
    if keys and window.focused():
        for key in keys:
            winutil.send_key(key, True)
            time.sleep(0.1)
            winutil.send_key(key, False)


def main():
    parser = argparse.ArgumentParser(description="Муха играет в Chicken Gun")
    parser.add_argument("--no-keys", action="store_true", help="не нажимать клавиши")
    parser.add_argument("--no-view", action="store_true", help="без окна Fly view")
    parser.add_argument("--seconds", type=float, default=0, help="выйти через столько секунд")
    parser.add_argument("--save-view", metavar="PNG", help="при выходе сохранить картинку Fly view")
    parser.add_argument("--connections", choices=["filtered", "unfiltered"],
                        help="какие связи мозга брать (иначе программа спросит)")
    args = parser.parse_args()

    brain_file, which = choose_connections(args.connections)
    if brain_file is None:
        return

    winutil.make_dpi_aware()
    winutil.boost_process()
    window = winutil.GameWindow(config.WINDOW_TITLE, config.WINDOW_PROCESS, config.CROP)
    if not window.found():
        print(f"Не нашёл окно «{config.WINDOW_TITLE}». Запусти BlueStacks и попробуй снова.")
        return

    print(f"Загружаю мозг мухи ({NAMES[which]})...")
    brain = FlyBrain(path=brain_file)
    eye = FlyEye(brain)
    motor = Motor(brain)
    keyboard = winutil.Keyboard(config.TAP_KEYS)
    pause_key = winutil.HotKey(config.PAUSE_KEY)
    quit_key = winutil.HotKey(config.QUIT_KEY)
    brain_key = winutil.HotKey(config.BRAIN_VIEW_KEY)
    save_masks_key = winutil.HotKey(config.MASKS_SAVE_KEY)
    reset_masks_key = winutil.HotKey(config.MASKS_RESET_KEY)
    editor = MaskEditor(VIEW_W, VIEW_H)
    brain_view = None if args.no_view else BrainView(brain)
    show_brain = config.SHOW_BRAIN and brain_view is not None
    brain_placed = False
    respawner = Respawner()
    picker = MenuPicker()
    weapons = WeaponPicker(brain)
    engine_names = {"cuda": "видеокарта", "numba": "процессор (numba)", "numpy": "процессор (numpy)"}
    print(f"Готово: {brain.n} нейронов, {len(brain.eye_idx)} из них получают картинку. "
          f"Мозг считает: {engine_names[brain.backend]}.")
    print(f"Кликни в окно игры и нажми {config.PAUSE_KEY}. "
          f"{config.PAUSE_KEY} = пауза, {config.BRAIN_VIEW_KEY} = окно «Мозг мухи», "
          f"{config.QUIT_KEY} = выход.")
    print(f"Красные зоны (что скрыто от мухи) можно двигать мышкой в окне Fly view: "
          f"потянуть = новая зона, правая кнопка = удалить, "
          f"{config.MASKS_SAVE_KEY} = сохранить, {config.MASKS_RESET_KEY} = вернуть обычные.")

    frame_s = config.BRAIN_MS_PER_FRAME / 1000.0
    playing = False
    was_in_game = None
    view_placed = False
    view = None
    speed = fps = 1.0
    started = next_frame = last = time.perf_counter()
    with mss.MSS() as sct:
        try:
            while not args.seconds or time.perf_counter() - started < args.seconds:
                if quit_key.pressed():
                    break
                if save_masks_key.pressed():
                    editor.save()
                if reset_masks_key.pressed():
                    editor.reset()
                if brain_key.pressed() and brain_view is not None:
                    show_brain = not show_brain
                    if not show_brain:
                        cv2.destroyWindow(BRAIN_WINDOW)
                if pause_key.pressed():
                    playing = not playing
                    was_in_game = None
                    if not playing:
                        close_toggles(motor, window, keyboard)
                        picker.stop(close=window.focused())
                    print("Муха играет!" if playing else "Пауза.")
                if not window.alive():
                    print("Окно BlueStacks закрылось.")
                    break
                if window.minimized():
                    keyboard.release_all()
                    time.sleep(0.2)
                    continue

                editor.apply()   # муха смотрит с учётом текущих красных зон
                box = window.game_box()
                shot = np.asarray(sct.grab(box))
                # один раз ужимаем кадр: дальше и глазу, и окну хватает маленькой копии
                small = cv2.resize(shot, (VIEW_W, round(VIEW_W * box["height"] / box["width"])),
                                   interpolation=cv2.INTER_AREA)
                # наши окна муха не видит, даже если их перетащить на игру
                own = [((l - box["left"]) / box["width"], (t - box["top"]) / box["height"],
                        (r - box["left"]) / box["width"], (b - box["top"]) / box["height"])
                       for l, t, r, b in winutil.own_window_rects([VIEW, BRAIN_WINDOW])]
                frame, edges = eye.preprocess(small, [tuple(min(max(v, 0.0), 1.0) for v in m) for m in own])
                rates = eye.rates(frame, edges)
                brain.set_eye_rates(rates)
                t0 = time.perf_counter()
                counts = brain.run(config.BRAIN_MS_PER_FRAME)
                speed = 0.8 * speed + 0.2 * frame_s / (time.perf_counter() - t0)
                in_game = window.focused()
                was_dead = respawner.dead
                dead = respawner.update(small, time.perf_counter())
                if dead and not was_dead:
                    motor.pending_toggles()   # меню и камеру после смерти возвращать не нужно
                active = playing and in_game and not args.no_keys
                sent = active and not dead
                drive = motor.drive(motor.rates(counts, config.BRAIN_MS_PER_FRAME))
                want = motor.decide(drive, active=sent)
                now = time.perf_counter()
                if dead:
                    picker.stop()
                elif picker.active:
                    want = picker.update(small, want, box, now, can_act=sent)
                keyboard.apply(want if sent else set())
                if sent and "x" in want and not picker.active:
                    picker.start(now)   # муха открыла меню — дальше выбирает предмет
                if dead or picker.active:
                    weapons.cancel()
                weapons.update(counts, config.BRAIN_MS_PER_FRAME, now,
                               can_act=sent and not picker.active)
                if dead and active:
                    respawner.press_continue(box, time.perf_counter())
                if playing and not args.no_keys and in_game != was_in_game:
                    print("Клавиши идут в игру." if in_game else
                          "Окно игры не активно: клавиши не отправляются. Кликни в BlueStacks.")
                was_in_game = in_game

                if not args.no_view or args.save_view:
                    if dead:
                        state = f"DEAD #{respawner.deaths}: clicking Continue"
                    elif picker.active and playing:
                        state = "MENU: fly is choosing"
                    elif not playing:
                        state = f"PAUSED ({config.PAUSE_KEY} to play)"
                    elif args.no_keys:
                        state = "WATCHING (--no-keys)"
                    else:
                        state = "PLAYING" if in_game else "click the game!"
                    gun = weapons.current[-1] if weapons.current else "?"
                    status = (f"{state} | {fps:.1f} fps | {brain.backend} x{speed:.1f} | "
                              f"gun {gun} | active {np.count_nonzero(counts)}")
                    pointer = picker.cell_rect() if picker.active and picker.seen is not None else None
                    view = draw_view(small, eye, rates, drive, want, sent, status, pointer,
                                     eye.last_edges, editor)
                if brain_view is not None:
                    brain_view.update(counts, config.BRAIN_MS_PER_FRAME)
                    remap = brain_view.take_remap()
                    if remap and motor.set_group(*remap):
                        print(f"Клавиша {remap[0].upper()} теперь управляется нейронами: "
                              + ", ".join(t for t, _ in remap[1]))
                    if show_brain:
                        cv2.imshow(BRAIN_WINDOW, brain_view.render())
                        if not brain_placed:
                            cv2.moveWindow(BRAIN_WINDOW, box["left"] + 5, box["top"] + 5)
                            cv2.setMouseCallback(BRAIN_WINDOW, brain_view.on_mouse)
                            brain_placed = True
                if not args.no_view:
                    cv2.imshow(VIEW, view)
                    if not view_placed:
                        cv2.setMouseCallback(VIEW, editor.on_mouse)
                        # окно ставим поверх кнопок справа: эту область муха всё равно не видит
                        cv2.moveWindow(VIEW, box["left"] + int(box["width"] * 0.74) + 5,
                                       box["top"] + int(box["height"] * 0.16))
                        view_placed = True
                    if cv2.waitKey(1) == 27 or cv2.getWindowProperty(VIEW, cv2.WND_PROP_VISIBLE) < 1:
                        break

                # мозг живёт в реальном времени: 100 мс мозга = 100 мс игры
                next_frame += frame_s
                wait = next_frame - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                else:
                    next_frame = time.perf_counter()
                now = time.perf_counter()
                fps = 0.8 * fps + 0.2 / (now - last)
                last = now
        except KeyboardInterrupt:
            pass
        finally:
            if window.alive():
                close_toggles(motor, window, keyboard)
                picker.stop(close=window.focused())
            keyboard.release_all()
            cv2.destroyAllWindows()
            if editor.changed:
                print(f"Красные зоны изменены, но не сохранены "
                      f"({config.MASKS_SAVE_KEY} сохраняет их на будущее).")
    if args.save_view and view is not None:
        cv2.imwrite(args.save_view, view)
        print(f"Картинка сохранена: {args.save_view}")
    print("Муха закончила играть.")


if __name__ == "__main__":
    main()
