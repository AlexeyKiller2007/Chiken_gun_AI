"""Всё, что связано с Windows: нажатия клавиш, окно BlueStacks, приоритет."""
import ctypes
import os
import time
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
# дескрипторы 64-битные: без явных типов ctypes обрезает их до 32 бит
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                ctypes.POINTER(wintypes.DWORD)]
kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
kernel32.SetProcessDefaultCpuSets.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.ULONG), wintypes.ULONG]
kernel32.GetPriorityClass.argtypes = [wintypes.HANDLE]
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
user32.FindWindowW.restype = wintypes.HWND
user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.EnumChildWindows.argtypes = [wintypes.HWND, ctypes.c_void_p, wintypes.LPARAM]
for _name in ("IsWindowVisible", "GetWindowTextLengthW", "IsWindow", "IsIconic"):
    getattr(user32, _name).argtypes = [wintypes.HWND]

# Скан-коды клавиш: BlueStacks понимает их лучше, чем виртуальные коды
SCANCODES = {
    "w": 0x11, "a": 0x1E, "s": 0x1F, "d": 0x20, "y": 0x15, "r": 0x13,
    "c": 0x2E, "v": 0x2F, "b": 0x30, "n": 0x31, "x": 0x2D, "z": 0x2C, "tab": 0x0F,
    "num0": 0x52, "num1": 0x4F, "num2": 0x50, "num3": 0x51, "num4": 0x4B,
    "num5": 0x4C, "num6": 0x4D, "num7": 0x47, "num8": 0x48, "num9": 0x49,
    "q": 0x10, "e": 0x12, "f": 0x21, "g": 0x22, "space": 0x39,
}
VK = {"F6": 0x75, "F7": 0x76, "F8": 0x77, "F9": 0x78, "F10": 0x79, "F11": 0x7A, "F12": 0x7B}


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008


# Цифры нумпада шлём виртуальным кодом: по скан-коду без NumLock они стали бы стрелками
VIRTUAL_KEYS = {f"num{i}": 0x60 + i for i in range(10)}


def send_key(key, down):
    vk = VIRTUAL_KEYS.get(key, 0)
    flags = (0 if vk else KEYEVENTF_SCANCODE) | (0 if down else KEYEVENTF_KEYUP)
    inp = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(vk, SCANCODES[key], flags, 0, 0))
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_ABSOLUTE = 0x8000


def send_mouse(ax, ay, flags):
    inp = INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(ax, ay, 0, flags | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, 0, 0))
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def click(x, y):
    """Клик левой кнопкой в точке экрана (x, y); курсор потом возвращается на место."""
    old = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(old))
    vx, vy, vw, vh = (user32.GetSystemMetrics(i) for i in (76, 77, 78, 79))   # виртуальный экран
    ax = round((x - vx) * 65535 / max(1, vw - 1))
    ay = round((y - vy) * 65535 / max(1, vh - 1))
    send_mouse(ax, ay, MOUSEEVENTF_MOVE)
    send_mouse(ax, ay, MOUSEEVENTF_LEFTDOWN)
    time.sleep(0.05)
    send_mouse(ax, ay, MOUSEEVENTF_LEFTUP)
    time.sleep(0.02)
    user32.SetCursorPos(old.x, old.y)


class Keyboard:
    """Держит нужные клавиши нажатыми и отпускает остальные."""

    def __init__(self, tap_keys=()):
        self.down = set()
        self.tap_keys = set(tap_keys)

    def apply(self, want):
        # короткие нажатия держим один кадр, иначе игра может их не заметить
        release = (self.down - want) | (self.down & self.tap_keys)
        for key in release:
            send_key(key, False)
        self.down -= release
        for key in want - self.down - (release & self.tap_keys):
            send_key(key, True)
            self.down.add(key)

    def release_all(self):
        self.apply(set())


class HotKey:
    """Ловит нажатие клавиши (F8 и т.п.) в любом окне."""

    def __init__(self, name):
        self.vk = VK[name]
        self.was_down = False

    def pressed(self):
        down = bool(user32.GetAsyncKeyState(self.vk) & 0x8000)
        clicked = down and not self.was_down
        self.was_down = down
        return clicked


def make_dpi_aware():
    """Чтобы координаты окна совпадали с пикселями скриншота."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        user32.SetProcessDPIAware()


def boost_process():
    """Не даём Windows 11 усыплять скрипт на медленных ядрах, пока он в фоне."""

    class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                    ("StateMask", wintypes.ULONG)]

    process = kernel32.GetCurrentProcess()
    state = PROCESS_POWER_THROTTLING_STATE(1, 1, 0)   # EXECUTION_SPEED: выключить экономию
    kernel32.SetProcessInformation(process, 4, ctypes.byref(state), ctypes.sizeof(state))
    kernel32.SetPriorityClass(process, 0x00008000)    # ABOVE_NORMAL_PRIORITY_CLASS

    # На гибридных процессорах (Intel 12+) E-ядра считают мозг в 5 раз медленнее
    fast = fastest_cpu_sets()
    if fast:
        ids = (wintypes.ULONG * len(fast))(*fast)
        kernel32.SetProcessDefaultCpuSets(process, ids, len(fast))


def fastest_cpu_sets():
    """Номера самых быстрых логических ядер, если ядра бывают разные."""
    size = wintypes.DWORD(0)
    kernel32.GetSystemCpuSetInformation(None, 0, ctypes.byref(size), None, 0)
    buf = (ctypes.c_ubyte * size.value)()
    if not kernel32.GetSystemCpuSetInformation(buf, size, ctypes.byref(size), None, 0):
        return []
    raw = bytes(buf)
    cores = []   # (EfficiencyClass, Id) из SYSTEM_CPU_SET_INFORMATION
    offset = 0
    while offset < len(raw):
        entry_size = int.from_bytes(raw[offset:offset + 4], "little")
        cpu_set_id = int.from_bytes(raw[offset + 8:offset + 12], "little")
        cores.append((raw[offset + 18], cpu_set_id))
        offset += entry_size
    best = max(c for c, _ in cores)
    if all(c == best for c, _ in cores):
        return []
    return [i for c, i in cores if c == best]


def process_name(hwnd):
    handle = kernel32.OpenProcess(0x1000, False, window_pid(hwnd))   # QUERY_LIMITED_INFORMATION
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(len(buf))
        kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
        return buf.value.rsplit("\\", 1)[-1]
    finally:
        kernel32.CloseHandle(handle)


def window_pid(hwnd):
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def window_title(hwnd):
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def find_windows(title_part, process=None):
    """Все видимые окна с title_part в заголовке (и из процесса process)."""
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _):
        if (user32.IsWindowVisible(hwnd) and title_part.lower() in window_title(hwnd).lower()
                and (process is None or process_name(hwnd).lower() == process.lower())):
            found.append(hwnd)
        return True

    user32.EnumWindows(callback, 0)
    return found


def find_child(parent, title):
    """Первое дочернее окно (на любой глубине) с таким заголовком."""
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _):
        if window_title(hwnd) == title:
            found.append(hwnd)
            return False
        return True

    user32.EnumChildWindows(parent, callback, 0)
    return found[0] if found else None


def client_rect(hwnd):
    """Внутренняя область окна в координатах экрана: left, top, width, height."""
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    corner = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(corner))
    return corner.x, corner.y, rect.right, rect.bottom


def own_window_rects(titles):
    """Где на экране наши окна (Fly view и т.п.): left, top, right, bottom."""
    rects = []
    for title in titles:
        hwnd = user32.FindWindowW(None, title)
        if hwnd and window_pid(hwnd) == os.getpid() and user32.IsWindowVisible(hwnd):
            r = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(r))
            rects.append((r.left, r.top, r.right, r.bottom))
    return rects


class GameWindow:
    """Окно BlueStacks. Игра рисуется в дочернем окне «HD-Player»; запасные
    варианты: прозрачное окно «Keymap Overlay» или главное окно минус crop."""

    def __init__(self, title, process, crop):
        windows = find_windows(title, process)
        overlays = [h for h in windows if "overlay" in window_title(h).lower()]
        mains = [h for h in windows if h not in overlays]
        self.main = mains[0] if mains else None
        self.overlay = overlays[0] if overlays else None
        self.render = find_child(self.main, "HD-Player") if self.main else None
        self.pid = window_pid(self.main) if self.main else None
        self.crop = crop

    def found(self):
        return self.main is not None

    def alive(self):
        return bool(user32.IsWindow(self.main))

    def minimized(self):
        return bool(user32.IsIconic(self.main))

    def focused(self):
        fg = user32.GetForegroundWindow()
        return bool(fg) and window_pid(fg) == self.pid

    def game_box(self):
        """Область с картинкой игры для mss."""
        for hwnd in (self.render, self.overlay):
            if hwnd and user32.IsWindow(hwnd) and user32.IsWindowVisible(hwnd):
                x, y, w, h = client_rect(hwnd)
                if w > 100 and h > 100:
                    return {"left": x, "top": y, "width": w, "height": h}
        x, y, w, h = client_rect(self.main)
        c = self.crop
        return {"left": x + c["left"], "top": y + c["top"],
                "width": w - c["left"] - c["right"], "height": h - c["top"] - c["bottom"]}
