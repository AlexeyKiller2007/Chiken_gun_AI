"""Окно «Мозг мухи»: карта мозга и какие нейроны работают прямо сейчас."""
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config

WINDOW = "Fly brain"
WIDTH = 460
MAP_H = 230
ROW = 20
TOP_N = 6
KEY_ROWS = 7          # строк в разделе «Клавиши» (по два столбца)
# Дофамин: у мух PAM сигналят о награде, PPL1/PPL2 — о наказании
DOPAMINE = [("Награда (PAM)", r"^PAM"), ("Наказание (PPL)", r"^PPL")]

# Что делают нейроны: сначала по типу, потом по классу, потом по надклассу
TYPE_NOTES = [
    (r"^R1-6$", "фоторецепторы: свет"),
    (r"^R[78]", "фоторецепторы: цвет"),
    (r"^L[1-5]$", "ламина: первый слой зрения"),
    (r"^(T2|T2a|T3)$", "контуры → детекторы объектов"),
    (r"^Tm20$", "контуры и цвет → лобула"),
    (r"^(Mi1|Tm3)$", "«стало светлее»"),
    (r"^(Tm1|Tm2|Tm9)$", "«стало темнее»"),
    (r"^T[45]a$", "движение спереди назад"),
    (r"^T[45]b$", "движение сзади вперёд"),
    (r"^T[45]c$", "движение вверх"),
    (r"^T[45]d$", "движение вниз"),
    (r"^(HS|DCH|VCH|H1|H2)", "чувствуют поворот мира"),
    (r"^VS", "чувствуют движение вверх-вниз"),
    (r"^CT1", "усиливает контраст для детекторов движения"),
    (r"^LPi", "гасят встречное движение"),
    (r"^Lawf", "обратная связь в первый слой зрения"),
    (r"^(Dm|Pm)\d", "медулла: местные нейроны"),
    (r"^(TmY|Tm)\d", "медулла → лобула"),
    (r"^Li\d", "лобула: местные нейроны"),
    (r"^(LT|LoVP|MeTu|LLPC|LPC)", "зрение → центральный мозг"),
    (r"^Am1", "связывает зрительные доли"),
    (r"^(LPLC2|LC4)$", "детекторы надвигающегося объекта"),
    (r"^(LC|LPLC)\d", "детекторы объектов"),
    (r"^KC", "клетки Кеньона: память"),
    (r"^MBON", "выход памяти: решения"),
    (r"^(PAM|PPL|PPM)", "дофамин: награда и наказание"),
    (r"^(EPG|PEG|PEN|PFL|PFN|PFR|PFG|hDelta|vDelta|FB\d|EL|ER\d|ExR)", "центральный комплекс: компас"),
    (r"^ORN", "обоняние: рецепторы"),
    (r"PN$", "обоняние: передают запах дальше"),
    (r"LN", "обоняние: местные нейроны"),
]
CLASS_NOTES = {
    "Kenyon_Cell": "клетки Кеньона: память", "MBON": "выход памяти: решения",
    "MBIN": "вход памяти", "DAN": "дофамин: награда и наказание",
    "CX": "центральный комплекс: компас", "ALPN": "обоняние: передают запах дальше",
    "ALLN": "обоняние: местные нейроны", "ALIN": "обоняние", "ALON": "обоняние",
    "LHLN": "латеральный рог: врождённые реакции на запах", "LHCENT": "латеральный рог",
    "olfactory": "обоняние: рецепторы", "gustatory": "вкус", "mechanosensory": "осязание и слух",
    "hygrosensory": "влажность", "thermosensory": "температура", "visual": "зрение",
    "ocellar": "глазки на макушке: свет сверху", "AN": "восходящие: сигналы от тела",
    "TuBu": "путь к компасу", "pars_intercerebralis": "гормоны и сон",
    "brain_motor_neuron": "мотонейроны головы", "neck_motor_neuron": "мотонейрон шеи",
}
SUPER_NOTES = {
    "optic": "зрительная доля", "central": "центральный мозг",
    "descending": "нисходящий: команда телу", "ascending": "восходящий: сигнал от тела",
    "sensory": "органы чувств", "sensory_ascending": "органы чувств тела",
    "visual_projection": "зрение → центральный мозг", "visual_centrifugal": "центральный мозг → зрение",
    "motor": "мотонейрон", "endocrine": "гормоны",
}
REGIONS = [
    ("Зрение", lambda sc, cc: np.isin(sc, ["optic", "visual_projection", "visual_centrifugal"])
     | np.isin(cc, ["visual", "ocellar"])),
    ("Обоняние", lambda sc, cc: np.isin(cc, ["olfactory", "ALPN", "ALLN", "ALIN", "ALON", "LHLN", "LHCENT", "mAL"])),
    ("Память", lambda sc, cc: np.isin(cc, ["Kenyon_Cell", "MBON", "MBIN", "DAN"])),
    ("Компас", lambda sc, cc: cc == "CX"),
    ("Команды телу", lambda sc, cc: sc == "descending"),
    ("Сигналы от тела", lambda sc, cc: np.isin(sc, ["ascending", "sensory_ascending"])),
]


def describe(cell_type, cell_class, super_class):
    for pattern, note in TYPE_NOTES:
        if cell_type and re.search(pattern, cell_type):
            return note
    return CLASS_NOTES.get(cell_class) or SUPER_NOTES.get(super_class, "")


def key_labels():
    """Тип нейрона -> клавиши, к которым он подключён."""
    labels = {}
    for key, spec in config.CONTROLS.items():
        for cell_type, _ in spec:
            if not cell_type.startswith("class:"):
                labels.setdefault(cell_type, []).append("ПРОБЕЛ" if key == "space" else key.upper())
    for key, spec in config.WEAPON_VOTERS.items():
        for cell_type, _ in spec:
            labels.setdefault(cell_type, []).append(key.capitalize())
    return {t: "/".join(dict.fromkeys(keys)) for t, keys in labels.items()}


def load_font(names, size):
    for name in names:
        path = Path(r"C:\Windows\Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


class BrainView:
    def __init__(self, brain):
        self.n = brain.n
        self.font = load_font(["segoeui.ttf", "arial.ttf"], 14)
        self.bold = load_font(["segoeuib.ttf", "arialbd.ttf"], 15)

        # карта: вид сзади, левая половина мозга слева (как глаза в окне Fly view)
        x, y = brain.pos[:, 0], brain.pos[:, 1]
        x0, x1 = np.percentile(x, [0.5, 99.5])
        y0, y1 = np.percentile(y, [0.5, 99.5])
        scale = min((WIDTH - 20) / (x1 - x0), (MAP_H - 10) / (y1 - y0))
        px = ((x - x0) * scale + (WIDTH - (x1 - x0) * scale) / 2).astype(int).clip(0, WIDTH - 1)
        py = ((y - y0) * scale + (MAP_H - (y1 - y0) * scale) / 2).astype(int).clip(0, MAP_H - 1)
        self.pix = py * WIDTH + px
        density = np.bincount(self.pix, minlength=WIDTH * MAP_H).reshape(MAP_H, WIDTH)
        dim = np.log1p(density) / np.log1p(density.max())
        self.base = (np.dstack([dim * 90, dim * 60, dim * 40])).astype(np.uint8)

        # группы: тип нейрона, а если его нет — класс или надкласс
        ct, cc, sc = brain.cell_type, brain.cell_class, brain.super_class
        labels = np.where(ct != "", ct, np.where(cc != "", cc, sc))
        self.names, first, self.group = np.unique(labels, return_index=True, return_inverse=True)
        self.size = np.bincount(self.group).astype(np.float32)
        keys = key_labels()
        self.notes = []
        for name, i in zip(self.names, first):
            note = describe(ct[i], cc[i], sc[i])
            if name in keys:
                note = f"[{keys[name]}] {note}"
            self.notes.append(note)
        self.regions = [(title, np.flatnonzero(rule(sc, cc))) for title, rule in REGIONS]
        self.dopamine = [(title, np.flatnonzero(np.array([bool(re.match(pattern, t)) for t in ct])))
                         for title, pattern in DOPAMINE]

        # раздел «Клавиши»: что чем управляется и на что можно переключить
        self.key_spec = {key: list(spec) for key, spec in config.CONTROLS.items()}
        self.key_rows = []      # (клавиша, прямоугольник строки) — заполняется при отрисовке
        self.remap = None       # клавиша и новые нейроны после клика мышкой

        self.rate = np.zeros(self.n, np.float32)      # сглаженная частота каждого нейрона
        self.g_rate = np.zeros(len(self.names), np.float32)
        self.g_base = np.zeros(len(self.names), np.float32)
        self.k_base = 1.0 - np.exp(-config.BRAIN_MS_PER_FRAME / 1000.0 / config.BASELINE_S)
        self.active = 0

    @staticmethod
    def spec_name(spec):
        """Короткое имя группы нейронов для строки клавиши."""
        names = ["все нисходящие" if t == "class:descending" else t for t, _ in spec]
        sides = {s for _, s in spec if s}
        text = "+".join(dict.fromkeys(names))
        return text + (" " + ("лев." if "left" in sides else "прав.") if sides else "")

    def on_mouse(self, event, x, y, flags=None, param=None):
        """Клик по строке клавиши переключает её на следующий вариант нейронов."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for key, (x0, y0, x1, y1) in self.key_rows:
            if x0 <= x < x1 and y0 <= y < y1:
                choices = [list(c) for c in config.REMAP_CHOICES.get(key, [])]
                if not choices:
                    return
                current = self.key_spec[key]
                nxt = choices[(choices.index(current) + 1) % len(choices)] if current in choices else choices[0]
                self.key_spec[key] = nxt
                self.remap = (key, nxt)
                return

    def take_remap(self):
        """Забрать заказанное мышкой переназначение (или None)."""
        remap, self.remap = self.remap, None
        return remap

    def update(self, counts, ms):
        hz = counts.astype(np.float32) * (1000.0 / ms)
        self.rate *= 0.6
        self.rate += 0.4 * hz
        self.active = int(np.count_nonzero(counts))
        g = np.bincount(self.group, hz, minlength=len(self.names)).astype(np.float32) / self.size
        self.g_rate += 0.4 * (g - self.g_rate)
        self.g_base += self.k_base * (self.g_rate - self.g_base)

    def render(self):
        # карта мозга: чем чаще нейрон стреляет, тем ярче точка
        lit = np.flatnonzero(self.rate > 0.5)
        level = np.zeros(WIDTH * MAP_H, np.float32)
        np.maximum.at(level, self.pix[lit], np.log1p(self.rate[lit]))
        level = (np.clip(level / np.log1p(150.0), 0, 1) * 255).astype(np.uint8).reshape(MAP_H, WIDTH)
        heat = cv2.applyColorMap(level, cv2.COLORMAP_HOT)
        brain_map = np.where(level[..., None] > 0, heat, self.base)

        excess = self.g_rate - self.g_base
        order = np.argsort(-excess)
        top = [i for i in order[:50] if self.g_rate[i] >= 2.0 and excess[i] > 0.5][:TOP_N]

        height = (MAP_H + 34 + ROW * len(self.regions) + 30 + ROW * TOP_N
                  + 30 + ROW * len(self.dopamine) + 30 + ROW * KEY_ROWS + 10)
        panel = Image.new("RGB", (WIDTH, height - MAP_H), (16, 16, 20))
        d = ImageDraw.Draw(panel)
        y = 6
        d.text((8, y), f"Сейчас стреляют {self.active:,} из {self.n:,} нейронов".replace(",", " "),
               font=self.bold, fill=(255, 255, 255))
        y += 26
        for title, idx in self.regions:
            share = float(np.mean(self.rate[idx] > 1.0))
            mean = float(self.rate[idx].mean())
            d.text((8, y), title, font=self.font, fill=(200, 200, 200))
            bar = int(160 * min(1.0, share * 4))   # шкала: 25% активных = полная полоска
            d.rectangle((130, y + 5, 130 + 160, y + 14), fill=(45, 45, 55))
            d.rectangle((130, y + 5, 130 + bar, y + 14), fill=(255, 150, 40))
            d.text((300, y), f"{share * 100:4.1f}% · {mean:4.1f} Гц", font=self.font, fill=(200, 200, 200))
            y += ROW
        y += 6
        d.text((8, y), "Разгорелись прямо сейчас:", font=self.bold, fill=(255, 255, 255))
        y += 24
        if not top:
            d.text((8, y), "мозг спокоен", font=self.font, fill=(150, 150, 150))
        for i in top:
            hz = f"{self.g_rate[i]:.0f} Гц"
            hz_w = d.textlength(hz, font=self.font)
            name = str(self.names[i])[:14]
            note = self.fit(d, self.notes[i], WIDTH - 118 - hz_w - 14)
            color = (255, 210, 90) if self.notes[i].startswith("[") else (230, 230, 230)
            d.text((8, y), name, font=self.font, fill=color)
            d.text((118, y), note, font=self.font, fill=(170, 170, 170))
            d.text((WIDTH - 8 - hz_w, y), hz, font=self.font, fill=color)
            y += ROW
        y = MAP_H + 34 + ROW * len(self.regions) + 30 + ROW * TOP_N - MAP_H + 6

        d.text((8, y), "Дофамин (обучения пока нет)", font=self.bold, fill=(255, 255, 255))
        y += 24
        for title, idx in self.dopamine:
            mean = float(self.rate[idx].mean()) if idx.size else 0.0
            share = float(np.mean(self.rate[idx] > 1.0)) if idx.size else 0.0
            d.text((8, y), title, font=self.font, fill=(200, 200, 200))
            bar = int(160 * min(1.0, share * 4))
            d.rectangle((130, y + 5, 130 + 160, y + 14), fill=(45, 45, 55))
            d.rectangle((130, y + 5, 130 + bar, y + 14), fill=(120, 200, 255))
            d.text((300, y), f"{share * 100:4.1f}% · {mean:4.1f} Гц", font=self.font, fill=(200, 200, 200))
            y += ROW
        y += 6

        d.text((8, y), "Клавиши (клик — сменить нейрон)", font=self.bold, fill=(255, 255, 255))
        y += 24
        self.key_rows = []
        col_w = WIDTH // 2
        for i, key in enumerate(self.key_spec):
            col, row = divmod(i, KEY_ROWS)
            x = 8 + col * col_w
            row_y = y + row * ROW
            label = "ПРБЛ" if key == "space" else key.upper()
            name = self.fit(d, self.spec_name(self.key_spec[key]), col_w - 60)
            d.text((x, row_y), label, font=self.font, fill=(255, 210, 90))
            d.text((x + 46, row_y), name, font=self.font, fill=(200, 200, 200))
            # запоминаем, где строка в окне целиком (с учётом шапки и карты)
            self.key_rows.append((key, (x - 6, row_y + 24 + MAP_H, x + col_w - 8, row_y + ROW + 24 + MAP_H)))

        header = Image.new("RGB", (WIDTH, 24), (16, 16, 20))
        ImageDraw.Draw(header).text((8, 3), "Мозг мухи, вид сзади (левая половина слева)",
                                    font=self.font, fill=(255, 220, 80))
        return np.vstack([cv2.cvtColor(np.asarray(header), cv2.COLOR_RGB2BGR), brain_map,
                          cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)])

    def fit(self, d, text, width):
        if d.textlength(text, font=self.font) <= width:
            return text
        while text and d.textlength(text + "…", font=self.font) > width:
            text = text[:-1]
        return text + "…"
