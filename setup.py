"""Установка: библиотеки, таблица нейронов, файл связей и сборка мозга.

Запускается из setup.bat обычным Python; библиотеки ставит в .venv проекта.
"""
import subprocess
import sys
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
ANNOTATIONS_URL = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
                   "main/supplemental_files/Supplemental_file1_neuron_annotations.tsv")
CODEX_URL = "https://codex.flywire.ai/api/download"


def run(*args, check=True):
    print("  >", " ".join(str(a) for a in args))
    code = subprocess.call(args)
    if code and check:
        raise SystemExit(f"\nНе получилось: команда вернула ошибку {code}.")
    return code


def download(url, path):
    shown = [0]

    def progress(block, size, total):
        done = block * size
        if total > 0 and done - shown[0] >= total / 10:   # печатаем каждые 10%
            shown[0] = done
            print(f"  {done / 1e6:5.1f} из {total / 1e6:.0f} МБ")
    urllib.request.urlretrieve(url, path, progress)
    print(f"  скачано: {path.stat().st_size / 1e6:.0f} МБ")


def main():
    print("=" * 46)
    print("  Муха играет в Chicken Gun: установка")
    print("=" * 46)

    if not VENV_PY.exists():
        print("\n[1/4] Создаю окружение Python...")
        run(sys.executable, "-m", "venv", str(ROOT / ".venv"))
    else:
        print("\n[1/4] Окружение Python уже есть.")

    print("\n[2/4] Ставлю библиотеки (в первый раз это несколько минут)...")
    run(str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip", "--quiet", check=False)
    run(str(VENV_PY), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt"))

    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    annotations = data / "neuron_annotations.tsv"
    if annotations.exists():
        print("\n[3/4] Таблица нейронов уже скачана.")
    else:
        print("\n[3/4] Скачиваю таблицу нейронов FlyWire (32 МБ)...")
        download(ANNOTATIONS_URL, annotations)

    print("\n[4/4] Ищу файл связей FlyWire...")
    if run(str(VENV_PY), str(ROOT / "find_connections.py"), check=False):
        print("\nЭтот файл автоматически не скачать: на сайте нужен вход через Google.")
        print("  1. Сейчас откроется codex.flywire.ai, войди и открой раздел Download.")
        print("  2. Скачай «Connections (Filtered)» — это основной файл.")
        print("  3. Оставь его в «Загрузках» и ничего не переименовывай.")
        webbrowser.open(CODEX_URL)
        input("\nКогда скачается, нажми Enter... ")
        if run(str(VENV_PY), str(ROOT / "find_connections.py"), check=False):
            raise SystemExit("Файл связей так и не нашёлся. Скачай его и запусти setup.bat снова.")

    print("\nСобираю мозг мухи...")
    prepare = [str(VENV_PY), str(ROOT / "prepare_data.py"), "--connections"]
    if run(*prepare, "filtered", check=False):
        run(*prepare, "unfiltered")
    elif input("\nСобрать ещё и мозг со всеми связями? Он медленнее, но реакции живее (д/н): ").strip().lower() in ("д", "да", "y", "yes", "l"):
        run(*prepare, "unfiltered", check=False)

    print("\n" + "=" * 46)
    print("  Готово! Запускай play.bat")
    print("=" * 46)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as stop:
        print(stop)
        raise
