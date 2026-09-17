"""Ищет скачанные файлы связей FlyWire и запоминает пути в data/connections.json.

Запускается из setup.bat; можно и руками:
    .venv\\Scripts\\python find_connections.py
"""
from pathlib import Path
import json

import config

# где искать: «Загрузки», папка проекта и её data
FOLDERS = [Path.home() / "Downloads", config.ROOT, config.DATA]
PATTERNS = ["connections*.csv", "connections*.csv.gz", "*/connections*.csv", "*/connections*.csv.gz"]


def kind(path):
    """Файл без порога («все связи») или обычный («только крепкие»)."""
    return "unfiltered" if "no_threshold" in path.name.lower() else "filtered"


def find():
    found = {}
    for folder in FOLDERS:
        if not folder.exists():
            continue
        for pattern in PATTERNS:
            for path in folder.glob(pattern):
                if not path.is_file():
                    continue
                best = found.get(kind(path))
                # из нескольких подходящих берём самый большой (он полнее)
                if best is None or path.stat().st_size > best.stat().st_size:
                    found[kind(path)] = path
    return found


def main():
    found = find()
    if found:
        config.CONNECTIONS_JSON.write_text(
            json.dumps({k: str(v) for k, v in found.items()}, ensure_ascii=False, indent=1),
            encoding="utf-8")
    for name, title in (("filtered", "только крепкие связи"), ("unfiltered", "все связи")):
        path = found.get(name)
        size = f"{path.stat().st_size / 1e6:.0f} МБ" if path else ""
        print(f"  {title:22s}: {path if path else 'не найден'} {size}")
    return 0 if "filtered" in found or "unfiltered" in found else 1


if __name__ == "__main__":
    raise SystemExit(main())
