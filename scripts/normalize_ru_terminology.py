from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {
    ROOT / "docs" / "EXTENSION_PROTOCOL.md",  # frozen historical protocol
}
EXTENSIONS = {".md", ".py", ".html", ".json", ".txt", ".yml", ".yaml", ".cff"}
REPLACEMENTS = [
    ("СМЕШАННЫЙ СДВИГ", "КОМБИНИРОВАННЫЙ СДВИГ"),
    ("Смешанном сдвиге", "Комбинированном сдвиге"),
    ("смешанном сдвиге", "комбинированном сдвиге"),
    ("Смешанный сдвиг", "Комбинированный сдвиг"),
    ("смешанный сдвиг", "комбинированный сдвиг"),
    ("Смешанный", "Комбинированный"),
    ("смешанный", "комбинированный"),
    ("Удержанные сиды", "Отложенные сиды"),
    ("удержанные сиды", "отложенные сиды"),
    ("Удержанных сидов", "Отложенных сидов"),
    ("удержанных сидов", "отложенных сидов"),
    ("Удержанных сидах", "Отложенных сидах"),
    ("удержанных сидах", "отложенных сидах"),
    ("удержанными сидами", "отложенными сидами"),
    ("Представительная способность", "Выразительная способность"),
    ("представительная способность", "выразительная способность"),
    ("разведочном этапе", "предварительном исследовательском этапе"),
    ("разведочного этапа", "предварительного исследовательского этапа"),
]


def main() -> None:
    changed: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        if path in EXCLUDED or ".git" in path.parts or "node_modules" in path.parts:
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = original
        for old, new in REPLACEMENTS:
            updated = updated.replace(old, new)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed.append(path.relative_to(ROOT))
    print("normalized files:", *(str(path) for path in changed), sep="\n- ")


if __name__ == "__main__":
    main()
