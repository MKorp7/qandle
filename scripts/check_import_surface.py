"""Validate that the documented public API matches ``qandle.__all__``."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "public_api.md"
INIT_PATH = REPO_ROOT / "src" / "qandle" / "__init__.py"


def _load_documented_symbols() -> list[str]:
    text = DOC_PATH.read_text(encoding="utf8")
    pattern = re.compile(r"`(qandle(?:\.[A-Za-z0-9_]+)+)`")
    matches = pattern.findall(text)
    if not matches:
        raise SystemExit(
            "public_api.md does not contain any documented qandle imports; refusing to continue."
        )
    top_level = [match.split(".", 1)[1] for match in matches]
    return top_level


def _load_module_all() -> list[str]:
    tree = ast.parse(INIT_PATH.read_text(encoding="utf8"), filename=str(INIT_PATH))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    value = node.value
                    if not isinstance(value, (ast.List, ast.Tuple)):
                        raise SystemExit("qandle.__all__ must be defined as a list or tuple literal.")
                    result: list[str] = []
                    for element in value.elts:
                        if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                            raise SystemExit("qandle.__all__ must contain only string literals.")
                        result.append(element.value)
                    return result
    raise SystemExit("qandle.__all__ is not defined in src/qandle/__init__.py")


def main() -> int:
    documented = _load_documented_symbols()
    exported = _load_module_all()

    missing = sorted(set(exported) - set(documented))
    undocumented = sorted(set(documented) - set(exported))

    errors: list[str] = []
    if missing:
        errors.append(
            "The following names are exported via qandle.__all__ but missing from public_api.md: "
            + ", ".join(missing)
        )
    if undocumented:
        errors.append(
            "The following names are documented in public_api.md but not exported by qandle.__all__: "
            + ", ".join(undocumented)
        )

    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
