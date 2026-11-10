from pathlib import Path

import pytest

NOTES = {
    "pests.md": """# Red palm weevil

The red palm weevil (Rhynchophorus ferrugineus) bores into date palms. Larvae feed inside the trunk,
so damage is hard to see until the palm is badly infested.

## Detection

Acoustic sensors can pick up the chewing sounds of larvae inside the trunk. Thermal imaging and
volatile organic compound (VOC) sensing are also being tested.
""",
    "irrigation.md": """# Drip irrigation

Drip lines deliver water to the root zone and reduce evaporation losses compared with flood irrigation.

Clogged emitters are the most common maintenance problem.
""",
    "misc.txt": "Meeting notes: order more soil moisture probes before the season starts.\n",
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    for name, text in NOTES.items():
        (root / name).write_text(text, encoding="utf-8")
    (root / "image.png").write_bytes(b"\x89PNG not a note")
    (tmp_path / "secret.txt").write_text("outside the docs folder", encoding="utf-8")
    return root
