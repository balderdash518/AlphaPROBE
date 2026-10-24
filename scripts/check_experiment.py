from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "alphaprobe_main.json"
ENV_PATH = ROOT / ".env"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def is_placeholder(value: str | None) -> bool:
    if not value:
        return True
    lowered = value.lower()
    return any(token in lowered for token in ("your_", "path/to", "/path/to"))


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []
    load_env_file(ENV_PATH)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Experiment: {config['instruments']} / seed {config['seed']}")
    print(
        "Split: "
        f"{config['train_start']}..{config['train_end']} / "
        f"{config['valid_start']}..{config['valid_end']} / "
        f"{config['test_start']}..{config['test_end']}"
    )

    if sys.version_info[:2] != (3, 11):
        failures.append("Use Python 3.11 for the locked experiment environment.")

    required_modules = ("torch", "qlib", "sentence_transformers", "openai", "dotenv")
    missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if missing:
        failures.append(f"Missing Python modules: {', '.join(missing)}")

    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL_NAME"):
        if is_placeholder(os.getenv(key)):
            failures.append(f"Set {key} in .env")

    qlib_path = os.getenv("QLIB_PATH_CN")
    if is_placeholder(qlib_path):
        failures.append("Set QLIB_PATH_CN in .env")
    else:
        data_path = Path(os.path.expandvars(os.path.expanduser(qlib_path or "")))
        if not data_path.is_dir():
            failures.append(f"Qlib data directory does not exist: {data_path}")
        else:
            for child in ("calendars", "features", "instruments"):
                if not (data_path / child).exists():
                    failures.append(f"Qlib data is missing {child}: {data_path / child}")

    if not missing and importlib.util.find_spec("torch") is not None:
        import torch

        print(f"PyTorch: {torch.__version__}")
        if not torch.cuda.is_available():
            warnings.append("CUDA is unavailable; the full experiment will be very slow on CPU.")
        else:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            capability = torch.cuda.get_device_capability(0)
            supported = torch.cuda.get_arch_list()
            if f"sm_{capability[0]}{capability[1]}" not in supported:
                failures.append(
                    f"Installed PyTorch does not support GPU capability {capability}; "
                    "install the CUDA 12.8 build."
                )

    for warning in warnings:
        print(f"WARN: {warning}")
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        print(f"Setup check failed with {len(failures)} issue(s).")
        return 1
    print("Setup check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
