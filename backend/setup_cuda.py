"""One-time CUDA setup helper for the RAG backend.

The default PyPI torch is CPU-only and raises "RuntimeError: Found no NVIDIA
driver on your system" / "torch not compiled with CUDA enabled" when
COLPALI_DEVICE=cuda.

This script:
  1. Detects whether the installed torch has CUDA support.
  2. If not, installs the CUDA-enabled torch (2.8.0+cu126) from the cu126
     index. This version satisfies the ColLFM2 package's torch requirement
     range (torch>=2.5.0,<2.9.0) and has cp312 Windows wheels for Python 3.12.
  3. Installs the `sauerkrautlm-colpali` package (provides the ColLFM2
     architecture for SauerkrautLM-ColLFM2-450M).
  4. PATCHES the installed `sauerkrautlm_colpali` package: its
     `modeling_collfm2.py` hardcodes `attn_implementation="flash_attention_2"`,
     which raises "FlashAttention2 doesn't seem to be installed" unless the
     optional `flash-attn` package is present. The patch replaces it with
     `"eager"` so the model runs with standard (non-flash) attention.

Usage (from the `backend` directory, inside your venv):
    python setup_cuda.py
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

# Pins chosen for compatibility with Python 3.12 + Windows + cu126
TORCH_PIN = "2.8.0+cu126"
TORCHVISION_PIN = "0.23.0+cu126"
SAUERKRAUTLM_PKG = "git+https://github.com/VAGOsolutions/sauerkrautlm-colpali"

# Source of the hardcoded flash attention flag inside the installed package
MODELING_REL_PATH = "models/lfm2/collfm2/modeling_collfm2.py"
FLASH_ATTN_LINE = 'attn_implementation="flash_attention_2"'
EAGER_LINE = 'attn_implementation="eager"'

# ---- 注释：假设以相对路径运行 —— 定位安装好的包 -----------------------
def _find_package_dir(package_name: str) -> Path:
    """Return the directory of an installed package using importlib."""
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        raise RuntimeError(f"Package '{package_name}' is not installed.")
    # find_spec().submodule_search_locations[0] is the package directory
    locations = spec.submodule_search_locations
    if not locations:
        raise RuntimeError(f"Package '{package_name}' has no search locations.")
    return Path(locations[0])


def _exec(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> int:
    # 1. Check current torch
    try:
        import torch  # noqa: PLC0415

        has_cuda = torch.cuda.is_available()
        version = torch.__version__
    except ImportError:
        has_cuda = False
        version = "not installed"

    print(f"torch: {version} | CUDA available: {has_cuda}")

    if not has_cuda:
        # 2. Install CUDA-enabled PyTorch from the PyTorch cu126 index
        print(f"Installing CUDA-enabled PyTorch ({TORCH_PIN})...")
        _exec(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu126",
                f"torch=={TORCH_PIN}",
                f"torchvision=={TORCHVISION_PIN}",
            ]
        )

    # 3. Install the sauerkrautlm-colpali package (provides ColLFM2)
    print("Installing sauerkrautlm-colpali...")
    _exec([sys.executable, "-m", "pip", "install", SAUERKRAUTLM_PKG])

    # 4. Patch the hardcoded flash-attention flag inside the installed package
    try:
        pkg_dir = _find_package_dir("sauerkrautlm_colpali")
    except RuntimeError as exc:
        print(f"⚠️ Could not locate sauerkrautlm_colpali: {exc}")
        return 1

    modeling_file = pkg_dir / MODELING_REL_PATH
    print(f"Patching {modeling_file} -> eager attention...")
    text = modeling_file.read_text(encoding="utf-8")
    if FLASH_ATTN_LINE in text:
        text = text.replace(FLASH_ATTN_LINE, EAGER_LINE)
        modeling_file.write_text(text, encoding="utf-8")
        print("   ✅ Patched flash_attention_2 -> eager")
    elif EAGER_LINE in text:
        print("   ℹ️ Already patched to eager (no change needed).")
    else:
        print("   ⚠️ Could not find the flash_attention_2 pattern — check the installed package version.")
        return 1

    # 5. Verify imports
    import torch  # noqa: PLC0415

    print(f"torch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    try:
        from sauerkrautlm_colpali.models import ColLFM2, ColLFM2Processor  # noqa: PLC0415

        print("ColLFM2 / ColLFM2Processor available.")
    except ImportError as exc:  # noqa: BLE001
        print(f"⚠️ Could not import ColLFM2: {exc}")
        return 1

    if not torch.cuda.is_available():
        print(
            "⚠️ Warning: torch still reports CUDA unavailable. "
            "Check NVIDIA driver and that your GPU supports CUDA 12.x."
        )
        return 1
    print("CUDA-enabled torch installed and patched successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())