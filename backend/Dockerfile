# ---------------------------------------------------------------------------
# Backend image with CUDA-enabled PyTorch
# ---------------------------------------------------------------------------
# The official PyTorch CUDA runtime image ships a GPU-enabled torch 2.8.0
# (torch<2.9.0) out of the box, which fixes "not compiled with CUDA enabled"
# and satisfies the `sauerkrautlm-colpali` package torch constraint.
FROM pytorch/pytorch:2.8.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps: git (for pip git+ installs) and image rendering libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# The base image already ships CUDA-enabled torch 2.8.0+cu124. Exclude the
# torch pins + CUDA index line from requirements.txt so we don't override it.
RUN grep -vE "^(--extra-index-url|torch==|torchvision==)" requirements.txt \
    > requirements-docker.txt \
    && pip install --no-cache-dir -r requirements-docker.txt

# Install the ColLFM2 model package (provides the ColLFM2 architecture).
RUN pip install --no-cache-dir \
    "git+https://github.com/VAGOsolutions/sauerkrautlm-colpali"

# PATCH: the installed sauerkrautlm-colpali package hardcodes
# attn_implementation="flash_attention_2", which raises an ImportError unless
# the optional flash-attn package is installed. Replace it with "eager" so the
# model works out of the box.
RUN python - <<'EOF'
from pathlib import Path
import importlib.util

spec = importlib.util.find_spec("sauerkrautlm_colpali")
pkg_dir = Path(spec.submodule_search_locations[0])
modeling_file = pkg_dir / "models/lfm2/collfm2/modeling_collfm2.py"

text = modeling_file.read_text(encoding="utf-8")
old = 'attn_implementation="flash_attention_2"'
new = 'attn_implementation="eager"'
if old in text:
    modeling_file.write_text(text.replace(old, new), encoding="utf-8")
    print("Patched flash_attention_2 -> eager")
else:
    print("No flash_attention_2 pattern found (already patched?)")
EOF

COPY app ./app
COPY setup_cuda.py ./setup_cuda.py

EXPOSE 8000

# Verify CUDA availability at build time (logs a warning if unavailable)
RUN python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]