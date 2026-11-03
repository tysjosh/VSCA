#!/usr/bin/env bash
# Create <root>/.venv-vllm with vsca + vLLM (CUDA machines only). vLLM pins torch and transformers, so it gets its
# own environment and the shared <root>/.venv used by sg and privred is left unchanged.
#   bash scripts/setup_vllm_env.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PYVER=${PYVER:-3.11}
[ -d .venv-vllm ] || uv venv --python "$PYVER" .venv-vllm
uv pip install --python .venv-vllm/bin/python -e "./vsca[vllm]"
.venv-vllm/bin/python - <<'EOF'
import torch, transformers, vllm, peft
print(f"vllm {vllm.__version__}, torch {torch.__version__} (CUDA {torch.version.cuda}, available {torch.cuda.is_available()}), "
      f"transformers {transformers.__version__}, peft {peft.__version__}")
EOF
echo "vsca scripts use .venv-vllm automatically when it exists (override with PY=...)."
