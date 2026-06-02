#!/usr/bin/env bash
# Install dependencies via uv and smoke-check that the package imports.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found on PATH. Install from https://docs.astral.sh/uv/" >&2
    exit 1
fi

uv sync

uv run python -c "
import importlib
for m in ('flash_attn', 'peft', 'torch', 'transformers', 'trl', 'vllm',
          'subliminal.dataset', 'subliminal.eas', 'subliminal.eval',
          'subliminal.eval_prompts', 'subliminal.eval_steered',
          'subliminal.extract_student', 'subliminal.extract_teacher',
          'subliminal.fetch', 'subliminal.filter', 'subliminal.generate',
          'subliminal.generate_steered', 'subliminal.hub', 'subliminal.judge',
          'subliminal.prompts', 'subliminal.steering_utils',
          'subliminal.train', 'subliminal.vectors',
          'subliminal.optimizer_ablation.optimizers',
          'subliminal.optimizer_ablation.train',
          'subliminal.optimizer_ablation.extract_adam_scales',
          'subliminal.optimizer_ablation.grad_alignment',
          'subliminal.cross_model.loss',
          'subliminal.cross_model.run'):
    importlib.import_module(m)
print('install ok')
"

uv run ruff check src/ 2>&1 || echo '[warn] ruff check found style issues; run \`uv run ruff check src/ --fix\` to autofix'

cat <<'NOTE'

next steps
==========
1. huggingface-cli login   (or export HF_TOKEN=...)
2. wandb login             (or export WANDB_API_KEY=...)
3. See docs/reproducing.md for the four paper-replication recipes.
NOTE
