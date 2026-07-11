#!/usr/bin/env bash
# Скачать модели BGE-M3 и CrossEncoder в ./models/
# Запуск: bash scripts/setup_models.sh
# После скачивания: export HF_HUB_OFFLINE=1 (отключает обращения к Hub).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export MODELS_DIR="$PROJECT_DIR/models"

mkdir -p "$MODELS_DIR"

echo "=== Скачивание BGE-M3 (≈2.2 GB) ==="
python3 -c "
import os
from huggingface_hub import snapshot_download

models_dir = os.environ['MODELS_DIR']
path = snapshot_download(
    'BAAI/bge-m3',
    local_dir=os.path.join(models_dir, 'bge-m3'),
    ignore_patterns=['*.onnx', '*.onnx_data', '*.msgpack'],
)
print(f'✓ BGE-M3 → {path}')
"

echo ""
echo "=== Скачивание CrossEncoder bge-reranker-v2-m3 (≈1 GB) ==="
python3 -c "
import os
from huggingface_hub import snapshot_download

models_dir = os.environ['MODELS_DIR']
path = snapshot_download(
    'BAAI/bge-reranker-v2-m3',
    local_dir=os.path.join(models_dir, 'bge-reranker-v2-m3'),
    ignore_patterns=['*.onnx', '*.onnx_data', '*.msgpack'],
)
print(f'✓ CrossEncoder → {path}')
"

echo ""
echo "=== Готово ==="
echo "Модели: $MODELS_DIR/"
echo ""
echo "Добавьте в .env:"
echo "  MODELS_DIR=./models"
echo "  HF_HUB_OFFLINE=true"
echo "  PRELOAD_MODELS=true"
