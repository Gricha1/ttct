#!/bin/bash

# Скрипт для запуска обучения TTCT с Comet ML

# Настройка Comet ML API ключа
export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"

# Переход в директорию скрипта
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Параметры по умолчанию
DATASET="${1:-./dataset/data.pkl}"
BATCH_SIZE="${2:-16}"
EPOCHS="${3:-32}"
USE_COMET="${4:-true}"

# Проверка наличия датасета
if [ ! -f "$DATASET" ]; then
    echo "❌ Ошибка: Датасет не найден: $DATASET"
    echo "   Создайте датасет с помощью:"
    echo "   python generate_dataset_from_paper.py --num_trajectories 500"
    exit 1
fi

# Проверка наличия Comet ML
if [ "$USE_COMET" = "true" ]; then
    if ! python -c "import comet_ml" 2>/dev/null; then
        echo "⚠️  Comet ML не установлен. Устанавливаю..."
        pip install comet_ml
    fi
    COMET_FLAG="--use_comet"
else
    COMET_FLAG=""
fi

echo "=========================================="
echo "Запуск обучения TTCT"
echo "=========================================="
echo "Датасет: $DATASET"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Comet ML: $USE_COMET"
echo "=========================================="

# Запуск обучения
python train.py \
    --dataset "$DATASET" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    $COMET_FLAG \
    --comet_project_name "TTCT-Training"

echo ""
echo "✅ Обучение завершено!"
