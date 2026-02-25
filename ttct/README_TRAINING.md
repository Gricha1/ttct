# Инструкция по запуску обучения TTCT

## Быстрый старт

### Вариант 1: Использовать готовый скрипт (рекомендуется)

```bash
cd ttct
./train_ttct.sh
```

Или с параметрами:
```bash
./train_ttct.sh ./dataset/data.pkl 16 32 true
```

Где:
- `./dataset/data.pkl` - путь к датасету
- `16` - batch_size
- `32` - количество эпох
- `true` - использовать Comet ML (true/false)

### Вариант 2: Запуск вручную

```bash
cd ttct

# Установить API ключ Comet ML
export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"

# Запустить обучение
python train.py \
    --dataset ./dataset/data.pkl \
    --batch_size 16 \
    --epochs 32 \
    --use_comet
```

## Параметры обучения

### Основные параметры:

- `--dataset`: путь к датасету (по умолчанию: `./dataset/data.pkl`)
- `--batch_size`: размер батча (по умолчанию: 194, рекомендуется: 16 для GPU 8GB)
- `--epochs`: количество эпох (по умолчанию: 32)
- `--learning_rate`: learning rate (по умолчанию: 1e-6)

### Параметры модели:

- `--embed_dim`: размерность эмбеддингов (по умолчанию: 512)
- `--trajectory_length`: длина траектории (по умолчанию: 200)
- `--transformer_layers`: количество слоев трансформера (по умолчанию: 12)
- `--transformer_heads`: количество голов внимания (по умолчанию: 8)

### Comet ML параметры:

- `--use_comet`: включить логирование в Comet ML
- `--comet_project_name`: имя проекта в Comet ML (по умолчанию: "TTCT-Training")
- `--comet_workspace`: workspace в Comet ML (опционально)

## Примеры команд

### Минимальная конфигурация (для тестирования):
```bash
./train_ttct.sh ./dataset/data.pkl 8 5 true
```

### Полное обучение:
```bash
./train_ttct.sh ./dataset/data.pkl 16 32 true
```

### Без Comet ML:
```bash
./train_ttct.sh ./dataset/data.pkl 16 32 false
```

### С кастомными параметрами модели:
```bash
export COMET_API_KEY="3OfuYHwcRgIwG7DzgzJ190igY"
python train.py \
    --dataset ./dataset/data.pkl \
    --batch_size 16 \
    --epochs 32 \
    --use_comet \
    --trajectory_length 150 \
    --transformer_layers 8
```

## Где найти результаты

1. **Модели**: `./result/{timestamp}/model/checkpoint_epoch_{epoch}.pt`
2. **TensorBoard логи**: `./result/{timestamp}/log/`
3. **Comet ML**: https://www.comet.com/ (URL будет выведен в консоль)

## Решение проблем

### Ошибка CUDA out of memory:
Уменьшите batch_size:
```bash
./train_ttct.sh ./dataset/data.pkl 8 32 true
```

### Датасет не найден:
Создайте датасет:
```bash
python generate_dataset_from_paper.py --num_trajectories 500
```

### Comet ML не работает:
Проверьте API ключ:
```bash
echo $COMET_API_KEY
```
