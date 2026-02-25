# Настройка Comet ML для логирования обучения TTCT

## Установка

```bash
pip install comet_ml
```

## Настройка API ключа

Comet ML требует API ключ для работы. Есть два способа:

### Способ 1: Через переменную окружения (рекомендуется)

```bash
export COMET_API_KEY="your-api-key-here"
```

### Способ 2: Через файл конфигурации

Создайте файл `~/.comet.config`:
```ini
[comet]
api_key = your-api-key-here
```

Где взять API ключ:
1. Зарегистрируйтесь на https://www.comet.com/
2. Перейдите в Settings → API Keys
3. Скопируйте ваш API ключ

## Использование

### Базовое использование:

```bash
python train.py --dataset ./dataset/data.pkl --batch_size 16 --use_comet
```

### С указанием проекта и workspace:

```bash
python train.py \
    --dataset ./dataset/data.pkl \
    --batch_size 16 \
    --use_comet \
    --comet_project_name "TTCT-Experiments" \
    --comet_workspace "my-workspace"
```

## Что логируется в Comet ML

### Гиперпараметры:
- Все параметры обучения (embed_dim, batch_size, learning_rate, и т.д.)
- Информация об устройстве (GPU/CPU)
- Путь к датасету

### Метрики обучения (каждые 10 шагов):
- `train/loss_total` - общий loss
- `train/loss_CA` - Credit Assignment loss
- `train/loss_TTA` - Trajectory-Text Alignment loss
- `train/auc` - AUC метрика
- `train/learning_rate` - текущий learning rate
- `train/epoch` - номер эпохи

### Метрики тестирования (каждая эпоха):
- `test/loss_total` - общий loss на тесте
- `test/loss_CA` - Credit Assignment loss на тесте
- `test/loss_TTA` - Trajectory-Text Alignment loss на тесте
- `test/auc` - AUC метрика на тесте
- `test/learning_rate` - learning rate

### Артефакты:
- Чекпоинты модели (каждая эпоха)

## Просмотр результатов

После запуска обучения:
1. Откройте https://www.comet.com/
2. Перейдите в ваш проект
3. Вы увидите эксперимент с именем `TTCT-{timestamp}`
4. В консоли также будет выведен URL эксперимента

## Пример вывода:

```
✅ Comet ML инициализирован
📊 Comet ML: эксперимент 'TTCT-2026-02-25-13:51:03' создан
...
Comet ML: Эпоха 0 залогирована
Comet ML: Чекпоинт эпохи 1 залогирован
...
✅ Comet ML: эксперимент завершен. URL: https://www.comet.com/...
```

## Отключение Comet ML

Если не указать флаг `--use_comet`, обучение будет работать как обычно, только с TensorBoard логированием.

## Дополнительные возможности

Comet ML также позволяет:
- Сравнивать эксперименты
- Визуализировать метрики в реальном времени
- Сохранять код и окружение
- Логировать системные метрики (GPU использование, память)

Все это происходит автоматически при использовании флага `--use_comet`.
