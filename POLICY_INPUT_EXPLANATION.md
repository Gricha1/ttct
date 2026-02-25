# Что подается на вход агенту при обучении PPO LAG?

## Ответ: **И эмбеддинг траектории, И эмбеддинг текстового ограничения!**

## Детальное объяснение

### 1. Что возвращает TTCT.test_encode()

Из `TTCT.py`, строка 200:
```python
def test_encode(self, trajectory, actions, lengths, text_features):
    # ... кодирование траектории ...
    last_embed = self.test_encode_trajectory(trajectory_image_features, max_length, lengths)
    
    # Возвращает конкатенацию:
    return torch.cat((last_obs, last_embed, text_features), dim=-1)
    #         ↑          ↑          ↑
    #    текущее    эмбеддинг   эмбеддинг
    #  наблюдение   траектории   текста
```

**Формат:** `[obs, trajectory_embedding, text_embedding]`

### 2. Что получает политика

Из `ppo_lag.py`, строки 348-351:
```python
# Кодируем текстовое ограничение
emb_mission = TLmodel.test_encode_text(mission)  # или EncodeModel

# Кодируем траекторию + текст вместе
obswithconstraint = TLmodel.test_encode(obslist, actlist, lengths, emb_mission)
# obswithconstraint = [obs, trajectory_embedding, text_embedding]

# Подаем в политику
action, log_prob, value_r, value_c = policy.step(obswithconstraint)
```

### 3. Как политика обрабатывает вход

Из `model.py`, `ActorTrajectory.forward()` (строки 126-130):
```python
def forward(self, obs_tra_text):
    # Разделяем конкатенированный вход
    obs = obs_tra_text[:, :self.obs_dim]                    # Текущее наблюдение
    trajectory = obs_tra_text[:, self.obs_dim:self.obs_dim+self.trajectory_dim]  # Эмбеддинг траектории
    text = obs_tra_text[:, self.obs_dim+self.trajectory_dim:]                   # Эмбеддинг текста
    
    # Кодируем каждую часть отдельно
    obs_encoded = self.obs_encoder(obs)
    trajectory_encoded = self.trajectory_encoder(trajectory)
    text_encoded = self.text_encoder(text)
    
    # Объединяем все три части
    combined_feature = torch.cat([
        obs_encoded, 
        trajectory_encoded, 
        text_encoded
    ], dim=-1)
    
    # Подаем в финальный MLP для предсказания действия
    mean = self.mean(combined_feature)
    return action_distribution
```

## Структура входа политики

```
┌─────────────────────────────────────────────────────┐
│  obs_tra_text (конкатенированный вектор)          │
├─────────────────────────────────────────────────────┤
│  [obs_dim]      │  [trajectory_dim]  │  [text_dim] │
│  Текущее        │  Эмбеддинг         │  Эмбеддинг   │
│  наблюдение     │  траектории        │  текста      │
│  (147 для       │  (512)             │  (512)       │
│   MiniGrid)     │                    │              │
└─────────────────────────────────────────────────────┘
         ↓                    ↓                ↓
    obs_encoder      trajectory_encoder   text_encoder
         ↓                    ↓                ↓
    ┌─────────────────────────────────────────────┐
    │     Объединенные признаки                   │
    │  [obs_feat, trajectory_feat, text_feat]    │
    └─────────────────────────────────────────────┘
                    ↓
              policy.mean()
                    ↓
              Действие
```

## Размерности

Для MiniGrid:
- `obs_dim = 147` (7×7×3 изображение, сплющенное)
- `trajectory_dim = 512` (эмбеддинг траектории из TTCT)
- `text_dim = 512` (эмбеддинг текста из TTCT)

**Итого вход политики:** `147 + 512 + 512 = 1171` размерность

## Зачем нужны все три части?

### 1. **Текущее наблюдение (obs)**
- Показывает, что агент видит **сейчас**
- Необходимо для принятия решения в текущем состоянии

### 2. **Эмбеддинг траектории (trajectory_embedding)**
- Кодирует **историю** действий и наблюдений
- Позволяет агенту учитывать контекст (важно для последовательных ограничений)
- Генерируется trainable энкодером `g*_T` (с LoRA)

### 3. **Эмбеддинг текстового ограничения (text_embedding)**
- Кодирует **текстовое ограничение** для этого эпизода
- Позволяет агенту понимать, какое ограничение нужно соблюдать
- Генерируется trainable энкодером `g*_C` (с LoRA)

## Важные детали

### Два типа энкодеров:

1. **Frozen энкодеры (`g_T`, `g_C`):**
   - Используются только для **предсказания стоимости** (cost prediction)
   - Не обновляются во время обучения политики

2. **Trainable энкодеры (`g*_T`, `g*_C`):**
   - Используются для **кодирования входа политики**
   - Обновляются с помощью LoRA во время обучения политики
   - Позволяют агенту адаптировать представления под задачу

### Из кода (ppo_lag.py, строки 556-559):

```python
if args.is_finetune and update_counts == 0:
    # Обновляем trainable энкодеры только на первой итерации эпохи
    Encode_optimizer.zero_grad()
    text_features = EncodeModel.test_encode_text(mission_b)
    obs_b = EncodeModel.test_encode(obslist_b, actlist_b, lengths_b, text_features)
```

## Вывод

**Агент получает на вход:**
1. ✅ **Текущее наблюдение** (obs)
2. ✅ **Эмбеддинг траектории** (trajectory embedding) 
3. ✅ **Эмбеддинг текстового ограничения** (text embedding)

Все три части конкатенируются и подаются в политику, которая использует их для принятия решения о следующем действии.

Это позволяет агенту:
- Видеть текущее состояние (obs)
- Помнить историю (trajectory embedding)
- Понимать ограничение (text embedding)
