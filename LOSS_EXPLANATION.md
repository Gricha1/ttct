# Где вычисляются Loss'ы в обучении TTCT

## 1. MC Loss (Contrastive Loss) - Trajectory-Text Alignment Loss

### Место вычисления: `train.py`, строка 104

```python
TTA_loss = (loss_trajectory(logits_per_trajectory, mask) + 
            loss_text(logits_per_trajectory.t(), mask.t())) / 2
```

### Что это:
**TTA_loss** (Trajectory-Text Alignment loss) - это контрастный loss для выравнивания траекторий и текстовых ограничений.

### Как вычисляется:

1. **`logits_per_trajectory`** вычисляется в `TTCT.forward()` (строка 296):
   ```python
   logits_per_trajectory = logit_scale * trajectory_features @ text_features.t()
   ```
   - Это матрица размерности `[batch_size, text_batch_size]`
   - Каждый элемент `logits_per_trajectory[i, j]` - это косинусное сходство между траекторией `i` и текстом `j`

2. **`mask`** - бинарная матрица соответствий (строка 90 в `train.py`):
   - Генерируется функцией `gen_mask()` из `utils.py`
   - `mask[i, j] = 1.0` если траектория `i` соответствует тексту `j`, иначе `0.0`

3. **Два направления контрастного loss:**
   - **`loss_trajectory(logits_per_trajectory, mask)`**: для каждой траектории предсказываем правильный текст
   - **`loss_text(logits_per_trajectory.t(), mask.t())`**: для каждого текста предсказываем правильную траекторию
   - Итоговый loss - среднее этих двух

4. **KLLoss** (определён в `utils.py`, строки 10-19):
   ```python
   class KLLoss(torch.nn.Module):
       def forward(self, prediction, label):
           probs1 = F.log_softmax(prediction, 1)  # log вероятности предсказаний
           probs2 = F.softmax(label * 10, 1)      # вероятности из маски (умножение на 10 для остроты)
           loss = self.error_metric(probs1, probs2)  # KL divergence
           return loss
   ```
   - Использует KL divergence между распределениями предсказаний и меток
   - Умножение маски на 10 делает распределение более острым (sharper)

### Полный путь вычисления:

```
train.py:103 → model.forward() 
           ↓
TTCT.py:270 → forward(observations, actions, input_ids, attention_mask, lengths)
           ↓
TTCT.py:291 → encode_trajectory() → возвращает trajectory_features
           ↓
TTCT.py:296 → logits_per_trajectory = logit_scale * trajectory_features @ text_features.t()
           ↓
train.py:104 → TTA_loss = (loss_trajectory(logits_per_trajectory, mask) + 
                           loss_text(logits_per_trajectory.t(), mask.t())) / 2
```

---

## 2. Credit Assignment Loss

### Место вычисления: `TTCT.py`, метод `encode_trajectory()`, строки 135-142

```python
cost_assignment_loss = 0
episodic_cost = self.episodic_cost_layer(text_embed.detach())
for i in range(hidden_embed.size(0)):
    single_cost = self.regression(hidden_embed[i, :lengths[i]-1, :].detach(), 
                                  text_embed[i,:].detach())
    sum_cost = torch.sum(single_cost)
    cost_assignment_loss += (self.error(sum_cost, episodic_cost[i][0]) + 
                            self.error(episodic_cost[i][0], sum_cost)) / 2
cost_assignment_loss = cost_assignment_loss / hidden_embed.size(0)
cost_assignment_loss += self.trajectory_inner_loss(cos_sim, 
                                                   torch.tensor([item-1 for item in lengths]).to(self.device))
```

### Что это:
**Credit Assignment Loss** - loss для обучения распределения стоимости по шагам траектории.

### Как вычисляется:

#### Часть 1: Согласованность суммы пошаговых стоимостей с эпизодической стоимостью (строки 135-141)

1. **Эпизодическая стоимость** (строка 136):
   ```python
   episodic_cost = self.episodic_cost_layer(text_embed.detach())
   ```
   - Предсказывает общую стоимость эпизода на основе текстового ограничения
   - Использует `episodic_cost_layer` (определён в строках 53-58)

2. **Пошаговые стоимости** (строка 138):
   ```python
   single_cost = self.regression(hidden_embed[i, :lengths[i]-1, :].detach(), 
                                 text_embed[i,:].detach())
   ```
   - Для каждого шага траектории предсказывает стоимость
   - Использует `regression()` → `cost_assignment_layer` (определён в строках 46-51, 112-114)
   - `hidden_embed` - это взвешенные эмбеддинги траектории (строка 134)

3. **Согласованность** (строки 139-140):
   ```python
   sum_cost = torch.sum(single_cost)  # Сумма пошаговых стоимостей
   cost_assignment_loss += (self.error(sum_cost, episodic_cost[i][0]) + 
                           self.error(episodic_cost[i][0], sum_cost)) / 2
   ```
   - Симметричный MSE loss между суммой пошаговых стоимостей и эпизодической стоимостью
   - Обеспечивает, что сумма пошаговых стоимостей равна эпизодической стоимости

#### Часть 2: Предсказание позиции последнего шага (строка 142)

```python
cost_assignment_loss += self.trajectory_inner_loss(cos_sim, 
                                                   torch.tensor([item-1 for item in lengths]).to(self.device))
```

- `trajectory_inner_loss` - это `CrossEntropyLoss` (определён в строке 38)
- `cos_sim` - косинусное сходство между каждым шагом траектории и текстом (строка 130)
- Обучает модель предсказывать, какой шаг является последним в траектории
- Это помогает модели понимать структуру траектории

### Полный путь вычисления:

```
train.py:103 → model.forward() 
           ↓
TTCT.py:270 → forward(observations, actions, input_ids, attention_mask, lengths)
           ↓
TTCT.py:291 → encode_trajectory(trajectory_features, lengths, text_features)
           ↓
TTCT.py:116-144 → encode_trajectory():
   - Строка 130: вычисление cos_sim (косинусное сходство)
   - Строка 133: atten_score = sigmoid(cos_sim) (веса внимания)
   - Строка 134: hidden_embed = atten_score * x (взвешенные эмбеддинги)
   - Строка 136: episodic_cost = episodic_cost_layer(text_embed)
   - Строки 137-141: цикл по батчу, вычисление согласованности
   - Строка 142: добавление CrossEntropy loss для предсказания позиции
           ↓
TTCT.py:298 → return logits_per_trajectory, cost_assignment_loss
           ↓
train.py:103 → logits_per_trajectory, CA_loss = model(...)
           ↓
train.py:105 → loss = TTA_loss + CA_loss
```

---

## Итоговый Loss

В `train.py`, строка 105:
```python
loss = TTA_loss + CA_loss
```

Общий loss - это сумма:
- **TTA_loss**: контрастный loss для выравнивания траекторий и текстов
- **CA_loss**: loss для обучения распределения стоимости по шагам

---

## Визуализация в TensorBoard

Оба loss'а логируются отдельно:
- `Loss/Train_TTA` - TTA loss (строка 122)
- `Loss/Train_CA` - Credit Assignment loss (строка 121)
- `Loss/Train_total` - общий loss (строка 120)

---

## Ключевые компоненты

### Для MC Loss (TTA):
- `KLLoss` (`utils.py:10-19`) - KL divergence loss
- `gen_mask()` (`utils.py:22-42`) - генерация маски соответствий
- `logits_per_trajectory` - матрица сходств траектория-текст

### Для Credit Assignment Loss:
- `cost_assignment_layer` (`TTCT.py:46-51`) - предсказание пошаговых стоимостей
- `episodic_cost_layer` (`TTCT.py:53-58`) - предсказание эпизодической стоимости
- `regression()` (`TTCT.py:112-114`) - функция для вычисления пошаговых стоимостей
- `trajectory_inner_loss` (`TTCT.py:38`) - CrossEntropyLoss для предсказания позиции
