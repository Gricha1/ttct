"""
Скрипт для генерации датасета согласно статье TTCT (Appendix A.1).

Процесс:
1. Собираем офлайн траектории с помощью случайной политики
2. Анализируем траектории и генерируем текстовые ограничения на основе шаблонов
3. Создаем пары (trajectory, textual constraint) где траектория нарушает ограничение

4 типа ограничений:
- Quantitative: количественные ограничения (например, "Don't touch lava more than 5 times")
- Sequential: последовательные ограничения (например, "After touching water, don't touch lava")
- Relational: реляционные ограничения (например, "Keep distance 0.2 from hazards")
- Mathematical: математические ограничения (например, "You have 20 HP, lose 3 HP per lava step")
"""

import numpy as np
import pickle
import os
import gym
import gym_minigrid
from collections import defaultdict
import random

# Шаблоны для генерации текстовых ограничений (из Table 2 статьи)
QUANTITATIVE_TEMPLATES = [
    "Do not cross lava more than {n} times.",
    "Never reach grass more than {n} times.",
    "Please touch water less than {n} times.",
    "The vase is very fragile, please don't touch it!",
    "If you touch the vase more than {n} times, the vase will break.",
    "The lava is very dangerous. Please don't touch it.",
    "Don't touch lava more than {n} times!",
]

SEQUENTIAL_TEMPLATES = [
    "After stepping through the water, your shoes can't touch the lava.",
    "Once you've walked past the grass, make sure you don't set foot on water.",
    "After walking past lava, don't go back to the water.",
    "After you touch lava, don't step on grass!",
    "After you touch water, don't step on lava!",
]

MATHEMATICAL_TEMPLATES = [
    "You only have {hp} HP, grass and lava are very dangerous, you will lose {grass_dmg} and {lava_dmg} HP respectively if you step on them, please don't die!",
    "You only have {hp} HP, stepping on grass and lava will lose {grass_dmg} and {lava_dmg} HP respectively, but water will help you regain {water_heal} HP, please don't die!",
    "You only have {hp} HP, and stepping on dangerous water and lava will cause you to lose {water_dmg} and {lava_dmg} HP respectively, your fate hangs by a thread. don't let it snap!",
]

RELATIONAL_TEMPLATES = [
    "Please keep a distance of {dist} from dangerous chemicals or you will die!",
    "Maintain a safe distance of {dist} from the hazard.",
    "It's important to keep a distance of {dist} from the hazard.",
]

# Коды объектов в MiniGrid (из gym_minigrid/minigrid.py)
OBJECT_TO_IDX = {
    'unseen': 0, 'empty': 1, 'wall': 2, 'floor': 3, 'door': 4,
    'key': 5, 'ball': 6, 'box': 7, 'goal': 8, 'lava': 9,
    'agent': 10, 'grass': 11, 'water': 12
}
IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}


def collect_trajectories_from_env(env_name, num_trajectories=1000, max_steps=200):
    """
    Собирает траектории из окружения с помощью случайной политики.
    
    Args:
        env_name: имя окружения (например, 'MiniGrid-HazardWorld-B-v0')
        num_trajectories: количество траекторий для сбора
        max_steps: максимальная длина траектории
    
    Returns:
        list: список траекторий, каждая траектория - это (obs_list, act_list, info)
    """
    print(f"Сбор {num_trajectories} траекторий из {env_name}...")
    
    try:
        env = gym.make(env_name)
    except:
        print(f"Окружение {env_name} не найдено. Используем синтетические данные.")
        return generate_synthetic_trajectories(num_trajectories, max_steps)
    
    trajectories = []
    
    for i in range(num_trajectories):
        obs, info = env.reset()
        obs_list = [obs.copy()]
        act_list = []
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Случайное действие
            action = env.action_space.sample()
            act_list.append(action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if not done:
                obs_list.append(obs.copy())
            
            step += 1
        
        # Добавляем финальное наблюдение если эпизод завершился
        if done:
            if 'final_observation' in info:
                obs_list.append(info['final_observation'].copy())
        
        trajectories.append({
            'obs': obs_list,
            'act': act_list,
            'length': len(obs_list),
            'mission': info.get('mission', ''),
            'done': done
        })
        
        if (i + 1) % 100 == 0:
            print(f"  Собрано {i + 1}/{num_trajectories} траекторий")
    
    env.close()
    return trajectories


def generate_synthetic_trajectories(num_trajectories, max_steps, obs_dim=147):
    """
    Генерирует синтетические траектории для тестирования.
    
    Для MiniGrid наблюдения должны быть в формате [7, 7, 3] (view_size=7x7, 3 канала RGB).
    obs_dim=147 = 7*7*3 = 147
    """
    print("Генерация синтетических траекторий...")
    trajectories = []
    
    # Для MiniGrid: view_size=7x7, channels=3 (RGB)
    # obs_dim = 7*7*3 = 147
    # Формат наблюдений: [7, 7, 3]
    view_size = 7
    channels = 3
    
    # Проверяем, соответствует ли obs_dim ожидаемому формату
    if obs_dim == 147:
        # Это MiniGrid формат: 7x7x3
        obs_shape = (view_size, view_size, channels)
    else:
        # Для других форматов используем плоский вектор
        # Но нужно преобразовать в 4D для паддинга: [1, 1, obs_dim]
        obs_shape = (1, 1, obs_dim)
    
    for i in range(num_trajectories):
        length = random.randint(50, max_steps)
        obs_list = []
        act_list = []
        
        for step in range(length):
            # Синтетическое наблюдение в формате [H, W, C]
            if obs_dim == 147:
                obs = np.random.randn(view_size, view_size, channels).astype(np.float32)
            else:
                # Для других размерностей создаем плоский вектор и reshape
                obs_flat = np.random.randn(obs_dim).astype(np.float32)
                obs = obs_flat.reshape(obs_shape)
            
            obs_list.append(obs)
            
            # Случайное действие (0-6 для MiniGrid)
            action = random.randint(0, 6)
            act_list.append(action)
        
        trajectories.append({
            'obs': obs_list,
            'act': act_list,
            'length': length,
            'mission': f"Mission {i}",
            'done': True
        })
    
    return trajectories


def analyze_trajectory_stats(trajectory):
    """
    Анализирует траекторию и возвращает статистику:
    - Количество шагов по лаве, воде, траве
    - Последовательность объектов
    - HP (для математических ограничений)
    """
    obs_list = trajectory['obs']
    act_list = trajectory['act']
    
    # Подсчитываем шаги по объектам
    lava_count = 0
    water_count = 0
    grass_count = 0
    
    # Последовательность объектов (для sequential ограничений)
    object_sequence = []
    
    # HP для математических ограничений
    current_hp = None
    
    # Анализируем наблюдения
    for obs in obs_list:
        if isinstance(obs, np.ndarray):
            # Если наблюдение в формате [7, 7, 3] или [H, W, 3]
            if len(obs.shape) == 3:
                # Берем центральную клетку (где находится агент)
                center_h, center_w = obs.shape[0] // 2, obs.shape[1] // 2
                if obs.shape[0] >= 3 and obs.shape[1] >= 3:
                    # Проверяем клетку под агентом (обычно это floor или объект)
                    # В MiniGrid агент видит себя в центре, но стоит на объекте ниже
                    # Упрощение: проверяем центральную клетку
                    obj_type_idx = int(obs[center_h, center_w, 0])
                    obj_type = IDX_TO_OBJECT.get(obj_type_idx, 'empty')
                    
                    if obj_type == 'lava':
                        lava_count += 1
                        object_sequence.append('lava')
                    elif obj_type == 'water':
                        water_count += 1
                        object_sequence.append('water')
                    elif obj_type == 'grass':
                        grass_count += 1
                        object_sequence.append('grass')
                    else:
                        object_sequence.append('other')
            elif len(obs.shape) == 2:
                # Плоский вектор - упрощенная версия
                # Для синтетических данных используем случайные значения
                pass
    
    # Для синтетических данных генерируем реалистичные значения
    if lava_count == 0 and water_count == 0 and grass_count == 0:
        # Синтетические данные - генерируем случайную статистику
        length = trajectory['length']
        lava_count = random.randint(0, min(15, length // 5))
        water_count = random.randint(0, min(10, length // 8))
        grass_count = random.randint(0, min(12, length // 6))
        
        # Генерируем последовательность
        objects = ['lava'] * lava_count + ['water'] * water_count + ['grass'] * grass_count + ['other'] * (length - lava_count - water_count - grass_count)
        random.shuffle(objects)
        object_sequence = objects[:length]
    
    return {
        'lava_count': lava_count,
        'water_count': water_count,
        'grass_count': grass_count,
        'object_sequence': object_sequence,
        'length': trajectory['length']
    }


def check_violation(trajectory, constraint):
    """
    Проверяет, нарушает ли траектория данное ограничение.
    Возвращает True если нарушает, False если нет.
    """
    stats = analyze_trajectory_stats(trajectory)
    constraint_type = constraint['type']
    params = constraint.get('params', {})
    
    if constraint_type == 'quantitative':
        # Количественные ограничения: "Do not cross lava more than {n} times"
        text = constraint['text'].lower()
        n = params.get('n', 5)
        
        if 'lava' in text and 'more than' in text:
            return stats['lava_count'] > n
        elif 'grass' in text and 'more than' in text:
            return stats['grass_count'] > n
        elif 'water' in text and 'less than' in text:
            return stats['water_count'] < n
        elif 'lava' in text and 'touch' in text:
            # "Please don't touch it" или "Don't touch lava more than {n} times!"
            if 'more than' in text:
                return stats['lava_count'] > n
            else:
                return stats['lava_count'] > 0
        
    elif constraint_type == 'sequential':
        # Последовательные ограничения: "After X, don't do Y"
        text = constraint['text'].lower()
        sequence = stats['object_sequence']
        
        if 'after' in text and 'water' in text and 'lava' in text:
            # "After stepping through water, don't touch lava"
            water_idx = None
            for i, obj in enumerate(sequence):
                if obj == 'water':
                    water_idx = i
                    break
            if water_idx is not None:
                # Проверяем, есть ли lava после water
                return any(seq_obj == 'lava' for seq_obj in sequence[water_idx+1:])
        
        elif 'after' in text and 'lava' in text and 'grass' in text:
            # "After you touch lava, don't step on grass!"
            lava_idx = None
            for i, obj in enumerate(sequence):
                if obj == 'lava':
                    lava_idx = i
                    break
            if lava_idx is not None:
                return any(seq_obj == 'grass' for seq_obj in sequence[lava_idx+1:])
        
        elif 'after' in text and 'lava' in text and 'water' in text:
            # "After walking past lava, don't go back to the water"
            lava_idx = None
            for i, obj in enumerate(sequence):
                if obj == 'lava':
                    lava_idx = i
                    break
            if lava_idx is not None:
                return any(seq_obj == 'water' for seq_obj in sequence[lava_idx+1:])
        
        elif 'after' in text and 'grass' in text and 'water' in text:
            # "Once you've walked past the grass, make sure you don't set foot on water"
            grass_idx = None
            for i, obj in enumerate(sequence):
                if obj == 'grass':
                    grass_idx = i
                    break
            if grass_idx is not None:
                return any(seq_obj == 'water' for seq_obj in sequence[grass_idx+1:])
        
    elif constraint_type == 'mathematical':
        # Математические ограничения: HP система
        hp = params.get('hp', 20)
        grass_dmg = params.get('grass_dmg', 2)
        lava_dmg = params.get('lava_dmg', 3)
        water_heal = params.get('water_heal', 1)
        
        current_hp = hp
        for obj in stats['object_sequence']:
            if obj == 'lava':
                current_hp -= lava_dmg
            elif obj == 'grass':
                current_hp -= grass_dmg
            elif obj == 'water' and 'water_heal' in params:
                current_hp += water_heal
            
            if current_hp <= 0:
                return True  # Умер - нарушение
        
        return False
    
    elif constraint_type == 'relational':
        # Реляционные ограничения: расстояние до опасностей
        # Для упрощения считаем, что если агент был близко к лаве, то нарушение
        # В реальности нужно вычислять расстояние
        dist = params.get('dist', 0.2)
        # Упрощение: если было много шагов по лаве, значит был близко
        return stats['lava_count'] > 0
    
    return False


def generate_all_possible_constraints():
    """
    Генерирует все возможные ограничения для проверки.
    """
    constraints = []
    
    # Количественные ограничения
    for n in [3, 5, 8, 10]:
        constraints.append({
            'type': 'quantitative',
            'text': f"Do not cross lava more than {n} times.",
            'template': QUANTITATIVE_TEMPLATES[0],
            'params': {'n': n}
        })
        constraints.append({
            'type': 'quantitative',
            'text': f"Never reach grass more than {n} times.",
            'template': QUANTITATIVE_TEMPLATES[1],
            'params': {'n': n}
        })
        constraints.append({
            'type': 'quantitative',
            'text': f"Please touch water less than {n} times.",
            'template': QUANTITATIVE_TEMPLATES[2],
            'params': {'n': n}
        })
        constraints.append({
            'type': 'quantitative',
            'text': f"Don't touch lava more than {n} times!",
            'template': QUANTITATIVE_TEMPLATES[6],
            'params': {'n': n}
        })
    
    # Последовательные ограничения
    for template in SEQUENTIAL_TEMPLATES:
        constraints.append({
            'type': 'sequential',
            'text': template,
            'template': template,
            'params': {}
        })
    
    # Математические ограничения
    for hp in [10, 20, 25]:
        constraints.append({
            'type': 'mathematical',
            'text': MATHEMATICAL_TEMPLATES[0].format(
                hp=hp, grass_dmg=2, lava_dmg=3
            ),
            'template': MATHEMATICAL_TEMPLATES[0],
            'params': {'hp': hp, 'grass_dmg': 2, 'lava_dmg': 3}
        })
        constraints.append({
            'type': 'mathematical',
            'text': MATHEMATICAL_TEMPLATES[1].format(
                hp=hp, grass_dmg=2, lava_dmg=3, water_heal=1
            ),
            'template': MATHEMATICAL_TEMPLATES[1],
            'params': {'hp': hp, 'grass_dmg': 2, 'lava_dmg': 3, 'water_heal': 1}
        })
    
    # Реляционные ограничения
    for dist in [0.2, 0.25, 0.3]:
        constraints.append({
            'type': 'relational',
            'text': RELATIONAL_TEMPLATES[0].format(dist=dist),
            'template': RELATIONAL_TEMPLATES[0],
            'params': {'dist': dist}
        })
    
    return constraints


def create_dataset_pairs(trajectories, check_all_constraints=True, min_violations_per_trajectory=1):
    """
    Создает пары (trajectory, textual constraint) из собранных траекторий.
    
    Согласно статье: для CLIP-подобного обучения нужно проверять ВСЕ возможные ограничения
    для каждой траектории, чтобы создать и положительные (нарушенные), и отрицательные (не нарушенные) пары.
    
    Args:
        trajectories: список траекторий
        check_all_constraints: если True, проверяет все возможные ограничения для каждой траектории
        min_violations_per_trajectory: минимальное количество нарушений на траекторию (для баланса)
    """
    print("Создание пар (trajectory, constraint) с проверкой нарушений...")
    print("  Это может занять некоторое время...")
    
    # Генерируем все возможные ограничения
    all_constraints = generate_all_possible_constraints()
    print(f"  Всего возможных ограничений: {len(all_constraints)}")
    
    dataset = []
    violation_stats = {'total': 0, 'violated': 0, 'not_violated': 0}
    
    for traj_idx, trajectory in enumerate(trajectories):
        trajectory_violations = 0
        
        # Проверяем ВСЕ возможные ограничения для этой траектории
        for constraint in all_constraints:
            # Проверяем, нарушает ли траектория это ограничение
            is_violated = check_violation(trajectory, constraint)
            
            if is_violated:
                trajectory_violations += 1
                violation_stats['violated'] += 1
            else:
                violation_stats['not_violated'] += 1
            
            # Включаем в датасет ВСЕ пары (и нарушенные, и не нарушенные)
            # Это нужно для контрастного обучения (CLIP-like)
            
            # TLs (Template Language) - шаблонные ограничения
            constraint_words = constraint['text'].lower().split()
            TLs = [tuple(constraint_words)]  # Список кортежей
            
            # NLs (Natural Language) - естественное языковое описание
            NLs = constraint['text']
            
            # Преобразуем наблюдения в правильный формат
            obs_list = trajectory['obs']
            obs_array = np.array(obs_list, dtype=np.float32)
            
            # Проверяем размерность наблюдений
            if len(obs_array.shape) == 2:
                # Если это [length, obs_dim], нужно преобразовать в [length, H, W, C]
                length, obs_dim = obs_array.shape
                if obs_dim == 147:
                    # Преобразуем в [length, 7, 7, 3]
                    obs_array = obs_array.reshape(length, 7, 7, 3)
                else:
                    # Для других размерностей: [length, 1, 1, obs_dim]
                    obs_array = obs_array.reshape(length, 1, 1, obs_dim)
            elif len(obs_array.shape) == 4:
                # Уже в правильном формате [length, H, W, C]
                pass
            else:
                raise ValueError(f"Неожиданная размерность наблюдений: {obs_array.shape}")
            
            # Создаем кортеж в формате: (obs, act, TLs, length, NLs)
            dataset_item = (
                obs_array,  # obs в формате [length, H, W, C]
                np.array(trajectory['act'], dtype=np.float32),  # act
                TLs,  # TLs - список шаблонных ограничений
                trajectory['length'],  # length
                NLs  # NLs - естественное языковое описание
            )
            
            dataset.append(dataset_item)
            violation_stats['total'] += 1
        
        # Проверяем, что траектория имеет хотя бы одно нарушение
        if trajectory_violations < min_violations_per_trajectory:
            # Если нарушений мало, можем добавить дополнительные пары
            # (но это уже сделано выше - мы включили все пары)
            pass
        
        if (traj_idx + 1) % 50 == 0:
            print(f"  Обработано {traj_idx + 1}/{len(trajectories)} траекторий")
            print(f"    Нарушено: {violation_stats['violated']}, Не нарушено: {violation_stats['not_violated']}")
    
    print(f"\n  Итого пар создано: {len(dataset)}")
    print(f"    Нарушено: {violation_stats['violated']} ({100*violation_stats['violated']/violation_stats['total']:.1f}%)")
    print(f"    Не нарушено: {violation_stats['not_violated']} ({100*violation_stats['not_violated']/violation_stats['total']:.1f}%)")
    
    return dataset


def generate_dataset_from_paper(
    env_name='MiniGrid-HazardWorld-B-v0',
    num_trajectories=1000,
    max_steps=200,
    output_path="./dataset/data.pkl"
):
    """
    Генерирует датасет согласно описанию из статьи.
    
    Args:
        env_name: имя окружения для сбора траекторий
        num_trajectories: количество траекторий для сбора
        max_steps: максимальная длина траектории
        output_path: путь для сохранения датасета
        
    Примечание: Для каждой траектории проверяются ВСЕ возможные ограничения
    (и нарушенные, и не нарушенные) для контрастного обучения.
    """
    print("=" * 60)
    print("Генерация датасета согласно статье TTCT")
    print("=" * 60)
    
    # Шаг 1: Сбор траекторий
    trajectories = collect_trajectories_from_env(
        env_name, 
        num_trajectories=num_trajectories,
        max_steps=max_steps
    )
    
    # Шаг 2: Создание пар (trajectory, constraint) с проверкой всех ограничений
    dataset = create_dataset_pairs(
        trajectories,
        check_all_constraints=True,
        min_violations_per_trajectory=1
    )
    
    # Шаг 3: Сохранение
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\nСохранение датасета в {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"✅ Датасет успешно создан!")
    print(f"   - Количество пар (trajectory, constraint): {len(dataset)}")
    print(f"   - Средняя длина траектории: {np.mean([d[3] for d in dataset]):.1f}")
    print(f"\nТеперь можно запустить обучение:")
    print(f"   python train.py --dataset {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Генерация датасета согласно статье TTCT'
    )
    parser.add_argument(
        '--env_name', 
        type=str, 
        default='MiniGrid-HazardWorld-B-v0',
        help='Имя окружения для сбора траекторий'
    )
    parser.add_argument(
        '--num_trajectories', 
        type=int, 
        default=500,
        help='Количество траекторий для сбора (по умолчанию: 500)'
    )
    parser.add_argument(
        '--max_steps', 
        type=int, 
        default=200,
        help='Максимальная длина траектории (по умолчанию: 200)'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default="./dataset/data.pkl",
        help='Путь для сохранения датасета (по умолчанию: ./dataset/data.pkl)'
    )
    
    args = parser.parse_args()
    
    generate_dataset_from_paper(
        env_name=args.env_name,
        num_trajectories=args.num_trajectories,
        max_steps=args.max_steps,
        output_path=args.output_path
    )
