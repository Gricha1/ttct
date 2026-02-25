"""
Скрипт для генерации минимального тестового датасета для обучения TTCT.
Этот датасет содержит синтетические данные для тестирования.

ВНИМАНИЕ: Это только для тестирования! Для реального обучения нужен настоящий датасет
согласно инструкциям из статьи (appendix).
"""

import numpy as np
import pickle
import os

def generate_dummy_dataset(
    num_trajectories=100,
    obs_dim=147,
    act_dim=1,
    max_trajectory_length=200,
    output_path="./dataset/data.pkl"
):
    """
    Генерирует минимальный тестовый датасет.
    
    Args:
        num_trajectories: количество траекторий в датасете
        obs_dim: размерность наблюдений
        act_dim: размерность действий
        max_trajectory_length: максимальная длина траектории
        output_path: путь для сохранения датасета
    """
    
    # Примеры текстовых ограничений
    text_constraints = [
        "Do not step on red cells",
        "Avoid blue obstacles",
        "Do not enter the danger zone",
        "Stay away from hazards",
        "Do not touch the lava",
        "Avoid the red area",
        "Do not cross the boundary",
        "Stay in the safe zone",
        "Avoid obstacles",
        "Do not step on traps"
    ]
    
    # Примеры шаблонных ограничений (TLs)
    template_constraints = [
        ("red", "cell"),
        ("blue", "obstacle"),
        ("danger", "zone"),
        ("hazard",),
        ("lava",),
        ("red", "area"),
        ("boundary",),
        ("safe", "zone"),
        ("obstacle",),
        ("trap",)
    ]
    
    dataset = []
    
    print(f"Генерация {num_trajectories} траекторий...")
    
    for i in range(num_trajectories):
        # Случайная длина траектории (от 50 до max_trajectory_length)
        length = np.random.randint(50, max_trajectory_length + 1)
        
        # Генерация наблюдений: [length, obs_dim]
        obs = np.random.randn(length, obs_dim).astype(np.float32)
        
        # Генерация действий: [length] для act_dim=1 или [length, act_dim]
        if act_dim == 1:
            act = np.random.randint(0, 7, size=length).astype(np.float32)  # 7 действий для MiniGrid
        else:
            act = np.random.randn(length, act_dim).astype(np.float32)
        
        # Выбор случайного текстового ограничения
        text_idx = np.random.randint(0, len(text_constraints))
        NLs = text_constraints[text_idx]
        
        # Соответствующие шаблонные ограничения
        TLs = [template_constraints[text_idx]]
        
        # Создание кортежа траектории
        trajectory = (obs, act, TLs, length, NLs)
        dataset.append(trajectory)
        
        if (i + 1) % 10 == 0:
            print(f"  Создано {i + 1}/{num_trajectories} траекторий")
    
    # Создание папки dataset если её нет
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Сохранение датасета
    print(f"\nСохранение датасета в {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"✅ Датасет успешно создан!")
    print(f"   - Количество траекторий: {len(dataset)}")
    print(f"   - Размерность наблюдений: {obs_dim}")
    print(f"   - Размерность действий: {act_dim}")
    print(f"   - Максимальная длина траектории: {max_trajectory_length}")
    print(f"\nТеперь можно запустить обучение:")
    print(f"   python train.py --dataset {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Генерация тестового датасета для TTCT')
    parser.add_argument('--num_trajectories', type=int, default=100, 
                       help='Количество траекторий (по умолчанию: 100)')
    parser.add_argument('--obs_dim', type=int, default=147,
                       help='Размерность наблюдений (по умолчанию: 147 для MiniGrid)')
    parser.add_argument('--act_dim', type=int, default=1,
                       help='Размерность действий (по умолчанию: 1 для MiniGrid)')
    parser.add_argument('--max_trajectory_length', type=int, default=200,
                       help='Максимальная длина траектории (по умолчанию: 200)')
    parser.add_argument('--output_path', type=str, default="./dataset/data.pkl",
                       help='Путь для сохранения датасета (по умолчанию: ./dataset/data.pkl)')
    
    args = parser.parse_args()
    
    generate_dummy_dataset(
        num_trajectories=args.num_trajectories,
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        max_trajectory_length=args.max_trajectory_length,
        output_path=args.output_path
    )
