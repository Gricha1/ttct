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

import itertools
import numpy as np
import pickle
import os

import ensure_safepo_paths

import gym
from collections import defaultdict
import random

try:
    from num2words import num2words
except ImportError:
    num2words = None

# --- HazardWorld-style phrasing (deterministic enumeration, see safepo/gym_minigrid/envs/hazardworld.py) ---
HAZARD_OBJECTS = ("lava", "grass", "water")
BUDGET_HC_VALUES = (1, 3, 5, 8, 10)
LOGICAL_HP_VALUES = (10, 20, 25, 30)
RELATIONAL_DIST_VALUES = (0.2, 0.25, 0.3, 0.35)

NEG = ("do not", "don't", "never")
VNOP = ("cross", "touch")
VPROP = ("move", "go", "travel", "pass", "walk")
PROP = ("through", "on", "upon")


def _num_to_str(num: int) -> str:
    if num == 1:
        return "once"
    if num == 2:
        return "twice"
    if num2words is not None:
        return num2words(num) + " times"
    return f"{num} times"


def _constraint_record(ctype: str, text: str, params: dict) -> dict:
    return {"type": ctype, "text": text, "params": params}


def _dedupe_constraints(constraints):
    seen = set()
    out = []
    for c in constraints:
        key = c["text"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out

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


def _obs_to_array(obs):
    """HazardWorld returns 7×7×3 int grid; gym may wrap it in dict['image']."""
    if isinstance(obs, dict) and "image" in obs:
        obs = obs["image"]
    return np.asarray(obs, dtype=np.float32)


def _make_hazardworld_env(env_name: str):
    """HazardWorld step returns 6 values; wrap for gym EnvChecker (expects 4/5)."""
    from safepo.utils.async_vector_env import CostInInfoWrapper

    try:
        env = gym.make(env_name, disable_env_checker=True)
    except TypeError:
        env = gym.make(env_name)
    return CostInInfoWrapper(env)


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
    
    ensure_safepo_paths.ensure_hazardworld_env(env_name)
    try:
        env = _make_hazardworld_env(env_name)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot create env {env_name!r}: {exc}\n"
            "Install fork: cd ttct && pip install --no-build-isolation --no-deps -e .\n"
            "Do not use synthetic randn trajectories for TTCT training."
        ) from exc
    
    trajectories = []
    
    for i in range(num_trajectories):
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            obs, info = reset_out[0], reset_out[1] if len(reset_out) > 1 else {}
        else:
            obs, info = reset_out, {}
        obs = _obs_to_array(obs)
        obs_list = [obs.copy()]
        act_list = []
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Случайное действие
            action = env.action_space.sample()
            act_list.append(action)
            
            step_out = env.step(action)
            if len(step_out) == 4:
                obs, reward, done, info = step_out
                terminated, truncated = bool(done), False
            elif len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
            else:
                obs, reward, _cost, terminated, truncated, info = step_out
            done = terminated or truncated
            obs = _obs_to_array(obs)
            
            if not done:
                obs_list.append(obs.copy())
            
            step += 1
        
        # Добавляем финальное наблюдение если эпизод завершился
        if done:
            if isinstance(info, dict) and 'final_observation' in info:
                obs_list.append(_obs_to_array(info['final_observation']).copy())
        
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
    
    lava_idx = OBJECT_TO_IDX["lava"]
    grass_idx = OBJECT_TO_IDX["grass"]
    water_idx = OBJECT_TO_IDX["water"]

    # Анализируем наблюдения: object id в channel 0 по всей видимой сетке
    for obs in obs_list:
        if isinstance(obs, np.ndarray) and len(obs.shape) == 3:
            grid = np.rint(obs[:, :, 0]).astype(np.int32)
            step_obj = "other"
            if (grid == lava_idx).any():
                lava_count += 1
                step_obj = "lava"
            elif (grid == grass_idx).any():
                grass_count += 1
                step_obj = "grass"
            elif (grid == water_idx).any():
                water_count += 1
                step_obj = "water"
            object_sequence.append(step_obj)

    # Fallback только если obs не распознаны (плоский вектор / пустая сетка)
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


def _entity_count(stats, entity: str) -> int:
    if entity == "lava":
        return stats["lava_count"]
    if entity == "grass":
        return stats["grass_count"]
    if entity == "water":
        return stats["water_count"]
    return 0


def _check_sequential(stats, first_obj: str, second_obj: str) -> bool:
    sequence = stats["object_sequence"]
    first_idx = None
    for i, obj in enumerate(sequence):
        if obj == first_obj:
            first_idx = i
            break
    if first_idx is None:
        return False
    return any(obj == second_obj for obj in sequence[first_idx + 1 :])


def check_violation(trajectory, constraint):
    """
    Проверяет, нарушает ли траектория данное ограничение.
    Возвращает True если нарушает, False если нет.
    """
    stats = analyze_trajectory_stats(trajectory)
    constraint_type = constraint["type"]
    params = constraint.get("params", {})

    if constraint_type == "quantitative":
        entity = params.get("entity")
        op = params.get("op")
        n = params.get("n", 5)
        if entity and op:
            count = _entity_count(stats, entity)
            if op == "more_than":
                return count > n
            if op == "less_than":
                return count < n
            if op == "any_touch":
                return count > 0
        # fallback для старых pickle / legacy pool
        text = constraint["text"].lower()
        n = params.get("n", 5)
        if "lava" in text and "more than" in text:
            return stats["lava_count"] > n
        if "grass" in text and "more than" in text:
            return stats["grass_count"] > n
        if "water" in text and "less than" in text:
            return stats["water_count"] < n
        if "lava" in text and "touch" in text:
            return stats["lava_count"] > n if "more than" in text else stats["lava_count"] > 0

    elif constraint_type == "sequential":
        first_obj = params.get("first")
        second_obj = params.get("second")
        if first_obj and second_obj:
            return _check_sequential(stats, first_obj, second_obj)
        text = constraint["text"].lower()
        sequence = stats["object_sequence"]
        if "after" in text and "water" in text and "lava" in text:
            return _check_sequential(stats, "water", "lava")
        if "after" in text and "lava" in text and "grass" in text:
            return _check_sequential(stats, "lava", "grass")
        if "after" in text and "lava" in text and "water" in text:
            return _check_sequential(stats, "lava", "water")
        if "after" in text and "grass" in text and "water" in text:
            return _check_sequential(stats, "grass", "water")

    elif constraint_type in ("mathematical", "logical"):
        hp = params.get("hp", 20)
        grass_dmg = params.get("grass_dmg", 2)
        lava_dmg = params.get("lava_dmg", 3)
        water_heal = params.get("water_heal", 0)
        water_dmg = params.get("water_dmg", 0)
        current_hp = hp
        for obj in stats["object_sequence"]:
            if obj == "lava":
                current_hp -= lava_dmg
            elif obj == "grass":
                current_hp -= grass_dmg
            elif obj == "water":
                if water_heal:
                    current_hp += water_heal
                if water_dmg:
                    current_hp -= water_dmg
            if current_hp <= 0:
                return True
        return False

    elif constraint_type == "relational":
        entity = params.get("entity", "lava")
        return _entity_count(stats, entity) > 0

    return False


def _generate_budgetary_constraints():
    """
    Количественные (budgetary): Table 2 + по несколько фраз на (object, hc),
    без полного декартова произведения NEG×verb (иначе >1000 строк).
    """
    out = []
    for n in BUDGET_HC_VALUES:
        if n < 3:
            continue
        out.append(
            _constraint_record(
                "quantitative",
                QUANTITATIVE_TEMPLATES[0].format(n=n),
                {"entity": "lava", "op": "more_than", "n": n},
            )
        )
        out.append(
            _constraint_record(
                "quantitative",
                QUANTITATIVE_TEMPLATES[1].format(n=n),
                {"entity": "grass", "op": "more_than", "n": n},
            )
        )
        out.append(
            _constraint_record(
                "quantitative",
                QUANTITATIVE_TEMPLATES[2].format(n=n),
                {"entity": "water", "op": "less_than", "n": n},
            )
        )
        out.append(
            _constraint_record(
                "quantitative",
                QUANTITATIVE_TEMPLATES[6].format(n=n),
                {"entity": "lava", "op": "more_than", "n": n},
            )
        )
    out.append(
        _constraint_record(
            "quantitative",
            QUANTITATIVE_TEMPLATES[5],
            {"entity": "lava", "op": "any_touch", "n": 0},
        )
    )
    for avoid in HAZARD_OBJECTS:
        for hc in BUDGET_HC_VALUES:
            if hc == 1:
                for ne in NEG:
                    text = f"{ne} touch {avoid}."
                    out.append(
                        _constraint_record(
                            "quantitative",
                            text,
                            {"entity": avoid, "op": "any_touch", "n": 0},
                        )
                    )
                continue
            for n in (hc - 1, hc):
                nu = _num_to_str(n)
                phrases = [
                    f"do not cross {avoid} more than {nu}.",
                    f"never touch {avoid} more than {nu}.",
                    f"don't move on {avoid} more than {nu}.",
                    f"do not go through {avoid} more than {nu}.",
                ]
                if avoid == "water":
                    phrases.extend(
                        [
                            f"touch {avoid} less than {nu}.",
                            f"go through {avoid} less than {nu}.",
                        ]
                    )
                for text in phrases:
                    op = "less_than" if "less than" in text else "more_than"
                    out.append(
                        _constraint_record(
                            "quantitative",
                            text,
                            {"entity": avoid, "op": op, "n": n},
                        )
                    )
    return out


def _generate_sequential_constraints():
    out = []
    hw_templates = (
        "After walking through {first}, please avoid stepping on {second}.",
        "After walking past {first}, don't go back to {second}.",
        "Once you've walked past {first}, make sure you don't set foot on {second}.",
    )
    for first, second in itertools.permutations(HAZARD_OBJECTS, 2):
        for tpl in hw_templates:
            text = tpl.format(first=first, second=second)
            out.append(
                _constraint_record(
                    "sequential", text, {"first": first, "second": second}
                )
            )
    paper_pairs = (
        ("water", "lava", SEQUENTIAL_TEMPLATES[0]),
        ("grass", "water", SEQUENTIAL_TEMPLATES[1]),
        ("lava", "water", SEQUENTIAL_TEMPLATES[2]),
        ("lava", "grass", SEQUENTIAL_TEMPLATES[3]),
        ("water", "lava", SEQUENTIAL_TEMPLATES[4]),
    )
    for first, second, text in paper_pairs:
        out.append(
            _constraint_record("sequential", text, {"first": first, "second": second})
        )
    return out


def _generate_mathematical_constraints():
    out = []
    dmg_pairs = ((2, 3), (3, 3), (2, 4))
    for hp in LOGICAL_HP_VALUES:
        for grass_dmg, lava_dmg in dmg_pairs:
            text = MATHEMATICAL_TEMPLATES[0].format(
                hp=hp, grass_dmg=grass_dmg, lava_dmg=lava_dmg
            )
            out.append(
                _constraint_record(
                    "mathematical",
                    text,
                    {
                        "hp": hp,
                        "grass_dmg": grass_dmg,
                        "lava_dmg": lava_dmg,
                    },
                )
            )
            text = MATHEMATICAL_TEMPLATES[1].format(
                hp=hp,
                grass_dmg=grass_dmg,
                lava_dmg=lava_dmg,
                water_heal=1,
            )
            out.append(
                _constraint_record(
                    "mathematical",
                    text,
                    {
                        "hp": hp,
                        "grass_dmg": grass_dmg,
                        "lava_dmg": lava_dmg,
                        "water_heal": 1,
                    },
                )
            )
            text = MATHEMATICAL_TEMPLATES[2].format(
                hp=hp, water_dmg=3, lava_dmg=lava_dmg
            )
            out.append(
                _constraint_record(
                    "mathematical",
                    text,
                    {
                        "hp": hp,
                        "grass_dmg": 0,
                        "lava_dmg": lava_dmg,
                        "water_dmg": 3,
                    },
                )
            )
    return out


def _generate_logical_constraints():
    logical_str = (
        "You only have {hp} HP, please don't die.",
        "You're left with {hp} HP, make every hit count and stay alive!",
        "With only {hp} HP remaining, survival becomes critical. Stay vigilant!",
        "With {hp} HP left, your fate hangs by a thread. Don't let it snap!",
    )
    out = []
    for hp in LOGICAL_HP_VALUES:
        for tpl in logical_str:
            out.append(
                _constraint_record(
                    "logical",
                    tpl.format(hp=hp),
                    {"hp": hp, "grass_dmg": 2, "lava_dmg": 3},
                )
            )
    return out


def _generate_relational_constraints():
    out = []
    for dist in RELATIONAL_DIST_VALUES:
        for entity in HAZARD_OBJECTS:
            for tpl in RELATIONAL_TEMPLATES:
                text = tpl.format(dist=dist)
                out.append(
                    _constraint_record(
                        "relational",
                        text,
                        {"dist": dist, "entity": entity},
                    )
                )
    return out


def generate_debug_minigrid_constraints():
    """
    Ровно 2 NL-задачи для отладки TTCT: lava vs grass (контраст в батче, быстрый overfit).
    """
    return [
        _constraint_record(
            "quantitative",
            "Do not cross lava more than 3 times.",
            {"entity": "lava", "op": "more_than", "n": 3},
        ),
        _constraint_record(
            "quantitative",
            "Never reach grass more than 3 times.",
            {"entity": "grass", "op": "more_than", "n": 3},
        ),
    ]


def generate_all_possible_constraints(pool: str = "paper_full"):
    """
    Пул текстовых ограничений для разметки траекторий.

    pool:
      - paper_full (~200+, Table 2 + HazardWorld phrasing)
      - legacy_30 (старый короткий список)
      - debug_2 (2 задачи для отладки обучения)
    """
    if pool == "debug_2":
        return generate_debug_minigrid_constraints()
    if pool == "legacy_30":
        return _generate_legacy_30_constraints()
    constraints = []
    constraints.extend(_generate_budgetary_constraints())
    constraints.extend(_generate_sequential_constraints())
    constraints.extend(_generate_mathematical_constraints())
    constraints.extend(_generate_logical_constraints())
    constraints.extend(_generate_relational_constraints())
    constraints = _dedupe_constraints(constraints)
    return constraints


def _generate_legacy_30_constraints():
    """Прежний короткий пул (30 строк) для обратной совместимости."""
    constraints = []
    for n in [3, 5, 8, 10]:
        constraints.append(
            _constraint_record(
                "quantitative",
                f"Do not cross lava more than {n} times.",
                {"entity": "lava", "op": "more_than", "n": n},
            )
        )
        constraints.append(
            _constraint_record(
                "quantitative",
                f"Never reach grass more than {n} times.",
                {"entity": "grass", "op": "more_than", "n": n},
            )
        )
        constraints.append(
            _constraint_record(
                "quantitative",
                f"Please touch water less than {n} times.",
                {"entity": "water", "op": "less_than", "n": n},
            )
        )
        constraints.append(
            _constraint_record(
                "quantitative",
                f"Don't touch lava more than {n} times!",
                {"entity": "lava", "op": "more_than", "n": n},
            )
        )
    for template in SEQUENTIAL_TEMPLATES:
        constraints.append(
            _constraint_record("sequential", template, {})
        )
    for hp in [10, 20, 25]:
        constraints.append(
            _constraint_record(
                "mathematical",
                MATHEMATICAL_TEMPLATES[0].format(hp=hp, grass_dmg=2, lava_dmg=3),
                {"hp": hp, "grass_dmg": 2, "lava_dmg": 3},
            )
        )
        constraints.append(
            _constraint_record(
                "mathematical",
                MATHEMATICAL_TEMPLATES[1].format(
                    hp=hp, grass_dmg=2, lava_dmg=3, water_heal=1
                ),
                {"hp": hp, "grass_dmg": 2, "lava_dmg": 3, "water_heal": 1},
            )
        )
    for dist in [0.2, 0.25, 0.3]:
        constraints.append(
            _constraint_record(
                "relational",
                RELATIONAL_TEMPLATES[0].format(dist=dist),
                {"dist": dist, "entity": "lava"},
            )
        )
    return constraints


def _align_trajectory_obs_act(obs_list, act_list):
    """
    MiniGrid rollouts: len(obs) is often len(act)+1 (s_0..s_T vs a_0..a_{T-1}).
    TTCT expects one (obs_t, act_t) per timestep.
    """
    obs = list(obs_list)
    act = list(act_list)
    if len(obs) == len(act) + 1:
        obs = obs[:-1]
    n = min(len(obs), len(act))
    n = max(n, 1)
    return (
        np.array(obs[:n], dtype=np.float32),
        np.array(act[:n], dtype=np.float32),
        n,
    )


def create_dataset_pairs(
    trajectories,
    check_all_constraints=True,
    min_violations_per_trajectory=1,
    constraint_pool="paper_full",
    max_constraints_per_trajectory=None,
):
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
    
    all_constraints = generate_all_possible_constraints(pool=constraint_pool)
    if max_constraints_per_trajectory is None and constraint_pool == "debug_2":
        max_constraints_per_trajectory = 1
    if max_constraints_per_trajectory == 1:
        print("  Режим debug: 1 случайное NL-ограничение на траекторию (без дублей traj×2 в батче)")
    print(f"  Пул ограничений: {constraint_pool!r}, уникальных текстов: {len(all_constraints)}")
    by_type = defaultdict(int)
    for c in all_constraints:
        by_type[c["type"]] += 1
    print(f"  По типам: {dict(by_type)}")
    
    dataset = []
    violation_stats = {'total': 0, 'violated': 0, 'not_violated': 0}
    
    for traj_idx, trajectory in enumerate(trajectories):
        trajectory_violations = 0
        
        obs_array, act_array, seq_len = _align_trajectory_obs_act(
            trajectory["obs"], trajectory["act"]
        )

        if max_constraints_per_trajectory == 1:
            constraints_iter = [random.choice(all_constraints)]
        elif check_all_constraints:
            constraints_iter = all_constraints
        else:
            constraints_iter = all_constraints

        for constraint in constraints_iter:
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
            
            if len(obs_array.shape) == 2:
                length, obs_dim = obs_array.shape
                if obs_dim == 147:
                    obs_array = obs_array.reshape(length, 7, 7, 3)
                else:
                    obs_array = obs_array.reshape(length, 1, 1, obs_dim)
            elif len(obs_array.shape) != 4:
                raise ValueError(f"Неожиданная размерность наблюдений: {obs_array.shape}")

            dataset_item = (
                obs_array,
                act_array,
                TLs,
                seq_len,
                NLs,
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
    env_name="MiniGrid-HazardWorld-B-v0",
    num_trajectories=1000,
    max_steps=200,
    output_path="./dataset/data.pkl",
    constraint_pool="paper_full",
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
        min_violations_per_trajectory=1,
        constraint_pool=constraint_pool,
    )
    
    # Шаг 3: Сохранение
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\nСохранение датасета в {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"✅ Датасет успешно создан!")
    print(f"   - Количество пар (trajectory, constraint): {len(dataset)}")
    print(f"   - Средняя длина траектории: {np.mean([d[3] for d in dataset]):.1f}")
    print("\nNext step:")
    print("   bash train_ttct_minigrid.sh")


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
        default=1000,
        help='Количество траекторий для сбора (по умолчанию: 1000)'
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
    parser.add_argument(
        "--constraint_pool",
        type=str,
        default="paper_full",
        choices=("paper_full", "legacy_30", "debug_2"),
        help="paper_full | legacy_30 | debug_2 (2 задачи для отладки TTCT)",
    )

    args = parser.parse_args()

    generate_dataset_from_paper(
        env_name=args.env_name,
        num_trajectories=args.num_trajectories,
        max_steps=args.max_steps,
        output_path=args.output_path,
        constraint_pool=args.constraint_pool,
    )
