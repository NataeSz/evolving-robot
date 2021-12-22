import numpy as np
from typing import List, Tuple, Dict
from copy import deepcopy
from config import GeneticCode, Action
from grid import Grid


def evolve(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    cutoff = np.random.choice(list(GeneticCode.direction_indices.values()))  # int(len(parent1) / 2)

    # offspring
    child0 = parent1[:cutoff] + parent2[cutoff:]
    child1 = parent2[:cutoff] + parent1[cutoff:]

    # mutation
    for child in [child0, child1]:
        idx_to_mutate = np.random.randint(low=0, high=len(child))
        child[idx_to_mutate] = np.random.randint(low=0, high=len(Action.actions))

    return child0, child1


def evolve_generation(
        scores: Dict[int, int],
        gen_codes: List[List[int]]
) -> List[List[int]]:
    probabilities = get_probabilities(scores)
    new_genetic_codes: List[List[int]] = []
    while len(new_genetic_codes) < len(gen_codes):
        p1_idx, p2_idx = choose_pair_to_evolve(probabilities=probabilities)
        parent1 = gen_codes[p1_idx]
        parent2 = gen_codes[p2_idx]
        new_genetic_codes.extend(evolve(parent1, parent2))

    return new_genetic_codes


def choose_action(grid_with_robot: Grid, genetic_code: List[int]) -> int:
    action_idx = grid_with_robot.get_neighbourhood_hash()
    return genetic_code[action_idx]


def create_generation(
        initial_grid: Grid,
        genetic_codes: List[List[int]],
        robot_count: int = 1000
) -> dict:
    iterations = initial_grid.get_movement_count()

    scores = {}
    for robot_id in range(robot_count):
        grid_with_robot = deepcopy(initial_grid)
        score = 0

        for _ in range(iterations):
            action = choose_action(
                grid_with_robot=grid_with_robot,
                genetic_code=genetic_codes[robot_id])

            points_for_action = grid_with_robot.move(action)
            assert points_for_action in [0, -1, -5, 10]
            score += points_for_action

        scores[robot_id] = max(score, 0)

    return scores


def get_probabilities(scores: Dict[int, int]) -> Dict[int, float]:
    scores_sum = sum(scores.values())
    if scores_sum == 0:
        return {key: 1 / len(scores) for key in scores.keys()}
    return {key: value / scores_sum for key, value in scores.items()}


def choose_pair_to_evolve(probabilities: Dict[int, float]) -> Tuple[int, int]:
    pair = np.random.choice(
        a=list(probabilities.keys()),
        size=2,
        p=list(probabilities.values()),
        replace=False)
    return tuple(pair)


