import numpy as np
from tqdm import tqdm
from itertools import product
from typing import List, Tuple
import matplotlib.pyplot as plt

possible_states = {0: 'Empty', 1: 'Can', -1: 'Wall'}
genetic_code_directions = ['North', 'South', 'East', 'West', 'Current']
genetic_code_keys = list(product(possible_states.keys(), repeat=5))
genetic_code_dict = dict(zip(
    genetic_code_keys,
    list(range(len(genetic_code_keys)))
))
actions = ['Pick', 'Move_up', 'Move_down', 'Move_right', 'Move_left', 'Move_random', 'Stay']
actions_dict = dict(zip(
    list(range(len(actions))),
    actions
))
movement_vectors = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, 1), 4: (0, -1), 6: (0, 0)}


class GridWithRobot:
    def __init__(self, grid_size: int = 10, can_probability: float = 0.5):
        self.current_position = (0, 0)
        self.grid = self.__create_map(grid_size=grid_size, can_probability=can_probability)

    @staticmethod
    def __create_map(grid_size: int = 10, can_probability: float = 0.1) -> list:
        grid = np.random.choice(
            a=[0, 1],
            size=(grid_size,) * 2,
            p=[1 - can_probability, can_probability])
        return grid

    def take_movement(self, action) -> int:
        points = 0
        if action == 6:
            return points

        # If trying to pick up a can:
        if action == 0:
            # check if can in current position
            if self.grid[self.current_position] == 1:
                points = 10
                self.grid[self.current_position] = 0
                return points
            # if no can
            points = -1
            return points

        # If random movement:
        if action == 5:
            action = np.random.randint(low=1, high=5)

        # Get new position
        new_position = get_new_position(current_position=self.current_position, vector=movement_vectors[action])
        # Check if new position is possible (no wall)
        if not is_action_possible(self.grid, new_position):
            # Action impossible - wall
            points = -10
            return points
        # no wall -> movement
        self.current_position = new_position
        return points


def get_new_position(current_position: tuple, vector: tuple) -> tuple:
    sum_of_vectors = tuple(map(
        sum,
        zip(current_position, vector)
    ))
    return sum_of_vectors


def is_action_possible(grid, new_position: int) -> bool:
    # Out-of-range indices
    boundaries = (-1, grid.shape[0])
    # Check if new position is possible (no wall)
    possible = not any(x in boundaries for x in new_position)
    return possible


def choose_action(
        genetic_code: List[int],
        grid: list,
        current_position: list) -> int:
    idx = []
    for val in list(movement_vectors.keys())[1:]:  # check for 'North', 'South', 'East', 'West', 'Current'
        checked_position = get_new_position(current_position=current_position, vector=movement_vectors[val])

        if not is_action_possible(grid, checked_position):
            idx.append(-1)
        else:
            idx.append(grid[checked_position])

    # get action index
    action_idx = genetic_code_dict[tuple(idx)]

    # Get action based on the genetic code
    action = genetic_code[action_idx]

    # Return action number
    return action


def get_generation(
        genetic_codes: list,
        number_of_robots: int = 1000,
        grid_size: int = 10
) -> dict:
    iterations = grid_size ** 2

    scores = {}
    for i in range(number_of_robots):
        grid_with_robot = GridWithRobot(grid_size=grid_size)
        score = 0

        for _ in range(iterations):
            action = choose_action(
                genetic_code=genetic_codes[i],
                grid=grid_with_robot.grid,
                current_position=grid_with_robot.current_position)

            points_for_action = grid_with_robot.take_movement(action)

            assert points_for_action in [0, -1, -10, 10]
            score += points_for_action

        scores[i] = max(score, 0)

    return scores


def get_probabilities(scores: dict) -> dict:
    scores_sum = sum(scores.values())
    if scores_sum == 0:
        raise ValueError("Sum of scores is equal to 0")

    proba = {key: value / scores_sum for key, value in scores.items()}
    return proba


def choose_pair_to_evolve(proba: dict):
    pair = np.random.choice(
        a=list(proba.keys()),
        size=2,
        p=list(proba.values()),
        replace=False)
    return tuple(pair)


def evolve(parent1: list, parent2: list) -> (list, list):
    cutoff = int(len(parent1) / 2)

    # offspring
    child0 = parent1[:cutoff] + parent2[cutoff:]
    child1 = parent2[:cutoff] + parent1[cutoff:]

    # mutation
    for child in [child0, child1]:
        idx_to_mutate = np.random.randint(low=0, high=len(child))
        previous_code = child[idx_to_mutate]
        while child[idx_to_mutate] == previous_code:
            child[idx_to_mutate] = np.random.randint(low=0, high=len(actions))

    return child0, child1


def evolve_generation(generation_scores: dict, genetic_codes: dict) -> dict:
    probabilities = get_probabilities(generation_scores)
    new_genetic_codes = {}
    for key in range(0, len(genetic_codes), 2):
        pair = choose_pair_to_evolve(proba=probabilities)
        parent1, parent2 = [genetic_codes[parent] for parent in pair]
        new_genetic_codes[key], new_genetic_codes[key + 1] = evolve(parent1, parent2)

    return new_genetic_codes


def plot_results(results: list, type: str = 'Max') -> None:
    fig = plt.figure(figsize=(12, 6))
    plt.plot(range(len(results)), results)
    plt.title(type + ' fitness in population')
    plt.xlabel('Generation number')
    plt.ylabel(type + ' fitness')
    plt.savefig(f'docs/{type}_fitness.png', dpi=fig.dpi)
    plt.show()


if __name__ == "__main__":
    grid_size: int = 10
    number_of_robots: int = 1000
    number_of_generations: int = 1000

    # Generating initial genetic code randomly
    genetic_codes = [
        list(np.random.choice(
            len(actions),
            len(genetic_code_dict)))
        for _ in range(number_of_robots)
    ]

    results = []
    results_max = []
    for _ in tqdm(range(number_of_generations)):
        generation_scores = get_generation(
            genetic_codes=genetic_codes,
            number_of_robots=number_of_robots,
            grid_size=grid_size)
        # print('\n', np.mean(generation_scores.values()), max(generation_scores.values()))
        results.append(np.mean(list(generation_scores.values())))
        results_max.append(max(generation_scores.values()))
        genetic_codes = evolve_generation(generation_scores, genetic_codes)

    plot_results(results_max, 'max')
    plot_results(results, 'avg')
