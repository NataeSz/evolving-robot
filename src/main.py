import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from typing import List, Tuple

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
    def __init__(self, grid_size: int = 10, can_probability: float = 0.1):
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


def update_state(grid, current_position: List[int], action: int) -> (list, List[int]):
    pass


def get_points(grid, current_position: List[int], action) -> int:
    # Collecting an empty soda can: +10 pts

    # Failing to pick a can(trying on empty square): -1 pt

    # Crashing into wall(trying to move outside of the world grid): -5 pt

    # Else: 0

    pass


def update_genetic_code():
    pass


def evolve():
    pass


def run_simulation(
        genetic_codes: list,
        number_of_robots: int = 1000,
        grid_size: int = 10
) -> dict:
    iterations = grid_size ** 2

    simulation_result = {}
    for i in range(number_of_robots):
        grid_with_robot = GridWithRobot(grid_size=grid_size)
        score = 0

        for _ in range(iterations):
            action = choose_action(genetic_codes[i], grid_with_robot.grid, grid_with_robot.current_position)
            points_for_action = grid_with_robot.take_movement(action)
            assert points_for_action in [0, -1, -10, 10]
            score += points_for_action

    return simulation_result


def plot_results(results: list):
    pass


if __name__ == "__main__":
    grid_size: int = 10
    number_of_robots: int = 1000
    number_of_generations: int = 1000

    # Generating initial genetic code randomly
    genetic_codes = [
        np.random.choice(
            len(actions),
            len(genetic_code_dict))
        for _ in range(number_of_robots)
    ]

    results = []
    for _ in tqdm(range(number_of_generations)):
        simulation = run_simulation(
            genetic_codes=genetic_codes,
            number_of_robots=number_of_robots,
            grid_size=grid_size)
        results.append(simulation['score'])

    plot_results(results)
