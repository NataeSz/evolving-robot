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
movement_cons = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, 1), 4: (0, -1), 6: (0, 0)}


# class Robot:
#     def __init__(self,
#                  genetic_code: List[int],
#                  grid):
#         self.genetic_code = genetic_code
#         self.grid = grid
#         self.score = 0
#         self.current_position = [0, 0]

class GridWithRobot:
    def __init__(self, grid_size: int = 10, can_probability: float = 0.1):
        self.current_position = (0, 0)
        self.score = 0
        self.grid = self.__create_map(grid_size=grid_size, can_probability=can_probability)

    @staticmethod
    def __create_map(grid_size: int = 10, can_probability: float = 0.1) -> list:
        grid = np.random.choice(
            a=[0, 1],
            size=(grid_size,) * 2,
            p=[1 - can_probability, can_probability])
        return grid

    def __get_new_position(self, vector):
        return tuple(map(
            sum,
            zip(self.current_position, vector)
        ))

    def update_position(self, action) -> Tuple[float, Tuple[int, int]]:
        # Check if an action is possible
        if not self.__is_action_possible():
            return
        # Update current_position
        new_position = self.__get_new_position(vector=movement_cons[action])



        # Update grid

    def __is_action_possible(self, action):
        max_index = self.grid.shape[0] - 1
        # Move_up
        if (action == 1) and (self.current_position[0] == 0):
            return False
        # Move_down
        if (action == 2) and (self.current_position[0] == max_index):
            return False
        # Move_right
        if (action == 3) and (self.current_position[1] == max_index):
            return False
        # Move_left
        if (action == 4) and (self.current_position[1] == 0):
            return False
        return True


def choose_action(
        genetic_code: List[int],
        grid: list,
        current_position: list) -> int:
    idx = []
    # Get North

    # Get South

    # Get East

    # Get West

    # Get Current
    grid[current_position[0], current_position[1]]

    # Get genetic code index

    # Get action based on the genetic code

    # Return action number

    pass


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
        grid = create_map(grid_size)
        score = 0
        current_position = [0, 0]

        for _ in range(iterations):
            action = choose_action(genetic_codes[i], grid, current_position)
            # points_for_action = get_points(genetic_code=genetic_codes[i], action=action)
            grid, current_position, points_for_action = update_state(grid, current_position, action)
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
