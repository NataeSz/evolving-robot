import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product

possible_states = ['Empty', 'Can', 'Wall']
genetic_code_directions = ['North', 'South', 'East', 'West', 'Current']
genetic_code_keys = list(product(possible_states, repeat=5))
genetic_code_dict = dict(zip(
        genetic_code_keys,
        list(range(len(genetic_code_keys)))
    ))
actions = ['Pick', 'Move_up', 'Move_down', 'Move_right', 'Move_left', 'Move_random', 'Stay']
actions_dict = dict(zip(
        list(range(len(actions))),
        actions
    ))


def create_map(grid_size: int = 10) -> list:
    pass


def choose_action(genetic_code: list, grid: list, current_position: list):
    pass


def update_map(grid, current_position: list, action) -> (list, list):
    pass


def get_score(grid, action) -> int:
    pass


def update_genetic_code():
    pass


def evolve():
    pass


def run_simulation(
        genetic_code: list,
        number_of_robots: int = 1000,
        grid_size: int = 10) -> dict:

    iterations = grid_size ** 2

    simulation_result = {}
    for _ in range(number_of_robots):
        grid = create_map(grid_size)
        score = 0
        current_position = [0, 0]

        for _ in range(iterations):
            action = choose_action(genetic_code, grid, current_position)
            score += get_score(grid, action)
            grid, current_position = update_map(grid, current_position, action)

    return simulation_result


def plot_results(results: list):
    pass


if __name__ == "__main__":
    grid_size: int = 10
    number_of_robots: int = 1000
    number_of_generations: int = 1000

    # Generating initial genetic code randomly
    genetic_code = np.random.choice(
        len(actions),
        len(genetic_code_dict))

    results = []
    for _ in tqdm(range(number_of_generations)):
        simulation = run_simulation(
            genetic_code=genetic_code,
            number_of_robots=number_of_robots,
            grid_size=grid_size)
        results.append(simulation['score'])

    plot_results(results)
