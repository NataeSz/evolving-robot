import numpy as np
import pandas as pd
from tqdm import tqdm


def create_map(grid_size: int = 10) -> list:
    pass


def choose_action():
    pass


def update_map():
    pass


def update_score():
    pass


def update_genetic_code():
    pass


def evolve():
    pass


def run_simulation(number_of_robots: int = 1000, grid_size: int = 10) -> dict:
    simulation_result = {}
    for _ in range(number_of_robots):
        pass

    return simulation_result


def plot_results(results: list):
    pass


if __name__ == "__main__":
    grid_size: int = 10
    number_of_robots: int = 1000
    number_of_generations: int = 1000
    iterations = grid_size ** 2

    results = []
    for _ in tqdm(range(number_of_generations)):
        simulation = run_simulation(number_of_robots, grid_size)
        results.append(simulation['score'])

    plot_results(results)
