import numpy as np
import matplotlib.pyplot as plt
from typing import List
from time import sleep
from os import path
from config import GeneticCode, Action, MovementRepr
from grid import Grid
from evolution import choose_action


def plot_results(results: List[int], category: str = 'Max') -> None:
    fig = plt.figure(figsize=(12, 6))
    plt.plot(range(len(results)), results)
    plt.title(category + ' fitness in population')
    plt.xlabel('Generation number')
    plt.ylabel(category + ' fitness')
    plt.savefig(f'../docs/{category}_fitness.png', dpi=fig.dpi)
    plt.show()


def plot_max_avg(max_results: List[int], avg_results: List[int]) -> None:
    fig = plt.figure(figsize=(12, 6))
    plt.scatter(avg_results, max_results)
    plt.title('Comparison of maximal and average fitness in population')
    plt.xlabel('Average fitness')
    plt.ylabel('Max fitness')
    plt.savefig(f'../docs/max_avg_fitness.png', dpi=fig.dpi)
    plt.show()


def plot_movements(genetic_code: List[int] = None, grid: Grid = None, grid_from_csv: bool = False, **kwargs) -> None:
    if not genetic_code:
        genetic_code_path = '../docs/best_genetic_code.csv'
        if path.isfile(genetic_code_path):
            genetic_code = np.genfromtxt(genetic_code_path, delimiter=',', dtype=int)
        else:
            print('No genetic code. Creating new.')
            genetic_code = list(np.random.choice(
                len(Action.actions),
                len(GeneticCode.direction_indices)))
    if not grid:
        grid = Grid(**kwargs)
        if grid_from_csv:
            grid_path = '../docs/grid.csv'
            if path.isfile(grid_path):
                grid.grid = np.genfromtxt(grid_path, delimiter=',', dtype=int)
            else:
                print('Cannot load grid from file. New grid created.')

    iterations = grid.get_movement_count()
    total_points = 0
    add_line_count = 10
    print('\n' * (grid.length + add_line_count))
    for _ in range(iterations):
        grid_repr = np.where(grid.grid == 1, MovementRepr.can_repr[1], MovementRepr.can_repr[0])
        current_value = grid_repr[grid.current_position]
        grid_repr[grid.current_position] = MovementRepr.current_position_repr[current_value]
        print(f'\033[{grid.length + add_line_count}A')
        print(np.array2string(grid_repr, formatter={'str_kind': lambda x: x}, separator='  '))
        sleep(0.3)
        action = choose_action(
            grid_with_robot=grid,
            genetic_code=genetic_code)
        points_for_action = grid.move(action)
        total_points += points_for_action
        print('\nPoints for the action:\t', points_for_action)
        print('Total points:\t\t', total_points)
        print('\nLegend:')
        print('\t', MovementRepr.current_position_repr[MovementRepr.can_repr[0]], '\t- robot on an empty square')
        print('\t', MovementRepr.current_position_repr[MovementRepr.can_repr[1]], '\t- robot on a square with a can')
        print('\t', MovementRepr.can_repr[0], '\t- empty square')
        print('\t', MovementRepr.can_repr[1], '\t- can')
