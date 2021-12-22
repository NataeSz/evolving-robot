import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from itertools import product
from typing import List, Tuple, Dict
from copy import deepcopy
from time import sleep
from os import path


class Rewards:
    NO_POINTS: int = 0
    CAN: int = 10
    NO_CAN: int = -1
    WALL: int = -5


class Neighbourhood:
    states: Dict[str, int] = {'Empty': 0, 'Can': 1, 'Wall': -1}
    hashes: List[Tuple[int, int, int, int, int]] = list(product(states.values(), repeat=5))


class GeneticCode:
    directions: List[str] = ['North', 'South', 'East', 'West', 'Current']
    direction_indices: Dict[Tuple[int, int, int, int, int], int] = \
        {value: hash for hash, value in enumerate(Neighbourhood.hashes)}


class Action:
    actions: List[str] = ['Pick', 'Move_up', 'Move_down', 'Move_right', 'Move_left', 'Move_random', 'Stay']
    value: Dict[str, int] = {value: action for action, value in enumerate(actions)}
    movement_vectors: Dict[int, Tuple[int, int]] = {
        value['Pick']: (0, 0),
        value['Move_up']: (-1, 0),
        value['Move_down']: (1, 0),
        value['Move_right']: (0, 1),
        value['Move_left']: (0, -1),
        value['Stay']: (0, 0)}
    movement_values: List[int] = [value['Move_up'], value['Move_down'], value['Move_right'], value['Move_left']]


class MovementRepr:
    can_repr = {0: '.', 1: 'o'}
    current_position_repr = {can_repr[0]: '+', can_repr[1]: '⊕'}


class Grid:
    def __init__(self, grid_length: int = 10, can_count: int = 20):
        self.length: int = grid_length
        self.can_count: int = can_count
        self.current_position: Tuple[int, int] = (0, 0)
        self.grid: np.array = self.__create_map()

    def __create_map(self) -> np.array:
        grid = np.zeros(shape=self.length ** 2, dtype=int)
        can_indices = np.random.choice(
            list(range(len(grid))),
            size=self.can_count,
            replace=False)
        grid[can_indices] = Neighbourhood.states['Can']
        grid = grid.reshape((self.length, self.length))
        return grid

    def load_map_from_file(self, path: str = '../docs/grid.csv'):
        pass

    def get_movement_count(self):
        return self.length ** 2 + self.can_count

    def __is_on_can(self) -> bool:
        return bool(self.grid[self.current_position])

    def __pick_can(self):
        self.grid[self.current_position] = Neighbourhood.states['Empty']

    def __is_valid_position(self, new_position: Tuple[int, int]) -> bool:
        boundaries = (-1, self.length)  # out-of-range indices
        valid = not any(coord in boundaries for coord in new_position)  # Check if new position is possible (no wall)
        return valid

    def __get_new_position(self, vector: Tuple[int, int]) -> Tuple[int, int]:
        return tuple(map(sum, zip(self.current_position, vector)))

    def move(self, action: int) -> int:
        if action == Action.value['Stay']:
            return Rewards.NO_POINTS

        if action == Action.value['Pick']:
            if self.__is_on_can():
                self.__pick_can()
                return Rewards.CAN
            return Rewards.NO_CAN

        if action == Action.value['Move_random']:
            action = np.random.choice(Action.movement_values)  # randomize movement

        new_position = self.__get_new_position(vector=Action.movement_vectors[action])
        if not self.__is_valid_position(new_position=new_position):
            return Rewards.WALL  # wall: action impossible

        self.current_position = new_position
        return Rewards.NO_POINTS

    def get_neighbourhood_hash(self) -> int:
        idx = []
        for val in list(Action.movement_vectors.keys())[1:]:  # check for 'North', 'South', 'East', 'West', 'Current'
            checked_position = self.__get_new_position(vector=Action.movement_vectors[val])

            if not self.__is_valid_position(new_position=checked_position):
                idx.append(-1)
            else:
                idx.append(self.grid[checked_position])
        # get action index
        action_idx = GeneticCode.direction_indices[tuple(idx)]
        return action_idx


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
    add_line_count = 4
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
        print()
        print('Points for the action:\t', points_for_action)
        print('Total points:\t\t', total_points)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generation-count', type=int, default=1000, help='Number of generations of robots')
    parser.add_argument('-r', '--robot-count', type=int, default=1000, help='Number of robots in each generation')
    parser.add_argument('-l', '--grid-length', type=int, default=10, help='Length of the grid side')
    parser.add_argument('-c', '--can-count', type=int, default=20, help='Number of cans on a grid')
    args = parser.parse_args()

    if args.can_count > args.grid_length ** 2:
        raise ValueError(
            f'Number of cans ({args.can_count}) cannot exceed number of squares on a grid ({args.grid_length ** 2}).')

    # Generate initial genetic code randomly
    genetic_codes: List[List[int]] = [
        list(np.random.choice(
            len(Action.actions),
            len(GeneticCode.direction_indices)))
        for _ in range(args.robot_count)
    ]

    initial_grid = Grid(grid_length=args.grid_length, can_count=args.can_count)
    avg_results: List[int] = []
    max_results: List[int] = []
    for _ in tqdm(range(args.generation_count)):
        generation_scores = create_generation(
            genetic_codes=genetic_codes,
            robot_count=args.robot_count,
            initial_grid=initial_grid)
        avg_results.append(np.mean(list(generation_scores.values())))
        max_results.append(max(generation_scores.values()))
        best_idx = max(generation_scores, key=generation_scores.get)
        best_genetic_code = genetic_codes[best_idx]

        genetic_codes = evolve_generation(scores=generation_scores,
                                          gen_codes=genetic_codes)

    np.savetxt('../docs/grid.csv', initial_grid.grid, delimiter=',', fmt='%i')
    np.savetxt('../docs/best_genetic_code.csv', np.array(best_genetic_code), delimiter=',', fmt='%i')

    plot_movements(grid=initial_grid)

    plot_results(max_results, 'max')
    plot_results(avg_results, 'avg')
    plot_max_avg(max_results, avg_results)
