import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from typing import List, Tuple, Dict
from copy import deepcopy


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


class Grid:
    def __init__(self, grid_length: int = 10, can_count: int = 20):
        self.length: int = grid_length
        self.can_count: int = can_count
        self.current_position: Tuple[int, int] = (0, 0)
        self.grid: np.array = self.__create_map(grid_length=grid_length, can_count=can_count)

    def __create_map(self) -> np.array:
        grid = [0] * self.length**2
        can_indices = np.random.choice(list(range(len(grid))), size=self.can_count, replace=False)
        grid[can_indices] = Neighbourhood.states['Can']
        grid = grid.reshape((self.length, self.length))
        return grid

    def __is_on_can(self) -> bool:
        return bool(self.grid[self.current_position])

    def __pick_can(self):
        self.grid[self.current_position] = Neighbourhood.states['Empty']

    def __is_valid_position(self, new_position: Tuple[int, int]) -> bool:
        boundaries = (-1, self.length)  # out-of-range indices
        valid = not any(coord in boundaries for coord in new_position) # Check if new position is possible (no wall)
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
            return Rewards.WALL  # action impossible - wall

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
        robots_count: int = 1000
) -> dict:
    iterations = initial_grid.length ** 2 + initial_grid.can_count

    scores = {}
    for robot_id in range(robots_count):
        grid_with_robot = deepcopy(initial_grid)
        score = 0

        for _ in range(iterations):
            action = choose_action(grid_with_robot=grid_with_robot, genetic_code=genetic_codes[robot_id])

            points_for_action = grid_with_robot.move(action)
            assert points_for_action in [0, -1, -5, 10]
            score += points_for_action

        scores[robot_id] = max(score, 0)

    return scores


def get_probabilities(scores: Dict[int, int]) -> Dict[int, float]:
    scores_sum = sum(scores.values())
    if scores_sum == 0:
        return {key: 1/len(scores) for key in scores.keys()}
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


def evolve_generation(scores: Dict[int, int], gen_codes: List[List[int]]) -> List[List[int]]:
    probabilities = get_probabilities(scores)
    new_genetic_codes: List[List[int]] = []
    while len(new_genetic_codes) < 1000:
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


if __name__ == "__main__":
    GRID_LENGTH: int = 10
    ROBOTS_COUNT: int = 1000
    CAN_COUNT: int = 20
    number_of_generations: int = 1000

    # Generating initial genetic code randomly
    genetic_codes: List[List[int]] = [
        list(np.random.choice(
            len(Action.actions),
            len(GeneticCode.direction_indices)))
        for _ in range(ROBOTS_COUNT)
    ]

    initial_grid = Grid(grid_length=GRID_LENGTH, can_count=CAN_COUNT)
    avg_results: List[int] = []
    max_results: List[int] = []
    for _ in tqdm(range(number_of_generations)):
        generation_scores = create_generation(
            genetic_codes=genetic_codes,
            robots_count=ROBOTS_COUNT,
            initial_grid=initial_grid)
        # print('\n', np.mean(generation_scores.values()), max(generation_scores.values()))
        avg_results.append(np.mean(list(generation_scores.values())))
        max_results.append(max(generation_scores.values()))
        genetic_codes = evolve_generation(scores=generation_scores, gen_codes=genetic_codes)

    plot_results(max_results, 'max')
    plot_results(avg_results, 'avg')
