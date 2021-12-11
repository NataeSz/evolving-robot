import numpy as np
from tqdm import tqdm
from itertools import product
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


class Settings:  # TODO: dodac przedrostek
    possible_states: Dict[int, str] = {0: 'Empty', 1: 'Can', -1: 'Wall'}  # TODO: dokoncz typy
    genetic_code_directions: List[str] = ['North', 'South', 'East', 'West', 'Current']  # TODO: genetic_code_x do nowej klasy + metody
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


class Action:  # ?
    STAY = 0
    pass


class Rewards:
    NO_POINTS: int = 0
    CAN: int = 10
    NO_CAN: int = -1
    WALL: int = -5


class GridWithRobot:
    def __init__(self, grid_length: int = 10, can_probability: float = 0.5):
        self.current_position = (0, 0)
        self.grid = self.__create_map(grid_length=grid_length, can_probability=can_probability)

    @staticmethod
    def __create_map(grid_length: int = 10, can_probability: float = 0.5) -> np.array:  # TODO: napisz to ladniej
        can_count = int((grid_length**2) * can_probability)
        grid = np.zeros(grid_length**2, dtype=int)
        can_indices = np.random.choice(list(range(grid_length**2)), size=can_count, replace=False)
        grid[can_indices] = 1
        grid = grid.reshape((grid_length, grid_length))
        return grid

    def is_on_can(self) -> bool:
        return bool(self.grid[self.current_position])

    def move(self, action: int) -> int:
        if action == Action.STAY:  # TODO: action to class - wyeliminuj magiczne liczby, poczytaj o enumach, etc.
            return Rewards.NO_POINTS  # TODO: eliminate magic constants

        # If trying to pick up a can:
        if action == 0:
            # check if can in current position
            if self.is_on_can():  # TODO: admire for 1 minute
                self.grid[self.current_position] = 0  # TODO abstract away: self.pick_can()
                return Rewards.CAN
            return Rewards.NO_CAN  # Rewards to tylko luzna sugestia, przekminic gdzie dodac te stale

        # If random movement:
        if action == 5:
            action = np.random.randint(low=1, high=5)  # TODO: Magiczne stale, potencjalnie wyabstachować: get_movements() -> List[Action]

        # TODO: too much comments
        # Get new position
        new_position = self.get_new_position(vector=Settings.movement_vectors[action])
        # Check if new position is possible (no wall)
        if not self.is_valid_position(self.grid, new_position):
            # Action impossible - wall
            return Rewards.WALL
        # no wall -> movement
        self.current_position = new_position
        return Rewards.NO_POINTS

    def get_new_position(self, vector: Tuple[int]) -> Tuple[int]:
        return tuple(map(sum, zip(self.current_position, vector)))

    def is_valid_position(self, new_position: int) -> bool:
        # Out-of-range indices
        boundaries = (-1, self.grid.shape[0])
        # Check if new position is possible (no wall)
        valid = not any(coord in boundaries for coord in new_position)
        return valid

    def get_neighbourhood_hash(self) -> int:
        idx = []
        for val in list(Settings.movement_vectors.keys())[1:]:  # check for 'North', 'South', 'East', 'West', 'Current'
            checked_position = self.get_new_position(vector=Settings.movement_vectors[val])

            if not self.is_valid_position(new_position=checked_position):
                idx.append(-1)
            else:
                idx.append(self.grid[checked_position])
        # get action index
        action_idx = Settings.genetic_code_dict[tuple(idx)]
        return action_idx


def choose_action(grid_with_robot: GridWithRobot, genetic_code: List[int]) -> int:
    action_idx = grid_with_robot.get_neighbourhood_hash()
    return genetic_code[action_idx]


def create_generation(
        genetic_codes: List[int],
        number_of_robots: int = 1000,
        grid_length: int = 10
) -> dict:
    iterations = grid_length ** 2

    scores = {}
    for robot_id in range(number_of_robots):
        grid_with_robot = GridWithRobot(grid_length=grid_length)  # TODO: jedna plansza dla wszystkich robotow
        score = 0

        for _ in range(iterations):
            # TODO: inline choose_action ?
            action = choose_action(grid_with_robot=grid_with_robot, genetic_code=genetic_codes[robot_id])

            points_for_action = grid_with_robot.move(action)

            assert points_for_action in [0, -1, -10, 10]
            score += points_for_action

        # TODO: Szansa dla najsłabszych ?
        scores[robot_id] = max(score, 0)

    return scores


def get_probabilities(scores: Dict[int, int]) -> Dict[int, float]:
    scores_sum = sum(scores.values())
    if scores_sum == 0:
        return {key: 1/len(scores) for key in scores.key()}
    return {key: value / scores_sum for key, value in scores.items()}


def choose_pair_to_evolve(proba: Dict[int, float]) -> Tuple[int, int]:
    pair = np.random.choice(
        a=list(proba.keys()),
        size=2,
        p=list(proba.values()),
        replace=False)
    return tuple(pair)


def evolve(parent1: List[int], parent2: List[int]) -> (list, list):
    cutoff = int(len(parent1) / 2)

    # offspring
    child0 = parent1[:cutoff] + parent2[cutoff:]
    child1 = parent2[:cutoff] + parent1[cutoff:]

    # mutation
    for child in [child0, child1]:
        idx_to_mutate = np.random.randint(low=0, high=len(child))
        child[idx_to_mutate] = np.random.randint(low=0, high=len(Settings.actions))

    return child0, child1


def evolve_generation(generation_scores: dict, genetic_codes: dict) -> List[]: # TODO: typy
    probabilities = get_probabilities(generation_scores)
    new_genetic_codes: List[List[int]] = []
    while len(new_genetic_codes) < 1000:
        p1_idx, p2_idx = choose_pair_to_evolve(proba=probabilities)
        parent1 = genetic_codes[p1_idx]
        parent2 = genetic_codes[p2_idx]
        new_genetic_codes.extend(evolve(parent1, parent2))

    return new_genetic_codes


def plot_results(results: List[int], type: str = 'Max') -> None:
    fig = plt.figure(figsize=(12, 6))
    plt.plot(range(len(results)), results)
    plt.title(type + ' fitness in population')
    plt.xlabel('Generation number')
    plt.ylabel(type + ' fitness')
    plt.savefig(f'../docs/{type}_fitness.png', dpi=fig.dpi)
    plt.show()


if __name__ == "__main__":
    GRID_LENGTH: int = 10
    number_of_robots: int = 1000
    number_of_generations: int = 1000

    # Generating initial genetic code randomly
    genetic_codes = [
        list(np.random.choice(
            len(Settings.actions),
            len(Settings.genetic_code_dict)))
        for _ in range(number_of_robots)
    ]

    results = []
    results_max = []
    for _ in tqdm(range(number_of_generations)):
        generation_scores = create_generation(
            genetic_codes=genetic_codes,
            number_of_robots=number_of_robots,
            grid_length=GRID_LENGTH)
        # print('\n', np.mean(generation_scores.values()), max(generation_scores.values()))
        results.append(np.mean(list(generation_scores.values())))
        results_max.append(max(generation_scores.values()))
        genetic_codes = evolve_generation(generation_scores, genetic_codes)

    plot_results(results_max, 'max')
    plot_results(results, 'avg')
