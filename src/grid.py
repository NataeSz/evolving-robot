import numpy as np
from typing import Tuple
from config import GeneticCode, Rewards, Action, Neighbourhood


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
