from itertools import product
from typing import List, Tuple, Dict


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
    can_repr: Dict[int, str] = {0: '.', 1: 'o'}
    current_position_repr: Dict[str, str] = {can_repr[0]: '+', can_repr[1]: '⊕'}
