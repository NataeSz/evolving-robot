#!/usr/bin/env python3
import numpy as np
import argparse
from tqdm import tqdm
from typing import List
from evolution import create_generation, evolve_generation
from plot import plot_movements, plot_results, plot_max_avg
from config import GeneticCode, Action
from grid import Grid


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
