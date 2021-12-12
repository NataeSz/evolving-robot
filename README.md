# Evolving Robby - Genetic Algorithm

Genetic algorithm for a simple rule-based robot on a 2D-grid

## Overview
### Robby and his World

Robby is a very simple robot. He lives in a simulated world, consisting of a small (10x10) grid of squares.
His life goal is to collect empty soda cans, which are distributed randomly in the world. Each square of the grid can contain zero or one empty soda can. Robby has very bad vision. He only sees the contents of the square he stands in and the four side-adjacent cells (a square can be empty, wall or contain empty soda can). Based on the input from 5 cells, in each simulation step, he can choose from one of the following actions:

- Move (x4 directions)
- Move random
- Stay put
- Try to pick up a can (may fail if there's none)

The simulation can continue for a fixed number of steps or until empty soda cans are collected.

Robby gets scores points based on his behavior during the simulation:
- Collecting an empty soda can: +10 pts
- Failing to pick a can (trying on empty square): -1 pt
- Crashing into wall (trying to move outside of the world grid): -5 pt

### Robby's genetic code

Robby makes his decision based on the rule table his inheriting from his parents.
The rule table consists of 243 entries corresponding to every possible combination of his inputs.
Each entry defines an action that Robby will perform when confronted with given input.

| North | South | East  | West  | Current | Action |
|-------|-------|-------|------ |---------|--------| 
| Empty | Empty | Empty | Empty | Empty   | Move R |
| Empty | Empty | Empty | Empty | Can     | Pick   |
| ...   | ...   | ...   | ...   | ...     | ...    |
| Wall  | Wall  | Wall  | Wall  | Empty   | Stay   |

The table makes up Robby's "genetic code", which he will pass onto his children.
Note: some of those rules (inputs) are impossible in the Robby's world, but that's OK, even humans have garbage DNA.

### Simulation

Robots are evolved in generations of N (implementation-dependent) individuals.
The first generation has random genetic codes.
Each individual is put a number of times (K) in a separate world to demonstrate his abilities.
Those world should differ in can distribution, so that each genetic code is tested in different conditions.
An average score from all worlds becomes individual's score. 
After all individuals have been tested, a new generation is created.

### Reproduction

Every generation (except the first one) are created from the previous one
by sampling two parents and producing a child until N children are created.
For a robot, the probability of being chosen to be a parent should depend on its score (better score = higher probability).
Child genetic code is created by concatenating a prefix from one parent and a suffix from the other one, to create a complete rule table.
Additionally, a small number of mutations should be introduced.



