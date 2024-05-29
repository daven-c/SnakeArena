import random
from itertools import product
import numpy as np
from abc import ABC, abstractmethod


class Snake(ABC):
    snakes_created = 0

    DIRECTIONS = {  # transformation, opposite
        'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)
    }

    def __repr__(self) -> str:
        return f'{{name: {self.name}, length: {self.length}, eaten: {self.fruits_eaten}, state: {self.state}}}'

    def __str__(self) -> str:
        return f'{{name: {self.name}, length: {self.length}, eaten: {self.fruits_eaten}, state: {self.state}}}'

    def __init__(self, name: str = None, spawn_point: tuple[int, int] or None = None, default_length: int = 3,
                 growth_rate: int = 1, default_direction: str = 'down', debug: bool = False):
        self.snakes_created += 1

        self.debug = debug

        self.name: str = name
        self.spawn_point = spawn_point

        self.head = None
        self.body = None

        self.length: int = default_length
        self.growth_rate = growth_rate

        if self.spawn_point is not None:
            self.init_body(self.spawn_point)

        self.direction: str = default_direction  # refers to keys in dict directions

        self.fruits_eaten: int = 0
        self.steps_taken = 0
        self.state: int = 1  # 1 alive 0 dead -1 survived
        self.double_backs = 0

    # leave for child classes to define
    @abstractmethod
    def get_move(self, *dummy):
        pass

    @staticmethod
    def print_debug(event, indent: int = 1, **kwargs):
        print('\t' * indent + f'- {event}:', kwargs)

    @staticmethod
    def get_adj_nodes(node: tuple, only_cardinal: bool = False, box_bounds: tuple[int, int, int, int] = None) -> list:
        neighbors = list(product([node[0] - 1, node[0], node[0] + 1],  # generate 3x3
                                 [node[1] - 1, node[1], node[1] + 1]))
        neighbors.remove(node)  # remove center point
        if only_cardinal:  # only cardinal points
            neighbors = filter(
                lambda n: n[0] == node[0] or n[1] == node[1], neighbors)
        if box_bounds is not None:  # check in bounds if there is bounds
            neighbors = [n for n in neighbors if
                         (box_bounds[0] <= n[0] < box_bounds[1]) and (box_bounds[2] <= n[1] < box_bounds[3])]
        return neighbors

    @classmethod
    def get_directionals(cls, only_cardinal: bool = False) -> list[tuple[int, int]]:
        return cls.get_adj_nodes(node=(0, 0), only_cardinal=only_cardinal)

    @classmethod
    def get_key_from_transformation(cls, transformation: tuple) -> str:
        all_transformations = list(cls.DIRECTIONS.values())
        key_idx = all_transformations.index(transformation)
        move = list(cls.DIRECTIONS.keys())[key_idx]
        return move

    def fitness(self) -> float:
        return max(self.fruits_eaten * 100 - 0.2 * self.steps_taken, 1)

    def hot_encode_direction(self) -> list[int, int, int, int]:
        return [1 if self.direction == d else 0 for d in self.DIRECTIONS]

    def hot_encode_tail_direction(self) -> list[int, int, int, int]:
        tail_direction: tuple[int, int] = (
            self.body[-2][0] - self.body[-1][0], self.body[-2][1] - self.body[-1][1])
        if tail_direction == (0, 0):
            return [0, 0, 0, 0]
        return [1 if self.get_key_from_transformation(tail_direction) == d else 0 for d in self.DIRECTIONS]

    def init_body(self, spawn_point: tuple[int, int]):
        self.head = spawn_point
        self.body = [spawn_point for _ in range(self.length)]

    def move(self):  # prevents double backing
        transformation = self.DIRECTIONS.get(
            self.direction)  # get tuple from dict
        prev_transformation = tuple(
            map(lambda x, y: x - y, self.head, self.body[1]))

        if prev_transformation != (0, 0):
            if self.is_opposite_direction(self.get_key_from_transformation(prev_transformation)):
                transformation = prev_transformation
        self.head = self.head[0] + \
            transformation[0], self.head[1] + transformation[1]
        self.body.insert(0, self.head)
        self.body.pop(-1)
        self.steps_taken += 1

    def grow(self, growth_rate: int = 1):
        self.body += [self.body[-1]] * growth_rate
        self.length += growth_rate

    def is_opposite_direction(self, new_direction: str) -> bool:
        if self.direction == 'up':
            return new_direction == 'down'
        elif self.direction == 'down':
            return new_direction == 'up'
        elif self.direction == 'left':
            return new_direction == 'right'
        elif self.direction == 'right':
            return new_direction == 'left'


class RandomPlayer(Snake):

    # inherits constructor

    def get_move(self) -> str:
        move = list(self.DIRECTIONS.keys())[random.randint(
            0, len(self.DIRECTIONS) - 1)]  # get a random move from moves
        if self.debug:
            self.print_debug('Get Move', move=move)
        return move


class PathingPlayer(Snake):

    def __init__(self, name: str = None, spawn_point: tuple[int, int] = None, default_length: int = 3,
                 growth_rate: int = 1, default_direction: str = 'down', debug: bool = False):
        super().__init__(name, spawn_point, default_length,
                         growth_rate, default_direction, debug)

    def get_move(self, all_distances: list[tuple[tuple[int, int], float]]) -> str:
        sorted_distances = sorted(
            filter(lambda n: n[0] not in self.body, all_distances), key=lambda d: d[1])
        if len(sorted_distances) == 0:
            return self.direction
        direction = (sorted_distances[0][0][0] - self.head[0]), (
            sorted_distances[0][0][1] - self.head[1])  # [0][0][0] = first tuple set, coord tuple, x coord
        move = self.get_key_from_transformation(transformation=direction)
        if self.is_opposite_direction(new_direction=move):
            direction = (sorted_distances[1][0][0] - self.head[0]
                         ), (sorted_distances[1][0][1] - self.head[1])
            move = self.get_key_from_transformation(transformation=direction)
        if self.debug:
            self.print_debug('Get Move', move=move)
        return move


class GeneticPlayer(Snake):
    # input, hid, hid, output
    architecture = [16, 24, 24, 24, len(Snake.DIRECTIONS)]

    def __init__(self, brain, name: str = None, spawn_point: tuple[int, int] or None = None, default_length: int = 3,
                 growth_rate: int = 1, default_direction: str = 'down', debug: bool = False):
        super().__init__(name, spawn_point, default_length,
                         growth_rate, default_direction, debug)

        self.current_brain = brain

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def ReLu(x):
        return max(0, x)

    @classmethod
    def generate_brain(cls):
        return [np.random.uniform(low=-1, high=1, size=(cls.architecture[i], cls.architecture[i - 1] + 1)) for i in
                range(1, len(cls.architecture))]

    def get_move(self, inputs: list):
        hidden_act = self.ReLu
        output_act = self.sigmoid

        input_vector = np.array(inputs + [1])  # [1] represents bias
        hidden_layer1, hidden_layer2, hidden_layer3, output_layer = self.current_brain

        # Forward propagation
        Ho1 = np.append(
            np.apply_along_axis(func1d=lambda row: hidden_act(np.dot(input_vector, row)), axis=1, arr=hidden_layer1), 1)
        Ho2 = np.append(np.apply_along_axis(func1d=lambda row: hidden_act(np.dot(Ho1, row)), axis=1, arr=hidden_layer2),
                        1)
        Ho3 = np.append(np.apply_along_axis(func1d=lambda row: hidden_act(np.dot(Ho2, row)), axis=1, arr=hidden_layer2),
                        1)
        Zo = np.apply_along_axis(func1d=lambda row: output_act(
            np.dot(Ho3, row)), axis=1, arr=output_layer)
        max_index = np.argmax(Zo)
        move = list(self.DIRECTIONS.keys())[max_index]

        # Debug
        if self.debug:
            self.print_debug('DEBUG', indent=1, input_shape=input_vector.shape, HL1_shape=hidden_layer1.shape,
                             HL2_shape=hidden_layer2.shape, HL3_shape=hidden_layer3.shape, output_shape=output_layer.shape, Ho1_shape=Ho1.shape,
                             Ho2_shape=Ho2.shape, Zo_shape=Zo.shape,  # Zo=Zo,
                             max_idx=max_index, key=move)
        return move


if __name__ == '__main__':
    pass
