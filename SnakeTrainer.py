import time
from numpy.random import choice, uniform
from SnakeAgents import *
from SnakeEnvironment import SnakeGame
from matplotlib import pyplot as plt
from random import sample
import statistics
import pickle


class TrainGeneticPlayers:

    def __init__(self, pop_size: int = 100, generations: int = 100, epochs: int = 1, mutation_rate: float = 0.1,
                 mutation_size: float = 0.1, elite_ratio: float = 0.25):
        self.pop_size = pop_size
        self.generations = generations
        self.epochs = epochs

        self.mutation_rate = mutation_rate
        self.mutation_size = mutation_size
        self.elite_ratio = elite_ratio
        if int(elite_ratio * pop_size) < 2:
            raise ValueError("elite size must be greater than 2")

        self.population = [GeneticPlayer.generate_brain()
                           for _ in range(self.pop_size)]
        self.gen_max_scores = []
        self.best_brain = None

    @staticmethod
    def print_event(event, *args, indent: int = 1):
        print('\t' * indent + f'- {event}:', *args)

    @staticmethod
    def sum_arrays_together(*arrays) -> list:
        return list(map(lambda *arr: sum(arr), *arrays))

    @staticmethod
    def average_array(array, array_len) -> list:
        return list(map(lambda ele: ele / array_len, array))

    def reproduce(self, top_brains: list, fitness_scores: list) -> list:

        # returns two parent brains
        def selection_pair(brains: list, scores: list) -> tuple[list, list]:
            # normalize 0-1
            scores = list(map(lambda x: x / sum(scores), scores))
            i1, i2 = choice(a=range(len(brains)), size=2,
                            replace=False, p=scores)
            return brains[i1], brains[i2]

        # produces a single child with parts from each parent
        def multipoint_layer_crossover(parent1_layer: np.array, parent2_layer: np.array,
                                       crossovers: int = 1) -> np.array:
            child1_layer = parent1_layer.copy()
            for row_idx in range(len(parent1_layer)):
                row_len = len(parent1_layer[row_idx])

                splices = sorted(
                    [0] + sample(range(1, row_len - 1), crossovers) + [row_len])
                for i in range(crossovers + 1):
                    if i % 2 == 1:
                        child1_layer[row_idx][splices[i]: splices[i + 1]] = parent2_layer[row_idx][
                            splices[i]: splices[i + 1]]

            return child1_layer

        # adjusts each value in a layer based on rate and size
        def mutate_layer(layer) -> np.array:
            def mutate_row(row: np.array) -> np.array:
                return np.array(list(
                    map(lambda x: x + uniform(-1, 1) * self.mutation_size if uniform(0, 1) < self.mutation_rate else x,
                        row)))

            return np.apply_along_axis(mutate_row, 1, layer)

        new_pop = top_brains.copy()
        for _ in range(self.pop_size - len(top_brains)):
            new_brain_A = []

            # Selection
            parent1, parent2 = selection_pair(top_brains, fitness_scores)

            # Crossover and Mutation
            for layer_idx in range(len(GeneticPlayer.architecture) - 1):
                crossed_layer_A = multipoint_layer_crossover(
                    parent1[layer_idx], parent2[layer_idx], crossovers=2)
                child_layer_A = mutate_layer(crossed_layer_A)
                new_brain_A.append(child_layer_A)
            new_pop.append(new_brain_A)

        if self.pop_size - len(new_pop) == 1:
            new_pop.append(self.best_brain)
        return new_pop

    def evolve(self):
        total_start = time.time()

        train_type = 'single'

        for gen in range(1, self.generations + 1):
            self.print_event(
                'Generation', f'{gen}/{self.generations}', indent=0)
            fitness_scores = [0 for _ in range(self.pop_size)]
            start = time.time()

            for e in range(1, self.epochs + 1):
                snakes = [GeneticPlayer(brain=self.population[i], name=f'{gen}-{i}', spawn_point=None,
                                        default_length=default_len, growth_rate=gr, debug=False) for i in
                          range(self.pop_size)]
                for idx, snake in enumerate(snakes):
                    game = SnakeGame(snakes=[snake], snake_speed=fps, fruit_count=fruits, moves_cap=250, width=dim,
                                     tile_size=tile_size, display=False, auto_close=True, print_events=False)
                    state = game.run()
                    fitness_scores[idx] += snake.fitness()

            # round fitness scores
            fitness_scores = list(
                map(lambda fs: round(fs / self.epochs, 2), fitness_scores))

            # Select best brains
            top_brain_indexes = list(np.argsort(fitness_scores))[::-1][
                :int(self.elite_ratio * self.pop_size)]  # top 25% of brains based on score
            top_brains = [self.population[idx] for idx in top_brain_indexes]
            top_brains_fitness = [fitness_scores[idx]
                                  for idx in top_brain_indexes]

            # save best snake from generation
            self.gen_max_scores.append(top_brains_fitness[0])
            self.print_event('mean fitness', statistics.mean(fitness_scores))
            self.print_event('max fitness', top_brains_fitness[0])
            if len(self.gen_max_scores) >= 2 and self.gen_max_scores[-1] > self.gen_max_scores[-2]:
                self.best_brain = top_brains[0]

            # Create new population using crossover, mutation
            self.population = self.reproduce(top_brains, top_brains_fitness)

            self.print_event('runtime', round(time.time() - start, 2), 'secs')

            play_gen_game = True
            if play_gen_game and gen % 50 == 0:
                self.print_event(
                    'gen game', f'{gen}/{self.generations}', indent=1)
                gen_snake = GeneticPlayer(brain=self.best_brain, name='gen', spawn_point=None,
                                          default_length=default_len, growth_rate=gr, debug=False)
                game = SnakeGame(snakes=[gen_snake], snake_speed=30, fruit_count=fruits, moves_cap=1000, width=dim,
                                 tile_size=tile_size, display=True, auto_close=True, print_events=False)
                state = game.run()
                self.print_event('fitness', gen_snake.fitness(), indent=2)

        self.print_event('Completed Evolution',
                         f'{round(time.time() - total_start, 2)} secs', indent=0)


if __name__ == '__main__':

    dim = 40
    tile_size = 20
    fps = 300
    population_size = 1
    fruits = 10
    default_len = 3
    gr = 1
    moves_limit = None

    trainer = TrainGeneticPlayers(pop_size=10, generations=50, epochs=5, mutation_rate=0.1, mutation_size=0.03,
                                  elite_ratio=.2)

    trainer.evolve()

    pickle.dump(trainer.best_brain, open('model.obj', 'wb'))

    graph_y = trainer.gen_max_scores
    graph_x = list(range(1, trainer.generations + 1))
    plt.title('Snake Performance')
    plt.ylabel('average fitness score')
    plt.xlabel('generation')
    plt.plot(graph_x, graph_y)
    plt.show()

    # print(trainer.gen_avg_scores)
    print("playing final game", input('press enter '))

    apex_snake = GeneticPlayer(brain=trainer.best_brain, name='apex', spawn_point=None, default_length=default_len,
                               growth_rate=gr, debug=False)
    game = SnakeGame(snakes=[apex_snake], snake_speed=30, fruit_count=5, moves_cap=None, width=dim, tile_size=tile_size,
                     display=True, auto_close=False, print_events=True)
    final_state = game.run()
