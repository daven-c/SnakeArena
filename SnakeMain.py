from SnakeAgents import *
from SnakeEnvironment import SnakeGame
import pickle

if __name__ == '__main__':

    dim = 40
    tile_size = 20
    fps = 60
    population_size = 10
    fruits = 5
    default_len = 3
    gr = 1
    moves_limit = None

    idx = 2
    agent_type = [RandomPlayer, PathingPlayer, GeneticPlayer][idx]

    if idx <= 1:
        snakes = [agent_type(name=i + 1, spawn_point=None, default_length=default_len, growth_rate=gr, debug=False) for i in
                  range(population_size)]
        game = SnakeGame(snakes=snakes, snake_speed=fps, fruit_count=fruits, moves_cap=None, width=dim, tile_size=tile_size,
                         display=True, auto_close=False, print_events=True)
        final_state = game.run()
    elif idx == 2:
        brain = pickle.load(open('model.obj', 'rb'))
        snake = [GeneticPlayer(brain=brain, name=1, spawn_point=None,
                               default_length=default_len, growth_rate=gr, debug=False)]
        game = SnakeGame(snakes=snake, snake_speed=fps, fruit_count=fruits, moves_cap=None, width=dim, tile_size=tile_size,
                         display=True, auto_close=False, print_events=True)
        final_state = game.run()
