import math
import time
from itertools import chain
import pygame
from SnakeAgents import *

BLACK = pygame.Color(0, 0, 0)
GREY = pygame.Color(150, 150, 150)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
CYAN = pygame.Color(0, 255, 255)
DARK_GREEN = pygame.Color(0, 170, 0)
BLUE = pygame.Color(0, 0, 255)
YELLOW = pygame.Color(255, 255, 0)
ORANGE = pygame.Color(255, 150, 0)


class Fruit:
    fruitsCreated = 0

    def __repr__(self):
        return str(self.pos)

    def __str__(self):
        return str(self.pos)

    def __init__(self, pos):
        Fruit.fruitsCreated += 1
        self.pos: tuple[int, int] = pos


class SnakeGame:

    def __repr__(self) -> str:
        return str(self.getGameState())

    def __str__(self) -> str:
        return str(self.getGameState())

    def __init__(self, snakes: list[Snake] = None, snake_speed: int = 30, fruit_count: int = 1, moves_cap: int = None,
                 width: int = 10, tile_size: int = 10, display: bool = True, auto_close: bool = False,
                 print_events: bool = True, ):

        # Window size
        self.width: int = width
        self.stats_window_dim = self.width // 2
        self.block_size: int = tile_size

        # game options
        self.display: bool = display
        self.snake_speed: int = snake_speed
        # Initialising pygame
        if self.display:
            self.game_window = None
            self.__initDisplay()
        self.auto_close = auto_close
        self.print_events = print_events

        # game variables
        self.moves_remaining = moves_cap

        self.start_time: float = -1
        self.total_runtime: float = -1

        self.snakes = snakes
        for snake in snakes:
            if snake.spawn_point is None:
                snake.init_body(self.get_board_center())

        self.living_leader: Snake = self.snakes[0] if len(
            self.snakes) != 0 else None

        self.fruits: list[Fruit] = [Fruit(pos=self.get_random_location(0, self.width)) for _ in
                                    range(max(1, fruit_count))]  # prevents 0 fruits
        self.score: int = 0

        self.running: bool = True

    # -CLASS BASED METHODS- #####################################################################################################################

    @staticmethod
    def print_event(event, *args, indent: int = 1):
        print("\t" * indent + f"- {event}:", *args)

    @staticmethod
    def add_tuple(*tuples):
        return tuple(map(lambda *t: sum(t), *tuples))

    @staticmethod
    def multiply_tuples(*tuples):
        return tuple(map(lambda x, y: x * y, *tuples))

    # -GAME BASED METHODS- #############################################################################################################################

    # returns a dict of key game variables
    def getGameState(self) -> dict:
        return {
            "Score": self.score, "Runtime": str(self.getRunTime(place=1)) + "s", "Leader": self.living_leader,
            "Remaining": len(self.get_living_snakes()), "Moves Left": self.moves_remaining, "Running": self.running,
        }

    def getRunTime(self, place=1) -> float:  # return in seconds
        if self.running:
            self.total_runtime = round(time.time() - self.start_time, place)
        return self.total_runtime

    # returns a random coordinate scaled with tile size
    @staticmethod
    def get_random_location(lower, upper) -> tuple[int, int]:
        return random.randrange(lower, upper), random.randrange(lower, upper)

    def get_board_center(self):
        return self.width // 2, self.width // 2

    def add_snake(self, snake: Snake):
        if isinstance(snake, Snake):
            self.print_event("Added", snake)
            self.snakes.append(snake)
            self.updateLeader()
        else:
            print(snake, "is not a valid snake")

    def get_living_snakes(self) -> list[Snake]:
        return [snake for snake in self.snakes if snake.state == 1]

    # return a list of tiles which are occupied
    def get_occupied_tiles(self) -> list:
        fruits = [fruit.pos for fruit in self.fruits]
        snakes = list(chain.from_iterable(
            [snake.body for snake in self.get_living_snakes()]))
        return fruits + snakes

    def regen_fruit(self, fruit):
        taken = self.get_occupied_tiles()
        attempts = 0
        while fruit.pos in taken:  # regenerate fruit pos
            attempts += 1
            fruit.pos = self.get_random_location(0, self.width)
            if attempts > 1000:
                break

    # updates the leader based on number of fruits eaten and alive
    def updateLeader(self):
        living = self.get_living_snakes()
        if len(living) > 0:
            self.living_leader = max(living, key=lambda s: s.fruits_eaten)
        else:
            self.living_leader = self.living_leader = max(
                self.snakes, key=lambda s: s.fruits_eaten)

    # checks the death cases, returns True if snake should have died
    def has_collided(self, snake: Snake, self_collisions: bool = True) -> bool:
        # Touch edges
        if snake.head[0] < 0 or snake.head[0] >= self.width:
            return True
        if snake.head[1] < 0 or snake.head[1] >= self.width:
            return True
        # self collision
        if self_collisions:
            for block in snake.body[1:]:
                if snake.head == block:
                    return True
        return False

    # helper function to move snake to dead snakes and updates leader
    def killSnake(self, snake: Snake):
        snake.state = 0
        if snake == self.living_leader:
            self.updateLeader()
        if self.print_events:
            self.print_event("Death", snake)

    # -SNAKE INPUT BASED METHODS- ###############################################################################################################

    @staticmethod
    def distance(point1, point2, scale=1) -> float:
        return math.sqrt(((point2[1] - point1[1]) ** 2) + ((point2[0] - point1[0]) ** 2)) * scale

    def get_closest_fruit(self, snake: Snake) -> Fruit:
        return min(self.fruits, key=lambda fruit: self.distance(snake.head, fruit.pos))

    def get_closest_wall_point(self, node) -> float():
        wall_cords = [(node[0], 0), (node[0], self.width), (0, node[1]),
                      (self.width, node[1]), ]  # top, bottom, left, right
        return min(wall_cords, key=lambda w: self.distance(w, node))

    # d-food, d-tail, d-wall, in 8 directions
    def check_direction(self, direction, origin, snake: Snake) -> list:
        distance = [0]
        items: list = [0, 1, 1]

        def look_in_direction(current):
            if items[0] == 0:
                fruit = next(
                    (f for f in self.fruits if current == f.pos), None)
                if fruit is not None:
                    items[0] = 1 - distance[0] / self.width
            if items[1] == 1:
                body = next((b for b in snake.body[1:] if current == b), None)
                if body is not None:
                    items[1] = distance[0] / self.width
            if current[0] < 0 or current[0] >= self.width or current[1] < 0 or current[1] >= self.width:
                items[2] = distance[0] / self.width
                return
            next_pos = self.add_tuple(current, direction)
            distance[0] += 1
            return look_in_direction(next_pos)

        look_in_direction(current=origin)
        return items

    def get_nn_inputs(self, snake) -> list:  # d-food, d-tail, d-wall, in 8 directions
        closest_fruit = self.get_closest_fruit(snake)
        fruit_pos_encoded = list(map(lambda x: 1 if x else -1, [snake.head[0] < closest_fruit.pos[0], snake.head[0] > closest_fruit.pos[0],
                                                                snake.head[1] < closest_fruit.pos[1],
                                                                snake.head[1] > closest_fruit.pos[1]]))  # (4,)
        safe_dirs_encoded = [-1 if (node in snake.body or any(
            [node[0] < 0, node[0] >= self.width, node[1] < 0, node[1] >= self.width])) else 1 for node in
            snake.get_adj_nodes(snake.head, only_cardinal=True)]  # (4,)
        direction_encoded = snake.hot_encode_direction()  # (4,)
        tail_direction_encoded = snake.hot_encode_tail_direction()  # (4,)
        # (16,)
        return fruit_pos_encoded + safe_dirs_encoded + direction_encoded + tail_direction_encoded

    # (node, distance to fruit)
    def get_pathing_inputs(self, snake) -> list[tuple]:
        target_fruit = self.get_closest_fruit(snake=snake)
        surrounding_nodes = Snake.get_adj_nodes(snake.head, only_cardinal=True,
                                                box_bounds=(0, self.width, 0, self.width))
        surrounding_nodes = list(
            filter(lambda node: node not in snake.body, surrounding_nodes))  # prevents running into body
        return [(node, self.distance(node, target_fruit.pos)) for node in surrounding_nodes]

    # -GRAPHICS- #############################################################################################################################

    # initialize display
    def __initDisplay(self):
        self.display = True
        if pygame.display.get_init() is False:
            pygame.init()
            self.game_window = pygame.display.set_mode(
                ((self.width + self.stats_window_dim) * self.block_size, self.width * self.block_size,))
            pygame.display.set_caption(
                f"Snakes {pygame.display.get_window_size()}, {self.width}x{self.width}")
            self.fps = pygame.time.Clock()

    # displays game state on screen
    def display_stats(self):
        font_size = 20
        stat_font = pygame.font.SysFont("times new roman", font_size)
        game_state = self.getGameState()
        displacement = 0
        for k, v in game_state.items():
            stat_key = stat_font.render(str(k) + ":", True, ORANGE)
            key_rect = stat_key.get_rect()
            key_rect.topleft = (
                self.width * self.block_size + self.block_size + self.block_size // 2 - 1, displacement,)
            displacement += font_size
            self.game_window.blit(stat_key, key_rect)

            stat_value = stat_font.render(str(v), True, WHITE)
            value_rect = stat_value.get_rect()
            value_rect.topleft = (
                self.width * self.block_size + self.block_size + self.block_size // 2 - 1, displacement,)
            displacement += font_size + 5
            self.game_window.blit(stat_value, value_rect)
        return displacement

    # draws the frame with snakes, fruits, board
    def draw_frame(self, all_snakes: bool = False):

        # helper function to draw the snake body
        def draw_snake(snake: Snake):
            color = GREEN
            if snake.state == 0:  # draw dead snakes at end of game
                color = GREY
            for pos in snake.body[1:]:
                pygame.draw.rect(self.game_window, color,
                                 pygame.Rect(pos[0] * self.block_size, pos[1] * self.block_size, self.block_size,
                                             self.block_size, ), )
            pygame.draw.rect(self.game_window, WHITE,
                             pygame.Rect(snake.head[0] * self.block_size, snake.head[1] * self.block_size,
                                         self.block_size, self.block_size, ), )  # draw head

        def draw_leader(snake: Snake):
            color = ORANGE
            if snake.state == 0:  # draw dead snakes at end of game
                color = GREY
            for pos in snake.body[1:]:
                pygame.draw.rect(self.game_window, color,
                                 pygame.Rect(pos[0] * self.block_size, pos[1] * self.block_size, self.block_size,
                                             self.block_size))
            pygame.draw.rect(self.game_window, WHITE,
                             pygame.Rect(snake.head[0] * self.block_size, snake.head[1] * self.block_size,
                                         self.block_size, self.block_size))  # draw head

        def draw_neurons():
            neurons = self.get_nn_inputs(self.snakes[0])
            x_loc = (self.width + self.stats_window_dim - 1)
            for i in range(0, 16, 4):
                pygame.draw.rect(self.game_window, BLACK,
                                 pygame.Rect((x_loc - .5) * self.block_size, (i + .5) * self.block_size,
                                             self.block_size, self.block_size * 4))
                pygame.draw.rect(self.game_window, WHITE,
                                 pygame.Rect((x_loc - .5) * self.block_size, (i + .5) * self.block_size,
                                             self.block_size, self.block_size * 4), 1)
            for i, n in enumerate(neurons):
                pygame.draw.circle(self.game_window, (0, 255, 0) if n == 1 else (255, 255, 255),
                                   (x_loc * self.block_size, (i + 1) * self.block_size), self.block_size // 6)

        self.game_window.fill(BLACK)

        self.display_stats()

        snakes = self.get_living_snakes()
        if all_snakes:
            snakes = self.snakes
        for snake in snakes:  # only draws living snakes
            """for direction in Snake.get_directionals():  # draw directions snake looking in
                draw_direction(direction, snake)"""
            if snake == self.living_leader:
                draw_leader(self.living_leader)
                continue
            draw_snake(snake)

        if isinstance(self.snakes[0], GeneticPlayer):
            draw_neurons()
        for fruit in self.fruits:
            pygame.draw.rect(self.game_window, RED,
                             pygame.Rect(fruit.pos[0] * self.block_size, fruit.pos[1] * self.block_size,
                                         self.block_size, self.block_size))

        pygame.draw.line(surface=self.game_window, color=WHITE,
                         start_pos=(self.width * self.block_size + self.block_size // 2 - 1, 0), end_pos=(
                             self.width * self.block_size + self.block_size // 2 - 1, self.width * self.block_size),
                         width=self.block_size)

        pygame.display.update()
        self.fps.tick(self.snake_speed)

    # -MAIN- #########################################################################################################################

    # plays a single frame

    def step(self) -> dict:

        if self.start_time == -1:
            self.start_time = time.time()

        if self.display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # exit button
                    self.display = False
                    pygame.quit()

        # move the snake
        ####################################################################
        for idx, snake in enumerate(self.get_living_snakes()):
            if isinstance(snake, RandomPlayer):
                snake.direction = snake.get_move()
            elif isinstance(snake, PathingPlayer):
                snake.direction = snake.get_move(
                    all_distances=self.get_pathing_inputs(snake))
            elif isinstance(snake, GeneticPlayer):
                snake.direction = snake.get_move(
                    inputs=self.get_nn_inputs(snake))
            snake.move()
            if self.has_collided(snake, self_collisions=True):
                self.killSnake(snake)
                continue
            # check if eaten fruit
            ####################################################################
            for fruit in self.fruits:
                if snake.head == fruit.pos:
                    self.regen_fruit(fruit=fruit)
                    self.score += 1
                    snake.fruits_eaten += 1
                    snake.grow()
                    self.updateLeader()
                    break
        if self.moves_remaining is not None:
            self.moves_remaining -= 1

        # update the display
        ####################################################################
        if self.display:
            self.draw_frame()

        # All Snakes Dead or no more moves
        #####################################################################
        if len(self.get_living_snakes()) == 0 or self.moves_remaining == 0:
            self.running = False
            if self.print_events:
                self.print_event("Game Over", self.getGameState())
            if self.display:
                self.draw_frame(all_snakes=True)
                if not self.auto_close:
                    while True:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:  # exit button
                                break
                        else:
                            continue
                        break  # will not trigger unless inner if is triggered
                pygame.quit()
        return self.getGameState()

    # auto play steps until game over
    def run(self):
        while True:
            state = self.step()
            if state["Running"] is False:
                break
        return state
