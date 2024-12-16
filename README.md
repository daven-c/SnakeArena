# Snake AI - Genetic Algorithm

This project implements a Snake game and trains an AI agent to play it using a genetic algorithm.

## Project Structure

*   `SnakeAgents.py`: Contains the different snake agent classes (`Snake`, `RandomPlayer`, `PathingPlayer`, `GeneticPlayer`).
*   `SnakeEnvironment.py`: Defines the `SnakeGame` environment, game logic, and rendering.
*   `SnakeTrainer.py`: Implements the genetic algorithm for training `GeneticPlayer` agents.
*   `SnakeMain.py`:  Allows the user to run the game with different types of snakes.
*   `model.obj`: Stores the trained neural network weights.

## How to Run

1.  **Training:**
    *   Run `SnakeTrainer.py` to train the AI agent.
    *   The training process will save the best trained neural network weights in `model.obj`
    *   A graph will display the performance of the best snake agent in each generation.

    ```bash
    python SnakeTrainer.py
    ```

2.  **Playing the Game:**
    *   Run `SnakeMain.py` to play the game using different agents.
    *   To run the best agent, select index `2`.

    ```bash
    python SnakeMain.py
    ```

## Key Concepts

*   **Genetic Algorithm:** The AI is trained using a genetic algorithm which evolves a population of neural networks.
*   **Neural Network:** The `GeneticPlayer` uses a neural network to determine its next move.
*  **Multiprocessing:** The training algorithm utilises the multiprocessing library to decrease training time.
*   **Fitness Function:** A custom fitness function rewards snakes for eating, length, and survival.
*   **Mutation and Crossover:** Techniques used to generate new AI agents from current ones.

## Optimization notes
*    **Vectorization**: The training algorithm implements vectorization to improve the speed of the neural network calculations.
*    **Memory Use**: Many changes have been made to decrease unnecessary memory use.
*   **Data Structures:** `sets` have been implemented in place of `lists` for collision detection and tracking of fruit location.

## Requirements

*   Present in requirements.txt

##  Further Improvements

*   Experiment with different neural network architectures, mutation strategies, and crossover methods to improve performance.
*   Implement a more sophisticated evaluation method.
*   Try using advanced genetic algorithm techniques.
