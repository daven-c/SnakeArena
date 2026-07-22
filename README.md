# SnakeArena — Neuroevolution for Snake

A from-scratch Snake environment plus a **genetic algorithm** that evolves neural-network controllers to play it. No RL library, no PyTorch — just NumPy, `multiprocessing`, and a fitness function that rewards eating fruit while surviving.

The goal was to see how far a *gradient-free* evolutionary approach could push a small MLP in a classic RL playground, and to squeeze every drop of speed out of pure Python for the training loop.

## What's here

| File                    | Role                                                                                |
| ----------------------- | ----------------------------------------------------------------------------------- |
| `SnakeEnvironment.py`   | Grid, food spawning, collision, rendering — the Snake game itself                   |
| `SnakeAgents.py`        | `Snake` base class + `RandomPlayer`, `PathingPlayer` (A*-ish baseline), `GeneticPlayer` (NN) |
| `SnakeTrainer.py`       | Genetic algorithm: population, fitness eval, tournament selection, crossover, mutation |
| `SnakeMain.py`          | Interactive runner — pick an agent, watch it play                                    |
| `model.obj`             | Serialised weights of the best evolved snake                                          |

## Approach

- **Genome = flattened weights** of a small MLP that maps local grid observations → next-move logits.
- **Fitness** rewards fruit eaten, length, and survival time; punishes double-backs and long fruitless wanders.
- **Selection** is tournament-based; **crossover** blends parent weight matrices; **mutation** adds Gaussian noise with a configurable rate.
- **Multiprocessing pool** evaluates the population in parallel — the biggest wall-clock win.
- **Vectorised NumPy** for the forward pass instead of per-neuron Python loops.
- **Sets over lists** for collision + fruit-location lookups (O(1) instead of O(n)).

## Run it

Install deps and train:

```bash
pip install -r requirements.txt
python SnakeTrainer.py    # trains, saves model.obj, plots per-generation best fitness
```

Watch the best agent play:

```bash
python SnakeMain.py       # then choose agent index 2 (GeneticPlayer)
```

## Baseline comparison

`SnakeMain.py` ships three agents so you can eyeball the delta:

- `RandomPlayer` — pure noise, sanity check for the environment.
- `PathingPlayer` — greedy shortest-path to fruit, ignores tail collisions.
- `GeneticPlayer` — evolved MLP, learns to keep escape routes open.

## Possible next steps

- Try different MLP architectures + activation functions.
- Add novelty search or MAP-Elites to reduce fitness plateauing.
- Port the hot loop to Cython/Numba for another order-of-magnitude speedup.

## Status

Personal ML project. Complete and reproducible; not a package.
