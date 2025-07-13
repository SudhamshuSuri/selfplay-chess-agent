# AlphaZero-Style Chess Reinforcement Learning Agent

This project implements a simplified AlphaZero-style agent that learns to play chess through self-play and a deep neural network, enhanced by Monte Carlo Tree Search (MCTS). The agent is designed to be lightweight enough to run and train on platforms like Google Colab's free tier.

## Table of Contents

1.  [Project Overview](#1-project-overview)
2.  [Features](#2-features)
3.  [Getting Started](#3-getting-started)
    *   [Colab Setup](#colab-setup)
    *   [Local Setup (Optional)](#local-setup-optional)
4.  [Core Components Explained](#4-core-components-explained)
    *   [Chess Environment (`ChessEnv`)](#chess-environment-chessenv)
    *   [Move Mapper (`AlphaZeroMoveMapper`)](#move-mapper-alphazeromovemapper)
    *   [Neural Network (`ChessNet`)](#neural-network-chessnet)
    *   [Monte Carlo Tree Search (`MCTS`)](#monte-carlo-tree-search-mcts)
    *   [Self-Play & Training Loop](#self-play--training-loop)
    *   [Evaluation with Stockfish](#evaluation-with-stockfish)
5.  [How Learning Works](#5-how-learning-works)
6.  [Hyperparameter Optimization](#6-hyperparameter-optimization)
7.  [Next Steps & Possible Improvements](#7-next-steps--possible-improvements)
8.  [Credits](#8-credits)

---

## 1. Project Overview

This project provides a functional, albeit simplified, implementation of the AlphaZero algorithm applied to the game of chess. The agent learns from scratch without human supervision or pre-existing chess knowledge, relying solely on the rules of the game and self-play reinforcement learning.

The primary goal is to demonstrate the core principles of AlphaZero in an accessible manner, allowing for experimentation and understanding even on limited computational resources.

## 2. Features

*   **Self-Play Reinforcement Learning:** Agent plays against itself to generate training data.
*   **Monte Carlo Tree Search (MCTS):** Uses a sophisticated search algorithm to guide exploration and produce improved policy targets for training.
*   **Deep Neural Network (`ChessNet`):** A convolutional residual network that acts as both the policy (move prediction) and value (game outcome prediction) head.
*   **AlphaZero-style Observation:** Board state encoded into a multi-channel tensor, flipped for player perspective.
*   **AlphaZero-style Action Space:** Moves mapped to a fixed 64x73 (4672) discrete action space.
*   **Temperature Annealing & Dirichlet Noise:** Exploration strategies to ensure diverse self-play data collection.
*   **Stockfish Evaluation:** Built-in benchmarking against the Stockfish chess engine at various ELO levels.
*   **Hyperparameter Optimization Script:** A utility to test different neural network architectures (number of residual blocks and channels).

## 3. Getting Started

The easiest way to get started is by using Google Colab.

### Colab Setup

1.  **Open a New Colab Notebook:** Go to [Colab](https://colab.research.google.com/) and create a new Python 3 notebook.
2.  **Enable GPU (Recommended):** Go to `Runtime` -> `Change runtime type` -> select `T4 GPU` (or equivalent) for hardware accelerator.
3.  **Copy-Paste Cells:** Copy the code blocks provided in the previous conversational turns into separate cells in your Colab notebook. Ensure you paste them in the correct order:
    *   `1. Install Dependencies and Stockfish`
    *   `2. Core Chess Environment and Utilities (Modified for AlphaZeroMoveMapper)`
    *   `3. Neural Network (ChessNet)`
    *   `4. Monte Carlo Tree Search (MCTS) - Modified for AlphaZeroMoveMapper`
    *   `5. Self-Play and Training Loop (with Debug Logs)`
    *   `6. Evaluation with Stockfish (Modified for AlphaZeroMoveMapper)`
    *   `7. Hyperparameter Optimization & Analysis`
4.  **Run All Cells:** Go to `Runtime` -> `Run all`. The training process will begin, followed by evaluation and hyperparameter analysis.

### Local Setup (Optional)

If you prefer to run this project locally, follow these steps:

1.  **Clone the Repository (if applicable):** If this README is part of a repo, clone it. Otherwise, create a project directory.
2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install torch numpy python-chess tqdm matplotlib pandas
    ```
    (Note: PyTorch installation might vary based on your OS and CUDA version. Refer to [PyTorch website](https://pytorch.org/get-started/locally/) for specific instructions.)
4.  **Install Stockfish:**
    *   **Linux/WSL:** `sudo apt-get install stockfish`
    *   **macOS:** `brew install stockfish`
    *   **Windows:** Download the executable from the [Stockfish website](https://stockfishchess.org/download/) and place it in your system's PATH, or provide the full path to the executable in the `engine_path` argument in the evaluation code.
5.  **Organize Code:** Create Python files for each logical component (e.g., `chess_env.py`, `models.py`, `mcts.py`, `train.py`, `evaluate.py`, `hyperparams.py`) and import them accordingly.
6.  **Run:** Execute your main training/evaluation script.

## 4. Core Components Explained

### Chess Environment (`ChessEnv`)
*   **Role:** Acts as the interface between the agent and the game of chess.
*   **Backend:** Uses the `python-chess` library for game logic (moves, legality, termination).
*   **Observation:** Translates the `chess.Board` state into an 8x8x20 NumPy array (tensor), which is the input to the neural network. This includes planes for pieces (current player and opponent), turn, and placeholders for history. Critically, it **flips the board's perspective for Black** so the network always "sees" its pieces at the bottom, promoting rotational invariance in learned features.
*   **Actions:** Takes an integer action ID from the agent and converts it into a `chess.Move` object to interact with the `python-chess` board.
*   **Reward:** Provides immediate reward (0.0 for non-terminal, later adjusted to ±1/0 for win/loss/draw for training) and `done` flags.

### Move Mapper (`AlphaZeroMoveMapper`)
*   **Role:** Translates between `chess.Move` objects and a fixed, numerical action space expected by the neural network.
*   **Action Space:** Implements the conceptual **AlphaZero 64x73 action space (4672 total actions)**. This means an action is encoded as `(source_square_index * 73 + move_type_index)`.
    *   The 73 `move_type_index` categories abstract various chess moves (sliding piece moves, knight moves, pawn underpromotions).
*   **Encoding (`move_to_action`):** Takes a `chess.Move` object and the current `board` context, and algorithmically derives its corresponding `move_type_index` and `from_square_index` to produce the flat action ID.
*   **Decoding (`action_to_move`):** Takes a flat action ID and the current `board` context, and reverses the process to reconstruct the `chess.Move` object. The `board` context is essential for promotions and correctly handling perspective flips.

### Neural Network (`ChessNet`)
*   **Role:** The "brain" of the agent, serving as both the policy and value network.
*   **Architecture:** A Convolutional Neural Network (CNN) incorporating **Residual Blocks (`ResBlock`)**.
    *   **`ResBlock`:** Allows for the construction of very deep networks by using "skip connections" (`x + out`), which help mitigate vanishing gradients and improve training stability.
*   **Input:** An 8x8x20 tensor representing the board state observation.
*   **Outputs (Two Heads):**
    *   **Policy Head:** Outputs a log-probability distribution over the 4672 possible actions ($\log\pi(a|s)$).
    *   **Value Head:** Outputs a single scalar value between -1 and 1 ($V(s)$), estimating the game's outcome from the current player's perspective (-1 for certain loss, 0 for draw, 1 for certain win).

### Monte Carlo Tree Search (`MCTS`)
*   **Role:** A planning algorithm that uses the `ChessNet` to guide a simulated search of the game tree, producing a more informed policy and value estimate for the current move.
*   **Simulations (`sims`):** For each real game move, MCTS runs a specified number of simulations (e.g., 50).
*   **Four Phases of Each Simulation:**
    1.  **Selection:** Traverses the search tree using the **PUCT (Polynomial Upper Confidence Trees)** formula, which balances exploiting high-value paths with exploring promising but less-visited paths.
    2.  **Expansion:** When a new (unvisited) node is reached, the `ChessNet` is queried for its policy priors ($P$) and value estimate ($V$), expanding the tree.
    3.  **Simulation (Network Evaluation):** The `ChessNet`'s value estimate for the expanded leaf node acts as the outcome for that simulation.
    4.  **Backpropagation:** The simulation's outcome is propagated back up the traversed path, updating visit counts ($N(s,a)$) and action-value estimates ($Q(s,a)$) for all state-action pairs.
*   **Improved Policy (`pi`):** After all simulations, MCTS generates a final `pi` (policy) distribution over legal moves based on the visit counts. This `pi` is a stronger, more refined policy than the raw `ChessNet` output and serves as the training target.

### Self-Play & Training Loop
*   **Self-Play:** The agent continuously plays games against itself.
    *   **Exploration:** Guided by MCTS, actions are chosen using a **temperature-controlled sampling** from `pi`.
        *   **Temperature Annealing:** Starts with a high `temp` (e.g., 1.0) for early moves to encourage diverse exploration (trying different openings).
        *   Reduces `temp` (e.g., to 0.5) for later moves to promote more greedy, optimal play, providing clearer win/loss signals.
    *   **Dirichlet Noise:** Added to the root node's policy priors to further encourage exploration of varied initial game lines across different self-play games.
    *   **Data Collection:** For each move in a game, the `(observation, MCTS_generated_pi, current_player_turn)` is recorded.
*   **Training:** After a set number of self-play games (`GAMES_PER_EPOCH`), the collected data is used to train the `ChessNet`.
    *   **Loss Function:** A combined loss function is used:
        *   **Value Loss (Mean Squared Error):** Compares `ChessNet`'s predicted value ($V(s)$) to the actual game outcome ($Z$) (win/loss/draw) for that state.
        *   **Policy Loss (Negative Log Likelihood / Cross-Entropy):** Compares `ChessNet`'s predicted policy ($\pi(a|s)$) to the MCTS-generated improved policy (`pi`).
    *   **Optimization:** The network weights are updated using `Adam` optimizer via backpropagation to minimize the total loss.
*   **Iterative Improvement:** The improved `ChessNet` then plays more self-play games, generating higher-quality data, leading to a continuous cycle of learning.

### Evaluation with Stockfish
*   **Role:** Benchmarks the trained agent's performance against a strong, well-established chess engine.
*   **Stockfish Integration:** Uses `python-chess` to communicate with the Stockfish engine executable.
*   **ELO Setting:** Stockfish's playing strength can be adjusted using the `UCI_Elo` parameter (minimum 1350).
*   **Game Play:** The agent plays as White (using MCTS with `temp=0` for greedy play), and Stockfish plays as Black.
*   **Results:** Tracks wins, losses, and draws against different ELO levels, providing a quantitative measure of the agent's progress.

## 5. How Learning Works

The learning process in this AlphaZero-style agent is a powerful form of reinforcement learning:

1.  **Initial State:** The `ChessNet` starts with randomly initialized weights, essentially knowing nothing about chess. MCTS initially explores almost randomly, as `ChessNet`'s policy provides little guidance.
2.  **Self-Play Experience Generation:** The agent plays games against itself. For each move:
    *   MCTS explores possible lines based on its current understanding (informed by the `ChessNet`).
    *   It updates its internal tree statistics ($N, Q$).
    *   It selects a move using a temperature-controlled sampling based on visit counts, adding some randomness (especially early in games) to explore more states.
    *   The `(observation, MCTS_pi, game_outcome_for_state_player)` is recorded.
3.  **Network Training (Policy and Value Iteration):** After a batch of self-play games, the collected data is used to train the `ChessNet`.
    *   The **policy head** learns to mimic the `pi` generated by MCTS. This teaches the network which moves are "good" according to the deeper search.
    *   The **value head** learns to predict the actual outcome ($Z$) of the game from a given position. This teaches the network "how good" a state is.
4.  **Feedback Loop:** The trained `ChessNet` (now a slightly better chess player) is immediately used in the *next* set of self-play games.
    *   A stronger `ChessNet` provides better `policy priors` to MCTS, making MCTS's simulations more efficient and effective.
    *   A more effective MCTS generates even higher-quality `pi` targets and value estimates, leading to more robust training data.
    *   This continuous cycle of self-improvement allows the agent to learn from scratch and progressively increase its playing strength.

## 6. Hyperparameter Optimization

The provided `7. Hyperparameter Optimization & Analysis` cell is designed to help you find suitable architectural parameters for your `ChessNet`.

*   **Combinations:** It tests different numbers of `ResBlocks` and `channels` per layer.
*   **Reduced Training:** It uses fewer self-play games and epochs per combination to make the search feasible on limited hardware.
*   **Metrics:** It tracks average training loss and actual wins/draws/losses against a fixed ELO Stockfish.
*   **Analysis:** The generated graphs (training loss and evaluation results) will help you visualize which network sizes perform better under the given training constraints. Look for models that show reasonable training loss and a positive trend in wins/draws.

## 7. Next Steps & Possible Improvements

This project provides a solid foundation. Here are some ideas for further development:

*   **Increase Training Scale:**
    *   More `EPOCHS` and `GAMES_PER_EPOCH`. This is the most direct way to improve performance.
    *   Increase `MCTS.sims`. More simulations mean better targets for training, but also slower per move.
*   **Observation Channels:**
    *   Implement additional AlphaZero observation planes (e.g., 8-ply move history, half-move clock, repetition count, castling rights, en passant target square). This provides more context to the network.
*   **Advanced Replay Buffer:**
    *   Implement a true "Experience Replay Buffer" that stores many past games and samples from them randomly during training. This breaks correlations in the training data, improving stability.
    *   Use a "Prioritized Experience Replay" to prioritize training on more impactful (e.g., high-error or recent) experiences.
*   **Learning Rate Schedule:** Implement a decaying learning rate, common in deep learning to allow for larger updates early and finer adjustments later.
*   **Regularization:** Add L2 regularization (weight decay is already there) or dropout layers to `ChessNet` to combat overfitting as the network gets larger.
*   **Distributed Training:** For serious performance, AlphaZero relies on massive parallelism (many self-play workers, many training workers). This is beyond Colab Free Tier but is the next logical step.
*   **More Robust Error Handling:** Improve logging and error handling, especially for edge cases during move mapping or environment steps.
*   **Visualizations:** Integrate a chess board visualization library (e.g., `chess.svg` with `IPython.display`) to see games unfold.
*   **Save/Load MCTS Tree:** For very long-running agents, the MCTS tree can be partially persisted between games to retain knowledge, rather than clearing it completely.

## 8. Credits

*   Inspired by the DeepMind AlphaZero paper: "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play" (Silver et al., 2018).
*   Uses the excellent `python-chess` library for game logic.
*   Built with `PyTorch` for deep learning.
