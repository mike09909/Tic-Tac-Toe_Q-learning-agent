# Tic Tac Toe AI Project

A comprehensive implementation of Tic-Tac-Toe featuring various AI agents, including Q-learning agents and Minimax agents. The code is structured to facilitate training, evaluation, and gameplay of these agents. The project aims to compare the performance of reinforcement learning algorithms (Q-learning) with traditional planning/searching algorithms (Minimax) in the context of a simple yet strategic game.

## Table of Contents
1. [Overview](#overview)
2. [Game Rules](#game-rules)
3. [Project Structure](#project-structure)
4. [Project Dependencies and Environment Setup](#project-dependencies-and-environment-setup)
5. [Model Training, Saving, and Loading](#model-training-saving-and-loading)
6. [Game Startup and Agent Selection](#game-startup-and-agent-selection)
7. [Game Interface and Gameplay](#game-interface-and-gameplay)
8. [Code Documentation](#code-documentation)

## Overview

The main entry point of the application is the `tic_tac_toe.py` file. This file initializes the game interface and manages the overall flow of the application. It serves as the central hub, coordinating between the user interface, game logic, and AI agents.

## Game Rules

Tic-Tac-Toe is a two-player game played on a 3x3 grid. Players take turns marking empty cells with their symbol (X or O). The first player to align three of their symbols horizontally, vertically, or diagonally wins the game. If all cells are filled and no player has won, the game is a draw.

## Project Structure

### tic_tac_toe.py
This is the main file that runs the game interface. It uses Pygame to create a graphical user interface for the Tic-Tac-Toe game. Key functions and features include:

- Initializing the game window and UI elements
- Managing game states (start screen, game screen, end screen)
- Handling user inputs for menu navigation and move selection
- Coordinating gameplay between different AI agents
- Displaying game results and statistics
- Loading and initializing different AI agents based on user selection
- Implementing the game loop for smooth gameplay and transitions

### tic_tac_toe_game.py
This file contains the core game logic for Tic-Tac-Toe. It defines the TicTacToe class, which encapsulates the game state and rules. Key methods include:

- `make_move`: Executes a player's move and updates the game state
- `check_winner`: Determines if there's a winner or if the game is a draw
- `get_available_moves`: Returns a list of valid moves
- `is_game_over`: Checks if the game has ended
- `reset`: Resets the game board for a new game
- `get_board_state`: Returns the current state of the game board

### train_rl_agent.py
This file is responsible for training the Q-learning agent against a random opponent. It includes:

- The OptimizedQLearningAgent class with:
  - Double Q-learning implementation to reduce overestimation bias
  - Experience replay for improved learning efficiency
  - Epsilon-greedy strategy for balancing exploration and exploitation
  - Methods for state representation, action selection, and Q-value updates
- Training loop that simulates games between the Q-learning agent and a random opponent
- Evaluation functions to track the agent's performance over time
- Performance plotting functions to visualize learning progress:
  - Win rate over episodes
  - Average Q-value over episodes
  - Q-table size over episodes
  - New states added per episode
- Functions to save and load trained models

### train_lr_minimax.py
This file trains a Q-learning agent specifically against a Minimax opponent. It contains:

- The QLearningAgentMinimax class, tailored for learning against Minimax:
  - Specialized state representation to capture Minimax opponent patterns
  - Adaptive learning rate and exploration strategies
- Training and evaluation functions for Minimax-specific scenarios:
  - Simulated games against Minimax opponents of varying depths
  - Performance tracking against Minimax strategies
- Analysis tools to compare Q-learning performance with Minimax benchmarks

### train_lr_hybrid.py
This file implements a hybrid Q-learning agent that can play against both random and Minimax opponents. It includes:

- The QLearningAgentHybrid class:
  - Combines strategies learned from both random and Minimax opponents
  - Implements a flexible state representation to handle diverse opponents
  - Uses a dynamic action selection mechanism based on opponent modeling
- Training functions that alternate between random and Minimax opponents
- Evaluation methods to assess performance against various opponent types
- Comparative analysis tools to measure the hybrid agent's versatility

### minimax_agent.py
This file contains the implementation of the Minimax algorithm for Tic-Tac-Toe. It includes:

- The MinimaxAgent class:
  - Implements the Minimax algorithm
  - Includes depth-limited search for performance optimization
- Customizable search depth to adjust the agent's look-ahead capability

### agents.py
This file contains a simple implementation of a random agent for Tic-Tac-Toe:

- RandomAgent class:
  - Implements a `make_move` method that selects a random move from the available moves in the game
  - Used for baseline comparison and initial training of other AI agents

## Project Dependencies and Environment Setup

To run this project, you need to set up a Python environment with the following dependencies:

- Python 3.7 or higher
- Pygame (2.1.0 or higher): For creating the graphical user interface
- NumPy (1.19.0 or higher): For numerical computations and array operations
- Matplotlib (3.3.0 or higher): For generating performance plots and visualizations
- Pickle: For saving and loading trained models (included in Python standard library)
- Functools: For using the lru_cache decorator (included in Python standard library)

### Setup Instructions

1. Download the source files
2. Install the required dependencies:
   ```bash
   pip install pygame numpy matplotlib
   ```
3. Ensure all Python files are in the same directory or adjust import statements accordingly
4. Run tic_tac_toe.py to start the game interface:
   ```bash
   python tic_tac_toe.py
   ```

## Model Training, Saving, and Loading

The project implements a workflow for training, saving, and loading reinforcement learning models:

### Training
The training process is handled by three main files:
- `train_rl_agent.py`: Trains the Q-learning agent against random opponents
- `train_lr_minimax.py`: Trains the Q-learning agent specifically against Minimax opponents
- `train_lr_hybrid.py`: Trains a hybrid Q-learning agent that can play against both random and Minimax opponents

### Saving
After training, each script saves the trained model to the qr-model folder in the project directory. The saved models include:
- Q-tables (for single or double Q-learning)
- Winning moves sets
- Other relevant agent parameters

### Loading
The main game interface file, `tic_tac_toe.py`, handles the loading of trained models:
- When a player selects a reinforcement learning agent for gameplay, `tic_tac_toe.py` loads the corresponding model from the qr-model folder
- The loaded model is then used to initialize the agent, allowing it to make decisions based on its trained knowledge

## Game Startup and Agent Selection

When the game is launched by running `tic_tac_toe.py`, players are presented with an agent selection interface. This interface allows users to choose any two agents to compete against each other. The available agents include:

- Random Agent
- Three pre-trained reinforcement learning agents:
  - Q-Learning Agent (trained against random opponents)
  - Q-Learning Agent (trained against Minimax)
  - Hybrid Q-Learning Agent (trained against both random and Minimax opponents)
- Minimax Agent

When selecting the Minimax agent, players can adjust two key parameters:
- Depth: This setting determines how many moves ahead the Minimax algorithm will consider. A higher depth generally results in stronger play but requires more computation time
- Time Limit: This sets the maximum time (in seconds) that the Minimax agent is allowed to spend on each move. This feature ensures that the game remains responsive, even when using high depth settings

## Game Interface and Gameplay

After selecting two agents from the agent selection screen, the game proceeds to the main gameplay interface. This interface provides a comprehensive view of the ongoing match:

### Game Board
The central feature of the interface is a 3x3 grid representing the Tic-Tac-Toe board. This board dynamically updates to show the current state of the game, with 'X' and 'O' symbols representing each player's moves.

### Match Information
Below the game board, players can find detailed information about the current match:
- Current Game Number: Displays which game out of the total 50 games is currently being played (e.g., "Game 9/50")
- Agent Information: For each agent (X and O), the interface shows:
  - The type of agent (e.g., "Random Agent", "Q-Learning Agent(for random)", etc.)
  - Current win/loss/draw statistics for the ongoing series of games

### Real-time Updates
As the games progress, the interface updates in real-time, allowing players to observe:
- Each move made by the agents
- The outcome of each game
- Cumulative statistics across the 50-game series

### Series Progress
The interface automatically cycles through all 50 games, providing a comprehensive view of the performance of both agents over an extended series.

### Visual Clarity
Different colors are used to distinguish between the agents (typically blue for 'X' and red for 'O'), making it easy to follow the progress of each game at a glance.

The 50-game series provides a statistically significant sample to evaluate the performance of the chosen agents, offering a balanced view of their capabilities beyond the outcome of a single game. This setup is particularly useful for comparing the long-term effectiveness of different AI approaches in the context of Tic-Tac-Toe.

## Code Documentation

An important feature of this project is that each Python file contains detailed comments. These comments provide the following benefits:

- Code Understanding: Every function, class, and key code block has comments explaining its purpose and functionality
- Algorithm Explanation: For complex algorithms (such as Q-learning or Minimax), comments provide detailed explanations of how they work
- Parameter Description: Function parameters and return values are clearly described
- Usage Examples: Where appropriate, comments provide examples of how to use specific functions or classes
- Design Decisions: Comments explain why certain implementation methods or data structures were chosen
- Optimization Notes: For performance-optimized sections, comments explain the reasons and methods for optimization

