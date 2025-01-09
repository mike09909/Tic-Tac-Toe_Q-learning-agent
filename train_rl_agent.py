import numpy as np
import pickle
import os
from tic_tac_toe_game import TicTacToe
from collections import deque
import random
import matplotlib.pyplot as plt

# Define model parameters for different training configurations
# These configurations allow for experimenting with various hyperparameters
MODEL_PARAMS = [
    {
        "name": "baseline",
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "exploration_rate": 0.1,
        "num_episodes": 50000,
        "filename": "model_baseline.pkl"
    },
    {
        "name": "high_learning_rate",
        "learning_rate": 0.5,
        "discount_factor": 0.95,
        "exploration_rate": 0.1,
        "num_episodes": 50000,
        "filename": "model_high_learning_rate.pkl"
    },
    {
        "name": "low_discount_factor",
        "learning_rate": 0.1,
        "discount_factor": 0.5,
        "exploration_rate": 0.1,
        "num_episodes": 50000,
        "filename": "model_low_discount_factor.pkl"
    },
    {
        "name": "high_exploration_rate",
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "exploration_rate": 0.5,
        "num_episodes": 50000,
        "filename": "model_high_exploration_rate.pkl"
    },
    {
        "name": "very_high_episodes",
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "exploration_rate": 0.1,
        "num_episodes": 100000,
        "filename": "model_very_high_episodes.pkl"
    }
]

class OptimizedQLearningAgent:
    """
    An optimized Q-learning agent for playing Tic-Tac-Toe.
    This agent uses double Q-learning and experience replay for improved learning.
    """

    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        """
        Initialize the Q-learning agent with optimized learning strategies.

        Args:
            learning_rate (float): The rate at which the agent learns from new information.
            discount_factor (float): The factor by which future rewards are discounted.
            epsilon (float): The initial exploration rate for the epsilon-greedy strategy.

        The agent uses two Q-tables (q_table1 and q_table2) for double Q-learning,
        which helps to reduce overestimation of Q-values.
        """
        self.q_table1 = {}  # First Q-table for double Q-learning
        self.q_table2 = {}  # Second Q-table for double Q-learning
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.99995  # Factor to decay epsilon over time
        self.epsilon_min = 0.1  # Minimum value for epsilon
        self.experience_replay = deque(maxlen=50000)  # Experience replay buffer
        self.winning_moves = set()  # Set to store winning moves for quick reference
    
    def get_state_key(self, game):
        """
        Get a normalized representation of the board state.

        This method creates a unique key for each board state, considering rotations
        and reflections to reduce the state space and improve learning efficiency.
        It's crucial for generalizing learning across symmetric board states.

        Args:
            game (TicTacToe): The current game state.

        Returns:
            tuple: A tuple representing the normalized board state.
        """
        board = [0 if cell == ' ' else (1 if cell == 'X' else 2) for cell in game.board]
        
        # Basic transformations: original, horizontal flip, vertical flip, diagonal flip
        transformations = [
            board,
            [board[6], board[7], board[8], board[3], board[4], board[5], board[0], board[1], board[2]],  # Vertical flip
            [board[2], board[1], board[0], board[5], board[4], board[3], board[8], board[7], board[6]],  # Horizontal flip
            [board[0], board[3], board[6], board[1], board[4], board[7], board[2], board[5], board[8]]   # Transpose
        ]
        return tuple(min(transformations))  # Return the minimum transformation as the canonical state
    
    def choose_action(self, game):
        """
        Choose an action based on the current game state.

        This method implements an epsilon-greedy strategy with additional heuristics:
        - Exploration: Random moves with preference for center and corners.
        - Exploitation: Combines Q-value based selection with strategic move checking.

        Args:
            game (TicTacToe): The current game state.

        Returns:
            int: The chosen action (move) for the current state.
        """
        state = self.get_state_key(game)
        available_moves = game.get_available_moves()
        
        # Exploration phase
        if random.random() < self.epsilon:
            # Strategic position priority during exploration
            if 4 in available_moves and random.random() < 0.6:  # Center priority
                return 4
            corners = [move for move in [0, 2, 6, 8] if move in available_moves]
            if corners and random.random() < 0.4:  # Corners second priority
                return random.choice(corners)
            return random.choice(available_moves)
        
        # Exploitation phase: Check for winning moves
        winning_move = self.get_winning_move(game, 'X')
        if winning_move is not None:
            return winning_move
            
        # Block opponent's winning moves
        blocking_move = self.get_winning_move(game, 'O')
        if blocking_move is not None:
            return blocking_move
            
        # Check for fork opportunities
        fork_move = self.check_fork_opportunity(game, 'X')
        if fork_move is not None:
            return fork_move
            
        # Choose best action based on average Q-values from both tables
        q_values = {move: (self.q_table1.get((state, move), 0) + 
                         self.q_table2.get((state, move), 0)) / 2 
                   for move in available_moves}
        max_q = max(q_values.values())
        best_moves = [move for move, q in q_values.items() if q == max_q]
        
        # Prioritize center and corners among best moves
        strategic_moves = [move for move in best_moves if move in [4, 0, 2, 6, 8]]
        if strategic_moves:
            return random.choice(strategic_moves)
        return random.choice(best_moves)

    def get_winning_move(self, game, player):
        """
        Check if there's a winning move for the given player.

        This method simulates each possible move to find an immediate winning move.
        It's a crucial part of the agent's strategic decision-making.

        Args:
            game (TicTacToe): The current game state.
            player (str): The player to check for ('X' or 'O').

        Returns:
            int or None: The winning move if exists, None otherwise.
        """
        for move in game.get_available_moves():
            temp_board = game.board.copy()
            game.board[move] = player
            if game.check_winner() == player:
                game.board = temp_board
                return move
            game.board = temp_board
        return None

    def check_fork_opportunity(self, game, player):
        """
        Check if there's an opportunity to create a fork (two winning paths).

        A fork is a powerful strategic move that creates two winning paths,
        forcing the opponent to block one and allowing the player to win on the next move.

        Args:
            game (TicTacToe): The current game state.
            player (str): The player to check for ('X' or 'O').

        Returns:
            int or None: The fork move if exists, None otherwise.
        """
        for move in game.get_available_moves():
            winning_paths = 0
            temp_board = game.board.copy()
            game.board[move] = player
            
            # Check if this move creates multiple winning opportunities
            for next_move in game.get_available_moves():
                game.board[next_move] = player
                if game.check_winner() == player:
                    winning_paths += 1
                game.board[next_move] = ' '
            
            game.board = temp_board
            if winning_paths >= 2:
                return move
        return None

    def calculate_reward(self, game, action, winner):
        """
        Calculate the reward for an action based on the game outcome and strategic value.

        This method implements a sophisticated reward system that considers:
        - Game outcome (win, loss, draw)
        - Strategic value of the move (center, corners)
        - Blocking opponent's winning moves
        - Creating winning opportunities

        Args:
            game (TicTacToe): The current game state.
            action (int): The action (move) taken.
            winner (str or None): The winner of the game, if any.

        Returns:
            float: The calculated reward value.
        """
        if winner:
            if winner == 'X':
                self.winning_moves.add((self.get_state_key(game), action))
                return 15.0  # High reward for winning
            elif winner == 'O':
                return -15.0  # High penalty for losing
            return 0.5  # Small positive reward for a draw
        
        reward = 0.0
        board = game.board
        
        # Reward for strategic positions
        if action == 4:  # Center
            reward += 2.0
        elif action in [0, 2, 6, 8]:  # Corners
            reward += 1.0
        
        # Check if the move blocked opponent's winning opportunity
        was_blocking = False
        temp_board = board.copy()
        for opponent_move in game.get_available_moves():
            board[opponent_move] = 'O'
            if game.check_winner() == 'O':
                was_blocking = True
                reward += 3.0  # Reward for blocking opponent's win
                break
            board[opponent_move] = ' '
        board = temp_board
        
        # If not blocking, check if it created a winning opportunity
        if not was_blocking:
            temp_board = board.copy()
            winning_opportunities = 0
            for next_move in game.get_available_moves():
                board[next_move] = 'X'
                if game.check_winner() == 'X':
                    winning_opportunities += 1
                board[next_move] = ' '
            reward += winning_opportunities * 2.0  # Reward for each winning opportunity
            board = temp_board
        
        return reward

    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-values based on the agent's experience.

        This method implements experience replay and double Q-learning:
        - Experiences are stored in a replay buffer.
        - Batch learning is performed on random samples from the buffer.
        - Two Q-tables are used alternately to reduce overestimation bias.

        Args:
            state (tuple): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (tuple): The resulting state after taking the action.
            done (bool): Whether the episode is finished.
        """
        # Add experience to replay buffer
        self.experience_replay.append((state, action, reward, next_state, done))
        
        # Batch learning
        if len(self.experience_replay) >= 32:
            batch = random.sample(self.experience_replay, 32)
            for state, action, reward, next_state, done in batch:
                # Randomly choose which Q-table to update
                if random.random() < 0.5:
                    self._update_q_value(self.q_table1, self.q_table2, state, action, reward, next_state, done)
                else:
                    self._update_q_value(self.q_table2, self.q_table1, state, action, reward, next_state, done)
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _update_q_value(self, q1, q2, state, action, reward, next_state, done):
        """
        Update Q-value using double Q-learning approach.

        This method implements the core Q-learning update, using a second Q-table
        to estimate the value of the next state, which helps to reduce overestimation.

        Args:
            q1 (dict): The Q-table being updated.
            q2 (dict): The other Q-table used for next state evaluation.
            state (tuple): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (tuple): The resulting state after taking the action.
            done (bool): Whether the episode is finished.
        """
        current_q = q1.get((state, action), 0)
        if done:
            target = reward
        else:
            next_actions = [a for a in range(9) if next_state[a] == 0]
            next_q_values = [q2.get((next_state, a), 0) for a in next_actions]
            target = reward + self.gamma * max(next_q_values) if next_q_values else reward
        
        # Update using Huber loss for robustness against outliers
        error = target - current_q
        if abs(error) < 1:
            q1[(state, action)] = current_q + self.lr * error
        else:
            q1[(state, action)] = current_q + self.lr * (error / abs(error))

def train_agent(agent, num_episodes):
    """
    Train the Q-learning agent.

    This function simulates games between the Q-learning agent and a random opponent,
    allowing the Q-learning agent to learn and improve its strategy over time.
    The agents now alternate who goes first.

    Args:
        agent (OptimizedQLearningAgent): The agent to train.
        num_episodes (int): Number of episodes to train for.

    Returns:
        tuple: Lists containing win rates, draw rates, Q-table sizes, and new states added over training.
    """
    env = TicTacToe()
    win_rates = []
    draw_rates = []
    q_table_sizes = []
    new_states_added = []
    eval_interval = 100  # Changed evaluation interval from 1000 to 100
    eval_games = 30  # Reduced number of evaluation games to speed up training
    previous_q_table_size = 0

    for episode in range(1, num_episodes + 1):
        env.reset()
        state = agent.get_state_key(env)
        done = False
        
        agent_first = episode % 2 == 1
        
        while not done:
            current_player = 'X' if (agent_first and env.current_player == 'X') or (not agent_first and env.current_player == 'O') else 'O'
            
            # Check if the game has ended
            if env.check_winner() is not None or len(env.get_available_moves()) == 0:
                done = True
                continue
            
            if current_player == 'X':
                action = agent.choose_action(env)
                env.make_move(action)
                next_state = agent.get_state_key(env)
                winner = env.check_winner()
                reward = agent.calculate_reward(env, action, winner)
                done = winner is not None or len(env.get_available_moves()) == 0
                agent.learn(state, action, reward, next_state, done)
                state = next_state
            else:
                available_moves = env.get_available_moves()
                if available_moves:
                    action = random_agent(env)
                    env.make_move(action)
                    winner = env.check_winner()
                    done = winner is not None or len(env.get_available_moves()) == 0
                else:
                    done = True
        
        # Evaluate every 100 episodes
        if episode % eval_interval == 0:
            wins = draws = 0
            for _ in range(eval_games):
                env.reset()
                game_done = False
                eval_agent_first = _ % 2 == 0  # Alternate who goes first in evaluation games
                while not game_done:
                    current_player = 'X' if (eval_agent_first and env.current_player == 'X') or (not eval_agent_first and env.current_player == 'O') else 'O'
                    if current_player == 'X':
                        action = agent.choose_action(env)
                    else:
                        action = random_agent(env)
                    env.make_move(action)
                    winner = env.check_winner()
                    if winner is not None:
                        game_done = True
                        if (winner == 'X' and eval_agent_first) or (winner == 'O' and not eval_agent_first):
                            wins += 1
                        elif winner == ' ':
                            draws += 1
            
            win_rate = wins / eval_games
            draw_rate = draws / eval_games
            win_rates.append(win_rate)
            draw_rates.append(draw_rate)
            
            current_q_table_size = len(agent.q_table1)
            q_table_sizes.append(current_q_table_size)
            new_states = current_q_table_size - previous_q_table_size
            new_states_added.append(new_states)
            previous_q_table_size = current_q_table_size
            
            print(f"\nEpisode {episode}")
            print(f"Win Rate: {win_rate*100:.2f}%")
            print(f"Draw Rate: {draw_rate*100:.2f}%")
            print(f"Loss Rate: {((1-win_rate-draw_rate)*100):.2f}%")
            print(f"Q-table size: {current_q_table_size}")
            print(f"Epsilon: {agent.epsilon:.4f}")
            print(f"Winning moves learned: {len(agent.winning_moves)}")

    return win_rates, draw_rates, q_table_sizes, new_states_added

def random_agent(game):
    """
    A simple agent that makes random moves.

    This function represents the behavior of a random opponent,
    which is used during training to provide diverse experiences for the Q-learning agent.

    Args:
        game (TicTacToe): The current game state.

    Returns:
        int: A randomly chosen valid move.
    """
    return random.choice(game.get_available_moves())

def plot_performance(params, agent, win_rates, draw_rates, q_table_sizes, new_states_added):
    """
    Generate and save performance plots for a single trained agent.
    """
    # Modify x-axis range
    x_range = range(100, params['num_episodes'] + 1, 100)

    # Win rate plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, win_rates)
    plt.title('Win Rate over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.savefig(os.path.join('pic-random', f'win_rate_over_episodes_{params["name"]}.png'))
    plt.close()

    # Average Q-value plot
    plt.figure(figsize=(10, 6))
    avg_q_values = [sum(agent.q_table1.values()) / len(agent.q_table1) for _ in range(len(win_rates))]
    plt.plot(x_range, avg_q_values)
    plt.title('Average Q-value over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Q-value')
    plt.savefig(os.path.join('pic-random', f'average_q_value_over_episodes_{params["name"]}.png'))
    plt.close()

    # Q-table size plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, q_table_sizes)
    plt.title('Q-table Size over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Q-table Size')
    plt.savefig(os.path.join('pic-random', f'q_table_size_over_episodes_{params["name"]}.png'))
    plt.close()

    # New states added plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, new_states_added)
    plt.title('New States Added per 100 Episodes')
    plt.xlabel('Episode')
    plt.ylabel('New States Added')
    plt.savefig(os.path.join('pic-random', f'new_states_added_per_100_episodes_{params["name"]}.png'))
    plt.close()

def plot_combined_performance(all_params, all_results):
    """
    Generate and save combined performance plots for all trained agents.

    Args:
        all_params (list): List of parameter dictionaries for all models.
        all_results (list): List of result tuples for all models.
    """
    plt.figure(figsize=(12, 8))
    for params, results in zip(all_params, all_results):
        x_range = range(100, params['num_episodes'] + 1, 100)
        plt.plot(x_range, results[0], label=params['name'])
    plt.title('Win Rate over Episodes for All Models')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.savefig(os.path.join('pic-random', 'combined_win_rate_over_episodes.png'))
    plt.close()

    plt.figure(figsize=(12, 8))
    for params, results in zip(all_params, all_results):
        x_range = range(100, params['num_episodes'] + 1, 100)
        avg_q_values = [sum(results[4].q_table1.values()) / len(results[4].q_table1) for _ in range(len(results[0]))]
        plt.plot(x_range, avg_q_values, label=params['name'])
    plt.title('Average Q-value over Episodes for All Models')
    plt.xlabel('Episode')
    plt.ylabel('Average Q-value')
    plt.legend()
    plt.savefig(os.path.join('pic-random', 'combined_average_q_value_over_episodes.png'))
    plt.close()

    plt.figure(figsize=(12, 8))
    for params, results in zip(all_params, all_results):
        x_range = range(100, params['num_episodes'] + 1, 100)
        plt.plot(x_range, results[2], label=params['name'])
    plt.title('Q-table Size over Episodes for All Models')
    plt.xlabel('Episode')
    plt.ylabel('Q-table Size')
    plt.legend()
    plt.savefig(os.path.join('pic-random', 'combined_q_table_size_over_episodes.png'))
    plt.close()

    plt.figure(figsize=(12, 8))
    for params, results in zip(all_params, all_results):
        x_range = range(100, params['num_episodes'] + 1, 100)
        plt.plot(x_range, results[3], label=params['name'])
    plt.title('New States Added per 100 Episodes for All Models')
    plt.xlabel('Episode')
    plt.ylabel('New States Added')
    plt.legend()
    plt.savefig(os.path.join('pic-random', 'combined_new_states_added_per_100_episodes.png'))
    plt.close()

def main():
    """
    Main function to run the training process for different model configurations.

    This function orchestrates the entire training process:
    1. Sets up the training environment
    2. Trains agents with different hyperparameters
    3. Saves the trained models
    4. Generates and saves performance plots

    The use of different configurations allows for experimentation and
    comparison of various hyperparameter settings.
    """
    print("Starting training...")
    
    # Ensure pic-random folder exists for saving plots
    pic_dir = "pic-random"
    os.makedirs(pic_dir, exist_ok=True)

    # Ensure model save directory exists
    model_dir = "qr-model"
    os.makedirs(model_dir, exist_ok=True)

    all_results = []

    for params in MODEL_PARAMS:
        print(f"\nTraining model: {params['name']}")
        print(f"Learning rate: {params['learning_rate']}")
        print(f"Discount factor: {params['discount_factor']}")
        print(f"Exploration rate: {params['exploration_rate']}")
        print(f"Training episodes: {params['num_episodes']}")
        
        agent = OptimizedQLearningAgent(
            learning_rate=params['learning_rate'],
            discount_factor=params['discount_factor'],
            epsilon=params['exploration_rate']
        )
        win_rates, draw_rates, q_table_sizes, new_states_added = train_agent(agent, params['num_episodes'])

        # Save the trained model
        model_path = os.path.join(model_dir, params['filename'])
        with open(model_path, 'wb') as f:
            pickle.dump((agent.q_table1, agent.q_table2, agent.winning_moves), f)
        print(f"\nModel saved to '{model_path}'")

        plot_performance(params, agent, win_rates, draw_rates, q_table_sizes, new_states_added)
        all_results.append((win_rates, draw_rates, q_table_sizes, new_states_added, agent))

    plot_combined_performance(MODEL_PARAMS, all_results)

    print("All training completed and charts have been saved.")

if __name__ == "__main__":
    main()

