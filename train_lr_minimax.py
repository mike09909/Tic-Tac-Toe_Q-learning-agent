import random
import pickle
import os
from minimax_agent import MinimaxAgent
from tic_tac_toe_game import TicTacToe
import matplotlib.pyplot as plt
import numpy as np
from functools import lru_cache

class QLearningAgentMinimax:
    """
    A Q-learning agent specifically designed to learn against a Minimax opponent in Tic-Tac-Toe.
    This agent uses Q-learning to improve its strategy over time.
    """

    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        """
        Initialize the Q-learning agent.

        Args:
            learning_rate (float): The rate at which the agent learns from new information.
            discount_factor (float): The factor by which future rewards are discounted.
            exploration_rate (float): The probability of choosing a random action for exploration.

        The Q-table is initialized as an empty dictionary to store state-action values.
        """
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def get_q_value(self, state, action):
        """
        Get the Q-value for a given state-action pair.

        Args:
            state (tuple): The current game state.
            action (int): The action (move) to evaluate.

        Returns:
            float: The Q-value for the state-action pair, or 0.0 if not yet encountered.
        """
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, game):
        """
        Choose an action based on the current game state.

        Args:
            game (TicTacToe): The current game state.

        Returns:
            int: The chosen action (move).

        This method implements an epsilon-greedy strategy:
        - With probability 'exploration_rate', choose a random action.
        - Otherwise, choose the action with the highest Q-value.
        """
        state = self.get_optimized_state(tuple(game.board))
        available_moves = game.get_available_moves()
        if random.random() < self.exploration_rate:
            return random.choice(available_moves)
        else:
            q_values = [self.get_q_value(state, action) for action in available_moves]
            max_q = max(q_values)
            best_actions = [action for action, q in zip(available_moves, q_values) if q == max_q]
            return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        """
        Update the Q-value for a state-action pair based on the observed reward and next state.

        Args:
            state (tuple): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (tuple): The resulting state after taking the action.
            done (bool): Whether the episode is finished.

        This method implements the Q-learning update rule:
        Q(s,a) = Q(s,a) + learning_rate * (reward + discount_factor * max(Q(s',a')) - Q(s,a))
        """
        old_q = self.get_q_value(state, action)
        if done:
            target_q = reward
        else:
            next_q_values = [self.get_q_value(next_state, a) for a in range(9) if next_state[a] == 0]
            next_max_q = max(next_q_values) if next_q_values else 0
            target_q = reward + self.discount_factor * next_max_q
        new_q = old_q + self.learning_rate * (target_q - old_q)
        self.q_table[(state, action)] = new_q

    @lru_cache(maxsize=None)
    def get_optimized_state(self, board_tuple):
        """
        Get an optimized representation of the board state, considering symmetries.

        Args:
            board_tuple (tuple): The current board state.

        Returns:
            tuple: An optimized (minimized) representation of the board state.

        This method reduces the state space by considering board symmetries,
        which helps in faster learning and reduced memory usage.
        """
        board = list(board_tuple)
        transformations = [
            board,
            [board[2], board[1], board[0], board[5], board[4], board[3], board[8], board[7], board[6]],
            [board[6], board[7], board[8], board[3], board[4], board[5], board[0], board[1], board[2]],
            [board[8], board[7], board[6], board[5], board[4], board[3], board[2], board[1], board[0]],
            [board[0], board[3], board[6], board[1], board[4], board[7], board[2], board[5], board[8]],
            [board[8], board[5], board[2], board[7], board[4], board[1], board[6], board[3], board[0]],
        ]
        return tuple(min(map(tuple, transformations)))

class CachedMinimaxAgent(MinimaxAgent):
    """
    An extension of the MinimaxAgent that uses caching to improve performance.
    """

    def __init__(self, depth=9):
        """
        Initialize the CachedMinimaxAgent.

        Args:
            depth (int): The maximum depth for the minimax algorithm.

        The cache is used to store and retrieve previously computed minimax values.
        """
        super().__init__(depth)
        self.cache = {}

    def make_move(self, game):
        """
        Determine the best move using the minimax algorithm with caching.

        Args:
            game (TicTacToe): The current game state.

        Returns:
            int: The best move according to the minimax algorithm.

        This method checks the cache before computing minimax values, significantly
        speeding up the decision-making process for previously encountered states.
        """
        board_tuple = tuple(game.board)
        if board_tuple in self.cache:
            return self.cache[board_tuple]

        best_move = super().make_move(game)
        self.cache[board_tuple] = best_move
        return best_move

def train_agent(agent, num_episodes, minimax_depth):
    """
    Train the Q-learning agent against a Minimax opponent.

    Args:
        agent (QLearningAgentMinimax): The Q-learning agent to train.
        num_episodes (int): The number of episodes to train for.
        minimax_depth (int): The depth of the Minimax opponent.

    Returns:
        tuple: Contains win rate, tie rate, Q-table size, Minimax cache size, average Q-value, and new states added.

    This function simulates games between the Q-learning agent and a Minimax opponent,
    allowing the Q-learning agent to learn and improve its strategy over time.
    """
    game = TicTacToe()
    minimax_opponent = CachedMinimaxAgent(depth=minimax_depth)
    wins = losses = ties = 0
    previous_q_table_size = len(agent.q_table)

    for _ in range(num_episodes):
        game.reset()
        game.current_player = random.choice(['O', 'X'])
        state = agent.get_optimized_state(tuple(game.board))
        done = False

        while not done:
            if game.current_player == 'O':
                action = agent.choose_action(game)
            else:
                action = minimax_opponent.make_move(game)
            
            game.make_move(action)
            winner = game.check_winner()

            if winner:
                if winner == 'O':
                    reward = 1
                    wins += 1
                elif winner == 'X':
                    reward = -1
                    losses += 1
                else:
                    reward = 0.5
                    ties += 1
                done = True
                next_state = None
            else:
                reward = 0
                next_state = agent.get_optimized_state(tuple(game.board))

            if game.current_player == 'X':  # Q-learning agent just moved
                agent.learn(state, action, reward, next_state, done)
                state = next_state

    total_games = wins + losses + ties
    win_rate = wins / total_games if total_games > 0 else 0
    tie_rate = ties / total_games if total_games > 0 else 0
    current_q_table_size = len(agent.q_table)
    avg_q = sum(agent.q_table.values()) / len(agent.q_table) if agent.q_table else 0
    new_states = current_q_table_size - previous_q_table_size

    return win_rate, tie_rate, current_q_table_size, len(minimax_opponent.cache), avg_q, new_states

def save_model(agent, filename):
    """
    Save the trained Q-learning agent model to a file.

    Args:
        agent (QLearningAgentMinimax): The trained agent to save.
        filename (str): The path where the model will be saved.

    This function serializes the agent's Q-table and saves it to a file,
    allowing the trained model to be reused later.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(agent.q_table, f)

def evaluate_agent(agent, num_games=1000, minimax_depth=3):
    """
    Evaluate the performance of the trained agent against a Minimax opponent.

    Args:
        agent (QLearningAgentMinimax): The agent to evaluate.
        num_games (int): The number of games to play for evaluation.
        minimax_depth (int): The depth of the Minimax opponent.

    Returns:
        tuple: Contains win rate, loss rate, and tie rate.

    This function assesses how well the trained agent performs against
    a Minimax opponent of a specified depth.
    """
    game = TicTacToe()
    minimax_opponent = CachedMinimaxAgent(depth=minimax_depth)
    wins = losses = ties = 0

    for _ in range(num_games):
        game.reset()
        game.current_player = random.choice(['O', 'X'])
        done = False
        while not done:
            if game.current_player == 'O':
                action = agent.choose_action(game)
            else:
                action = minimax_opponent.make_move(game)
            game.make_move(action)
            winner = game.check_winner()
            if winner:
                if winner == 'O':
                    wins += 1
                elif winner == 'X':
                    losses += 1
                else:
                    ties += 1
                done = True

    win_rate = wins / num_games
    loss_rate = losses / num_games
    tie_rate = ties / num_games
    return win_rate, loss_rate, tie_rate

def plot_training_results(win_rates, tie_rates, q_table_sizes, avg_q_values, new_states_added, total_episodes):
    """
    Plot and save the training results.

    Args:
        win_rates (list): List of win rates over training episodes.
        tie_rates (list): List of tie rates over training episodes.
        q_table_sizes (list): List of Q-table sizes over training episodes.
        avg_q_values (list): List of average Q-values over training episodes.
        new_states_added (list): List of new states added over training episodes.
        total_episodes (int): Total number of training episodes.

    This function creates and saves various plots to visualize the training progress:
    - Win and tie rates
    - Q-table size growth
    - Average Q-value changes
    - New states added per episode
    """
    os.makedirs("pic-minimax", exist_ok=True)
    
    # Calculate actual episode numbers for x-axis
    x_episodes = range(episodes_per_batch, total_episodes + 1, episodes_per_batch)
    
    # Plot win and tie rates
    plt.figure(figsize=(12, 8))
    plt.plot(x_episodes, win_rates, label='Win Rate')
    plt.plot(x_episodes, tie_rates, label='Tie Rate')
    plt.title('Win and Tie Rates over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Rate')
    plt.legend()
    plt.savefig('pic-minimax/win_tie_rates_final.png')
    plt.close()

    # Plot Q-table size
    plt.figure(figsize=(10, 6))
    plt.plot(x_episodes, q_table_sizes, color='blue')
    plt.title('Q-table Size over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Q-table Size')
    plt.xlim(0, total_episodes)
    plt.ylim(0, max(q_table_sizes) * 1.1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('pic-minimax/q_table_size_final.png')
    plt.close()

    # Plot average Q-value
    plt.figure(figsize=(10, 6))
    plt.plot(x_episodes, avg_q_values)
    plt.title('Average Q-value over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Q-value')
    plt.savefig('pic-minimax/average_q_value_final.png')
    plt.close()

    # Plot new states added
    plt.figure(figsize=(10, 6))
    plt.plot(x_episodes, new_states_added)
    plt.title('New States Added per Episode')
    plt.xlabel('Episode')
    plt.ylabel('New States Added')
    plt.savefig('pic-minimax/new_states_added_final.png')
    plt.close()

if __name__ == "__main__":
    """
    Main execution block for training and evaluating the Q-learning agent against Minimax.

    This section:
    1. Initializes the Q-learning agent
    2. Trains the agent against Minimax opponents of varying depths
    3. Periodically evaluates and logs the agent's performance
    4. Saves the trained model
    5. Conducts a final evaluation against various depths of Minimax
    6. Plots and saves the training results
    """
    agent = QLearningAgentMinimax(learning_rate=0.1, discount_factor=0.95, exploration_rate=0.1)
    total_episodes = 270000
    episodes_per_batch = 1000  # Train 1000 episodes per batch
    depths = list(range(1, 10))
    
    win_rates = []
    tie_rates = []
    q_table_sizes = []
    avg_q_values = []
    new_states_added = []
    
    for batch in range(total_episodes // episodes_per_batch):
        depth = random.choice(depths)
        
        win_rate, tie_rate, q_table_size, minimax_cache_size, avg_q, new_states = train_agent(agent, episodes_per_batch, depth)
        
        # Update statistics
        win_rates.append(win_rate)
        tie_rates.append(tie_rate)
        q_table_sizes.append(q_table_size)
        avg_q_values.append(avg_q)
        new_states_added.append(new_states)
        
        print(f"Completed {(batch+1)*episodes_per_batch}/{total_episodes} episodes...")
        
    # Save the model after training
    model_filename = f"qr-model/agent_minimax_depths_final.pkl"
    save_model(agent, model_filename)
    print(f"Final model saved as {model_filename}")

    # Final evaluation
    print("Final Evaluation:")
    for eval_depth in depths:
        print(f"Evaluating agent against Minimax with depth {eval_depth}...")
        eval_win_rate, eval_loss_rate, eval_tie_rate = evaluate_agent(agent, num_games=100, minimax_depth=eval_depth)
        print(f"Evaluation results (100 games against Minimax depth {eval_depth}):")
        print(f"Win rate: {eval_win_rate:.2%}")
        print(f"Loss rate: {eval_loss_rate:.2%}")
        print(f"Tie rate: {eval_tie_rate:.2%}")
    print("\n" + "="*50 + "\n")

    # Plot training results
    plot_training_results(win_rates, tie_rates, q_table_sizes, avg_q_values, new_states_added, total_episodes)
