import numpy as np
import pickle
import os
from tic_tac_toe_game import TicTacToe
from collections import deque
import random
import matplotlib.pyplot as plt
from minimax_agent import MinimaxAgent
from functools import lru_cache

class QLearningAgentByBrid:
    """
    A Q-learning agent that combines strategies for playing Tic-Tac-Toe.
    This agent uses a hybrid approach, incorporating both Q-learning and heuristic strategies.
    """

    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        """
        Initialize the Q-learning agent with hybrid strategies.

        Args:
            learning_rate (float): The learning rate for Q-value updates.
            discount_factor (float): The discount factor for future rewards.
            epsilon (float): The initial exploration rate.

        The agent uses two Q-tables for double Q-learning, which helps reduce overestimation
        of Q-values and improves learning stability.
        """
        self.q_table1 = {}
        self.q_table2 = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.99995
        self.epsilon_min = 0.1
        self.experience_replay = deque(maxlen=50000)
        self.winning_moves = set()
    
    def get_state_key(self, game):
        """
        Get a normalized representation of the board state.

        Args:
            game (TicTacToe): The current game state.

        Returns:
            tuple: A tuple representing the normalized board state.

        This method creates a unique key for each board state, considering rotations
        and reflections to reduce the state space and improve learning efficiency.
        """
        board = [0 if cell == ' ' else (1 if cell == 'X' else 2) for cell in game.board]
        
        # Basic transformations: original, horizontal flip, vertical flip, diagonal flip
        transformations = [
            board,
            [board[6], board[7], board[8], board[3], board[4], board[5], board[0], board[1], board[2]],  # Vertical flip
            [board[2], board[1], board[0], board[5], board[4], board[3], board[8], board[7], board[6]],  # Horizontal flip
            [board[0], board[3], board[6], board[1], board[4], board[7], board[2], board[5], board[8]]   # Transpose
        ]
        return tuple(min(transformations))
    
    def choose_action(self, game):
        """
        Choose an action based on the current game state.

        Args:
            game (TicTacToe): The current game state.

        Returns:
            int: The chosen action (move) for the current state.

        This method implements an epsilon-greedy strategy with additional heuristics:
        - Exploration: Random moves with preference for center and corners.
        - Exploitation: Combines Q-value based selection with strategic move checking.
        """
        state = self.get_state_key(game)
        available_moves = game.get_available_moves()
        
        # Exploration phase
        if random.random() < self.epsilon:
            # Strategic position priority
            if 4 in available_moves and random.random() < 0.6:  # Center priority
                return 4
            corners = [move for move in [0, 2, 6, 8] if move in available_moves]
            if corners and random.random() < 0.4:  # Corners second
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
            
        # Choose best action based on Q-values
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

        Args:
            game (TicTacToe): The current game state.
            player (str): The player to check for ('X' or 'O').

        Returns:
            int or None: The winning move if exists, None otherwise.

        This method simulates each possible move to find an immediate winning move.
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

        Args:
            game (TicTacToe): The current game state.
            player (str): The player to check for ('X' or 'O').

        Returns:
            int or None: The fork move if exists, None otherwise.

        A fork is a move that creates two winning paths, forcing the opponent to block
        one, allowing the player to win on the next move.
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

        Args:
            game (TicTacToe): The current game state.
            action (int): The action (move) taken.
            winner (str or None): The winner of the game, if any.

        Returns:
            float: The calculated reward value.

        This method implements a sophisticated reward system that considers:
        - Game outcome (win, loss, draw)
        - Strategic value of the move (center, corners)
        - Blocking opponent's winning moves
        - Creating winning opportunities
        """
        if winner:
            if winner == 'X':
                self.winning_moves.add((self.get_state_key(game), action))
                return 15.0  # Winning reward
            elif winner == 'O':
                return -15.0  # Losing penalty
            return 0.5  # Small reward for a draw
        
        reward = 0.0
        board = game.board
        
        # Base position rewards
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

        Args:
            state (tuple): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (tuple): The resulting state after taking the action.
            done (bool): Whether the episode is finished.

        This method implements experience replay and double Q-learning:
        - Experiences are stored in a replay buffer.
        - Batch learning is performed on random samples from the buffer.
        - Two Q-tables are used alternately to reduce overestimation bias.
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

        Args:
            q1 (dict): The Q-table being updated.
            q2 (dict): The other Q-table used for next state evaluation.
            state (tuple): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (tuple): The resulting state after taking the action.
            done (bool): Whether the episode is finished.

        This method implements the core Q-learning update, using a second Q-table
        to estimate the value of the next state, which helps to reduce overestimation.
        """
        current_q = q1.get((state, action), 0)
        if done:
            target = reward
        else:
            next_actions = [a for a in range(9) if next_state[a] == 0]
            next_q_values = [q2.get((next_state, a), 0) for a in next_actions]
            target = reward + self.gamma * max(next_q_values) if next_q_values else reward
        
        # Update using Huber loss
        error = target - current_q
        if abs(error) < 1:
            q1[(state, action)] = current_q + self.lr * error
        else:
            q1[(state, action)] = current_q + self.lr * (error / abs(error))

class QLearningAgentMinimax:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, game):
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
    def __init__(self, depth=9):
        super().__init__(depth)
        self.cache = {}

    def evaluate(self, game):
        # 实现评估逻辑
        winner = game.check_winner()
        if winner == 'O':
            return 1
        elif winner == 'X':
            return -1
        return 0

    def minimax(self, game, depth, maximizing_player):
        board_tuple = tuple(game.board)
        if (board_tuple, depth, maximizing_player) in self.cache:
            return self.cache[(board_tuple, depth, maximizing_player)]

        if depth == 0 or game.check_winner() is not None:
            score = self.evaluate(game)
            self.cache[(board_tuple, depth, maximizing_player)] = score
            return score

        if maximizing_player:
            best_score = float('-inf')
            for move in game.get_available_moves():
                game.make_move(move)
                score = self.minimax(game, depth - 1, False)
                game.undo_move(move)
                best_score = max(score, best_score)
        else:
            best_score = float('inf')
            for move in game.get_available_moves():
                game.make_move(move)
                score = self.minimax(game, depth - 1, True)
                game.undo_move(move)
                best_score = min(score, best_score)

        self.cache[(board_tuple, depth, maximizing_player)] = best_score
        return best_score

    def make_move(self, game):
        best_score = float('-inf')
        best_move = None
        for move in game.get_available_moves():
            game.make_move(move)
            score = self.minimax(game, self.depth - 1, False)
            game.undo_move(move)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

def train_agent(agent, num_episodes, opponent_type='random', minimax_depth=None):
    """
    Train the Q-learning agent against a specified opponent.

    Args:
        agent (QLearningAgentByBrid): The agent to train.
        num_episodes (int): Number of episodes to train for.
        opponent_type (str): Type of opponent ('random' or 'minimax').
        minimax_depth (int): Depth for minimax opponent, if applicable.

    Returns:
        tuple: Contains win rate, tie rate, Q-table size, average Q-value, and new states added.

    This function runs the training process, alternating between the Q-learning agent
    and the specified opponent. It tracks various metrics to monitor the learning progress.
    """
    game = TicTacToe()
    minimax_opponent = CachedMinimaxAgent(depth=minimax_depth) if opponent_type == 'minimax' else None
    wins = losses = ties = 0
    previous_q_table_size = len(agent.q_table1) + len(agent.q_table2)

    for episode in range(1, num_episodes + 1):
        game.reset()
        game.current_player = random.choice(['O', 'X'])
        state = agent.get_state_key(game)
        done = False

        while not done:
            if game.current_player == 'O':
                action = agent.choose_action(game)
            else:
                if opponent_type == 'minimax':
                    action = minimax_opponent.make_move(game)
                else:
                    action = random.choice(game.get_available_moves())
            
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
                next_state = agent.get_state_key(game)

            if game.current_player == 'X':  # Q-learning agent just moved
                agent.learn(state, action, reward, next_state, done)
                state = next_state

        # Log every 1000 games
        if episode % 1000 == 0:
            current_q_table_size = len(agent.q_table1) + len(agent.q_table2)
            total_games = wins + losses + ties
            win_rate = wins / total_games if total_games > 0 else 0
            print(f"Episode {episode}: Q-table size: {current_q_table_size}, Win rate: {win_rate:.2%}")
            # Reset counters
            wins, losses, ties = 0, 0, 0

    total_games = wins + losses + ties
    win_rate = wins / total_games if total_games > 0 else 0
    tie_rate = ties / total_games if total_games > 0 else 0
    current_q_table_size = len(agent.q_table1) + len(agent.q_table2)
    avg_q = (sum(agent.q_table1.values()) + sum(agent.q_table2.values())) / (len(agent.q_table1) + len(agent.q_table2)) if (agent.q_table1 or agent.q_table2) else 0
    new_states = current_q_table_size - previous_q_table_size

    return win_rate, tie_rate, current_q_table_size, avg_q, new_states

def random_agent(game):
    """随机代理"""
    return random.choice(game.get_available_moves())

def evaluate_against_random(agent, num_games=100):
    game = TicTacToe()
    wins = losses = ties = 0

    for _ in range(num_games):
        game.reset()
        game.current_player = random.choice(['O', 'X'])
        done = False
        while not done:
            if game.current_player == 'O':
                action = agent.choose_action(game)
            else:
                action = random.choice(game.get_available_moves())
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
            elif not game.get_available_moves():
                ties += 1
                done = True

    win_rate = wins / num_games
    loss_rate = losses / num_games
    tie_rate = ties / num_games
    return win_rate, loss_rate, tie_rate

def evaluate_against_minimax(agent, depth, num_games=100):
    game = TicTacToe()
    minimax_opponent = CachedMinimaxAgent(depth=depth)
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
            elif not game.get_available_moves():
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
        win_rates (list): List of win rates over training.
        tie_rates (list): List of tie rates over training.
        q_table_sizes (list): List of Q-table sizes over training.
        avg_q_values (list): List of average Q-values over training.
        new_states_added (list): List of new states added over training.
        total_episodes (int): Total number of training episodes.

    This function creates and saves various plots to visualize the training progress:
    - Win and tie rates
    - Q-table size growth
    - Average Q-value changes
    - New states added per batch
    """
    os.makedirs("pic-hybrid", exist_ok=True)
    
    # Calculate actual episode numbers for x-axis
    episodes_per_data_point = total_episodes // len(win_rates)
    x_episodes = range(episodes_per_data_point, total_episodes + 1, episodes_per_data_point)
    
    # Plot win and tie rates
    plt.figure(figsize=(12, 6))
    plt.plot(x_episodes, win_rates, label='Win Rate')
    plt.plot(x_episodes, tie_rates, label='Tie Rate')
    plt.title('Win and Tie Rates over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Rate')
    plt.legend()
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    plt.savefig('pic-hybrid/win_tie_rates.png')
    plt.close()

    # Plot Q-table size
    plt.figure(figsize=(12, 6))
    plt.plot(x_episodes, q_table_sizes)
    plt.title('Q-table Size over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Q-table Size')
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    plt.savefig('pic-hybrid/q_table_size.png')
    plt.close()

    # Plot average Q-value
    plt.figure(figsize=(12, 6))
    plt.plot(x_episodes, avg_q_values)
    plt.title('Average Q-value over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Q-value')
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    plt.savefig('pic-hybrid/avg_q_values.png')
    plt.close()

    # Plot new states added
    plt.figure(figsize=(12, 6))
    plt.plot(x_episodes, new_states_added)
    plt.title('New States Added per {} Episodes'.format(episodes_per_data_point))
    plt.xlabel('Episode')
    plt.ylabel('New States Added')
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    plt.savefig('pic-hybrid/new_states_added.png')
    plt.close()

if __name__ == "__main__":
    """
    Main execution block for training and evaluating the hybrid Q-learning agent.

    This section:
    1. Initializes the hybrid Q-learning agent
    2. Trains the agent against both random and minimax opponents
    3. Periodically evaluates and logs the agent's performance
    4. Saves the trained model
    5. Conducts a final evaluation against random and various depths of minimax opponents
    6. Plots and saves the training results
    """
    agent = QLearningAgentByBrid(learning_rate=0.1, discount_factor=0.95, epsilon=0.1)
    total_episodes = 500000
    episodes_per_batch = 1000
    minimax_ratio = 0.25  # 25% chance to use Minimax, 75% chance to use random opponent
    
    win_rates = []
    tie_rates = []
    q_table_sizes = []
    avg_q_values = []
    new_states_added = []
    
    for batch in range(total_episodes // episodes_per_batch):
        if batch % 4 == 0:  # Use Minimax every 4 batches
            depth = random.randint(1, 9)
            win_rate, tie_rate, q_table_size, avg_q, new_states = train_agent(agent, episodes_per_batch, 'minimax', depth)
            print(f"Completed {(batch+1)*episodes_per_batch}/{total_episodes} episodes (Minimax, depth {depth})...")
        else:
            win_rate, tie_rate, q_table_size, avg_q, new_states = train_agent(agent, episodes_per_batch, 'random')
            print(f"Completed {(batch+1)*episodes_per_batch}/{total_episodes} episodes (Random)...")
        
        # Update statistics
        win_rates.append(win_rate)
        tie_rates.append(tie_rate)
        q_table_sizes.append(q_table_size)
        avg_q_values.append(avg_q)
        new_states_added.append(new_states)
        
        print(f"Current Q-table size: {q_table_size}")
        print(f"Win rate: {win_rate:.2%}")
        print(f"Tie rate: {tie_rate:.2%}")
        print(f"Average Q-value: {avg_q:.4f}")
        print(f"New states added: {new_states}")
        print("---")

    # Save the model
    model_path = os.path.join("qr-model", "agent_hybrid_random_minimax.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump((agent.q_table1, agent.q_table2, agent.winning_moves), f)
    print(f"\nModel saved to '{model_path}'")

    # Plot training results
    plot_training_results(win_rates, tie_rates, q_table_sizes, avg_q_values, new_states_added, total_episodes)

    print("Training completed, charts saved.")

    # Final evaluation
    print("\n=== Final Evaluation ===")

    # Against random agent
    print("\nAgainst random agent (1000 games):")
    random_win_rate, random_loss_rate, random_tie_rate = evaluate_against_random(agent, num_games=1000)
    print(f"Win rate: {random_win_rate:.2%}")
    print(f"Loss rate: {random_loss_rate:.2%}")
    print(f"Tie rate: {random_tie_rate:.2%}")

    # Against different depths of Minimax
    for depth in range(1, 10):
        print(f"\nAgainst Minimax (depth {depth}, 100 games):")
        minimax_win_rate, minimax_loss_rate, minimax_tie_rate = evaluate_against_minimax(agent, depth, num_games=100)
        print(f"Win rate: {minimax_win_rate:.2%}")
        print(f"Loss rate: {minimax_loss_rate:.2%}")
        print(f"Tie rate: {minimax_tie_rate:.2%}")

    print("\nEvaluation completed.")
