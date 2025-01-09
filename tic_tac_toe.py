import pygame
import sys
from tic_tac_toe_game import TicTacToe
from agents import RandomAgent
from train_lr_minimax import QLearningAgentMinimax
from minimax_agent import MinimaxAgent, get_best_move
import time
import logging
import os
import pickle
from train_rl_agent import OptimizedQLearningAgent

# Set up logging configuration
# This allows for detailed tracking of the program's execution, which is crucial for debugging and performance analysis
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Pygame
# Pygame is used for creating the graphical user interface of the game
pygame.init()

# Define colors for the game interface
# These color constants are used throughout the game for various UI elements
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
LIGHT_GRAY = (230, 230, 230)
DARK_GRAY = (100, 100, 100)
LIGHT_BLUE = (173, 216, 230)

# Set up the game window
# This defines the size of the game window and sets the caption
WIDTH, HEIGHT = 600, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic-Tac-Toe")

# Define fonts for different text elements
# Different font sizes are used for various text elements in the game
title_font = pygame.font.Font(None, 42)
font = pygame.font.Font(None, 28)
small_font = pygame.font.Font(None, 22)

# Define game states
# These states are used to control the flow of the game between the start screen and the game screen
START_SCREEN = 0
GAME_SCREEN = 1
current_state = START_SCREEN

# Initialize agent selection flags
# These flags keep track of which agents have been selected by the user
random_agent_selected = False
optimized_ql_agent_selected = False
ql_hybrid_agent_selected = False
ql_agent_selected = False
minimax_agent_selected = False
minimax_depth = 5
minimax_time_limit = 20

# Global variables for game management
# These variables keep track of the game state, agents, and game statistics
game = None
agent1 = None
agent2 = None
games_played = 0
max_games = 50
agent1_wins = 0
agent2_wins = 0
ties = 0

# Additional global variables
ql_agent = None
current_first_player = 'X'  # Initialize the first player

def load_model_from_file(filename):
    """
    Load a trained model from a file.

    This function is crucial for loading pre-trained Q-learning models,
    allowing the use of previously learned strategies. It uses Python's
    pickle module to deserialize the model data.

    Args:
        filename (str): Path to the model file.

    Returns:
        object: The loaded model (typically a Q-table or neural network).

    Note:
        Ensure that the file exists and contains valid pickle data to avoid errors.
    """
    with open(filename, 'rb') as file:
        agent = pickle.load(file)
    return agent

def load_ql_agent(agent_type):
    """
    Load a Q-learning agent based on the specified type.

    This function handles the loading of different types of Q-learning agents,
    including hybrid, optimized, and standard versions. It's a key part of
    the system's flexibility in using different AI strategies.

    Args:
        agent_type (str): Type of Q-learning agent to load ('ql_hybrid', 'optimized_ql', or 'ql').

    Returns:
        object: The loaded Q-learning agent, ready for gameplay.

    Raises:
        ValueError: If an unsupported agent type is specified.

    Note:
        The function configures the agent differently based on its type,
        setting up the appropriate Q-tables and other necessary attributes.
    """
    filename = {
        'ql_hybrid': 'qr-model/agent_hybrid_random_minimax.pkl',
        'optimized_ql': 'qr-model/model_very_high_episodes.pkl',
        'ql': 'qr-model/agent_minimax_depths_final.pkl'
    }.get(agent_type)

    if not filename:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    agent = load_model_from_file(filename)

    # Configure the agent based on its type
    if agent_type in ['ql_hybrid', 'optimized_ql']:
        q_table1, q_table2, winning_moves = agent
        agent = OptimizedQLearningAgent()
        agent.q_table1 = q_table1
        agent.q_table2 = q_table2
        agent.winning_moves = winning_moves
    elif agent_type == 'ql':
        q_table = agent
        agent = QLearningAgentMinimax()
        agent.q_table = q_table

    # Set agent properties
    agent.exploration_rate = 0  # Ensure deterministic behavior during gameplay
    agent.filename = filename
    agent.agent_type = agent_type

    # Log agent information for debugging and analysis
    logging.info(f"Loaded Q-learning agent from {filename}")
    if isinstance(agent, OptimizedQLearningAgent):
        logging.info(f"Q-table1 size: {len(agent.q_table1)}, Q-table2 size: {len(agent.q_table2)}")
    else:
        logging.info(f"Q-table size: {len(agent.q_table)}")

    return agent

def draw_start_screen():
    """
    Draw the start screen of the game.

    This function creates the initial user interface, including:
    - Game title
    - Agent selection checkboxes
    - Minimax depth and time limit sliders
    - Confirm button

    It's crucial for setting up the game parameters before starting.
    The start screen allows users to select agents and configure game settings.
    """
    screen.fill(WHITE)
    
    # Draw title
    title = title_font.render("Tic-Tac-Toe", True, BLACK)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 40))
    
    # Draw agent selection checkboxes
    draw_checkbox("Random Agent", 80, 130, random_agent_selected)
    draw_checkbox("Q-Learning Agent(for random)", 80, 180, optimized_ql_agent_selected)
    draw_checkbox("Q-Learning Agent(for random and minimax)", 80, 230, ql_hybrid_agent_selected)
    draw_checkbox("Q-Learning Agent(for MiniMax)", 80, 280, ql_agent_selected)
    draw_checkbox("Minimax Agent", 80, 330, minimax_agent_selected)
    
    # Draw Minimax parameter sliders
    draw_slider("Depth", 80, 400, minimax_depth, 1, 10)
    draw_slider("Time Limit (s)", 80, 470, minimax_time_limit, 1, 60)
    
    # Draw confirm button
    draw_button("Confirm", WIDTH // 2 - 60, 550, 120, 40)

def draw_checkbox(text, x, y, checked):
    """
    Draw a checkbox with label.
    
    This function creates a visual checkbox with an associated label.
    It's used in the start screen for agent selection.

    Args:
        text (str): Label for the checkbox.
        x (int): X-coordinate of the checkbox.
        y (int): Y-coordinate of the checkbox.
        checked (bool): Whether the checkbox is checked.

    The checkbox is drawn as a square with a checkmark if selected.
    """
    pygame.draw.rect(screen, LIGHT_GRAY, (x, y, 25, 25))
    pygame.draw.rect(screen, BLACK, (x, y, 25, 25), 2)
    if checked:
        pygame.draw.line(screen, BLACK, (x + 5, y + 12), (x + 10, y + 20), 2)
        pygame.draw.line(screen, BLACK, (x + 10, y + 20), (x + 20, y + 5), 2)
    label = font.render(text, True, BLACK)
    screen.blit(label, (x + 35, y + 3))

def draw_slider(text, x, y, value, min_value, max_value):
    """
    Draw a slider with label and current value.
    
    This function creates a visual slider for adjusting numerical values.
    It's used for setting Minimax depth and time limit in the start screen.

    Args:
        text (str): Label for the slider.
        x, y (int): Coordinates of the slider.
        value (int): Current value of the slider.
        min_value, max_value (int): Range of the slider.

    The slider includes a label, a track, and a movable handle indicating the current value.
    """
    label = font.render(f"{text}: {value}", True, BLACK)
    screen.blit(label, (x, y))
    
    slider_width = 180
    pygame.draw.rect(screen, LIGHT_GRAY, (x, y + 25, slider_width, 8))
    slider_pos = x + int((value - min_value) / (max_value - min_value) * slider_width)
    pygame.draw.circle(screen, LIGHT_BLUE, (slider_pos, y + 29), 12)
    pygame.draw.circle(screen, BLACK, (slider_pos, y + 29), 12, 2)

def draw_button(text, x, y, width, height):
    """
    Draw a button with text.

    This function creates a clickable button with text.
    It's used for the confirm button in the start screen.

    Args:
        text (str): Text to display on the button.
        x, y (int): Coordinates of the top-left corner of the button.
        width, height (int): Dimensions of the button.

    The button is drawn as a rectangle with centered text.
    """
    pygame.draw.rect(screen, LIGHT_BLUE, (x, y, width, height))
    pygame.draw.rect(screen, BLACK, (x, y, width, height), 2)
    label = font.render(text, True, BLACK)
    screen.blit(label, (x + width // 2 - label.get_width() // 2, y + height // 2 - label.get_height() // 2))

def handle_start_screen_events(event):
    """
    Handle events on the start screen.

    This function is responsible for processing user interactions on the start screen,
    including agent selection, parameter adjustments, and game initiation.

    Key features:
    - Manages agent selection and deselection
    - Adjusts Minimax algorithm parameters
    - Initiates the game when conditions are met

    Args:
        event (pygame.event): The event to handle.

    This function is crucial for setting up the game based on user preferences.
    It ensures that the correct number of agents are selected and parameters are set
    before starting the game.
    """
    global random_agent_selected, optimized_ql_agent_selected, minimax_agent_selected, ql_agent_selected, ql_hybrid_agent_selected, minimax_depth, minimax_time_limit, current_state, ql_agent, current_first_player
    
    if event.type == pygame.MOUSEBUTTONDOWN:
        x, y = event.pos
        
        # Calculate the number of currently selected agents
        selected_agents = sum([random_agent_selected, optimized_ql_agent_selected, ql_hybrid_agent_selected, ql_agent_selected, minimax_agent_selected])

        # Handle agent selection
        if 80 <= x <= 105:
            if 130 <= y <= 160:
                handle_agent_selection('random')
            elif 180 <= y <= 210:
                handle_agent_selection('optimized_ql')
            elif 230 <= y <= 260:
                handle_agent_selection('ql_hybrid')
            elif 280 <= y <= 310:
                handle_agent_selection('ql')
            elif 330 <= y <= 360:
                handle_agent_selection('minimax')
        
        # Adjust Minimax depth
        if 80 <= x <= 260 and 400 <= y <= 430:
            minimax_depth = max(1, min(10, int((x - 80) / 180 * 9) + 1))
            logging.info(f"Minimax depth set to {minimax_depth}")
        
        # Adjust Minimax time limit
        if 80 <= x <= 260 and 470 <= y <= 500:
            minimax_time_limit = max(1, min(60, int((x - 80) / 180 * 59) + 1))
            logging.info(f"Minimax time limit set to {minimax_time_limit}s")
        
        # Handle game start
        if WIDTH // 2 - 60 <= x <= WIDTH // 2 + 60 and 550 <= y <= 590:
            if sum([random_agent_selected, optimized_ql_agent_selected, ql_hybrid_agent_selected, ql_agent_selected, minimax_agent_selected]) == 2:
                logging.info(f"Starting game with Random: {random_agent_selected}, Minimax: {minimax_agent_selected}, "
                             f"Q-Learning(MiniMax): {ql_agent_selected}, Q-Learning(Random): {optimized_ql_agent_selected}, "
                             f"Q-Learning(Hybrid): {ql_hybrid_agent_selected}")
                current_state = GAME_SCREEN
                start_game()
                logging.info("Starting 50 games")
            else:
                show_message("Please select exactly two agents to start the game.")

def handle_agent_selection(agent_type):
    """
    Handle the selection or deselection of an agent.

    This function toggles the selection state of the specified agent
    and loads the appropriate model if necessary. It ensures that
    the correct number of agents are selected for the game.

    Args:
        agent_type (str): The type of agent being selected or deselected.

    This function is crucial for managing the agent selection process,
    including loading Q-learning models when necessary and enforcing
    the rule of selecting exactly two agents for the game.
    """
    global random_agent_selected, optimized_ql_agent_selected, ql_hybrid_agent_selected, ql_agent_selected, minimax_agent_selected, ql_agent
    
    selected_agents = sum([random_agent_selected, optimized_ql_agent_selected, ql_hybrid_agent_selected, ql_agent_selected, minimax_agent_selected])
    
    if selected_agents < 2 or agent_type in ['random', 'optimized_ql', 'ql_hybrid', 'ql', 'minimax']:
        if agent_type == 'random':
            random_agent_selected = not random_agent_selected
        elif agent_type == 'optimized_ql':
            optimized_ql_agent_selected = not optimized_ql_agent_selected
            ql_agent = load_ql_agent(agent_type) if optimized_ql_agent_selected else None
        elif agent_type == 'ql_hybrid':
            ql_hybrid_agent_selected = not ql_hybrid_agent_selected
            ql_agent = load_ql_agent(agent_type) if ql_hybrid_agent_selected else None
        elif agent_type == 'ql':
            ql_agent_selected = not ql_agent_selected
            ql_agent = load_ql_agent(agent_type) if ql_agent_selected else None
        elif agent_type == 'minimax':
            minimax_agent_selected = not minimax_agent_selected
    else:
        show_message("You can select a maximum of two agents.")

def start_game():
    """
    Initialize and start a new game session.

    This function sets up the game environment, selects agents based on user choices,
    and prepares for the game to begin. It's crucial for transitioning from the
    start screen to actual gameplay.

    Key steps:
    1. Create a new TicTacToe game instance
    2. Select and initialize agents based on user choices
    3. Reset game statistics
    4. Log initial game information

    This function is essential for setting up the game state and ensuring
    that all necessary components are properly initialized before gameplay begins.
    """
    global game, agent1, agent2, games_played, agent1_wins, agent2_wins, ties, ql_agent, current_first_player
    
    game = TicTacToe()
    
    def print_agent_info(agent, name):
        """Helper function to log detailed agent information"""
        logging.info(f"{name}: {get_agent_display_name(agent)}")
        if agent.agent_type == 'random':
            logging.info("  Random agent (no additional info)")
        elif agent.agent_type == 'minimax':
            logging.info(f"  Depth: {agent.depth}")
        elif agent.agent_type in ['optimized_ql', 'ql_hybrid']:
            logging.info(f"  Q-table1 size: {len(agent.q_table1)}")
            logging.info(f"  Q-table2 size: {len(agent.q_table2)}")
            logging.info(f"  Winning moves: {len(agent.winning_moves)}")
            logging.info(f"  Model loaded from: {agent.filename}")
        elif agent.agent_type == 'ql':
            logging.info(f"  Q-table size: {len(agent.q_table)}")
            logging.info(f"  Model loaded from: {agent.filename}")
        else:
            logging.info("  Unknown agent type")
    
    logging.info(f"Creating agents based on selection:")
    logging.info(f"Random: {random_agent_selected}, Minimax: {minimax_agent_selected}, "
                 f"Q-Learning(MiniMax): {ql_agent_selected}, Q-Learning(Random): {optimized_ql_agent_selected}, "
                 f"Q-Learning(Hybrid): {ql_hybrid_agent_selected}")

    agent1, agent2 = select_agents()

    logging.info("Agent details:")
    print_agent_info(agent1, "Agent 1 (X)")
    print_agent_info(agent2, "Agent 2 (O)")
    
    # Reset game statistics
    games_played = 0
    agent1_wins = 0
    agent2_wins = 0
    ties = 0
    current_first_player = 'X'
    
    logging.info(f"Preparing to start 50 games.")
    if isinstance(agent1, MinimaxAgent) or isinstance(agent2, MinimaxAgent):
        logging.info(f"Minimax depth: {minimax_depth}, Time limit: {minimax_time_limit}s")

def get_agent_display_name(agent):
    """
    Get a display name for an agent based on its type.

    This function returns a human-readable name for each agent type,
    which is used for logging and display purposes.

    Args:
        agent (object): The agent object.

    Returns:
        str: A string representing the agent's type.

    This function helps in providing clear and consistent naming for different
    agent types throughout the application.
    """
    if agent.agent_type == 'random':
        return "Random Agent"
    elif agent.agent_type == 'minimax':
        return f"Minimax Agent"
    elif agent.agent_type == 'optimized_ql':
        return "Q-Learning Agent(for random)"
    elif agent.agent_type == 'ql_hybrid':
        return "Q-Learning Agent(for random and minimax)"
    elif agent.agent_type == 'ql':
        return "Q-Learning Agent(for MiniMax)"
    else:
        return "Unknown Agent"

def print_final_results():
    """
    Print the final results of the game session.

    This function calculates and logs the overall performance statistics
    for both agents after all games have been played. It provides a
    comprehensive summary of the game outcomes.

    The function calculates:
    - Total games played
    - Win rates for both agents
    - Tie rate
    - Detailed results for each agent including wins, losses, and ties

    This information is crucial for analyzing the performance of different
    agent types and strategies over multiple games.
    """
    total_games = agent1_wins + agent2_wins + ties
    agent1_win_rate = agent1_wins / total_games if total_games > 0 else 0
    agent2_win_rate = agent2_wins / total_games if total_games > 0 else 0
    tie_rate = ties / total_games if total_games > 0 else 0
    
    logging.info("Final Results:")
    logging.info(f"Total games played: {total_games}")
    
    def print_agent_results(agent, wins, losses, name):
        agent_name = get_agent_display_name(agent)
        if isinstance(agent, MinimaxAgent):
            agent_name += f" (Depth: {agent.depth})"
        logging.info(f"{name} - {agent_name}:")
        logging.info(f"  Wins: {wins}, Losses: {losses}, Ties: {ties}")
        logging.info(f"  Win Rate: {wins/total_games:.2%}")
    
    print_agent_results(agent1, agent1_wins, agent2_wins, "Agent 1 (X)")
    print_agent_results(agent2, agent2_wins, agent1_wins, "Agent 2 (O)")
    logging.info(f"Tie Rate: {tie_rate:.2%}")

def play_game():
    """
    Execute a single game turn and manage the overall game flow.

    This function is the core of the game logic, handling:
    - Turn-based gameplay between two AI agents
    - Game state updates and winner checks
    - Logging of game progress and results
    - Management of multiple games in a session

    Returns:
        bool: True if the game should continue, False if the session is over.

    This function is crucial for the actual gameplay, managing the flow of the game,
    updating scores, and determining when the entire session (50 games) is complete.
    It also handles the alternation of the first player between games for fairness.
    """
    global game, agent1, agent2, games_played, agent1_wins, agent2_wins, ties, current_first_player
    
    # Check if the game has ended
    if game.check_winner():
        games_played += 1
        winner = game.check_winner()
        if winner == 'X':
            agent1_wins += 1
        elif winner == 'O':
            agent2_wins += 1
        else:
            ties += 1
        
        # Prepare for the next game or end the session
        if games_played < max_games:
            game.reset()
            current_first_player = 'O' if current_first_player == 'X' else 'X'
            game.current_player = current_first_player
            
            first_agent = "Agent 1 (X)" if current_first_player == 'X' else "Agent 2 (O)"
            logging.info(f"Game {games_played + 1} starts, {first_agent} goes first")
            
            if games_played % 10 == 0:
                logging.info(f"Completed {games_played}/{max_games} games. Current results - X: {agent1_wins}, O: {agent2_wins}, Ties: {ties}")
        else:
            logging.info(f"All 50 games completed. Final results - X: {agent1_wins}, O: {agent2_wins}, Ties: {ties}")
            return False
    
    # Execute the current player's move
    current_agent = agent1 if game.current_player == 'X' else agent2
    if isinstance(current_agent, (OptimizedQLearningAgent, QLearningAgentMinimax)):
        move = current_agent.choose_action(game)
    else:
        move = current_agent.make_move(game)
    
    game.make_move(move)
    
    return True

def draw_game_screen():
    """
    Draw the main game screen.
    
    This function renders the game board, current scores, and agent information.
    It's responsible for the visual representation of the game state and results.

    The function draws:
    - The game title
    - The Tic-Tac-Toe board with current moves
    - Game progress (current game number out of total games)
    - Agent names and their current scores

    This visual representation is crucial for observing the game progress
    and understanding the performance of different agents.
    """
    screen.fill(WHITE)
    
    # Title
    title = title_font.render("Tic-Tac-Toe", True, BLACK)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 20))
    
    # Draw the board
    board_size = 400
    board_top_left = ((WIDTH - board_size) // 2, 100)
    pygame.draw.rect(screen, WHITE, (*board_top_left, board_size, board_size))
    
    # Draw board lines
    for i in range(1, 3):
        pygame.draw.line(screen, BLACK, (board_top_left[0] + i * board_size // 3, board_top_left[1]),
                         (board_top_left[0] + i * board_size // 3, board_top_left[1] + board_size), 4)
        pygame.draw.line(screen, BLACK, (board_top_left[0], board_top_left[1] + i * board_size // 3),
                         (board_top_left[0] + board_size, board_top_left[1] + i * board_size // 3), 4)
    
    # Draw board border
    pygame.draw.rect(screen, BLACK, (*board_top_left, board_size, board_size), 4)
    
    # Draw X's and O's
    for i in range(9):
        x = i % 3
        y = i // 3
        cell_size = board_size // 3
        if game.board[i] == 'X':
            pygame.draw.line(screen, BLUE, 
                             (board_top_left[0] + x * cell_size + 20, board_top_left[1] + y * cell_size + 20),
                             (board_top_left[0] + (x + 1) * cell_size - 20, board_top_left[1] + (y + 1) * cell_size - 20), 4)
            pygame.draw.line(screen, BLUE, 
                             (board_top_left[0] + (x + 1) * cell_size - 20, board_top_left[1] + y * cell_size + 20),
                             (board_top_left[0] + x * cell_size + 20, board_top_left[1] + (y + 1) * cell_size - 20), 4)
        elif game.board[i] == 'O':
            pygame.draw.circle(screen, RED, 
                               (board_top_left[0] + x * cell_size + cell_size // 2, 
                                board_top_left[1] + y * cell_size + cell_size // 2),
                               cell_size // 2 - 20, 4)
    
    # Display game results
    result_text = font.render(f"Game {games_played}/50", True, BLACK)
    screen.blit(result_text, (20, HEIGHT - 120))
    
    def get_agent_name(agent):
        agent_type = agent.agent_type;
        
        if agent_type == 'random':
            return "Random Agent"
        elif agent_type == 'minimax':
            return f"Minimax_D{minimax_depth}_T{minimax_time_limit}"
        elif agent_type == 'ql':
            return "Q-Learning Agent(for MiniMax)"
        elif agent_type == 'optimized_ql':
             return "Q-Learning Agent(for random)"
        elif agent_type == 'ql_hybrid':
             return "Q-Learning Agent(for random and minimax)"

    agent1_name = get_agent_name(agent1)
    agent2_name = get_agent_name(agent2)

    # Use smaller font for detailed information
    very_small_font = pygame.font.Font(None, 18)

    # Display Agent 1 (X) information
    agent1_text = very_small_font.render(f"X: {agent1_name}", True, BLUE)
    screen.blit(agent1_text, (10, HEIGHT - 90))
    agent1_score = very_small_font.render(f"Win: {agent1_wins}, Lose: {agent2_wins}, Draw: {ties}", True, BLUE)
    screen.blit(agent1_score, (10, HEIGHT - 60))

    # Display Agent 2 (O) information
    agent2_text = very_small_font.render(f"O: {agent2_name}", True, RED)
    screen.blit(agent2_text, (WIDTH - agent2_text.get_width() - 10, HEIGHT - 90))
    agent2_score = very_small_font.render(f"Win: {agent2_wins}, Lose: {agent1_wins}, Draw: {ties}", True, RED)
    screen.blit(agent2_score, (WIDTH - agent2_score.get_width() - 10, HEIGHT - 60))

def evaluate_loaded_agent(agent, num_games=1000):
    game = TicTacToe()
    minimax_opponent = MinimaxAgent(depth=3)  # Use the same settings as in training
    wins = losses = ties = 0

    for _ in range(num_games):
        game.reset()
        while True:
            if game.current_player == 'O':  # Q-learning agent's turn
                move = agent.choose_action(game)
            else:  # Minimax agent's turn
                move = minimax_opponent.make_move(game)
            
            game.make_move(move)
            winner = game.check_winner()
            if winner:
                if winner == 'O':
                    wins += 1
                elif winner == 'X':
                    losses += 1
                else:
                    ties += 1
                break

    win_rate = wins / num_games
    loss_rate = losses / num_games
    tie_rate = ties / num_games
    return win_rate, loss_rate, tie_rate

def get_optimized_state(game):
    board = [0 if cell == ' ' else (1 if cell == 'X' else 2) for cell in game.board]
    transformations = [
        board,
        [board[2], board[1], board[0], board[5], board[4], board[3], board[8], board[7], board[6]],
        [board[6], board[7], board[8], board[3], board[4], board[5], board[0], board[1], board[2]],
        [board[8], board[7], board[6], board[5], board[4], board[3], board[2], board[1], board[0]],
        [board[0], board[3], board[6], board[1], board[4], board[7], board[2], board[5], board[8]],
        [board[8], board[5], board[2], board[7], board[4], board[1], board[6], board[3], board[0]],
    ]
    return tuple(min(transformations))

def show_message(message):
    """
    Display a message box on the screen.
    
    Args:
        message (str): The message to display.
    
    This function creates a temporary surface to show a message,
    handling text wrapping for longer messages.
    """
    message_box = pygame.Surface((320, 100))  # Slightly increase width
    message_box.fill(WHITE)
    pygame.draw.rect(message_box, BLACK, message_box.get_rect(), 2)
    
    # Use smaller font
    very_small_font = pygame.font.Font(None, 20)  # Further reduce font size
    text = very_small_font.render(message, True, BLACK)
    
    # Handle line wrapping if text is too long
    if text.get_width() > message_box.get_width() - 20:
        words = message.split()
        lines = []
        current_line = words[0]
        for word in words[1:]:
            if very_small_font.size(current_line + ' ' + word)[0] <= message_box.get_width() - 20:
                current_line += ' ' + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        
        y_offset = (message_box.get_height() - len(lines) * very_small_font.get_linesize()) // 2
        for i, line in enumerate(lines):
            text_surface = very_small_font.render(line, True, BLACK)
            message_box.blit(text_surface, (message_box.get_width() // 2 - text_surface.get_width() // 2, 
                                            y_offset + i * very_small_font.get_linesize()))
    else:
        message_box.blit(text, (message_box.get_width() // 2 - text.get_width() // 2, 
                                message_box.get_height() // 2 - text.get_height() // 2))
    
    screen.blit(message_box, (WIDTH // 2 - message_box.get_width() // 2, 
                              HEIGHT // 2 - message_box.get_height() // 2))
    pygame.display.flip()
    
    start_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start_time < 1000:  # Display for 1 second
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.time.wait(10)  # Small delay to avoid excessive CPU usage

def select_agent(agent_type):
    """
    Create and return an agent of the specified type.
    
    Args:
        agent_type (str): The type of agent to create.
    
    Returns:
        object: An instance of the specified agent type.
    """
    if agent_type == 'random':
        agent = RandomAgent();
        agent.agent_type = 'random';
        return agent
    elif agent_type == 'minimax':
        
        agent = MinimaxAgent(depth=minimax_depth);
        agent.agent_type = 'minimax';
        return agent
    elif agent_type in ['ql', 'optimized_ql', 'ql_hybrid']:
        return load_ql_agent(agent_type)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def get_agent_types():
    agent_types = []
    if random_agent_selected:
        agent_types.append('random')
    if minimax_agent_selected:
        agent_types.append('minimax')
    if ql_agent_selected:
        agent_types.append('ql')
    if optimized_ql_agent_selected:
        agent_types.append('optimized_ql')
    if ql_hybrid_agent_selected:
        agent_types.append('ql_hybrid')
    return agent_types

def select_agents():
    agent_types = get_agent_types()
    if len(agent_types) < 1:
        logging.error("No valid agent combination selected")
        return None, None

    agent1_type = agent_types[0]
    agent1 = select_agent(agent1_type)

    if len(agent_types) > 1:
        agent2_type = agent_types[1]
    else:
        agent2_type = agent_types[0]
    agent2 = select_agent(agent2_type)

    return agent1, agent2

def main():
    """
    Main game loop and program entry point.

    This function orchestrates the overall flow of the application, including:
    - Handling the transition between start screen and game screen
    - Managing the game state and user interactions
    - Controlling the game's frame rate

    It demonstrates the structure of a Pygame-based application and how
    different components (UI, game logic, AI) are integrated.

    The main loop continuously checks for events, updates the game state,
    and redraws the screen, providing a smooth and responsive user experience.
    """
    global current_state, game
    
    clock = pygame.time.Clock()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if current_state == START_SCREEN:
                handle_start_screen_events(event)
        
        if current_state == START_SCREEN:
            draw_start_screen()
        elif current_state == GAME_SCREEN:
            if games_played < max_games:
                game_continues = play_game()
                if not game_continues:
                    print_final_results()
                    current_state = START_SCREEN
            draw_game_screen()
        
        pygame.display.flip()
        clock.tick(10)  # Control frame rate to 10 FPS for better observation of game progress

if __name__ == "__main__":
    main()
