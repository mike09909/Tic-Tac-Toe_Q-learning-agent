def minimax(game, depth, is_maximizing):
    """
    Implement the minimax algorithm for the Tic-Tac-Toe game.
    
    Args:
        game (TicTacToe): The current game state.
        depth (int): The current depth in the game tree.
        is_maximizing (bool): True if it's the maximizing player's turn, False otherwise.
    
    Returns:
        int: The best score for the current game state.
    
    This function recursively evaluates all possible moves up to a certain depth,
    alternating between maximizing and minimizing players.
    """
    winner = game.check_winner()
    if winner == 'X':
        return -1  # X wins, so it's a loss for O (minimizing score)
    elif winner == 'O':
        return 1   # O wins, so it's a win for O (maximizing score)
    elif winner == 'Tie':
        return 0   # Tie game
    
    if depth == 0:
        return 0   # Reached maximum depth, return neutral score

    if is_maximizing:
        best_score = -float('inf')
        for move in game.get_available_moves():
            new_game = game.copy()
            new_game.make_move(move)
            score = minimax(new_game, depth - 1, False)
            best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for move in game.get_available_moves():
            new_game = game.copy()
            new_game.make_move(move)
            score = minimax(new_game, depth - 1, True)
            best_score = min(score, best_score)
        return best_score

def get_best_move(game, depth):
    """
    Determine the best move for the current player using the minimax algorithm.
    
    Args:
        game (TicTacToe): The current game state.
        depth (int): The maximum depth to search in the game tree.
    
    Returns:
        int: The index of the best move (0-8).
    
    This function evaluates all possible moves using the minimax algorithm
    and returns the move with the highest score for the current player.
    """
    best_score = -float('inf')
    best_move = None
    for move in game.get_available_moves():
        new_game = game.copy()
        new_game.make_move(move)
        score = minimax(new_game, depth - 1, False)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move

class MinimaxAgent:
    """
    An AI agent that uses the minimax algorithm to make decisions in Tic-Tac-Toe.
    """

    def __init__(self, depth):
        """
        Initialize the MinimaxAgent.
        
        Args:
            depth (int): The maximum depth for the minimax algorithm to search.
        """
        self.depth = depth
    
    def make_move(self, game):
        """
        Determine the best move for the current game state.
        
        Args:
            game (TicTacToe): The current game state.
        
        Returns:
            int: The index of the chosen move (0-8).
        
        This method uses the get_best_move function to determine the optimal move
        based on the minimax algorithm with the specified search depth.
        """
        return get_best_move(game, self.depth)
