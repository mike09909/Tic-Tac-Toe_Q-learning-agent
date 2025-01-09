import numpy as np

class TicTacToe:
    def __init__(self):
        """
        Initialize the Tic-Tac-Toe game.
        Creates an empty 3x3 board represented as a list of 9 spaces.
        Sets the current player to 'X' to start the game.
        """
        self.board = [' ' for _ in range(9)]
        self.current_player = 'X'
    
    def make_move(self, position):
        """
        Attempt to make a move on the board.
        
        Args:
            position (int): The index (0-8) where the current player wants to place their mark.
        
        Returns:
            bool: True if the move was successful, False if the position was already occupied.
        
        Side effects:
            If the move is valid, updates the board and switches the current player.
        """
        if self.board[position] == ' ':
            self.board[position] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False
    
    def undo_move(self, position):
        """
        Undo a move on the board.
        
        Args:
            position (int): The index (0-8) of the move to undo.
        
        Side effects:
            Removes the mark from the specified position and switches the current player back.
            This method is useful for AI algorithms that need to explore different game states.
        """
        self.board[position] = ' '
        self.current_player = 'O' if self.current_player == 'X' else 'X'
    
    def check_winner(self):
        """
        Check if there's a winner or if the game is a tie.
        
        Returns:
            str or None: 
                - 'X' if X wins
                - 'O' if O wins
                - 'Tie' if the game is a draw
                - None if the game is still ongoing
        
        This method checks all possible winning combinations and the tie condition.
        """
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Horizontal rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Vertical columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]
        
        for combo in winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != ' ':
                return self.board[combo[0]]
        
        if ' ' not in self.board:
            return 'Tie'  # or return ' ' if you prefer
        
        return None
    
    def get_available_moves(self):
        """
        Get a list of all available (empty) positions on the board.
        
        Returns:
            list: Indices of empty spots on the board.
        
        This method is particularly useful for AI agents to determine possible moves.
        """
        return [i for i, spot in enumerate(self.board) if spot == ' ']
    
    def reset(self):
        """
        Reset the game to its initial state.
        
        Side effects:
            Clears the board and sets the current player back to 'X'.
        
        This method is useful for starting a new game without creating a new instance.
        """
        self.board = [' ' for _ in range(9)]
        self.current_player = 'X'

    def copy(self):
        """
        Create a deep copy of the current game state.
        
        Returns:
            TicTacToe: A new TicTacToe instance with the same board and current player.
        
        This method is crucial for AI algorithms that need to simulate future game states
        without modifying the current game state.
        """
        new_game = TicTacToe()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        return new_game

    def get_state(self):
        """
        Get a string representation of the current board state.
        
        Returns:
            str: A string where each character represents a cell on the board.
        
        This method is useful for creating unique keys for game states, which can be
        used in reinforcement learning algorithms or for caching purposes.
        """
        return ''.join(self.board)

    def __str__(self):
        """
        Create a string representation of the board for display purposes.
        
        Returns:
            str: A formatted string showing the current state of the board.
        
        This method allows for easy printing of the board state, which is useful
        for debugging or creating a text-based interface for the game.
        """
        return '\n'.join([' '.join(self.board[i:i+3]) for i in range(0, 9, 3)])
