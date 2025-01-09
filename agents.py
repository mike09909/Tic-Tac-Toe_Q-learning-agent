import random
import time
import logging
from minimax_agent import MinimaxAgent

class RandomAgent:
    def make_move(self, game):
        # Choose a random move from the available moves
        return random.choice(game.get_available_moves())
