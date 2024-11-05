# %%
from stockfish import Stockfish
from pydantic import BaseModel, Field
import re
from typing import List



class StockfishPlayer:
    def __init__(self, stockfish_path: str):
        self.stockfish_path = stockfish_path
        self.stockfish = None
        self.elo = None
  
    def init_stockfish(self,
                       set_elo_rating: int = 1000,
                       set_depth: int = 15,):
        self.stockfish = Stockfish(path=self.stockfish_path)
        self.elo = set_elo_rating
        self.stockfish.set_elo_rating(set_elo_rating)
        self.stockfish.set_depth(set_depth)
        print("> Stockfish is well initialized and have ELO of ", set_elo_rating)
    
    def set_position_with_fen(self, fen_position: str):
        self.stockfish.set_fen_position(fen_position)

    def get_stockfish_best_move(self):
         self.color_turn = "white"
         return self.stockfish.get_best_move()
    
    def get_params(self):
        return self.stockfish.get_parameters()
    
    def get_board_visual(self, Boolean=True):
        return self.stockfish.get_board_visual(Boolean)
    
    def get_fen_position(self):
         return self.stockfish.get_fen_position()
    
        
    def reset_game(self):
        self.stockfish.set_position()
        print("--- GAME IS RESET ---")