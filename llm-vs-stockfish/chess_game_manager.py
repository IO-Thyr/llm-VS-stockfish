# %%
import chess
import chess.svg
from typing import List, Tuple
from llm_player import LLMPlayer
from stockfish_player import StockfishPlayer
from IPython.display import display, clear_output
import pandas as pd

class ChessGameManager:
    def __init__(self, llm_player: LLMPlayer, stockfish_player: StockfishPlayer):
        self.board = chess.Board()
        self.llm_player = llm_player
        self.stockfish_player = stockfish_player
        self.move_history: List[str] = []
        self.board_svg = None
        self.jupyter_board = True
        self.turn = 1
        self.record_fen:List[str] = [self.get_current_fen()]
        self.color_winner = None

    def reset_game(self):
        self.board.reset()
        self.move_history.clear()
        self.stockfish_player.reset_game()
        print("Game has been reset.")

    def make_move(self, move: str) -> bool:
        try:
            self.record_fen.append(self.get_current_fen())
            chess_move = self.board.parse_san(move)
            if chess_move in self.board.legal_moves:
                self.board.push(chess_move)
                print(chess_move)
                self.move_history.append(str(self.turn) + ". " + move)
                self.turn += 1
                return True
            else:
                print(f"Illegal move: {move}")
                return False
        except ValueError:
            print(f"Invalid move format: {move}")
            return False

    def get_current_fen(self) -> str:
        return self.board.fen()

    def get_board_visual(self) -> str:
        return str(self.board)

    def get_legal_move_list(self):
        return [i.uci() for i in self.board.legal_moves]

    def is_game_over(self) -> Tuple[bool, str]:
        if self.board.is_game_over():
            result = self.board.result()
            df_histo = pd.read_csv("ChessGameExperiment.csv")
            if result == "1-0":
                self.color_winner = "White"
                return True, "White wins"
            elif result == "0-1":
                self.color_winner = "Black"
                return True, "Black wins"
            else:
                return True, "Draw"
        return False, ""

    def play_llm_turn(self):
        legal_move_list = self.get_legal_move_list()
        print("> List of legal movement", legal_move_list)
        print("> List of past movement", self.move_history)

        self.llm_player.current_fen_board = self.get_current_fen()
        self.llm_player.current_chess_board = self.get_board_visual()
        self.llm_player.movement_history = " ".join(self.move_history) if self.move_history else "[]"
        self.llm_player.legal_moves = " ".join(legal_move_list)

        llm_output = self.llm_player.get_llm_best_move()
        llm_move = llm_output

        if self.make_move(llm_move):
            print(f"LLM move: {llm_move}")

            if self.jupyter_board == True:
                self._get_board_visualization_jupyter()
            else: 
                print(self.board)
            return True
        return False

    def play_stockfish_turn(self):
        current_fen = self.get_current_fen()
        self.stockfish_player.set_position_with_fen(current_fen)
        stockfish_move = self.stockfish_player.get_stockfish_best_move()

        if self.make_move(stockfish_move):
            print(f"Stockfish move: {stockfish_move}")
            
            if self.jupyter_board == True:
                self._get_board_visualization_jupyter()
            else: 
                print(self.board)
            
            return True
        return False

    def play_game(self, num_moves: int = 2):
        for _ in range(num_moves):

            if not self.play_stockfish_turn():
                print("Stockfish made an illegal move. Game over.")
                break

            game_over, result = self.is_game_over()

            if game_over:
                print(f"Game over: {result}")
                break


            if not self.play_llm_turn():
                print("LLM made an illegal move. Game over.")
                break

            game_over, result = self.is_game_over()
            if game_over:
                print(f"Game over: {result}")
                break

        print("Final board position:")
        print(self.get_board_visual())
        print(f"Move history: {', '.join(self.move_history)}")

    def _get_board_visualization_jupyter(self):
        self.board_svg = chess.svg.board(self.board, size=350)
        clear_output(wait=True)
        display(self.board_svg)

    def show_replay(self, liste_of_fen_move):
        for i in liste_of_fen_move:
            board_vierge = chess.Board(i)
            display(board_vierge)


# %%


def test():
    llm_player = LLMPlayer()
    llm_player.init_llm_model("phi3:3.8b", temperature=0.9)

    stockfish_player = StockfishPlayer("/Users/tancredeh/Desktop/DSProject/AI-Chess/llmVSstockfish/llm-vs-stockfish/stockfish/stockfish-macos-m1-apple-silicon")
    stockfish_player.init_stockfish(set_elo_rating=1000, set_depth=15)

    game_manager = ChessGameManager(llm_player, stockfish_player)
    game_manager.play_game(num_moves=50)