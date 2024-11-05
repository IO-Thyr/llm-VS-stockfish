# %%
from stockfish import Stockfish
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
import re
import chess
import prompter
from import prompt
from typing import List

class ChessMove(BaseModel):
    move: str = Field(description="Chess move in long algebraic notation")
    explanation: str = Field(description="Brief explanation of the move")
    

class ChessOutput(BaseModel):
    position_analysis: str = Field(description="Analysis of the current position, including strengths and weaknesses for both sides.")
    candidate_moves: List[ChessMove] = Field(description="List of 3-5 candidate moves with brief explanations.")
    move_evaluation: str = Field(description="Detailed evaluation of the chosen move, including potential variations and counter-moves.")
    final_response: str = Field(description="Final move response in long algebraic notation. Example output: e7e5, g8f6, e8g8")
    position_evaluation: float = Field(description="Numerical evaluation of the position after your move. Positive for black advantage, negative for white advantage.")

    @field_validator('position_analysis', 'move_evaluation')
    @classmethod
    def validate_explanation_length(cls, v: str) -> str:
        if len(v.split()) < 10:
            raise ValueError("Explanation too short. Provide at least 10 words.")
        return v

    #@field_validator('final_response')
    #@classmethod
    #def validate_final_move(cls, v: str) -> str:
    #    pattern = r'^[a-h][1-8][a-h][1-8][qrbnQRBN]?$'
    #    if not re.match(pattern, v):
    #        raise ValueError("Invalid move format. Use long algebraic notation (e.g., e2e4, g8f6, e1g1)")
    #    return v.lower()

class ChessVSLLM:
    def __init__(self, stockfish_path: str):
        self.stockfish_path = stockfish_path
        self.stockfish = None
        self.llm = None
        self.model_name = None
        self.color_turn = None
        self.game_over = False
        self.move_history = []
        self.board = chess.Board()

    def init_stockfish(self,
                       set_elo_rating: int = 1000,
                       set_depth: int = 15,):
        self.stockfish = Stockfish(path=self.stockfish_path)
        self.stockfish.set_elo_rating(set_elo_rating)
        self.stockfish.set_depth(set_depth)
        print("> Stockfish is well initialized and have ELO of ", set_elo_rating)
         
    def init_llm_model(self,
                       model_name: str,
                       temperature: int
                       ):
        self.model_name = model_name
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=temperature,            
        )
        print("> New LLM model **{}** enter in competion".format(self.model_name))

    def get_params(self):
        return self.stockfish.get_parameters()
    
    def reset_game(self):
        self.stockfish.set_position()
        print("--- GAME IS RESET ---")
    
    def get_board_visual(self, Boolean=True):
        return self.stockfish.get_board_visual(Boolean)
    
    def get_fen_position(self):
         return self.stockfish.get_fen_position()

    def get_stockfish_best_move(self):
         self.color_turn = "white"
         return self.stockfish.get_best_move()
    
    def get_llm_best_move(self, last_movement="None"):
        if self.llm is None:
            raise ValueError("LLM model is not initialized. Call init_llm_model() first.")
        self.color_turn = "black"
        parser = PydanticOutputParser(pydantic_object=ChessOutput)
        visualisation = self.get_board_visual(Boolean=False)
        fen = self.get_fen_position()

        prompt = ChatPromptTemplate([
            ("system", 
             """You are a world-class chess player with a 3000 Elo rating. Your task is to analyze the current board position and generate a highly strategic chess move using a step-by-step thought process.

                Chess Rules and Notation:
                1. You MUST use long algebraic notation WITHOUT hyphens or spaces for all moves.
                2. Format: [start square][end square][optional promotion piece]
                3. Examples: e2e4, g7g5, e1g1 (for castling), e7e8q (for promotion)
                4. Lowercase letters only.
                5. For pawn promotion, append the promotion piece (q, r, b, or n) to the move.
                6. Always provide all four characters for the move, even for captures or special moves.
                7. Only output legal moves based on the current board state.
                8. Respect all standard chess rules including castling, en passant, and pawn promotion.

                Strategic Considerations:
                - Control of the center
                - Piece development and activity
                - King safety
                - Pawn structure
                - Tactical opportunities (forks, pins, skewers)
                - Long-term positional advantages
                - Adapt strategy based on the game phase (opening, middlegame, endgame)
                - Be mindful of opponent's potential threats and counter-moves
                - Consider the overall game strategy and how this move fits into it

                Your analysis should reflect deep positional understanding and precise calculation.
                Ensure all explanations are at least 10 words long to provide sufficient detail."""),

            ("human", 
             """You are playing as Black against Stockfish, the world's strongest chess engine. 
                
                The current board state is:
                {viz}
                
                The FEN visualisation is : {FEN}

                Move history: {move_history}

                Last move played: {last_mov}

                Analyze the position carefully using the following Chain of Thought process:

                1. Position Analysis (minimum 10 words):
                   Provide a detailed analysis of the current position. Discuss the strengths and weaknesses of both sides, key features of the position, and any immediate threats or opportunities.

                2. Candidate Moves:
                   Generate a list of 3-5 candidate moves. For each move, provide a brief explanation of its purpose and potential consequences. Use long algebraic notation for these moves.

                3. Move Evaluation (minimum 10 words):
                   Choose the best move from your candidates. Explain your reasoning in detail, including:
                   - Why this move is superior to the other candidates
                   - Potential variations and counter-moves you've calculated
                   - How this move addresses the strategic considerations of the position
                   - Any potential risks or drawbacks of the move

                4. Final Move Selection:
                   State your chosen move in long algebraic notation. This MUST be in the format [start square][end square], e.g., e7e5, g8f6, e8g8. Do not use short algebraic notation like Nf6 or exd5.

                5. Position Evaluation:
                   Provide a numerical evaluation of the position after your move. Use a float where positive numbers indicate black advantage and negative numbers indicate white advantage.

                {format_instructions}

                Remember:
                - Explain your thought process thoroughly at each step.
                - Ensure all explanations are at least 10 words long.
                - Use long algebraic notation for ALL moves.
                - Consider multiple possibilities and their implications.
                - Calculate variations at least 7-8 moves deep before deciding.
                - Ensure your final move is legal and properly formatted in long algebraic notation.
                - Do not repeat the last move played.
            """
            ),
        ],
        input_variables=["viz", "FEN", "move_history", "last_mov"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        )


        chain = prompt | self.llm | parser
        result = chain.invoke({
            "viz": visualisation,
            "FEN": fen,
            "move_history": ", ".join(self.move_history),
            "last_mov": last_movement
        })
        return result

    def make_a_move(self, movement):
        try:
            move = chess.Move.from_uci(movement)
            if move not in self.board.legal_moves:
                raise chess.IllegalMoveError
        except (chess.IllegalMoveError, ValueError):
            raise ValueError(f"Invalid or illegal move: {movement}")

        self.board.push(move)
        self.stockfish.make_moves_from_current_position([move.uci()])
        self.move_history.append(move.uci())
        return True      
# %%


def em():
    chessgame = ChessVSLLM("/Users/tancredeh/Desktop/DSProject/AI-Chess/llmVSstockfish/llm-vs-stockfish/stockfish/stockfish-macos-m1-apple-silicon")
    chessgame.init_llm_model(model_name="phi3:3.8b", temperature=0.7)
    chessgame.init_stockfish()

    result_stoc = chessgame.get_stockfish_best_move()
    print("result stock : ", result_stoc)
    chessgame.make_a_move(result_stoc)

    print(chessgame.get_board_visual())

    result_llm = chessgame.get_llm_best_move()
    print("> llm :", result_llm)
    chessgame.make_a_move(result_llm.final_response)

    print(chessgame.get_board_visual())

    v = result_llm.final_response
    result_llm = chessgame.get_llm_best_move(last_movement=str(v))
    print("> llm :", result_llm)
    chessgame.make_a_move(result_llm.final_response)

    print(chessgame.get_board_visual())
# %%
