from langchain_ollama import ChatOllama
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import re
from typing import List, Optional
import random
import time
import os

class ChessMove(BaseModel):
    move: str = Field(description="Coup d'échec au format case de départ case de fin.")
    explanation: str = Field(description="Explication complète du mouvement, y compris le raisonnement stratégique, les considérations tactiques et les résultats potentiels.")

class ChessOutput(BaseModel):
    #position_analysis: str = Field(description="Detailed analysis of the current position, including a calculation of the sum of your pieces' values. This number should stay high, reflecting material advantage.")
    candidate_moves: List[ChessMove] = Field(description="List of 5 legal candidate moves to play this turn, chosen from the list. Each move should be presented in the format start square end square (e.g., e2e4) without using dashes, along with a detailed explanation.")
    #move_evaluation: str = Field(description="Detailed evaluation of the chosen move, including its strategic impact and any anticipated responses from the opponent.")
    best_move: List[ChessMove] = Field(description="Coup à jouer au format uci")

class LLMPlayer:
    def __init__(self, color: str = "Black"):
        self.llm: Optional[ChatOllama] = None
        self.model_name: Optional[str] = None
        self.current_chess_board: Optional[str] = None
        self.current_fen_board: Optional[str] = None
        self.movement_history: List[str] = []
        self.legal_moves: List[str] = []
        self.color: str = color
        self.move_number: int = 0
        self.history = []
        self.prompt_schema = self._create_chess_prompt()
        self.parser = None
        self.temp = None
       

    def init_llm_model(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temp = temperature
        self.llm = ChatOllama(model=self.model_name, temperature=temperature, format='json')
        #os.environ["MISTRAL_API_KEY"]="9m0UbfklcS177Zu642F5WVFONA0XeRN3"
        #self.llm = ChatMistralAI(model= self.model_name, temperature=self.temp).with_structured_output(method="json_mode", include_raw=True)
        print(f"> New LLM model **{self.model_name}** enters competition as {self.color}")

    def get_llm_best_move(self) -> ChessOutput:
        if self.llm is None:
            raise ValueError("LLM model is not initialized. Call init_llm_model() first.")

        parser = PydanticOutputParser(pydantic_object=ChessOutput)
        self.parser = parser
        prompt = self._create_chess_prompt()
        prompt = prompt.format(**self._get_prompt_variables())
        #print(prompt)

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                print(self.movement_history)
                result = self.llm.invoke(prompt)
                print(result)
                result = result.content
                print(">>>> Result\n",result)
                result = parser.parse(result)
                print(">>>> PARSER\n", result)
                self._print_move_analysis(result)
                self.history.append(result)
                
                if result.best_move[0].move.replace("-", "") not in self.legal_moves:
                    raise ValueError(f"Selected move {result.best_move[0].move} is not in the legal moves list.")

                #self.movement_history.append(result.final_move)
                return result.best_move[0].move.replace("-", "") 

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_attempts - 1:
                    print("Max attempts reached. Selecting a random legal move.")
                    time.sleep(5)
                    self.history.append("random")
                    return self._select_random_move()

    def _create_chess_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [("system",
             """Tu es un grand maitre d'echec et j'ai besoin d'un conseil pour mon coup à jouer, je joue les pieces noires.
             """+ """Voici une liste des coups que j'envisage de jouer [{legal_moves}]"""),
             ("human",
             self._get_human_message() + self._get_board_representation()
            )
            ])
    #self._get_system_message() +

    def _get_system_message(self) -> str:
        #<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        return """
        
        You are a world-class grandmaster chess player with deep strategic knowledge. Your task is to analyze the given chess position and select the best possible move from the provided list of legal moves. The output should be a valid JSON object with no additional text. Each field must strictly follow the format provided.

        We are playing a crucial chess game, and it is essential to make the most strategically sound move. You should consider both immediate tactical advantages and long-term strategic goals.

        **Strategic Principles**:
        - **Material Balance**: Prioritize keeping more valuable pieces on the board. Avoid losing pieces unless the trade offers a significant tactical or strategic gain.
        - **Control of the Center**: Focus on controlling the central squares (d4, d5, e4, e5) as this provides greater mobility and strategic advantages.
        - **King Safety**: Always consider the safety of your king. Avoid moves that expose your king to checks or attacks.
        - **Piece Activity**: Aim to improve the activity of your pieces. Place your pieces on squares where they control the most territory and have the most influence.
        - **Threats and Counter-Threats**: Always consider both your potential threats and your opponent’s. Look for moves that create double attacks or force the opponent to respond defensively.

        **Your Task**:
        Analyze the position and evaluate each move from the provided list of legal moves. The list of legal moves is as follows: [{legal_moves}]. Choose the best move from this list. Your decision should be based on the following criteria:
        1. **Maximizing Material Gain**: Choose moves that capture opponent pieces without losing equal or greater material.
        2. **Positional Advantage**: Select moves that improve your control of the board, especially the center.
        3. **Avoiding Blunders**: Eliminate moves that result in unnecessary loss of material or expose your king to danger.

        Provide a detailed evaluation of each move and explain why your chosen move is the best option given the current position. Your output should include:
        1. Evaluate potential candidate moves from the provided list, each with a detailed explanation.
        2. An evaluation of your final chosen move, including its strategic impact and any anticipated responses from the opponent.
        3. Choose ONE best_move, in the format start square end square.
                """
    def _get_board_representation(self):
        return """
        Pour te repèrer voici à les coordonnée d'un jeu d'échec.
          a b c d e f g h\n
        8 . . . . . . . .\n
        7 . . . . . . . .\n
        6 . . . . . . . .\n
        5 . . . . . . . .\n
        4 . . . . . . . .\n
        3 . . . . . . . .\n
        2 . . . . . . . .\n
        1 . . . . . . . .\n
        """ + """
        Le plateau de jeu ressemble actuellement à ça:
        \n{viz}\n
        Pour context, les lettres minuscule [k, q, b, n, r, p] représentent les noirs ; les lettres majuscules [K, Q, B, N, R, P] représentent les blancs. 
        """
    #Pour le contexte, la lettre k q r b n p représentent les pièces du joueur noir, alors que K Q R B N P représentent les pièces du joueur blanc.
        
    def _get_human_message(self) -> str:
        #<|start_header_id|>user<|end_header_id|>
        # 
        # Sur la base de la séquence de mouvements ci-dessus, quel coup de la liste [{legal_moves}] est la meilleure option à jouer ?

        return """
        \n{format_instructions}\n
        [Event "Chess Tournament"]
        [Site "Paris, FR"]
        [Date "2024.12.08"]
        [Round "5"]
        [White "Magnus"]
        [Black "LLM"]
        [Result "1-0"]
        [WhiteElo "2885"]
        [BlackElo "2812"]

        {movement_history}
        """
## INPUT
#Based on the current sequence of mouvement above, which move from this list [{legal_moves}] is the best option to play? **The objectif is to win the game by taking white king (K)**
#The current board visualasation is \n{viz}\n.
        #For context the letter k q r b n p represent the pieces of the black player, when K Q R B N P represent the pieces of the white player.
        
#The board is \n{viz}\n. Based on the current position and the strategic principles provided, which move from this list [{legal_moves}] is the best option to play? **Focus on maintaining material balance, controlling the board, and ensuring king safety. Evaluate each move from the list carefully and choose the best one.**

    # The FEN visualisation is {FEN}
    #        Chess board is {viz}.  Move selection {legal_moves}
##Question: What is the best next move to play for {color} ?
    # As {color} player, choose one move from this list {legal_moves}, based on the sequence history to increase the probability of winning the game by taking the white King. Prioritize preserving material, and avoid losing pieces unless you have a strong strategic reason to do so.
    # To help your decision this is a list of all available movement {legal_moves}.

    def _get_prompt_variables(self) -> dict:
        return {
            "FEN": self.current_fen_board,
            "legal_moves": self.legal_moves,
            "viz": self.current_chess_board,
            "movement_history": self.movement_history,
            "format_instructions": self.parser.get_format_instructions(),
            "color": self.color
        }

    def _print_move_analysis(self, result: ChessOutput) -> None:
        print(f"Move {self.move_number} analysis:")
        #print(f"Position analysis: {result.position_analysis}")
        #print("Candidate moves:")
        #print(result.candidate_moves)
        #for move in result.candidate_moves:
        #    print(f"- {move.move}: {move.explanation}")
        #print(f"Move evaluation: {result.move_evaluation}")
        print(f"Final move: {result.best_move}")

    def _select_random_move(self) -> ChessOutput:
        random_move = random.choice(self.legal_moves)
        self.movement_history = random_move
        return random_move
    
    """ChessOutput(
            position_analysis="Random move selected due to error.",
            candidate_moves=[],
            move_evaluation="No evaluation available for random move.",
            final_move=random_move,
        )
    """


    # Your task is to analyze the current board position and generate a highly strategic chess move.
    """ Key Concepts to Consider:
                        - Control of the center
                        - Piece activity (mobility)
                        - Pawn structure (strength and weaknesses)
                        - King safety
                        - Material balance (avoid unnecessary sacrifices unless strategically justified)"""

    """Ensure all explanations are at least 10 words long to provide sufficient detail.
                
                Some rules:
                - Explain your thought process thoroughly at each step.
                - Ensure all explanations are at least 10 words long.
                - Use long algebraic notation for ALL moves.
                - Consider multiple possibilities and their implications.
                - Calculate variations at least 7-8 moves deep before deciding.
                - Ensure your final move is legal and properly formatted in long algebraic notation."""
    

    """                How The Chess Pieces Move.
                King (K) Moves one square in any direction.
                Queen (Q) Moves any number of squares diagonally, horizontally, or vertically.
                Rook (R) Moves any number of squares horizontally or vertically.
                Bishop (B) Moves any number of squares diagonally.
                Knight (N) Moves in an ‘L-shape,’ two squares in a straight direction, and then one square perpendicular to that.
                Pawn (P) Moves one square forward, but on its first move, it can move two squares forward. It captures diagonally one square forward.
    """


    """1. Position Analysis (minimum 10 words):
            Provide a detailed analysis of the current position. Discuss the strengths and weaknesses of both sides, key features of the position, 
            and any immediate threats or opportunities.
            Chess Piece Values (lower is less important):
                - Pawn = 1
                - Knight = 3
                - Bishop = 3
                - Rook = 5
                - Queen = 9
                - King = invaluable (game ends if captured)
        2. Candidate Moves:
            Generate a list of 3 to 5 candidate moves in the format 'start square end square' without any dashes (e.g., 'd7d5'). 
            For each move, provide a brief explanation of its purpose and potential consequences.
        3. Move Evaluation (minimum 20 words):
            Choose the best move from your candidates. Explain your reasoning in detail, including:
            - Why this move is superior to the other candidates
            - Potential variations and counter-moves you've calculated
            - How this move addresses the strategic considerations of the position
            - Any potential risks or drawbacks of the move

        4. Final Move Selection:
            State your chosen move in the format 'start square end square' without using dashes."""