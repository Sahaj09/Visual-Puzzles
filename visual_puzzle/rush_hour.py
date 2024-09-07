import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from typing import Optional


class RushHourEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        board_description: Optional[str] = None,
        obs_type: str = "rgb",
        rush_txt_path: str = "experiment_data/rush.txt",
    ):
        """
        Initialize the Rush Hour environment.

        This environment simulates the Rush Hour puzzle game, where the goal is to
        move vehicles to allow the main car (usually represented by 'A') to exit.

        Parameters:
        -----------
        board_description : str, Optional
            A string representation of the initial board state. The string should
            contain 36 characters representing a 6x6 grid. Use 'o' for empty spaces,
            'x' for walls, 'A' for the main car, and other letters for other vehicles.
            Eg.- ooIBBBGoIJCCGAAJKLoHDDKLxHFFKMoooooM
            If None, a random board will be loaded from rush.txt (which should be located at rush_txt_path) file.

        obs_type : str
            The type of observation to return. Must be either 'rgb' or 'text'.
            Default is 'rgb'.

        rush_txt_path : str
            The path to the rush.txt file containing the board descriptions.

        Attributes:
        -----------
        board : numpy.ndarray
            A 6x6 numpy array representing the current board state.
        pieces : list
            A sorted list of unique vehicle identifiers on the board.
        piece_orientations : dict
            A dictionary mapping each piece to its orientation ('h' for horizontal, 'v' for vertical).
        action_space : gym.spaces.MultiDiscrete
            The action space, representing which piece to move and in which direction.
        observation_space : gym.spaces.Box
            The observation space, representing the 6x6 game board.
        colors : dict
            A dictionary mapping piece identifiers to RGB color tuples.

        Raises:
        -------
        AssertionError
            If the obs_type is not 'rgb' or 'text'.

        Notes:
        ------
        - The action space is defined as follows:
          - First value: index of the piece to move (in self.pieces)
          - Second value: direction to move (0: up, 1: right, 2: down, 3: left)
        - Colors are predefined for empty spaces ('o'), walls ('x'), and the main car ('A').
          Other pieces are assigned random RGB colors.
        """
        super(RushHourEnv, self).__init__()

        assert obs_type in ["rgb", "text"], "Observation type must be 'rgb' or 'text'"
        if board_description is not None:
            self.board_description = board_description
            self.num_steps_to_finish = None
        else:
            self.num_steps_to_finish, self.board_description = self.load_board_randomly(
                rush_txt_path
            )

        self.board = np.array(list(self.board_description)).reshape(6, 6)
        self.pieces = set(self.board.flatten()) - set("ox")
        self.pieces = sorted(list(self.pieces))
        self.piece_orientations = self._get_piece_orientations()
        self.obs_type = obs_type
        self.cell_size = 50
        # print(self.pieces)

        # piece description, direction: 0 - up, 1 - right, 2 - down, 3 - left
        self.action_space = spaces.MultiDiscrete(np.array([len(self.pieces), 4]))

        # Define observation space
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(6 * self.cell_size, 6 * self.cell_size, 3),
            dtype=np.uint8,
        )

        # Define colors for pieces
        self.colors = {
            "o": (255, 255, 255),  # White for empty spaces
            "x": (0, 0, 0),  # Black for walls
            "A": (255, 0, 0),  # Red for the main car
        }
        # Generate random colors for other pieces
        for piece in self.pieces:
            if piece not in self.colors:
                self.colors[piece] = tuple(np.random.randint(0, 256, 3))

    # Load a board from rush.txt file were each sentence is shortest_path, board, id
    def load_board_randomly(self, file_path: str):
        with open("experiment_data/rush.txt", "r") as f:
            lines = f.readlines()
            random_board = lines[np.random.randint(0, len(lines))].split(" ")
            # print(random_board)
            return int(random_board[0]), random_board[1]

    def _get_piece_orientations(self):
        orientations = {}
        for piece in self.pieces:
            positions = np.argwhere(self.board == piece)
            if positions[0][0] == positions[-1][0]:
                orientations[piece] = "H"  # Horizontal
            else:
                orientations[piece] = "V"  # Vertical
        return orientations

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.array(list(self.board_description)).reshape(6, 6)
        return self._get_obs(), {"num_steps_to_finish": self.num_steps_to_finish}

    def step(self, action):
        piece = list(self.pieces)[action[0]]
        direction = action[1]
        moved = self._move_piece(piece, direction)
        done = self._check_win()
        reward = 0 if done else -1
        return (
            self._get_obs(),
            reward,
            done,
            False,
            {"num_steps_to_finish": self.num_steps_to_finish},
        )

    def _get_obs(self):
        if self.obs_type == "rgb":
            # Create an image representation of the board
            cell_size = self.cell_size
            img = Image.new("RGB", (6 * cell_size, 6 * cell_size), color="white")
            draw = ImageDraw.Draw(img)

            for i in range(6):
                for j in range(6):
                    color = self.colors[self.board[i, j]]
                    draw.rectangle(
                        [
                            j * cell_size,
                            i * cell_size,
                            (j + 1) * cell_size,
                            (i + 1) * cell_size,
                        ],
                        fill=color,
                        outline="black",
                    )
                    if (
                        str(self.board[i, j].lower()) != "o"
                        and str(self.board[i, j]) != "x"
                    ):
                        draw.text(
                            (
                                j * cell_size + cell_size // 2,
                                i * cell_size + cell_size // 2,
                            ),
                            str(self.pieces.index(self.board[i, j])),
                            fill="black",
                            anchor="mm",
                        )
            return np.array(img)
        else:
            return self.board.copy()

    def _move_piece(self, piece, direction):
        positions = np.argwhere(self.board == piece)
        if len(positions) == 0:
            return False

        if self.piece_orientations[piece] == "H":
            if direction == 0 or direction == 1:  # up or down
                new_pos = positions
            elif direction == 2:  # left
                new_pos = positions - [0, 1]
            elif direction == 3:  # right
                new_pos = positions + [0, 1]
        else:
            if direction == 0:  # up
                new_pos = positions - [1, 0]
            elif direction == 1:  # down
                new_pos = positions + [1, 0]
            elif direction == 2 or direction == 3:  # left or right
                new_pos = positions

        # print("pre-valid - ", self.board)
        if self._is_valid_move(positions, new_pos):
            self.board[tuple(positions.T)] = "o"
            self.board[tuple(new_pos.T)] = piece
            # print("valid move -", self.board)
            return True
        return False

    def _is_valid_move(self, current_pos, new_pos):
        if np.any(new_pos < 0) or np.any(new_pos >= 6):
            return False
        for pos in new_pos:
            if (
                tuple(pos) not in [tuple(p) for p in current_pos]
                and self.board[tuple(pos)] != "o"
            ):
                return False
        return True

    def _check_win(self):
        return self.board[2, 5] == "A"

    def render(self):
        if self.obs_type == "rgb":
            img = self._get_obs()
            plt.imshow(img)
            plt.show()
        else:
            for row in self.board:
                print(" ".join(row))
            print()
