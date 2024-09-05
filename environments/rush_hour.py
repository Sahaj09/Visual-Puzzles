import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

class RushHourEnv(gym.Env):
    def __init__(self, board_description, obs_type="rgb"):
        super(RushHourEnv, self).__init__()

        assert obs_type in ["rgb", "text"], "Observation type must be 'rgb' or 'text'"

        self.board_description = board_description
        self.board = np.array(list(board_description)).reshape(6, 6)
        self.pieces = set(self.board.flatten()) - set("ox")
        self.pieces = sorted(list(self.pieces))
        self.piece_orientations = self._get_piece_orientations()
        self.obs_type = obs_type
        print(self.pieces)

        # piece description, direction: 0 - up, 1 - right, 2 - down, 3 - left
        self.action_space = spaces.MultiDiscrete(np.array([len(self.pieces), 4]))

        # Define observation space
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(6, 6), dtype=np.uint8
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

    def _get_piece_orientations(self):
        orientations = {}
        for piece in self.pieces:
            positions = np.argwhere(self.board == piece)
            if positions[0][0] == positions[-1][0]:
                orientations[piece] = "H"  # Horizontal
            else:
                orientations[piece] = "V"  # Vertical
        return orientations

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.array(list(self.board_description)).reshape(6, 6)
        return self._get_obs(), {}

    def step(self, action):
        piece = list(self.pieces)[action[0]]
        direction = action[1]
        moved = self._move_piece(piece, direction)
        done = self._check_win()
        reward = 1 if done else 0
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        if self.obs_type == "rgb":
            # Create an image representation of the board
            cell_size = 50
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

        print("pre-valid - ", self.board)
        if self._is_valid_move(positions, new_pos):
            self.board[tuple(positions.T)] = "o"
            self.board[tuple(new_pos.T)] = piece
            print("valid move -", self.board)
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
