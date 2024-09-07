import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image, ImageTk, ImageFilter, ImageDraw
import tkinter as tk
from typing import Optional


class n_PuzzleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        image_path: str,
        render_mode: Optional[str] = None,
        n_puzzle: int = 15,
        image_size: int = 240,
        filter_effects: Optional[str] = None,
        time_steps_limit: Optional[int] = None,
    ):
        """Initialize the n-Puzzle environment.

        This method sets up the n-Puzzle game environment using an input image.
        It preprocesses the image, creates tiles, and initializes the game state.

        Args:
            image_path (str): Path to the image file to be used for the puzzle.
                The image will be resized to self.image_sizexself.image_size pixels.
            render_mode (str, optional): Specifies how to render the environment.
                Supported modes: "human" for rendering to a window using tkinter. ascii only for debugging.
                Defaults to None.
            n_puzzle (int): Number of tiles in the puzzle (e.g., 15 for 15-Puzzle).
                Defaults to 15.
            image_size (int): Size of the resized image in pixels.
                Defaults to 240.
            filter_effects (str, optional): Specifies the filter effects to apply to the image.
                Supported effects: "BLUR", "CONTOUR", "DETAIL", "EDGE_ENHANCE", "EDGE_ENHANCE_MORE",
                "EMBOSS", "FIND_EDGES", "SHARPEN", "SMOOTH", "SMOOTH_MORE".
                Defaults to None.
            time_steps_limit (int, optional): Maximum number of time steps for each episode. If None, there is no limit.
                Defaults to None.

        Attributes:
            size (int): The size of the puzzle grid (sqrt(n+1)xsqrt(n+1) for n-Puzzle, Eg. 4x4 for 15-puzzle).
            n (int): Total number of tiles, including the empty space.
            image_size (int): The size of the resized image in pixels.
            original_image (PIL.Image): The resized input image (image_sizeximage_size pixels).
            tile_size (int): The size of each puzzle tile in pixels.
            tiles (list): List of PIL.Image objects representing the puzzle tiles.
            blank_tile (PIL.Image): A black image representing the empty space.
            action_space (gym.spaces.Discrete): The space of possible actions.
            observation_space (gym.spaces.Box): The space of possible observations.
            render_mode (str): The specified render mode.
            terminated (bool): Flag indicating if the game is over.
            truncated (bool): Flag indicating if the game is truncated due to the time steps limit.
            window (tkinter.Tk): The main window for rendering (initialized later).
            canvas (tkinter.Canvas): The canvas for drawing the puzzle (initialized later).


        Note:
            The environment uses a discrete action space with 4 possible actions:
            0: move up, 1: move right, 2: move down, 3: move left.
            The observation space is a self.image_sizexself.image_sizex3 RGB image of the current puzzle state.
        """

        super(n_PuzzleEnv, self).__init__()

        self.size = np.sqrt(n_puzzle + 1).astype(int)
        self.n = self.size**2
        self.image_size = image_size
        # Load and preprocess the input image
        self.original_image = Image.open(image_path)
        self.original_image = self.original_image.resize(
            (self.image_size, self.image_size)
        )
        self.original_image_before_shuffle_or_filter = self.original_image.copy()

        assert filter_effects in [
            "BLUR",
            "CONTOUR",
            "DETAIL",
            "EDGE_ENHANCE",
            "EDGE_ENHANCE_MORE",
            "EMBOSS",
            "FIND_EDGES",
            "SHARPEN",
            "SMOOTH",
            "SMOOTH_MORE",
            None,
        ], "Invalid filter effect."
        if filter_effects:
            self.original_image = self.original_image.filter(
                getattr(ImageFilter, filter_effects.upper())
            )

        self.tile_size = int(self.image_size / self.size)  # 100

        assert self._check_if_valid_n_puzzle(
            self.image_size, n_puzzle
        ), "Invalid combination of image size and number of tiles."

        # Create image tiles
        self.tiles = []
        for i in range(self.size):
            for j in range(self.size):
                tile = self.original_image.crop(
                    (
                        j * self.tile_size,
                        i * self.tile_size,
                        (j + 1) * self.tile_size,
                        (i + 1) * self.tile_size,
                    )
                )
                self.tiles.append(tile)

        # Create a blank tile for the empty space
        self.blank_tile = Image.new(
            "RGB", (self.tile_size, self.tile_size), color="black"
        )

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # up, right, down, left
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.image_size, self.image_size, 3), dtype=np.uint8
        )

        self.render_mode = render_mode
        self.time_steps_limit = time_steps_limit if time_steps_limit else np.inf

        # Forced to reset the environment
        self.terminated = True
        self.truncated = True

        self.current_time_step = 0
        self.window = None
        self.canvas = None

        self.valid_positions = np.array(
            [[i, j] for i in range(self.size) for j in range(self.size)]
        )

        self.board = np.arange(self.n).reshape((self.size, self.size))

        self.final_image = Image.new("RGB", (self.image_size, self.image_size))
        _draw_ = ImageDraw.Draw(self.final_image)
        for i in range(self.size):
            for j in range(self.size):
                tile_index = self.board[i, j]
                if tile_index == 0:
                    tile = self.blank_tile
                else:
                    tile = self.tiles[tile_index]
                self.final_image.paste(tile, (j * self.tile_size, i * self.tile_size))
                x = j * self.tile_size
                y = i * self.tile_size
                _draw_.rectangle(
                    [x, y, x + self.tile_size, y + self.tile_size],
                    outline="black",
                    width=1,
                )
        

    @staticmethod
    def _check_if_valid_n_puzzle(image_size, n_puzzle):
        if (
            np.sqrt(n_puzzle + 1) % 1 != 0
            or n_puzzle < 1
            or image_size % np.sqrt(n_puzzle + 1) != 0
        ):
            return False
        return True

    def _get_obs(self):
        # Create the observation by assembling the image tiles
        obs = Image.new("RGB", (self.image_size, self.image_size))
        draw = ImageDraw.Draw(obs)
        for i in range(self.size):
            for j in range(self.size):
                tile_index = self.board[i, j]
                if tile_index == 0:
                    tile = self.blank_tile
                else:
                    tile = self.tiles[tile_index]
                obs.paste(tile, (j * self.tile_size, i * self.tile_size))
                x = j * self.tile_size
                y = i * self.tile_size
                draw.rectangle(
                    [x, y, x + self.tile_size, y + self.tile_size],
                    outline="black",
                    width=1,
                )
        return np.array(obs)

    def _get_info(self):
        return {
            "manhattan_distance": self._manhattan_distance(),
            "original_image": self.original_image_before_shuffle_or_filter,
            "goal_image": self.final_image,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time_step = 0
        # Initialize the board in solved state
        self.board = np.arange(self.n).reshape((self.size, self.size))

        # Shuffle the board
        self.np_random.shuffle(self.board.ravel())

        # Find the position of the empty tile (0)
        self.empty_pos = np.argwhere(self.board == 0)[0]

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "ascii":
            self._render_ascii()

        self.terminated = False
        self.truncated = False
        return observation, info

    def step(self, action):
        if self.terminated or self.truncated:
            # print("Invalid action. Environment has been terminated.")
            return self._get_obs(), 0, self.terminated, self.truncated, self._get_info()

        self.current_time_step += 1
        # Define movement directions
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
        dy, dx = directions[action]

        new_pos = self.empty_pos + np.array([dy, dx])

        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            # Swap the empty tile with the adjacent tile
            self.board[tuple(self.empty_pos)], self.board[tuple(new_pos)] = (
                self.board[tuple(new_pos)],
                self.board[tuple(self.empty_pos)],
            )
            self.empty_pos = new_pos

        self.terminated = self._is_solved()

        if self.current_time_step >= self.time_steps_limit:
            self.truncated = True

        reward = -1  # Small negative reward for each move
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "ascii":
            self._render_ascii()

        return observation, reward, self.terminated, self.truncated, info

    def _is_solved(self):
        return np.all(self._get_obs() == np.array(self.final_image))
        # return np.array_equal(
        #     self.board, np.arange(self.n).reshape((self.size, self.size))
        # )

    def _manhattan_distance(self):
        distance = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] != 0:
                    x, y = divmod(self.board[i, j] - 1, self.size)
                    distance += abs(x - i) + abs(y - j)
        return distance

    def render(self):
        if self.render_mode == "ascii":
            return self._render_ascii()
        elif self.render_mode == "human":
            return self._render_frame()

    def _render_ascii(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0:
                    print("  .", end=" ")
                else:
                    print(f"{self.board[i, j]:3d}", end=" ")
            print()
        print()

    def _render_frame(self):
        if self.window is None:
            self.window = tk.Tk()
            self.window.title("n-Puzzle")
            self.canvas = tk.Canvas(
                self.window, width=self.image_size, height=self.image_size
            )
            self.canvas.pack()

        img = Image.fromarray(self._get_obs())
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)  # type: ignore
        self.window.update()

    def close(self):
        pass
        # if self.window is not None:
        #     self.window.destroy()
        #     self.window = None
        #     self.canvas = None
