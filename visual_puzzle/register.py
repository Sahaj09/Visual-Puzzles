from gymnasium.envs.registration import register


def register_environments():
    register(
        id="n_Puzzle-v0",
        entry_point="visual_puzzle.n_puzzle:n_PuzzleEnv",
    )

    register(
        id="RushHour-v0",
        entry_point="visual_puzzle.rush_hour:RushHourEnv",
    )
