from gymnasium.envs.registration import register


def register_environments():
    register(
        id="n_Puzzle-v0",
        entry_point="environments.n_puzzle:n_PuzzleEnv",
    )

    register(
        id="RushHour-v0",
        entry_point="environments.rush_hour:RushHourEnv",
    )