import gymnasium as gym
from environments import RushHourEnv


def test_rush_hour_env():
    board_description = "ooxoKoCCoIKoGAAIKoGoHJDDEEHJoLoFFFoL"
    env = RushHourEnv(board_description)

    # Test reset
    obs, _ = env.reset()

    print("Reset test passed.")

    # Test step

    done = False

    # Test win condition
    while not done:
        # accept input from user

        piece = int(input("Enter piece to move: "))
        direction = int(input("Enter direction to move: "))

        action = [piece, direction]
        obs, reward, done, _, _ = env.step(action)
        print()
        print(f"{action=}")
        print(f"{reward=}")
        print(f"{done=}")
        env.render()

        if done:
            print("Game won!")
            env.render()
            break


if __name__ == "__main__":
    test_rush_hour_env()
