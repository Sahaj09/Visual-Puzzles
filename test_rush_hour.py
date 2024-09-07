import gymnasium as gym
import visual_puzzle


def test_rush_hour_env():
    board_description = None
    env = gym.make("RushHour-v0", board_description=board_description, obs_type="rgb")

    # Test reset
    obs, _ = env.reset()
    print(obs.shape)
    print("Reset test passed.")

    # Test step

    done = False

    # Test win condition
    while not done:
        # accept input from user
        env.render()
        piece = int(input("Enter piece to move: "))
        direction = int(input("Enter direction to move: "))

        action = [piece, direction]
        obs, reward, done, _, _ = env.step(action)
        print()
        print(f"{action=}")
        print(f"{reward=}")
        print(f"{done=}")

        if done:
            print("Game won!")
            env.render()
            break


if __name__ == "__main__":
    test_rush_hour_env()
