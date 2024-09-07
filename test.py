import gymnasium as gym
import visual_puzzle
from PIL import Image




def action_to_index(action):
    return {"w": 0, "d": 1, "s": 2, "a": 3}[action]


def get_user_action():
    while True:
        action = input(
            "Enter action (w: up, d: right, s: down, a: left, q: quit): "
        ).lower()
        if action in ["w", "d", "s", "a", "q"]:
            return action
        print("Invalid input. Please try again.")


def play_game_jigsaw():
    env = gym.make("jigsaw-v0", render_mode="human")
    
    observation, info = env.reset()

    print("Welcome to the JigSaw game!")
    print("Try to arrange the image pieces in the correct order.")

    total_reward = 0
    steps = 0

    while True:
        print(f"Steps: {steps}, Total Reward: {total_reward}")
        print(f"Manhattan distance: {info['manhattan_distance']}")

        pos_1_x = input("position_1, index_x :")
        pos_1_y = input("position_1, index_y :")
        pos_2_x = input("position_2, index_x :")
        pos_2_y = input("position_2, index_y :")
        action = [[int(pos_1_x), int(pos_1_y)], [int(pos_2_x), int(pos_2_y)]]
        if action == "q":
            print("Quitting the game.")
            break
        observation, reward, terminated, truncated, info = env.step(
            action
        )
        print(f"terminated: {terminated=}")
        print(f"truncated: {truncated=}")

        total_reward += reward
        steps += 1

        if terminated or truncated:
            env.render()
            print("Game over!")
            break

    


def play_game_n_puzzle():
    env = gym.make("n_Puzzle-v0", render_mode="human")
    # env = n_PuzzleEnv(image_path=image_path, render_mode="human")
    observation, info = env.reset(seed=0)

    print("Welcome to the n-Puzzle game!")
    print("Try to arrange the image pieces in the correct order.")

    total_reward = 0
    steps = 0

    while True:
        print(f"Steps: {steps}, Total Reward: {total_reward}")
        print(f"Manhattan distance: {info['manhattan_distance']}")

        action = get_user_action()
        if action == "q":
            print("Quitting the game.")
            break
        observation, reward, terminated, truncated, info = env.step(
            action_to_index(action)
        )
        print(f"terminated: {terminated=}")
        print(f"truncated: {truncated=}")

        total_reward += reward
        steps += 1

        if terminated or truncated:
            env.render()
            print("Game over!")
            break

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
    
    play_game_jigsaw()
    # play_game_n_puzzle()
    # test_rush_hour_env()
