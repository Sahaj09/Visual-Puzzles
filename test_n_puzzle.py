import gymnasium as gym
from environments import n_PuzzleEnv
from PIL import Image


def get_user_action():
    while True:
        action = input(
            "Enter action (w: up, d: right, s: down, a: left, q: quit): "
        ).lower()
        if action in ["w", "d", "s", "a", "q"]:
            return action
        print("Invalid input. Please try again.")


def action_to_index(action):
    return {"w": 0, "d": 1, "s": 2, "a": 3}[action]


def play_game(image_path):
    env = n_PuzzleEnv(image_path=image_path, render_mode="human")
    observation, info = env.reset()

    print("Welcome to the 15-Puzzle game!")
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

    env.close()


if __name__ == "__main__":
    image_path = "experiment_data/check.jpg"
    play_game(image_path)
