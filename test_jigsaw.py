import gymnasium as gym
import visual_puzzle
from PIL import Image




def action_to_index(action):
    return {"w": 0, "d": 1, "s": 2, "a": 3}[action]


def play_game(image_path):
    env = gym.make("jigsaw-v0", image_path=image_path, render_mode="human")
    
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

    


if __name__ == "__main__":
    image_path = "experiment_data/check.png"
    play_game(image_path)
