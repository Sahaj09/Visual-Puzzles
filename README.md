# Simple Puzzle(ing) Benchmark For Visual Reasoning in Vision Language Models

This repository provides some simple tasks based on sliding puzzles like 
- N-Puzzle
- N-Puzzle with modified image (blurring, contours, etc.) 
- N-Tile Jigsaw
- N-Tile Jigsaw with modified image (blurring, contours, etc.) 
- Rush-Hour (uses boards by [rush](https://github.com/fogleman/rush))

These tasks are set up as gymnasium environments as it allows for versatile evaluation (and can also be used to evaluate models other than VLMs).


## Installation -

```bash
pip install "visual_puzzle @ git+https://github.com/Sahaj09/Visual-Puzzles.git@main"
```
## Usage

```python
import gymnasium as gym
import visual_puzzle


env = gym.make("n_Puzzle-v0", render_mode="human")
# env = gym.make("jigsaw-v0", render_mode="human")
# env = gym.make("RushHour-v0", render_mode = "human")

observation, info = env.reset(seed=42)

for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
```

## N-puzzle

The observation is an RGB image. The actions - up, down, left, and right (which moves the blank tile in that direction). The goal of the puzzle is to manipulate the tiles in order to get the goal format.



| Start State | Goal State|
   |-----------------------|-------------|
   | ![image_2](./images/n_puzzle_shuffle.png) | ![image_1](./images/n_puzzle_final.png "check") |

## Jigsaw
The observation is an RGB image. The actions - [[a, b], [c, d]] where $a,b,c,d \in [0,N)$ in a $N\times N$ puzzle. [a, b] is the position of the first tile, [c, d] is the position of the second tile, and choosing these tiles leads to a swap between them. 

The goal of the puzzle is to manipulate the tiles in order to get the goal format.


| Start State | Goal State|
   |-----------------------|-------------|
   | ![image_4](./images/jigsaw_shuffled_image.png) | ![image_3](./images/jigsaw_final_image.png) |


## Variations of N-Puzzle


## Rush-Hour
The rush-hour game is sliding puzzle where the goal is to have the red tile (in our case, also indexed as 0) reach the right end of the board. More about this game can be read [here](https://en.wikipedia.org/wiki/Rush_Hour_(puzzle)) and in this amazing [blog](https://www.michaelfogleman.com/rush/).

### Third-Party Content
This project uses rush.txt file from [rush](https://github.com/fogleman/rush) 
under the MIT-License.