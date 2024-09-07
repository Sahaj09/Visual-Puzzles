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

### Third-Party Content
This project uses rush.txt file from [rush](https://github.com/fogleman/rush) 
under the MIT-License.