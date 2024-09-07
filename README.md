# Simple benchmark for visual reasoning in VLMs

This repository provides some simple tasks based on sliding puzzles like 
- n-puzzle
- n-puzzle with modified image (blurring, contours, etc.) 
- n-tile Jigsaw
- n-tile Jigsaw with modified image (blurring, contours, etc.) 
- Rush-Hour (uses boards by [rush](https://github.com/fogleman/rush))

These tasks are set up as gymnasium environments as it allows for versatile evaluation (and can also be used to evaluate models other than VLMs).

Installation -

```bash
pip install "visual_puzzle @ git+https://github.com/Sahaj09/Visual-Puzzles.git@main"
```


### Third-Party Content
This project uses rush.txt file from [rush](https://github.com/fogleman/rush) 
under the MIT-License.