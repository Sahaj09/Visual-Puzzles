import os

def get_asset_path(filename):
    """
    Returns the full path to an asset file.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, 'assets', filename)

from .n_puzzle import n_PuzzleEnv
from .rush_hour import RushHourEnv
from .register import register_environments

register_environments()