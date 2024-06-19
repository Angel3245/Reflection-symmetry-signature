# Copyright (C) 2024  Jose Ángel Pérez Garrido
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.centroid import *
from src.file import *
from src.symmetry import *
from src.skeleton import *

def get_args():
    parser = argparse.ArgumentParser(description="Detects reflection symmetry in shapes using shape signatures")

    """
    Data handling
    """
    parser.add_argument('--dataset-folder', type=str, default='./images',
                        help='dataset folder path (default: ./images)')

    """
    Other Parameters
    """
    parser.add_argument('--centroid-func', choices=["centroid","mass","min_circle"], default="centroid", help='Centroid computation\
                        function ["centroid","mass","min_circle"] (default: centroid)')
    
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold\
                        (default: 0.5)')


    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    # Load images
    images = load_images(args.dataset_folder)
    
    # Get centroid computation function
    if args.centroid_func == "centroid":
        centroid_func = find_centroid
    elif args.centroid_func == "mass":
        centroid_func = find_center_of_mass
    elif args.centroid_func == "min_circle":
        centroid_func = find_center_of_min_circle
    
    tau = args.threshold # Threshold

    # Execution on original images
    for image in images:
        plt.imshow(image, cmap=plt.cm.gray)
        plt.title("Original image")
        plt.show()

        print("- Image")
        mirror = SignatureSymmetry(image, tau, centroid_func=centroid_func)
        mirror.run()
