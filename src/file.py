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

import glob
import matplotlib.pyplot as plt

def load_images(folder):
    images_path = glob.glob(folder + '/*.png')
    images = [plt.imread(image) for image in images_path]
    for path,image in zip(images_path,images):
        print(path)
        print(image.shape)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()
    return images