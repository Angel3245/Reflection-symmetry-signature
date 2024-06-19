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

import cv2 as cv
import matplotlib.pyplot as plt

def find_centroid(images, debug=False):
    """Compute the centroid of the figure in the image."""
    centroids = []
    for image in images:
        M = cv.moments(image.astype('uint8'))
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        centroids.append((cx,cy))
    if debug:
        for image,centroid in zip(images,centroids):
            plt.imshow(image, cmap=plt.cm.gray)
            plt.scatter(centroid[0],centroid[1],c='r')
            plt.show()
    return centroids

def find_center_of_mass(images, debug=False):
    centers_of_mass = []
    for image in images:
        M = cv.moments(image.astype('uint8'))
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centers_of_mass.append((cx, cy))
        else:
            # If the image is empty, consider the center as (0, 0)
            centers_of_mass.append((0, 0))
    if debug:
        for image, center in zip(images, centers_of_mass):
            plt.imshow(image, cmap=plt.cm.gray)
            plt.scatter(center[0], center[1], c='r')
            plt.show()
    return centers_of_mass

def find_center_of_min_circle(images, debug=False):
    centers_of_min_circle = []
    for image in images:
        # Find contours
        contours, _ = cv.findContours(image.astype('uint8'), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get the minimum enclosing circle
            (x, y), radius = cv.minEnclosingCircle(contours[0])
            centers_of_min_circle.append((int(x), int(y)))
        else:
            # If no contour is found, consider the center as (0, 0)
            centers_of_min_circle.append((0, 0))
    if debug:
        for image, center in zip(images, centers_of_min_circle):
            plt.imshow(image, cmap=plt.cm.gray)
            plt.scatter(center[0], center[1], c='r')
            plt.show()
    return centers_of_min_circle