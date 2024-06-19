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

from scipy.signal import find_peaks
import numpy as np
from scipy.integrate import simps
from scipy.signal import find_peaks
from skimage.transform import radon
import matplotlib.pyplot as plt
import cv2 as cv

from .centroid import *

class SignatureSymmetry:
    """
    Detects symmetry in an image based on shape signatures.
    Based on Nguyen, T. P., Truong, H. P., Nguyen, T. T., & Kim, Y. (2022). Reflection symmetry detection of shapes based on shape signatures. Pattern Recognition, 128, 108667. https://doi.org/10.1016/j.patcog.2022.108667
    """

    def __init__(self, image, tau, centroid_func=find_centroid):
        """
        Initialize the SignatureSymmetry object with the image and the centroid function.

        Args:
            image (numpy.ndarray): The input image.
            tau (float): Threshold
            centroid_func: The centroid computation function (default is find_centroid).
        """
        self.image = image.astype('uint8')
        self.centroid = centroid_func([image])[0]
        self.tau = tau
        
        # Initially reflected image is not computed
        self.reflected_image = None

    # Function to calculate Rf2(θ)
    def Rf2(self,sinogram):
        """
        Calculate the squared Radon transform Rf2(θ).

        Parameters:
            sinogram (numpy.ndarray): The sinogram obtained from the Radon transform.

        Returns:
            numpy.ndarray: The squared Radon transform Rf2(θ).
        """
        # Compute the square of Rf(θ, ρ)
        sinogram_squared = sinogram**2

        # Integrate Rf^2(θ, ρ) with respect to ρ for each θ
        # We'll use the composite trapezoidal rule for numerical integration
        return simps(sinogram_squared, axis=0)

    def plot_r2_signature(self,signature, theta):
        """
        Plot the Rf2(θ) signature.

        Parameters:
            signature (numpy.ndarray): The Rf2(θ) signature.
            theta (numpy.ndarray): The angles at which the signature is computed.
        """
        # Print or plot the result
        plt.plot(theta, signature)
        plt.xlabel('Theta (degrees)')
        plt.ylabel('Rf2(θ)')
        plt.title('R-signature')
        plt.show()

    def plot_lip_values(self,signature, directions):
        """
        Plot the LIP values.

        Parameters:
            signature (numpy.ndarray): The LIP signature.
            directions (numpy.ndarray): The directions at which the signature is computed.
        """
        plt.plot(directions, signature)
        plt.title('LIP-Signature')
        plt.xlabel('Direction (degrees)')
        plt.ylabel('LIP Value')
        plt.show()

    def plot_sinogram(self,image, sinogram):
        """
        Plot the original image and its sinogram.

        Parameters:
            image (numpy.ndarray): The original image.
            sinogram (numpy.ndarray): The sinogram obtained from the Radon transform.
        """
        dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

        ax1.set_title("Original")
        ax1.imshow(image, cmap=plt.cm.Greys_r)

        ax2.set_title("Radon transform\n(Sinogram)")
        ax2.set_xlabel("Projection angle (deg)")
        ax2.set_ylabel("Projection position (pixels)")
        ax2.imshow(
            sinogram,
            cmap=plt.cm.Greys_r,
            extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
            aspect='auto',
        )

        fig.tight_layout()
        plt.show()

    def plot_symmetry_axis(self,image, symmetry_directions, centroid):
        """
        Plot the symmetry axes on the original image.

        Parameters:
            image (numpy.ndarray): The original image.
            symmetry_directions (list): List of angles representing symmetry directions.
            centroid (tuple): Coordinates of the centroid of the image.
        """
        # Get centroid of the image coordinates
        center_x, center_y = centroid
        
        # Plot the original image
        plt.imshow(image, cmap='gray')
        
        # Plot the symmetry axes
        for theta in symmetry_directions:
            theta_rad = np.deg2rad(theta)
            t = np.linspace(-max(image.shape), max(image.shape), 1000)
            x = center_x + t * np.cos(theta_rad)
            y = center_y + t * np.sin(theta_rad)

            # Clip the lines to stay within image boundaries
            x = np.clip(x, 0, image.shape[1]-1)
            y = np.clip(y, 0, image.shape[0]-1)
            
            plt.plot(x, y, 'r')
        
        #plt.axis('image')
        plt.axis('off')
        plt.show()

    # Function to plot merits with peaks
    def plot_merits_with_peaks(self,angles, merit, peaks):
        """
        Plot the merit values with peaks marked.

        Parameters:
            angles (numpy.ndarray): The angles at which the merit values are computed.
            merit (numpy.ndarray): The merit values.
            peaks (numpy.ndarray): Indices of the peaks in the merit values.
        """
        plt.plot(angles, merit)
        plt.plot(angles[peaks], merit[peaks], 'ro')  # Plot peaks as red points
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Merit')
        plt.title('Merit of signature with Peaks')
        plt.grid(True)
        plt.show()

    def largest_intersection(self,sinogram):
        """
        Find the largest intersection in the Radon image in a given direction.
        """
        return np.max(sinogram, axis=0)

    def projection(self,sinogram):
        """
        Compute the projection of the Radon image onto a line perpendicular to the given direction.
        """
        sinogram_nzero = sinogram > 0

        # Integrate Rf^2(θ, ρ) with respect to ρ for each θ
        # We'll use the composite trapezoidal rule for numerical integration
        return simps(sinogram_nzero, axis=0)

    def LIP_signature(self,sinogram):
        """
        Compute the LIP signature of the Radon image over a set of directions.
        """
        lip_value = self.largest_intersection(sinogram) / self.projection(sinogram)
            
        return np.array(lip_value)

    def run(self, method=True, visualize=True):
        """
        Detect reflection symmetry axes in an image.

        Parameters:
            image (numpy.ndarray): The input image.
            tau (float): Threshold for peak detection in the merit function.
            method (bool): If True, use LIP-signature method; otherwise, use R-signature method.
            plot (bool): If True, plot intermediate results.

        Returns:
            tuple: A tuple containing the number of symmetry axes found, a list of symmetry directions,
                and a list of symmetry measures.
        """
        # Compute the Radon transform Rf(θ, ρ)
        theta = np.arange(0, 180)  # angles from 0 to 179 degrees
        sinogram = radon(self.image, theta=theta, circle=True)

        if visualize:
            self.plot_sinogram(self.image, sinogram)

        merit = np.zeros_like(theta,dtype=np.float64)
        n = 0
        symmetry_directions = []
        symmetry_measures = []

        if method:
            signature = self.LIP_signature(sinogram)  # LIP-signature of D over angles
                
            if visualize:
                self.plot_lip_values(signature, theta)
        else:
            signature = self.Rf2(sinogram)  # R-signature of D over angles
            if visualize:
                self.plot_r2_signature(signature, theta)

        for angle in theta:
            merit[angle] = self.calculate_merit(angle, signature)

        peaks, _ = find_peaks(merit, height=self.tau)

        if visualize:
            print(f"{peaks=}")
        
        if visualize:
            self.plot_merits_with_peaks(theta,merit,peaks)

        for k in range(len(peaks)):
            angle = peaks[k]
            D_theta = self.symmetric_degree(sinogram, theta, angle)  # Calculate D(θ) using Eq. (4)

            if D_theta > self.tau:
                n += 1  # found a new axis of reflection symmetry
                symmetry_directions.append(angle)
                symmetry_measures.append(D_theta)

        if visualize:
            self.plot_symmetry_axis(self.image, symmetry_directions, self.centroid)

        if len(symmetry_directions) > 0:
            # Compute reflected image using the mirror line (with the most symmetric axis)
            idx_sym = max(range(len(symmetry_measures)), key=symmetry_measures.__getitem__)

            self.reflected_image = self.reflect_image(self.centroid, symmetry_directions[idx_sym])

        return n, symmetry_directions, symmetry_measures

    def reflect_image(self, centroid, angle):
        """
        Reflect the original image using the mirror line parameters (centroid_x, centroid_y, angle).

        Args:
            centroid (tuple): Coordinates of the centroid (centroid_x, centroid_y).
            angle (float): Angle of reflection (in radians).

        Returns:
            numpy.ndarray: Reflected image.
        """
        centroid_x, centroid_y = centroid

        # Compute the sine and cosine of the reflection angle
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # Translation matrix to move centroid to the origin
        translate_to_origin = np.array([[1, 0, -centroid_x],
                                        [0, 1, -centroid_y],
                                        [0, 0, 1]])

        # Rotation matrix to rotate about the origin
        rotate = np.array([[cos_angle, -sin_angle, 0],
                           [sin_angle, cos_angle, 0],
                           [0, 0, 1]])

        # Translation matrix to move the centroid back
        translate_back = np.array([[1, 0, centroid_x],
                                   [0, 1, centroid_y],
                                   [0, 0, 1]])

        # Affine transformation matrix
        transformation_matrix = translate_back @ rotate @ translate_to_origin

        # Get image dimensions
        rows, cols = self.image.shape[:2]

        # Create an empty array to store the reflected image
        reflected_image = np.zeros_like(self.image)

        # Iterate over each pixel in the original image
        for y in range(rows):
            for x in range(cols):
                # Apply the transformation matrix to get the reflected coordinates
                reflected_coords = np.dot(transformation_matrix, [x, y, 1]).astype(int)

                # Check if the reflected coordinates are within the image bounds
                if 0 <= reflected_coords[0] < cols and 0 <= reflected_coords[1] < rows:
                    # Assign the pixel value to the corresponding pixel in the reflected image
                    reflected_image[reflected_coords[1], reflected_coords[0]] = self.image[y, x]

        return reflected_image

    def pearson_correlation_coefficient(self,X, Y):
        """
        Compute the Pearson correlation coefficient between two NumPy arrays.
        
        Args:
        - X (numpy.ndarray): The first array.
        - Y (numpy.ndarray): The second array.
        
        Returns:
        - float: The Pearson correlation coefficient.
        """
        return np.corrcoef(X, Y)[0, 1]

    # Extract the Radon projection for a specific angle
    def get_radon_projection(self,sinogram, theta, angle):
        """
        Extract the Radon projection for a specific angle from the sinogram.

        Parameters:
            sinogram (numpy.ndarray): The sinogram obtained from the Radon transform.
            theta (numpy.ndarray): The array of projection angles.
            angle (float): The angle at which to extract the projection.

        Returns:
            numpy.ndarray: The Radon projection at the specified angle.
        """
        # Find the index of the specified angle in the theta array
        angle_index = np.argmin(np.abs(theta - angle))
        # Extract the corresponding column from the sinogram
        projection = sinogram[:, angle_index]
        return projection

    # Compute the inverse of an array
    def compute_inverse(self,array):
        """
        Compute the inverse of an array.

        Parameters:
            array (numpy.ndarray): The input array.

        Returns:
            numpy.ndarray: The array with elements in reverse order.
        """
        return array[::-1]

    # Function to calculate the intensity of the pixel at the centroid of shape D
    def symmetric_degree(self,sinogram, theta, peak):
        """
        Calculate the symmetric degree of a shape D.

        Parameters:
            sinogram (numpy.ndarray): The sinogram obtained from the Radon transform.
            theta (numpy.ndarray): The array of projection angles.
            peak (int): The index of the peak angle.

        Returns:
            float: The symmetric degree.
        """
        # Assuming D is an image and centroid is given as (x, y) coordinates
        c = self.get_radon_projection(sinogram, theta, peak)

        return self.pearson_correlation_coefficient(c, self.compute_inverse(c))  # Eq. (4)

    # Function to calculate merit value for a given angle
    def calculate_merit(self,theta, signature):
        """
        Calculate the merit value for a given angle.

        Parameters:
            theta (float): The angle for which to calculate the merit.
            signature (numpy.ndarray): The signature values.

        Returns:
            float: The merit value.
        """
        return self.pearson_correlation_coefficient(np.roll(signature, theta), np.roll(signature, -theta))  # Eq. (3)
    

    ###########
    # METRICS #
    ###########
    def symmetry_score(self):
        """
        Compute a symmetry score based on the distribution of matching points.

        Returns:
            float: Symmetry score.
        """
        matchpoints = self.find_matchpoints()

        if len(matchpoints) > 0:
            points_r, points_theta = self.find_points_r_theta(matchpoints)

            # Calculate the standard deviation of r and theta
            std_r = np.std(points_r)
            std_theta = np.std(points_theta)

            # Compute symmetry score
            symmetry_score = 1 / (std_r + std_theta)

            return symmetry_score
        else:
            return 0.0

    def edge_preservation(self):
        """
        Compute edge preservation metric.

        Returns:
            float: Edge preservation metric.
        """
        if self.reflected_image is None:
            return None
        
        # Compute edge maps for original and flipped images
        edge_map_orig = cv.Canny(self.image, 100, 200)
        edge_map_flip = cv.Canny(self.reflected_image, 100, 200)

        # Compute the percentage of preserved edges
        preserved_edges = np.sum(np.logical_and(edge_map_orig, edge_map_flip))
        total_edges = np.sum(edge_map_orig)

        edge_preservation = preserved_edges / total_edges if total_edges > 0 else 0.0

        return edge_preservation

    def information_loss(self):
        """
        Compute information loss metric.

        Returns:
            float: Information loss metric.
        """
        if self.reflected_image is None:
            return None
        # Compute pixel-wise absolute difference between original and reflected images
        abs_diff = np.abs(self.image - self.reflected_image)

        # Compute the average absolute difference
        avg_abs_diff = np.mean(abs_diff)

        # Normalize by the maximum pixel value (255 for uint8 images)
        information_loss = avg_abs_diff / 255.0

        return information_loss
    
    def jaccard_similarity(self):
        """
        Compute the Jaccard similarity between the original and reflected images.

        Returns:
            float: Jaccard similarity.
        """
        if self.reflected_image is None:
            return None
        # Compute Jaccard similarity
        intersection = np.sum(self.image & self.reflected_image)
        union = np.sum(self.image | self.reflected_image)

        if union == 0:
            return 0  # Handle division by zero
        else:
            return intersection / union

    def boundary_alignment(self):
        """
        Compute the boundary alignment metric.

        Returns:
            float: Boundary alignment metric.
        """
        if self.reflected_image is None:
            return None
        
        # Use edge detection to find object boundaries
        edges_original = cv.Canny(self.image, 100, 200)
        edges_reflected = cv.Canny(self.reflected_image, 100, 200)

        # Compute the percentage of aligned boundary pixels
        aligned_pixels = np.sum(edges_original == edges_reflected)
        total_boundary_pixels = np.sum(edges_original)
        boundary_alignment = aligned_pixels / total_boundary_pixels if total_boundary_pixels > 0 else 0
        return boundary_alignment
    