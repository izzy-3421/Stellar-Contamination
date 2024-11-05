#%% Estimating magnetic activity
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy.units import au
from astropy.constants import R_sun
from photutils.aperture import CircularAperture, aperture_photometry
from scipy.ndimage import rotate
from photutils.background import Background2D, MedianBackground
from matplotlib.colors import LogNorm
from skimage import filters, measure
import batman
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator

# %% FITS file imformation

fits_file = '2019_12_14_ChroTel_Ca.fits'  # replace with your FITS file name
with fits.open(fits_file) as hdul:
    data = hdul[0].data
    header = hdul[0].header
    
plt.figure(dpi=300)
plt.imshow(data, cmap='gray', origin='lower')
plt.colorbar(label='Intensity [DN]')
plt.title("Ca II K Intensity")
plt.show()

# Define circular mask parameters
center_x, center_y = 1000, 1000
radius = 1000

# Create a circular mask
y, x = np.ogrid[:data.shape[0], :data.shape[1]]
distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
circular_mask = distance_from_center <= radius

# Apply the mask to create masked data
masked_data = np.copy(data)
masked_data[~circular_mask] = 0  # Set pixels outside the mask to zero

# Perform Otsu thresholding only on the non-zero values inside the mask
masked_values = masked_data[circular_mask]
threshold_value = filters.threshold_otsu(masked_values)

# Create a binary image using the threshold, but only for the masked area
binary_image = np.zeros_like(masked_data, dtype=bool)
binary_image[circular_mask] = masked_data[circular_mask] > threshold_value
circular_mask = distance_from_center <= radius
background = np.zeros_like(masked_data, dtype=bool)
background[~circular_mask] = 0.5

plt.figure(dpi=300)
plt.imshow(binary_image + background, cmap='gray', origin='lower')
plt.colorbar(label='Binary Intensity')
plt.title("Plages and Enhanced Network (PEN)")
plt.show()

PEN_area = np.sum(binary_image)

# Apply Otsu's threshold to the entire image
threshold_value = filters.threshold_otsu(data)
binary_image = data > threshold_value  # Sun disk as True, background as False
# Count the pixels in the Sun's disk (where binary_image is True)

sun_disk_area = np.sum(binary_image)
A_PEN = PEN_area / sun_disk_area

print(f"A_PEN: {PEN_area / sun_disk_area}")

def APENToS(x):
    S = (x + 0.57) / 3.55
    error = S * np.sqrt((0.01/0.57) ** 2 + (0.06/3.55) ** 2)
    return S, error 

print(f"S-index: {APENToS(A_PEN)[0]:.5f} ± {APENToS(A_PEN)[1]:.5f}")









