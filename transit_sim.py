# Transit Simulation

import numpy as np
import matplotlib as plt 
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy.units import au
from astropy.constants import R_sun
from photutils.aperture import CircularAperture, aperture_photometry

# Step 1: Open the FITS file and load the data
fits_file = '18_07_13.fts'  # replace with your FITS file name
with fits.open(fits_file) as hdul:
    data = hdul[0].data

# Step 2: Extract the second plane (Continuum Intensity)
plane2 = data[1, :, :]  # Continuum Intensity (2D array)


# Step 2.1: Find centre and radius of Sun

from skimage import filters, measure

# Apply a threshold to segment the Sun from the background
threshold_value = filters.threshold_otsu(plane2)
binary_image = plane2 > threshold_value

# Find contours in the thresholded image
contours = measure.find_contours(binary_image, level=0.5)

# Identify the largest contour (should correspond to the Sun's disk)
largest_contour = max(contours, key=len)

# Calculate the center and radius
y_center = np.mean(largest_contour[:, 0])
x_center = np.mean(largest_contour[:, 1])
radii = np.sqrt((largest_contour[:, 0] - y_center) ** 2 + (largest_contour[:, 1] - x_center) ** 2)
radius = np.mean(radii)

# Step 3: Define the apertures
# Whole Sun aperture
sun_center = (x_center, y_center)  # example coordinates of the Sun's center
sun_radius = radius         # example radius of the Sun

# Occulting object aperture
object_center = (1000, 1000)  # example coordinates of the object's center
object_radius = 50          # exa  mple radius of the occulting object

# Create the apertures
sun_aperture = CircularAperture(sun_center, r=sun_radius)
object_aperture = CircularAperture(object_center, r=object_radius)

# Step 4: Perform aperture photometry on the Continuum Intensity plane
sun_photometry = aperture_photometry(plane2, sun_aperture)
object_photometry = aperture_photometry(plane2, object_aperture)

# Extract the summed flux values
sun_flux = sun_photometry['aperture_sum'][0]
object_flux = object_photometry['aperture_sum'][0]

# Step 5: Subtract the flux of the occulting object from the total flux of the Sun
visible_flux = sun_flux - object_flux

# Display the results
print(f"Total flux of the Sun (Continuum Intensity): {sun_flux}")
print(f"Flux of the occulting object (Continuum Intensity): {object_flux}")
print(f"Visible flux after occultation (Continuum Intensity): {visible_flux}")
normalized_flux_percentage = (object_flux / sun_flux) * 100
print(f"Normalized Flux Occulted: {normalized_flux_percentage:.4f}%")

# Optional: Display the Continuum Intensity image with the apertures overlaid
plt.imshow(plane2, cmap='gray', origin='lower')
sun_aperture.plot(color='red', lw=1, alpha=0.5)
object_aperture.plot(color='blue', lw=1, alpha=0.5)
plt.title("Continuum Intensity with Apertures")
plt.colorbar()
plt.show()

# Calculate Pixel to distance ratio
MetrePerPixel = R_sun.value / radius
AUPerPixel = MetrePerPixel / au.to(u.m)
print(f"Estimate of Metres per Pixel: {MetrePerPixel:.4f}")
#%% Masking Planet and Sun

import numpy as np
import matplotlib as plt 
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy.units import au
from astropy.constants import R_sun
from photutils.aperture import CircularAperture, aperture_photometry

# Step 1: Open the FITS file and extract the data
file_path = '18_07_13.fts'
with fits.open(file_path) as hdul:
    data = hdul[0].data  # Extract the 3D data array
    
# Extract the second 2D plane from your data (Continuum Intensity)
plane2 = data[1, :, :]  # Shape (2048, 2048)

from skimage import filters, measure

# Apply a threshold to segment the Sun from the background
threshold_value = filters.threshold_otsu(plane2)
binary_image = plane2 > threshold_value

# Find contours in the thresholded image
contours = measure.find_contours(binary_image, level=0.5)

# Identify the largest contour (should correspond to the Sun's disk)
largest_contour = max(contours, key=len)

# Calculate the center and radius
y_center = np.mean(largest_contour[:, 0])
x_center = np.mean(largest_contour[:, 1])
radii = np.sqrt((largest_contour[:, 0] - y_center) ** 2 + (largest_contour[:, 1] - x_center) ** 2)
radius = np.mean(radii)

print(f"Center: ({x_center}, {y_center}), Radius: {radius}")

# Create a circular mask for the Sun's disk
ny, nx = plane2.shape  # extracts the height (ny) and width (nx) of the 2D image (plane2)
y, x = np.ogrid[:ny, :nx]  # Create  two separate arrays that correspond to the row indices (y) and column indices (x) of the image

# Create a circular mask for Sun
distance_from_center = np.sqrt((x - x_center)**2 + (y - y_center)**2)
mask = distance_from_center <= radius

# Apply the mask to the image (set background pixels to zero)
sun_region = np.where(mask, plane2, 0)

# Calculate the integrated flux of the Sun (sum of the pixels within the disk)
integrated_flux_sun = sun_region.sum()
sun_flux = integrated_flux_sun

sun_region_masked = sun_region.copy()

# Create Planet Mask

object_center = (1000, 1000)  # example coordinates of the object's center
object_radius = 50          # exa  mple radius of the occulting object

distance_from_center2 = np.sqrt((x - object_center[1])**2 + (y - object_center[0])**2)
mask2 = distance_from_center2 <= object_radius
sun_region_masked[mask2] = 0

integrated_flux_masked = sun_region_masked.sum()

# Print the result
print(f"Integrated Flux of the Sun: {integrated_flux_sun} counts")
print(f"Integrated Flux of the Sun with Dot: {integrated_flux_masked} counts")

# Step 4: Plot the modified image with the circular black dot
plt.figure(figsize=(8, 8))
plt.imshow(sun_region_masked, cmap='gray', origin='lower')
plt.colorbar(label='Continuum Intensity [counts]')
plt.xlabel('X Pixels')
plt.ylabel('Y Pixels')
plt.show()
#%% Calculate Limb Darkening Coefficients

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.integrate import quad

def nonlinear(mu, c1, c2, c3, c4):
    return 1 - (c1 * (1 - mu**(1/2)) +
                 c2 * (1 - mu) +
                 c3 * (1 - mu**(3/2)) +
                 c4 * (1 - mu**2))

def quadratic(mu, u1, u2):
    """Quadratic limb darkening model."""
    return (1 - u1 * (1 - mu) - u2 * (1 - mu)**2)


intensity_array = sun_region
central_intensity = intensity_array[int(y_center), int(x_center)]
normalized_intensity_array = intensity_array / central_intensity


# Get the dimensions of the array
ny, nx = normalized_intensity_array.shape

# Create a grid of distances from the center
y, x = np.indices((ny, nx))
r = np.sqrt((x - x_center)**2 + (y - y_center)**2)
r_normalized = r / radius

# Mask to only include points within the Sun's disk
mask = r_normalized <= 1
  
# Average the intensity in radial bins
num_bins = 100
bins = np.linspace(0, 1, num_bins)
radial_profile = np.zeros(num_bins - 1)
radial_profile_error = np.zeros(num_bins - 1)
for i in range(num_bins - 1):
    bin_mask = (r_normalized >= bins[i]) & (r_normalized < bins[i + 1]) & mask
    bin_values = normalized_intensity_array[bin_mask]
    radial_profile[i] = np.mean(bin_values)
    
    # Calculate the standard error for the current bin
    if len(bin_values) > 0:
        radial_profile_error[i] = np.std(bin_values) / np.sqrt(len(bin_values))
    else:
        radial_profile_error[i] = 0

# Handle cases where bins might be empty (NaNs)
radial_profile = np.nan_to_num(radial_profile)

bin_centers = (bins[:-1] + bins[1:]) / 2
mu_values = np.sqrt(1 - bin_centers ** 2)
fit_quad, cov_quad = curve_fit(quadratic, mu_values, radial_profile, sigma=radial_profile_error)
fit_nonlinear, _ = curve_fit(nonlinear, mu_values, radial_profile, sigma=radial_profile_error)

print(f'Quadratic Fit: u1={fit_quad[0]:.3f}, u2={fit_quad[1]:.3f}')
print(f'Nonlinear Fit: u1={fit_nonlinear[0]:.3f}, u2={fit_nonlinear[1]:.3f}, u3={fit_nonlinear[2]:.3f}, u3={fit_nonlinear[3]:.3f}')
# Plot the results
plt.figure(dpi = 300)
plt.errorbar(mu_values, radial_profile, yerr=radial_profile_error,capsize = 2, fmt='o', ms = 2, label='Observed radial profile')
plt.plot(mu_values, nonlinear(mu_values, *fit_nonlinear), label=f'Nonlinear Fit')
plt.plot(mu_values, quadratic(mu_values, *fit_quad), label=f'Quadratic Fit')
plt.legend()
plt.xlabel('mu')
plt.ylabel('Normalized Intensity')
plt.legend()
plt.show()



#%% Orbital Sim

import numpy as np
import matplotlib.pyplot as plt
from astropy.units import au
from astropy import units as u
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def findMeanAnomaly(t, t0, P):
    return 2 * np.pi / P * (t-t0)

def f(E, M, e):
    return E - e * np.sin(E) - M

def f_prime(E, e):
    return 1 - e * np.cos(E)

def findEccentricAnomaly(M_array, e, initial_guess, tol=1e-8, max_iter=100): # through Newton method
    E_array = np.zeros_like(M_array)
    for i, M in enumerate(M_array):
        E = initial_guess[i]  
        for _ in range(max_iter):
            f_val = f(E, M, e)
            if np.abs(f_val) < tol:  
                E_array[i] = E
                break
            f_prime_val = f_prime(E, e)
            E = E - f_val / f_prime_val
        else:
            raise ValueError(f"Failed to converge for M = {M}")
    return E_array

def findTrueAnomalyDash(E, e):
    numerator = np.cos(E) - e
    denominator = 1 - e * np.cos(E)
    arg = numerator / denominator
    return np.arccos(arg)

def findTrueAnomaly(E_values, theta_prime_values):
    theta_values = np.zeros_like(E_values)  

    for i, E in enumerate(E_values):
        theta_prime = theta_prime_values[i]
    
        while E >= 2 * np.pi:  # ensures if/else conditions repreated every 2pi
            E -= 2 * np.pi

        if E <= np.pi:
            theta_values[i] = theta_prime
        elif np.pi < E < 2 * np.pi:
            theta_values[i] = 2 * np.pi - theta_prime

    return theta_values

def RadiusFromFocus(a, e, theta):
    r = (a * (1 - e**2)) / (1 + e * np.cos(theta))
    return r

def PolartoCartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def RotatePointsAroundY(x, y, z, inclination):
    # Convert angle from degrees to radians
    anlge_between_orbitalplane_refplane = 90 - inclination
    rad = np.radians(-anlge_between_orbitalplane_refplane)
    
    # Define the rotation matrix for counter-clockwise rotation around the y-axis
    x_rotated = x * np.cos(rad) + z * np.sin(rad)
    y_rotated = y
    z_rotated = - x * np.sin(rad) + z * np.cos(rad)
    
    return x_rotated, y_rotated, z_rotated

# AU per Pixel: 4.946822969244625e-06

a = 1/ AUPerPixel # au.to(u.km) is 1 AU 
e = 0.0
P = 365 # days
t = np.linspace(0,365,100000)

t0 = 0
i = 89.8

MeanAnomalyE = findMeanAnomaly(t,t0,P)
EccentricAnomaly = findEccentricAnomaly(MeanAnomalyE,e,MeanAnomalyE)
TrueAnomalyDashE = findTrueAnomalyDash(EccentricAnomaly, e)
TrueAnomalyE = findTrueAnomaly(EccentricAnomaly, TrueAnomalyDashE)
RadiusFromFocusE = RadiusFromFocus(a,e,TrueAnomalyE)
x_values, y_values = PolartoCartesian(RadiusFromFocusE, TrueAnomalyE)

x, y, z = RotatePointsAroundY(x_values, y_values, np.zeros_like(x_values), i)

fig = plt.figure(dpi=300)
ax = fig.add_subplot()
ax.axis('equal')
ax.plot(x_values,y_values,'.', ms = 1)
ax.plot(x,y,'.', color = 'red',ms = 1)
ax.plot(0,0,'.', color = 'orange')
ax.set_xlabel('x (Pixels)')
ax.set_ylabel('y (Pixels)')
plt.title("X-Y plane View")

fig = plt.figure(dpi=300)
ax = fig.add_subplot()
ax.axis('equal')
ax.plot(y_values,np.zeros_like(x_values),'.', ms = 1)
ax.plot(y,z,'.', color = 'red',ms = 1) # observer view
ax.plot(0,0,'.', color = 'orange')
ax.set_xlabel('z (Pixels)')
ax.set_ylabel('y (Pixels)')
plt.title("Observer View")

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_values, y_values, np.zeros_like(x_values), color='blue', label='Original Points', s = 3)

# Plot the rotated points in red
ax.axis('equal')
ax.scatter(x, y, z, color='red', label='Rotated Points', s = 1)
ax.set_xlabel('x (Pixels)')
ax.set_ylabel('y (Pixels)')
ax.set_zlabel('z (Pixels)')
plt.show()

#%% Lightcurve
import pandas as pd

# x,y,z, as the positions of centre of the occulting body (in pixels)
# we took positive x axis as direction towards observer so we discard that coordinate
# Transpose y and z array such that the initial point is at centre Sun
y_transpose = y + x_center
z_transpose = z + y_center
# Create a dataframe of orbital sim
df = pd.DataFrame({
    'x (Pixels)': x,
    'y (Pixels)': y_transpose,
    'z (Pixels)': z_transpose,
    't (Days)': t
})

n = len(df)
first_quarter_end_index = int(n // 4)
last_quarter_start_index = int((3 * n) // 4)

# Filter the DataFrame for points near transit
first_quarter = df.iloc[:first_quarter_end_index][(df['y (Pixels)'] >= x_center) & (df['y (Pixels)'] < 2200)]
last_quarter = df.iloc[last_quarter_start_index:][(df['y (Pixels)'] > -200) & (df['y (Pixels)'] <= x_center)]
last_quarter['t (Days)'] = last_quarter['t (Days)'] - 365

filtered_df= pd.concat([first_quarter, last_quarter]).sort_values(by='y (Pixels)', ascending=True).reset_index(drop=True)

# Plot graph
fig, ax = plt.subplots(dpi = 300)  # Create a figure and an axis

# Display the image on the axis
im = ax.imshow(plane2, cmap='gray', origin='lower')

# Set the title
ax.set_title("Position of Transit on Stellar Disk")

# Plot Centres of occulting body
ax.plot(filtered_df['y (Pixels)'], filtered_df['z (Pixels)'], '.', ms = 1)

# Add a colorbar to the figure
fig.colorbar(im, ax=ax)

# Show the plot
plt.show()

object_fluxes = []
object_positions = list(zip(filtered_df['y (Pixels)'], filtered_df['z (Pixels)']))
object_radius = 50
integrated_fluxes_masked = []
object_fluxes_mix = []

for position in object_positions:
    # Create a circular mask for the Sun's disk
    ny, nx = plane2.shape  # extracts the height (ny) and width (nx) of the 2D image (plane2)
    y_grid, x_grid = np.ogrid[:ny, :nx]  # Create  two separate arrays that correspond to the row indices (y) and column indices (x) of the image
    
    # create mask
    sun_region_masked = sun_region.copy()
    
    distance_from_center2 = np.sqrt((x_grid - position[1])**2 + (y_grid - position[0])**2)
    mask2 = distance_from_center2 <= object_radius
    sun_region_masked[mask2] = 0

    integrated_flux_masked = sun_region_masked.sum()
    integrated_fluxes_masked.append(integrated_flux_masked)

for position in object_positions:
    
    # Create an aperture for the current object position
    object_aperture = CircularAperture(position, r=object_radius)
    
    # Perform aperture photometry for the current object
    object_photometry = aperture_photometry(sun_region_masked, object_aperture)
    
    # Extract the summed flux value for the current object and store it in the list
    object_flux_mix = object_photometry['aperture_sum'][0]
    object_fluxes_mix.append(object_flux_mix)


# Iterate over each object position to create apertures and perform photometry
for position in object_positions:
    # Create an aperture for the current object position
    object_aperture = CircularAperture(position, r=object_radius)
    
    # Perform aperture photometry for the current object
    object_photometry = aperture_photometry(plane2, object_aperture)
    
    # Extract the summed flux value for the current object and store it in the list
    object_flux = object_photometry['aperture_sum'][0]
    object_fluxes.append(object_flux)

Transmission_array = (sun_flux - object_fluxes) / sun_flux
Transmission_array_2 = -(integrated_flux_sun -integrated_fluxes_masked) / integrated_flux_sun
Transmission_array_3 = (integrated_flux_sun - object_fluxes_mix) / integrated_flux_sun

filtered_df['Transmission (%)'] = Transmission_array - 1 # APERTURE METHOD
filtered_df['Transmission (%) 2'] = Transmission_array_2 # MASK METHOD
filtered_df['Transmission (%) 3'] = Transmission_array_3 - 1  # APERTURE + MASK METHOD
# Filter df to remove NaN
filtered_df['Transmission (%)'] = filtered_df['Transmission (%)'].fillna(0)
filtered_df['Transmission (%) 3'] = filtered_df['Transmission (%) 3'].fillna(0)
transmission_range = max(filtered_df['Transmission (%)']) - min(filtered_df['Transmission (%)'])


fig = plt.figure(dpi=300)
ax = fig.add_subplot()

ax.plot(filtered_df['t (Days)'], filtered_df['Transmission (%)'], label = "Apertrue")
ax.plot(filtered_df['t (Days)'], filtered_df['Transmission (%) 2'],label = "Mask")
ax.plot(filtered_df['t (Days)'], filtered_df['Transmission (%) 3'],label = "APERTURE + MASK")

ax.set_ylabel('Transmission Loss (%)')
ax.set_xlabel('Time (Days)')
ax.legend()
ax.set_xlim(filtered_df['t (Days)'].iloc[0], filtered_df['t (Days)'].iloc[-1])
ax.set_ylim(min(filtered_df['Transmission (%)']) - transmission_range * 0.1, 
            max(filtered_df['Transmission (%)']) + transmission_range * 0.1)
#%% Fitting Light Curve

import numpy as np
import batman
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Transmission and Time
Transmission = np.array(filtered_df['Transmission (%) 3']) + 1
time = np.array(filtered_df['t (Days)'])

# Define the transit model parameters
def fit_func(time, rp):
    # Create a TransitParams object
    params = batman.TransitParams()

    # Set fixed parameters
    params.t0 = t0           # Fixed time of central transit
    params.per = 365           # Fixed orbital period
    params.a = a / sun_radius  # Fixed semi-major axis / star radius
    params.inc = i          # Fixed inclination (degrees)
    params.ecc = e           # Fixed eccentricity
    params.w = 90.0        # Fixed longitude of periastron (degrees)

    # Set variable parameters
    params.rp = rp              # Variable: planet radius / star radius
    params.limb_dark = "nonlinear"  # Limb darkening model
    params.u = fit_nonlinear      # Variable: limb darkening coefficients

    # 0.9036 -0.2312 limb darkeining 


    # Calculate the light curve using the parameters
    m = batman.TransitModel(params, time)    
    return m.light_curve(params)

# Provide initial guesses for the parameters
initial_rp = object_radius / sun_radius   # Initial guess for the planet radius / star radius

# Initial parameter array for curve_fit and bounds
p0 = [initial_rp]
lower_bounds = [0]
upper_bounds = [1]

fit_params, covariance = curve_fit(fit_func, time, Transmission, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=10000)

fig = plt.figure(dpi=300)
ax = fig.add_subplot()

#ax.plot(filtered_df['t (Days)'], filtered_df['Transmission (%)'], 'x', ms = 3, label = "Aperture")
#ax.plot(filtered_df['t (Days)'], filtered_df['Transmission (%) 2'], 'x', ms = 3, label = "Mask")
ax.plot(filtered_df['t (Days)'], filtered_df['Transmission (%) 3'], 'x', ms = 3, label = "Aperture + Mask")
ax.set_ylabel('Transmission Loss (%)')
ax.set_xlabel('Time (Days)')    
ax.set_xlim(filtered_df['t (Days)'].iloc[0], filtered_df['t (Days)'].iloc[-1])
ax.set_ylim(min(filtered_df['Transmission (%)']) - transmission_range *  0.1, 
            max(filtered_df['Transmission (%)']) + transmission_range * 0.1)

ax.plot(time, fit_func(time, *fit_params)-1, label = "Batman Fit")
ax.legend()

transit_depth = fit_params[0] ** 2 * 100

print(f"Original Rp /Rs: {object_radius / sun_radius:.4f}")
print(f"Fitted Rp /Rs: {fit_params[0]:.4f}")
print(f"Transit Depth: {transit_depth:.4f}%")
formatted_coeffs = ', '.join([f'{coeff:.4f}' for coeff in fit_params[1:]])
print(f"Limb Darkening Coefficients: {formatted_coeffs}")







































