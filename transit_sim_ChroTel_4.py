# %% # Transit Simulation
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

fits_file = '2014_04_14_ChroTel_2.fits'  # replace with your FITS file name
with fits.open(fits_file) as hdul:
    data = hdul[0].data
    header = hdul[0].header

data1 = data[6, :, :] * header['CCDGAIN']

plt.figure(dpi=300)
plt.imshow(data1, cmap='gray', origin='lower')
plt.colorbar(label='Intensity [counts]')
plt.show()


#%% Step 1: Open the FITS file and load the data

fits_file = '2014_04_14_ChroTel_2.fits'  # replace with your FITS file name
with fits.open(fits_file) as hdul:
    data = hdul[0].data
    header = hdul[0].header

# Data has 7 planes representing the 7 bandpass Filters (see DOI: 10.1051/0004-6361/201117456)
# Each set of filtergrams inlevel 1.0 data is ordered with decreasing wavelength. (see https://doi.org/10.1002/asna.201813536)
# planes 0,1 and 5,6 will be used for continuum intensity
# plane 3 will be used for helium intensity

filter_data = {
    10833.15: data[0, :, :],
    10832.13: data[1, :, :],
    10831.00: data[2, :, :],
    10830.30: data[3, :, :],
    10829.60: data[4, :, :],
    10828.47: data[5, :, :],
    10827.45: data[6, :, :],
}

# Convert ADUs to electron count
factor = header['CCDGAIN']

# Rotate images so that solar north faces positive y
angle = 0

# Apply to all filters
for wavelength in filter_data:
    
    # Apply conversion and rotation
    filtered_image = filter_data[wavelength].astype(np.float64)  # Convert to float
    filtered_image *= factor  # Multiply by the factor

    background_removed_image = filtered_image

    # Plot the filtered image with background removed
    # plt.figure(dpi=300)
    # plt.imshow(background_removed_image, cmap='gray', origin='lower')
    # plt.colorbar(label='Intensity [counts]')
    # plt.title(f'Intensity at Wavelength {wavelength} Å')
    # plt.show()
    
    x_center = 1000
    y_center = 1000
    radius = 1000
    
    # Calculate Pixel to distance ratio
    MetrePerPixel = R_sun.value / radius
    AUPerPixel = MetrePerPixel / au.to(u.m)
    
    # Store the center, radius, and wavelength information directly in the filter_data dictionary
    filter_data[wavelength] = {
        'wavelength': wavelength,  # Store the wavelength
        'center': (y_center, x_center),
        'radius': radius,
        'AUPerPixel': AUPerPixel,
        'ImageData': background_removed_image,  # Keep the image data after background removal
    }


#%% Step 2: apply object aperture to test transit

normalised_flux_list = []
error_list = []

for wavelength in filter_data:
    
    # Define object at center
    object_center = filter_data[wavelength]['center']
    object_radius = 93   
    object_aperture = CircularAperture(object_center, r=object_radius)
    
    # Extract image data
    image_data = filter_data[wavelength]['ImageData']
    
    # Apply aperture photometry
    transit_object = aperture_photometry(image_data, object_aperture)
    transit_flux = transit_object['aperture_sum'][0]
    sun_flux = np.sum(image_data)
    normalised_flux =  (sun_flux - transit_flux) / sun_flux
    normalised_flux_list.append(normalised_flux)
    
    # Calculate errors
    numerator_error_per = np.sqrt(sun_flux + transit_flux) / (sun_flux + transit_flux)
    denominator_error_per = np.sqrt(sun_flux) / sun_flux
    normalised_flux_error_per = np.sqrt(numerator_error_per ** 2 + denominator_error_per ** 2)
    normalised_flux_error = normalised_flux * normalised_flux_error_per
    error_list.append(normalised_flux_error)
    
    # Create the plot
    plt.figure(dpi=300)
    plt.imshow(image_data, cmap='gray', origin='lower')
    
    # Overlay the aperture
    object_aperture.plot(color='red', lw=1.5, label='Aperture')
    
    # Add colorbar and title
    plt.colorbar(label='Intensity [counts]')
    plt.title(f'Intensity at Wavelength {wavelength} Å')
    
    # Add normalized flux loss to the legend
    plt.legend()
    
    # Show the plot
    plt.show()

normalised_flux_array = np.array(normalised_flux_list)
error = np.array(error_list)

#%% Orbital Sim
def findMeanAnomaly(t, t0, P):
    return 2 * np.pi / P * (t-t0)


def f(E, M, e):
    return E - e * np.sin(E) - M


def f_prime(E, e):
    return 1 - e * np.cos(E)


# through Newton method
def findEccentricAnomaly(M_array, e, initial_guess, tol=1e-8, max_iter=100):
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

#%% Step 3 :create position arrays for each filter

# Use semi major axis in units of pixels, Period in units of Hours
P = 24  # 24-hour period
T_earth_hours = 365.25 * 24  # Earth's period in hours
a_AU = (P / T_earth_hours) ** (2/3)
e = 0.
t = np.linspace(0, P, 1000)
t0 = 0
i = 90

for wavelength in filter_data:
    
    # define semi major axis in pixels
    a = a_AU / filter_data[wavelength]['AUPerPixel']
    
    MeanAnomalyE = findMeanAnomaly(t, t0, P)
    EccentricAnomaly = findEccentricAnomaly(MeanAnomalyE, e, MeanAnomalyE)
    TrueAnomalyDashE = findTrueAnomalyDash(EccentricAnomaly, e)
    TrueAnomalyE = findTrueAnomaly(EccentricAnomaly, TrueAnomalyDashE)
    RadiusFromFocusE = RadiusFromFocus(a, e, TrueAnomalyE)
    x_values, y_values = PolartoCartesian(RadiusFromFocusE, TrueAnomalyE)
    x, y, z = RotatePointsAroundY(x_values, y_values, np.zeros_like(x_values), i)
    
    x_center = filter_data[wavelength]['center'][0]
    y_center = filter_data[wavelength]['center'][1]
    y_transpose = y + x_center
    z_transpose = z + y_center
    
    df = pd.DataFrame({
        'y (Pixels)': y_transpose,
        'z (Pixels)': z_transpose,
        't (hours)': t
    })
    
    n = len(df)
    first_quarter_end_index = int(n // 4)
    last_quarter_start_index = int((3 * n) // 4)
    
    # Filter the DataFrame based on the conditions, then slice the resulting DataFrame
    first_quarter = df.iloc[:first_quarter_end_index].loc[
        (df['y (Pixels)'] >= x_center) & (
            df['y (Pixels)'] < 3100)
    ]

    last_quarter = df.iloc[last_quarter_start_index:].loc[
        (df['y (Pixels)'] > -
         1100) & (df['y (Pixels)'] < x_center)
    ]
    last_quarter['t (hours)'] = last_quarter['t (hours)'] - P

    filtered_df = pd.concat([first_quarter, last_quarter]).sort_values(
        by='y (Pixels)', ascending=True).reset_index(drop=True)
    
    # fig = plt.figure(dpi=300)
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(x, y, np.zeros_like(x), label = "continuum before rotation", s = 3)

    # # Plot the rotated points in red
    # ax.axis('equal')
    # ax.scatter(x, y, z,label = "continuum after rotation", s = 1)
    # ax.legend()
    # ax.set_xlabel('x (Pixels)')
    # ax.set_ylabel('y (Pixels)')
    # ax.set_zlabel('z (Pixels)')
    # plt.show()
    
    # put information into dictionary
    filter_data[wavelength].update({
        'y (Pixels)': filtered_df['y (Pixels)'],
        'z (Pixels)': filtered_df['z (Pixels)'],
        't (hours)': filtered_df['t (hours)']
    })

#%% Step 4: Simulate each light curve

# Ratios of Jupiter, Neptune, Earth size planets
Rj_Rs = 69911 / 696340
Rn_Rs = 24622 / 696340
Re_Rs = 6371 / 696340

for wavelength in filter_data:
    
    # define size regime
    radius_ratio = Rj_Rs
    
    # Create array for object fluxes and object positions
    object_fluxes = []
    object_positions = list(
        zip(filter_data[wavelength]['y (Pixels)'], filter_data[wavelength]['z (Pixels)']))
    object_radius = radius_ratio * filter_data[wavelength]['radius']
    a = a_AU / filter_data[wavelength]['AUPerPixel']
    radius = filter_data[wavelength]['radius'] 
    
    # APERTURE METHOD for object fluxes
    for position in object_positions:

        # Create an aperture for the current object position
        object_aperture = CircularAperture(position, r=object_radius)

        # Perform aperture photometry for the current object
        object_photometry = aperture_photometry(
            filter_data[wavelength]['ImageData'], object_aperture)

        # Extract the summed flux value for the current object and store it in the list
        object_flux = object_photometry['aperture_sum'][0]
        object_fluxes.append(object_flux)

    object_fluxes = [0 if np.isnan(
        x) else x for x in object_fluxes]

    object_fluxes_array = np.array(object_fluxes)
    
    # # Plot graph
    # fig, ax = plt.subplots(dpi = 300)  # Create a figure and an axis

    # # Display the image on the axis
    # im = ax.imshow(filter_data[wavelength]['ImageData'], cmap='gray', origin='lower')

    # # Set the title
    # ax.set_title("Position of Transit on Disk")

    # # Plot Centres of occulting body
    # ax.plot(filter_data[wavelength]['y (Pixels)'], filter_data[wavelength]['z (Pixels)'], '.', ms = 1)
    # # Add a colorbar to the figure
    # fig.colorbar(im, ax=ax)
    # #ax.set_xlim(0,2048)
    # # Show the plot
    # plt.show()

    # put information into dictionary
    filter_data[wavelength].update({
        'ObjectFLux': object_fluxes_array,
    })

#%% Plot each light curve

# plot seperately
for wavelength in filter_data:
    
    time = filter_data[wavelength]['t (hours)']
    sun_flux =  np.sum(filter_data[wavelength]['ImageData'])
    Normalised_flux = (sun_flux - filter_data[wavelength]['ObjectFLux']) / sun_flux
    
    plt.figure(dpi=300)
    plt.plot(time, Normalised_flux, label = f'{wavelength} Å')
    plt.xlabel('Time (hours)')
    plt.ylabel('Normalized Flux')
    plt.legend()
    plt.show

# plot on same axis

plt.figure(dpi=300)

for wavelength in filter_data:
    
    time = filter_data[wavelength]['t (hours)']
    sun_flux =  np.sum(filter_data[wavelength]['ImageData'])
    Normalised_flux = (sun_flux - filter_data[wavelength]['ObjectFLux']) / sun_flux
    
    plt.plot(time, Normalised_flux, label = f'{wavelength} Å')
    plt.show

# Add labels, legend, and title
plt.xlabel('Time (hours)')
plt.ylabel('Normalized Flux')
plt.legend()
plt.show()

#%% simulate noise in data

for wavelength in filter_data:
    
    # calculate errors for each flux measurement
    sun_flux =  np.sum(filter_data[wavelength]['ImageData'])
    Normalised_flux = (sun_flux - filter_data[wavelength]['ObjectFLux']) / sun_flux
    time = filter_data[wavelength]['t (hours)']
    
    sun_flux_error_per = np.sqrt(sun_flux) / sun_flux
    measured_flux_error_per = np.sqrt(sun_flux + filter_data[wavelength]['ObjectFLux']) / (sun_flux - filter_data[wavelength]['ObjectFLux'])
    normalized_flux_error_per = np.sqrt(measured_flux_error_per ** 2 + sun_flux_error_per ** 2)
    normalised_flux_error = normalized_flux_error_per * Normalised_flux * 0
    
    noisy_normalised_flux = np.random.normal(Normalised_flux, 0) #  error should be normalised_flux_error
    residuals = noisy_normalised_flux - Normalised_flux
    
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, dpi=300, gridspec_kw={'height_ratios': [4, 2]}, figsize = (5.5,5.5))
   
    # ax1.errorbar(time, noisy_normalised_flux,yerr = normalised_flux_error, fmt = 'o',
    #              ms = 2, label = f'{wavelength} Å')
    # ax1.set_ylabel('Normalized Flux')
    # ax1.legend()
    
    # ax2.errorbar(time, 
    #             residuals * 100, 
    #             yerr=normalised_flux_error * 100, 
    #             fmt='o',        
    #             ms = 2,
    #             )        
    
    # ax2.fill_between(time, 
    #                   - normalised_flux_error * 100,
    #                  + normalised_flux_error * 100,
    #                 color='yellow', alpha=0.75, label='1 $\sigma$')
    
    # ax2.set_ylabel('Residuals (%)')   
    # ax2.legend()
    
    # put information into dictionary
    filter_data[wavelength].update({
        'NoisyNormalisedFlux': noisy_normalised_flux,
    })

#%% Step 5: Fit light curves

for wavelength in filter_data:
    
    # Extract relevant information
    a = a_AU / filter_data[wavelength]['AUPerPixel'] 
    radius = filter_data[wavelength]['radius'] 
    noisy_data = filter_data[wavelength]['NoisyNormalisedFlux']
    time = np.array(filter_data[wavelength]['t (hours)'])
    
    # calculate errors for each flux measurement
    sun_flux =  np.sum(filter_data[wavelength]['ImageData'])
    Normalised_flux = (sun_flux - filter_data[wavelength]['ObjectFLux']) / sun_flux
    
    sun_flux_error_per = np.sqrt(sun_flux) / sun_flux
    measured_flux_error_per = np.sqrt(sun_flux + filter_data[wavelength]['ObjectFLux']) / (sun_flux - filter_data[wavelength]['ObjectFLux'])
    normalized_flux_error_per = np.sqrt(measured_flux_error_per ** 2 + sun_flux_error_per ** 2)
    normalised_flux_error = normalized_flux_error_per * Normalised_flux
    
    # define fit model
    def LightcurveFit(time, rp):
        # Create a TransitParams object
        params = batman.TransitParams()

        # Set fixed parameters.

        params.per = P          # Fixed orbital period
        params.a = a / radius  # Fixed semi-major axis / star radius
        params.inc = i          # Fixed inclination (degrees)
        params.ecc = e           # Fixed eccentricity
        params.w = 90.0        # Fixed longitude of periastron (degrees)
        
        # Set variable parameters
        params.t0 = t0           # Fixed time of central transit
        params.rp = rp              # Variable: planet radius / star radius
        params.limb_dark = "uniform"  # Limb darkening model
        params.u = []

        # Calculate the light curve using the parameters
        m = batman.TransitModel(params, time)
        return m.light_curve(params)

    # initial parameters and bounds
    initial_rp = radius_ratio
    initial_t0 = 0
    # Initial parameter array for curve_fit and bounds
    p0 = [initial_rp]
    lower_bounds = [0]
    upper_bounds = [1]

    # Fit noisy data
    fit, cov = curve_fit(LightcurveFit, time, noisy_data,p0 = p0, sigma = normalised_flux_error, bounds = (lower_bounds, upper_bounds))
    
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, dpi=300, gridspec_kw={'height_ratios': [4, 2]}, figsize = (5.5,5.5))
   
    ax1.errorbar(time, noisy_data,yerr = normalised_flux_error, fmt = 'o',
                 ms = 2, label = f'{wavelength} Å')
    ax1.errorbar(time, LightcurveFit(time, *fit))
    
    ax1.set_ylabel('Normalized Flux')
    ax1.legend()
    
    residuals = noisy_data - LightcurveFit(time, *fit)
    
    ax2.errorbar(time, 
                residuals * 100, 
                yerr=normalised_flux_error * 100, 
                fmt='o',        
                ms = 2,
                )        
    
    ax2.set_ylabel('Residuals (%)')   
    ax2.set_xlabel('Time (Hours)')   
    
    # put information into dictionary
    filter_data[wavelength].update({
        'Rp_Rs': fit[0],
        'Rp_Rs_error': np.sqrt(cov[0][0]),
    })
        
#%% Step 6: plot transmission spectrum

plt.figure(dpi=300)

for wavelength in filter_data:
    
    plt.errorbar(wavelength,  filter_data[wavelength]['Rp_Rs'],
                 yerr = filter_data[wavelength]['Rp_Rs_error'],
                 fmt = 'o')

# Add labels, legend, and title
plt.axhline(y = radius_ratio, color = "red")    
#plt.ylim(0.10035, 0.10045)
plt.xlabel('Wavelength Å')
plt.ylabel('PSRR')
plt.show()




#%% Take each PSRR value into array

PSRR_array = np.array([data['Rp_Rs'] for data in filter_data.values()])
wavelengths =  np.array([data['wavelength'] for data in filter_data.values()])

def gaussian(x, amplitude, mean, stddev, c):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) + c

p0 = [0.0002, 10830, 1, 0.1]
fit_gaus, cov_gaus = curve_fit(gaussian, wavelengths, PSRR_array, p0 = p0)

x_array = np.linspace(10827, 10834, 1000)
plt.plot(x_array, gaussian(x_array, *fit_gaus))
plt.plot(wavelengths, PSRR_array, 'x')
plt.xlabel('Wavelength Å')
plt.ylabel('PSRR')
plt.show()

print(f'error percentage in transit depth {(fit_gaus[0] / fit_gaus[3] * 100)**2} %')

# %%
'''



#%% FIND INCLINTATION ANGLE RANGE   FOR EACH PLANET REGIME

b = a * np.cos(i/ 180 * np.pi) / sun_radius # normalised impact parameter
D = Re_Rs * 2
b_array_e = np.arange(-1 + D/2, 1 - D/2, D)
cos_i = b_array_e * sun_radius / a
i_array_e_rad = np.arccos(cos_i)
i_array_e = np.degrees(i_array_e_rad)

# The inclination angles are found such that the full planet disk is captured transiting the
# stellar disk, i.e. no grazing transits
















#%% Find Rp/Rs for Different Inclinations

PSRRArray = []
ErrorArray = []
 
# Ratios of Jupiter, Neptune, Earth size planets
Jupiter = 69911 / 696340
Neptune = 24622 / 696340
Earth = 6371 / 696340

# Define which planet regime and its inclination array 
PSRR = Jupiter
InclinationArray = i_array_e
total_iterations = len(InclinationArray)

for iteration, inclination in enumerate(InclinationArray, start=1):

    ##################################### Simulate Orbit ###################################################
    
    a = 20 * radius # 20 Solar radii in pixels
    a_AU = a * AUPerPixel
    e = 0.0
    Te_ae = (365.25) ** 2 / 1 **3
    P = np.sqrt(a_AU ** 3) *365.25 *24 # period in hours
    t = np.linspace(0,P,5000)
    t0 = 0
    i = inclination
    
    MeanAnomalyE = findMeanAnomaly(t,t0,P)
    EccentricAnomaly = findEccentricAnomaly(MeanAnomalyE,e,MeanAnomalyE)
    TrueAnomalyDashE = findTrueAnomalyDash(EccentricAnomaly, e)
    TrueAnomalyE = findTrueAnomaly(EccentricAnomaly, TrueAnomalyDashE)
    RadiusFromFocusE = RadiusFromFocus(a,e,TrueAnomalyE)
    x_values, y_values = PolartoCartesian(RadiusFromFocusE, TrueAnomalyE)
    
    x, y, z = RotatePointsAroundY(x_values, y_values, np.zeros_like(x_values), i)
    
    ################################## Simulate Light Curve ################################################
    
    y_transpose = y + x_center
    z_transpose = z + y_center
    df = pd.DataFrame({
        'x (Pixels)': x,
        'y (Pixels)': y_transpose,
        'z (Pixels)': z_transpose,
        't (Days)': t
    })
    
    n = len(df)
    first_quarter_end_index = int(n // 4)
    last_quarter_start_index = int((3 * n) // 4)
    
    # Filter the DataFrame based on the conditions, then slice the resulting DataFrame
    first_quarter = df.iloc[:first_quarter_end_index].loc[
        (df['y (Pixels)'] >= x_center) & (df['y (Pixels)'] < 3000)
    ]

    last_quarter = df.iloc[last_quarter_start_index:].loc[
        (df['y (Pixels)'] > -1000) & (df['y (Pixels)'] < x_center)
    ]
    last_quarter['t (Days)'] = last_quarter['t (Days)'] - P
    
    filtered_df= pd.concat([first_quarter, last_quarter]).sort_values(by='y (Pixels)', ascending=True).reset_index(drop=True)
    
    object_positions = list(zip(filtered_df['y (Pixels)'], filtered_df['z (Pixels)']))
    object_radius = PSRR * sun_radius
    
    integrated_fluxes_masked = []
    
    for position in object_positions: # MASK METHOD
        # Create a circular mask for the Sun's disk
        ny, nx = plane2.shape  # extracts the height (ny) and width (nx) of the 2D image (plane2)
        y_grid, x_grid = np.ogrid[:ny, :nx]  # Create  two separate arrays that correspond to the row indices (y) and column indices (x) of the image
        
        # create mask
        sun_region_masked = sun_region.copy()
        
        distance_from_center2 = np.sqrt((x_grid - position[0])**2 + (y_grid - position[1])**2)
        mask2 = distance_from_center2 <= object_radius
        sun_region_masked[mask2] = 0
    
        integrated_flux_masked = sun_region_masked.sum()
        integrated_fluxes_masked.append(integrated_flux_masked)
        
    Transmission_array_2 = integrated_fluxes_masked / integrated_flux_sun
    
    per_error_integrated_fluxes_masked = np.sqrt(integrated_fluxes_masked) / integrated_fluxes_masked
    per_error_integrated_flux_sun = np.sqrt(integrated_flux_sun) / integrated_flux_sun
    per_error_transmission_array_2 = np.sqrt(per_error_integrated_fluxes_masked ** 2 + per_error_integrated_flux_sun ** 2)
    
    filtered_df['Transmission (%) 2'] = Transmission_array_2 # MASK METHOD
    filtered_df['Transmission (%) 2 error'] =per_error_transmission_array_2 * filtered_df['Transmission (%) 2']
    
    transmission_range = max(filtered_df['Transmission (%) 2']) - min(filtered_df['Transmission (%) 2'])
    
    noisy_transmission_2 = np.random.normal(filtered_df['Transmission (%) 2'], filtered_df['Transmission (%) 2 error'])
    filtered_df['Noisy Transmission (%) 2'] = noisy_transmission_2
    
    b = a * np.cos(i/ 180 * np.pi) / sun_radius
    arg = (sun_radius / a) * np.sqrt((1 + object_radius / sun_radius)**2 - b ** 2)
    
    T = (P / np.pi) * np.arcsin(arg)
    filtered_df = filtered_df[(filtered_df['t (Days)'] >= - T) & (filtered_df['t (Days)'] <= + T)].reset_index(drop=True)
    
    ##################################### Fit Light Curve ##################################################
    
    # Transmission and Time
    Transmission = np.array(filtered_df['Noisy Transmission (%) 2'])
    time = np.array(filtered_df['t (Days)'])
    errors = filtered_df['Transmission (%) 2 error']
    
    # Define the transit model parameters
    def fit_func(time, rp, c1, c2):  
        # Create a TransitParams object
        params = batman.TransitParams()
    
        # Set fixed parameters
        params.t0 = t0           # Fixed time of central  
        params.per = P          # Fixed orbital period
        params.a = a / sun_radius  # Fixed semi-major axis / star radius
        params.inc = i          # Fixed inclination (degrees)
        params.ecc = e           # Fixed eccentricity
        params.w = 90.0        # Fixed longitude of periastron (degrees)
    
        # Set variable parameters
        params.rp = rp              # Variable: planet radius / star radius
        params.limb_dark = "power2" # Limb darkening model
        params.u = [c1, c2]      # Variable: limb darkening coefficients
        
        # 0.9036 -0.2312 limb darkeining 
    
    
        # Calculate the light curve using the parameters
        m = batman.TransitModel(params, time)    
        return m.light_curve(params)
    
    # Provide initial guesses for the parameters
    initial_rp = object_radius / sun_radius   # Initial guess for the planet radius / star radius
    initia_c1 = 1
    initia_c2 = 1
    initia_c3 = 1
    initia_c4 = 1
    initial_t0 = 0
    # Initial parameter array for curve_fit and bounds
    p0 = [initial_rp, initia_c1, initia_c2]
    lower_bounds = [0, -np.inf, -np.inf]
    upper_bounds = [1, np.inf, np.inf]
    
    fit_params, covariance = curve_fit(fit_func, time, Transmission,sigma = errors, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=10000)
    
    PSRRArray.append(fit_params[0])
    ErrorArray.append(np.sqrt(covariance[0][0]))
    
    # Print the current iteration out of the total number of iterations
    print(f"Iteration {iteration} out of {total_iterations} complete")
    
#%% Save array as excel spreadsheet
parameter = 'J'
date = '15_04_14'
import pandas as pd






data = {
    'Inclination': InclinationArray,  
    'PSRR': PSRRArray,  
    'Error': ErrorArray  
}
df = pd.DataFrame(data)
filename = f"PSRR_{parameter}_{date}_power2.xlsx"
#df.to_excel(filename, index=False)

b_array = a * np.cos(df['Inclination']/ 180 * np.pi) / sun_radius


fig, ax1 = plt.subplots(dpi=300)

ax1.errorbar(b_array, 
            df['PSRR'], 
            yerr=df['Error'], 
            label="Data", 
            fmt='o',          # 'o' for circular markers
            ms = 2,
            color = 'black',#'w
            alpha=1,
            elinewidth=1, 
            capsize=0)   

ax1.axhline(y = PSRR,color = 'red', label = "True PSRR")
ax1.set_ylabel('PSRR')
ax1.set_xlabel('Impact Parameter')              
ax1.set_xlim(-1, 1)
#ax1.set_ylim(0.10025, 0.1005)
ax1.axhspan(PSRR * 0.999, PSRR * 1.001, color='red',lw = 0, alpha=0.2, label="")

ax1.legend(frameon = False)





'''
