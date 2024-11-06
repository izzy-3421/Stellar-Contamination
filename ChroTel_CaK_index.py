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
import os
from datetime import datetime
import matplotlib.dates as mdates

# %% FITS file imformation

fits_file = 'chrotel-ca_l2_20160817T083000_zic_v1.fits'  # replace with your FITS file name
with fits.open(fits_file) as hdul:
    data = hdul[0].data
    header = hdul[0].header

plt.figure(dpi=300)
plt.imshow(data, cmap='gray', origin='lower')
plt.colorbar(label='Intensity [DN]')
plt.title("Ca II K Intensity")
plt.show()

# Define the center coordinates
cx, cy = 1000, 1000  # Center x and y coordinates (provided)

# Define the radius as half the smaller dimension (to fit within the image)
ny, nx = data.shape
radius = 1000

# Define the 0.99 radius threshold
radius_threshold = 0.99 * radius

# Create a grid of coordinates with respect to the center
y, x = np.ogrid[:ny, :nx]
distance_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

# Create the mask
mask = distance_from_center <= radius_threshold

# Now 'mask' is True for pixels within 0.99 of the radius, False outside
filtered_data = np.where(mask, data, np.nan)

# Mask data to remove points where intensity is greater than 3
filtered_data = filtered_data[filtered_data <= 3]

# Plot histogram with bin size of 0.01
bin_heights, bin_edges, _ = plt.hist(filtered_data, bins=np.arange(min(filtered_data), max(filtered_data) + 0.01, 0.01),
                                     color ='blue', alpha = 0.5)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

plt.figure(dpi = 300)
# Add labels and title
plt.errorbar(bin_centers, bin_heights, yerr = np.sqrt(bin_heights), fmt = 'o', ms = 3)
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.title('Histogram of Selected Data (Intensity <= 3)')
plt.show()

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-0.5 * ((x - mean) / stddev)**2)

center_bin_index = np.argmax(bin_heights)

# Step 2: Select the first 35 bins on either side of the center bin
half_window = 35
start_bin = max(0, center_bin_index - half_window)
end_bin = min(len(bin_centers), center_bin_index + half_window + 1)

# Subset the data to only include these bins
selected_bin_centers = bin_centers[start_bin:end_bin]
selected_bin_heights = bin_heights[start_bin:end_bin]

# Step 3: Initial guess for the parameters: [amplitude, mean, stddev]
initial_guess = [max(selected_bin_heights), bin_centers[center_bin_index], np.std(selected_bin_centers)]

# Perform curve fitting
fit, cov = curve_fit(gaussian, selected_bin_centers, selected_bin_heights, p0=initial_guess)

# Extract the fitted parameters
amplitude, mean, stddev = fit

# Step 4: Filter the data based on (-2σ, +7σ) range around the mean
lower_bound = mean - 2 * stddev
upper_bound = mean + 7 * stddev

# Find number of disk pixels
disk_count = np.count_nonzero(data)

# Filter the data
filtered_data = filtered_data[(filtered_data >= lower_bound) & (filtered_data <= upper_bound)]

# Step 5: Plot the histogram of the filtered data
bin_heights2, bin_edges2, _ = plt.hist(filtered_data, bins=30,color='green', alpha=0.5)
bin_centers2 = 0.5 * (bin_edges2[:-1] + bin_edges2[1:])
# Add labels and title
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.title(f'Histogram of Filtered Data: Range ({lower_bound:.2f}, {upper_bound:.2f})')

# Show the plot
plt.show()

def CaII_K_Index(x, amplitude, mean, stddev, B):
    return amplitude * np.exp(-0.5 * ((x - mean) / stddev)**2) + B

initial_guess_2 = [0.1, 0.98, 0.05, 0.001]
fit2, cov2 = curve_fit(CaII_K_Index, bin_centers2, bin_heights2 / disk_count, sigma = np.sqrt(bin_heights2) / disk_count, p0 = initial_guess_2)

x_array = np.linspace(lower_bound, upper_bound, 1000)
plt.figure(dpi = 300)
plt.hist(filtered_data, bins=30,color='green', alpha=0.5, weights=np.ones_like(filtered_data) / disk_count)
plt.plot(x_array, CaII_K_Index(x_array, *fit2))
plt.legend()
plt.xlabel('Intensity')
plt.ylabel('Frequency')

B = fit2[3]
error = np.sqrt(cov2[3][3])
print(f"The CaK Index is:{B:.5f} ± {error:.5f}")



#%%

# Specify the root directory containing the 200 folders
root_directory = 'ChroTel_Ca_L2'  # replace with the path to your root folder

# Dictionary to store each dataset and header by date
fits_data_by_date = {}

# Loop through all subdirectories and files, no matter the depth
for dirpath, dirnames, filenames in os.walk(root_directory):
    for filename in filenames:
        if filename.endswith('.fits'):
            fits_file_path = os.path.join(dirpath, filename)
            print(f"Processing file: {fits_file_path}")
            
            # Extract the date from the filename
            try:
                # Extract date part '20130528' from the filename and parse it
                date_str = filename.split('_')[2][:8]  # '20130528'
                file_date = datetime.strptime(date_str, '%Y%m%d').date()
            except (IndexError, ValueError):
                print(f"Warning: Could not extract date from {filename}. Skipping file.")
                continue  # Skip files without a valid date
            
            # Open the FITS file and read the data and header
            with fits.open(fits_file_path) as hdul:
                data = hdul[0].data
                header = hdul[0].header

            # Store the data and header in dictionary with file_date as the key
            fits_data_by_date[file_date] = {
                'data': data,
                'header': header
            }

for date, content in fits_data_by_date.items():
    data = content['data']
    
    # Define the center coordinates
    cx, cy = 1000, 1000  # Center x and y coordinates (provided)

    # Define the radius as half the smaller dimension (to fit within the image)
    ny, nx = data.shape
    radius = 1000

    # Define the 0.99 radius threshold
    radius_threshold = 0.99 * radius

    # Create a grid of coordinates with respect to the center
    y, x = np.ogrid[:ny, :nx]
    distance_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Create the mask
    mask = distance_from_center <= radius_threshold

    # Now 'mask' is True for pixels within 0.99 of the radius, False outside
    filtered_data = np.where(mask, data, np.nan)

    # Mask data to remove points where intensity is greater than 3
    filtered_data = filtered_data[filtered_data <= 3]
    
    # Plot histogram with bin size of 0.01
    bin_heights, bin_edges = np.histogram(filtered_data, bins=np.arange(min(filtered_data), max(filtered_data) + 0.01, 0.01))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    center_bin_index = np.argmax(bin_heights)

    # Step 2: Select the first 35 bins on either side of the center bin
    half_window = 35
    start_bin = max(0, center_bin_index - half_window)
    end_bin = min(len(bin_centers), center_bin_index + half_window + 1)

    # Subset the data to only include these bins
    selected_bin_centers = bin_centers[start_bin:end_bin]
    selected_bin_heights = bin_heights[start_bin:end_bin]

    # Step 3: Initial guess for the parameters: [amplitude, mean, stddev]
    initial_guess = [max(selected_bin_heights), bin_centers[center_bin_index], np.std(selected_bin_centers)]

    # Perform curve fitting
    fit, cov = curve_fit(gaussian, selected_bin_centers, selected_bin_heights, p0=initial_guess)

    # Extract the fitted parameters
    amplitude, mean, stddev = fit

    # Step 4: Filter the data based on (-2σ, +7σ) range around the mean
    lower_bound = mean - 2 * stddev
    upper_bound = mean + 7 * stddev

    # Find number of disk pixels
    disk_count = np.count_nonzero(data)

    # Filter the data
    filtered_data = filtered_data[(filtered_data >= lower_bound) & (filtered_data <= upper_bound)]

    # Step 5: Plot the histogram of the filtered data
    bin_heights2, bin_edges2 = np.histogram(filtered_data, bins=30)
    bin_centers2 = 0.5 * (bin_edges2[:-1] + bin_edges2[1:])

    initial_guess_2 = [0.1, 0.98, 0.05, 0.001]
    fit2, cov2 = curve_fit(CaII_K_Index, bin_centers2, bin_heights2 / disk_count, sigma = np.sqrt(bin_heights2) / disk_count, p0 = initial_guess_2)
    
    # Extract the fitted parameter B (and its error)
    B = fit2[3]
    B_error = np.sqrt(cov2[3][3])

    # Store the B parameter and error in the dictionary for the corresponding date
    fits_data_by_date[date]['B'] = B
    fits_data_by_date[date]['B_error'] = B_error

    print(f"Date: {date} | B: {B:.5f} | B Error: {B_error:.5f}")


import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

#%% Extract the dates and B values from the fits_data_by_date dictionary
dates = []
B_values = []
errors = []

# Helper function to calculate date fraction
def date_to_fraction(date):
    # Ensure date is a datetime object
    if isinstance(date, datetime):
        date = date.date()  # Convert datetime to date if needed

    # Get the start of the year (January 1st)
    start_of_year = datetime(date.year, 1, 1)  # Start of the year as a datetime object
    
    # Calculate the day of the year
    day_of_year = (date - start_of_year.date()).days + 1  # Adding 1 to start from day 1
    
    # Get the total number of days in the year (handle leap years)
    total_days_in_year = 366 if date.year % 4 == 0 and (date.year % 100 != 0 or date.year % 400 == 0) else 365
    
    # Return the date fraction as year + day_of_year/total_days_in_year
    date_fraction = date.year + (day_of_year / total_days_in_year)
    return date_fraction

# Loop through fits_data_by_date dictionary to extract data
for date, content in fits_data_by_date.items():
    if 'B' in content:
        dates.append(date)  # Add datetime object
        B_values.append(content['B'])
        errors.append(content['B_error'])  # Assuming 'error' stores the error values

# Calculate the date fractions
date_fractions = [date_to_fraction(date) for date in dates]

# Plotting the B values with error bars against the date fractions
plt.figure(dpi = 300)
plt.errorbar(date_fractions, B_values, yerr=errors, fmt='o', color='b', label='B values', alpha=0.6, ms=3)

# Labeling the axes and adding title
plt.xlim(2012, 2021)
plt.xlabel('Date Fraction')
plt.ylabel('CaK')
plt.title('B Values over Time with Error Bars')
plt.legend()

# Display the plot
plt.show()
#%% Save data

# Create a DataFrame
data = {
    "Date Fraction": date_fractions,
    "CaK": B_values,
    "Error": errors
}
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
#df.to_excel("CaK_time_series.xlsx", index=False)  # index=False to omit the DataFrame index


#%%
# List of column names based on your data structure
column_names = ['Year', 'Month', 'Day', 'Date_Fraction', 'Spot Number', 'sigma', 'Value3', 'Value4']

# Replace 'your_file.csv' with the actual file path and read the CSV with the delimiter
df = pd.read_csv('SN_d_tot_V2.0.csv', delimiter=';', names=column_names, header=None)

time = df['Date_Fraction']
SN = df['Spot Number']
SN_error = df['sigma']

column_names2 = ['Year', 'Month', 'Date_Fraction', 'Spot Number', 'sigma', 'Value3', 'Value4']

# Replace 'your_file.csv' with the actual file path and read the CSV with the delimiter
df_M = pd.read_csv('SN_m_tot_V2.0.csv', delimiter=';', names=column_names2, header=None)

time_M = df_M['Date_Fraction']
SN_M = df_M['Spot Number']
SN_M_error = df_M['sigma']


column_names2 = ['Year', 'Month', 'Date_Fraction', 'Spot Number', 'sigma', 'Value3', 'Value4']

# Replace 'your_file.csv' with the actual file path and read the CSV with the delimiter
df_MS = pd.read_csv('SN_ms_tot_V2.0.csv', delimiter=';', names=column_names2, header=None)

time_MS = df_MS['Date_Fraction']
SN_MS = df_MS['Spot Number']
SN_MS_error = df_MS['sigma']

plt.figure(dpi = 300)
plt.plot(time, SN, label = 'Daily' )
plt.plot(time_M, SN_M, label = 'Monthly')
plt.plot(time_MS, SN_MS, label = 'Monthly smoothed')
plt.xlim(2012,2020)
plt.ylim(0,230)
plt.xlabel('Time (Years)')
plt.ylabel(r'Sunspot number $S_n$')
plt.legend(frameon = False)

#%%








































































































