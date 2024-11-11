import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# Planet regime
planet_size = 'Earth'

# Impact parameter
b = 0.25

# Time period
years = range(2012, 2021) # doesnt include 2021

dataframes = []

# Loop through each year and load the corresponding CSV
for year in years:
    # Construct the file path
    file_path = f"{planet_size}/b={b}/{year}.csv"
    
    # Check if the file exists (optional, in case some years are missing)
    if os.path.exists(file_path):
        # Read the CSV and append to the list
        df = pd.read_csv(file_path)
        dataframes.append(df)
    else:
        print(f"File not found: {file_path}")

combined_df = pd.concat(dataframes, ignore_index=True)

#%% Plot all transit depths against wavelength

rows_as_arrays = []

# Loop through each row in the DataFrame
for _, row in combined_df.iterrows():
    # Convert the row to a NumPy array and append it to the list
    rows_as_arrays.append(row.values)

wavelengths = np.array([10833.15, 10832.13, 10831.00, 10830.30, 10829.60, 10828.47, 10827.45])
Rj_Rs = 69911 / 696340
Rn_Rs = 24622 / 696340
Re_Rs = 6371 / 696340   
radius_ratio = Re_Rs
true_transit_depth = radius_ratio ** 2 *100 # as a percentage

fig, ax = plt.subplots(dpi=300)

for row in rows_as_arrays:
    
    PSRR_values = np.array(row[2:9])
    PSRR_error = np.array(row[9:])
    transit_depth = PSRR_values ** 2 * 100
    transit_depth_error_per = PSRR_error / PSRR_values * 2 
    transit_depth_error = transit_depth_error_per * transit_depth
    
    ax.errorbar(wavelengths, transit_depth, yerr = transit_depth_error, fmt = 'o', ms = 4)
    
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Transit Depth (%)')
ax.set_title('Transit Depth (%) vs Wavelength')
ax.axhline(y = true_transit_depth, color = "red") 
    
#%% Histograms

# PSRR extraction setup
PSRR_values_all = np.array([np.array(row[2:9]) for row in rows_as_arrays])  # Extract PSRR columns (columns 2-8)
transit_depth_all = PSRR_values_all ** 2 * 100
# Loop through each wavelength (0 to 6) and create a histogram for each corresponding PSRR values
for idx, wavelength in enumerate(wavelengths):  # Looping through each wavelength index
    # Extract the PSRR values corresponding to the current wavelength
    extracted_values = [array[idx] for array in transit_depth_all]  # Extract values at current index
    
    # Plot the histogram for this wavelength's PSRR values
    plt.figure(dpi=300)
    plt.hist(extracted_values, bins=10, edgecolor='black', alpha=0.7)  # Adjust bins as needed
    plt.xlim(1.007, 1.016)
    plt.axvline(x=true_transit_depth, color="red")
    plt.xlabel(f'PSRR Value at Wavelength {wavelength} Å')
    plt.ylabel('Frequency')
    plt.title(f'{wavelength} Å 2012')
    plt.grid(True)
    plt.show()

#%% Fit Light Curves

def gaussian(x, amplitude, stddev, c):
    
    mean = 10830.30
    
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) + c

def linear(x, h):
    return np.full_like(x, h)

wavelengths = np.array([10833.15, 10832.13, 10831.00, 10830.30, 10829.60, 10828.47, 10827.45])
p0 = [true_transit_depth * 0.002, 1., true_transit_depth]
p1 = [true_transit_depth]
lower_bound = [true_transit_depth * 0.9 - true_transit_depth, 0, true_transit_depth*0.995]
upper_bound = [true_transit_depth * 1.1 - true_transit_depth, 1, true_transit_depth*1.005]

results_dict = {}

for index, row in combined_df.iterrows():
    
    Date = row[0]
    date_fraction = row[1]
    PSRR_values = np.array(row[2:9], dtype=np.float64)
    PSRR_error = np.array(row[9:], dtype=np.float64)
    transit_depth = PSRR_values ** 2 * 100
    transit_depth_error_per = PSRR_error / PSRR_values * 2 
    transit_depth_error = transit_depth_error_per * transit_depth
    
    
    
    
    fit_gaussian, cov_gaussian = curve_fit(gaussian, wavelengths, transit_depth, sigma = transit_depth_error,
                                   bounds = (lower_bound, upper_bound))
    fit_linear, cov_linear = curve_fit(linear, wavelengths, transit_depth, sigma = transit_depth_error, p0 = p1)
    
    # gaussian_residuals = PSRR_values - gaussian(wavelengths, *fit_gaussian)
    # linear_residuals = PSRR_values - linear(wavelengths, *fit_linear)
    # linear_rms = np.sqrt(np.mean(linear_residuals ** 2))
    
    background_values = np.concatenate((transit_depth[:2], transit_depth[-2:]))
    background_stddev = np.std(background_values)
    
    x_array = np.linspace(10827, 10834, 1000)
    amplitude = abs(fit_gaussian[0])
    middle_filter_res = PSRR_values[3] - gaussian(wavelengths[3], *fit_gaussian)
    stddev_number = 3
    
    # Choose the better model
    if amplitude < background_stddev * stddev_number: # linear fit
        fit = fit_linear
        amplitude = 0
        base = fit[0]
        cov = cov_linear
        error_amplitude = 0
        error_base = np.sqrt(cov[0][0])
        model = "linear"
        fit_curve = linear(x_array, *fit)
    else:                                 # gaussiabn fit
        fit = fit_gaussian
        amplitude = fit[0]
        base = fit[2]
        cov = cov_gaussian
        error_amplitude = np.sqrt(cov[0][0])
        error_base = np.sqrt(cov[2][2])
        model = "gaussian"
        fit_curve = gaussian(x_array, *fit)
    
    # plt.figure(dpi = 300)
    # plt.errorbar(wavelengths, transit_depth, yerr = transit_depth_error,fmt = 'o', ms = 4)
    # plt.plot(x_array, fit_curve)
    # plt.title(f'{Date}')
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Transit Depth (%)')
    # plt.axhline(y = true_transit_depth, color = "red") 
    
    results_dict[Date] = {"Date": Date,
                          "date fraction": date_fraction,
                          "model": model,
                          "amplitude": amplitude,
                          "amplitude error": error_amplitude,
                          "base": base,
                          "base error": error_base
                          }


    if model ==  "gaussian":
        print(f"Date: {Date} | Model: {model} | Amplitude: {fit[0]}")
        
    else:
        print(f"Date: {Date} | Model: {model} | Amplitude: 0")

#%%

def APENToS(x):
    S = (x + 0.57) / 3.55
    error = S * np.sqrt((0.01/0.57) ** 2 + (0.06/3.55) ** 2)
    return S, error 

df_A_PEN = pd.read_csv('A_PEN_time_series.csv')

# Convert results_dict to DataFrame
df_results = pd.DataFrame.from_dict(results_dict, orient="index")

# Reset index to make the date a column (optional)
df_results.reset_index(inplace=True)
df_results.drop(columns=["index"], inplace=True)
df_gaussian = df_results[df_results['model'] == 'gaussian']

merged_df = pd.merge(df_gaussian, df_A_PEN, on="Date", how="inner")

# Calculate the magnetic index and contamination
magnetic_index, magnetic_index_error = APENToS(merged_df['CaK'])
merged_df['S-index'] = magnetic_index
merged_df['S-index error'] = magnetic_index_error
merged_df = merged_df.sort_values(by='S-index', ascending=True).reset_index()


contamination = merged_df['amplitude'] / merged_df['base'] * 100
contamination_error = 2 * contamination * \
                 np.sqrt( (merged_df['amplitude error'] / merged_df['amplitude']) ** 2 +
                         (merged_df['base error'] / merged_df['base']) ** 2)

s_index = merged_df['S-index']
s_index_error = merged_df['S-index error']
# Create the scatter plot with a color bar based on the date fraction

# Create the colormap
cmap = plt.cm.viridis  # Colormap
norm = plt.Normalize(np.min(merged_df['date fraction']), np.max(merged_df['date fraction']))  # Normalize the colormap



plt.figure(dpi=300)

scatter = plt.scatter(s_index, contamination, 
                       c=merged_df['date fraction'], 
                       cmap=cmap, 
                       s=4)

errors_x = s_index_error
errors_y = abs(contamination_error)

# Plot error bars with colormap
for x, y, xerr, yerr, color_val in zip(s_index, contamination, errors_x, errors_y, merged_df['date fraction']):
    plt.errorbar(x, y, 
                   yerr=yerr, 
                  fmt='none', 
                  ecolor=cmap(norm(color_val)),  # Map color based on the colormap
                  elinewidth=0.8, alpha=0.6)


# Add a color bar to the plot
cbar = plt.colorbar(scatter)
cbar.set_label("Year")  # Label for the color bar

# Add labels to the plot
plt.xlabel("S-index")
plt.ylabel("Contamination (%)")
plt.title(f"Contamination vs S-index b={b}")

# Show the plot
plt.show()
#%% Moving average

df_gaussian_positive = df_gaussian[df_gaussian['amplitude'] > 0]

merged_df = pd.merge(df_gaussian_positive, df_A_PEN, on="Date", how="inner")
magnetic_index, magnetic_index_error = APENToS(merged_df['CaK'])
merged_df['S-index'] = magnetic_index
merged_df['S-index error'] = magnetic_index_error
merged_df = merged_df.sort_values(by='S-index', ascending=True).reset_index()

contamination = merged_df['amplitude'] / merged_df['base'] * 100
contamination_error = 2 * contamination * \
                 np.sqrt( (merged_df['amplitude error'] / merged_df['amplitude']) ** 2 +
                         (merged_df['base error'] / merged_df['base']) ** 2)

s_index = merged_df['S-index']
s_index_error = merged_df['S-index error']

# Define the chunk size (10 points)
chunk_size = 10

# Group the DataFrame by every 'chunk_size' number of rows and calculate the mean for each group
averaged_df = merged_df.groupby(merged_df.index // chunk_size).mean()
smoothed_s_index = averaged_df['S-index']
smoothed_contamination = (averaged_df["amplitude"] / averaged_df["base"]) * 100


plt.figure(dpi = 300)

scatter = plt.scatter(s_index, contamination, 
                       c=merged_df['date fraction'], 
                       cmap=cmap, 
                       s=4)

errors_x = s_index_error
errors_y = abs(contamination_error)

# Plot error bars with colormap
for x, y, xerr, yerr, color_val in zip(s_index, contamination, errors_x, errors_y, merged_df['date fraction']):
    plt.errorbar(x, y, 
                   yerr=yerr, 
                  fmt='none', 
                  ecolor=cmap(norm(color_val)),  # Map color based on the colormap
                  elinewidth=0.8, alpha=0.6)

plt.plot(smoothed_s_index, smoothed_contamination, color = "red")

# Add a color bar to the plot
cbar = plt.colorbar(scatter)
cbar.set_label("Year")  # Label for the color bar
plt.ylim(0,10)
# Add labels to the plot
plt.xlabel("S-index")
plt.ylabel("Contamination (%)")
plt.title(f"Contamination vs S-index b={b}")












#%%



























































