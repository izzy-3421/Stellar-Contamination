import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from p_winds import tools



units = {
    'wavelength': u.angstrom,  # x-axis is wavelength in angstroms
    'flux': u.erg / u.s / u.cm**2 / u.angstrom  
}

spectrum = tools.make_spectrum_from_file('wasp107b_transmission_spectrum.dat',
                                    units)


def square_filter(wavelength, center, width):
    filter_response = np.where((wavelength >= center - width / 2) &
                               (wavelength <= center + width / 2), 1, 0)
    return filter_response

def convolve_with_filter(wavelength, spectrum, filter_response):
    assert len(wavelength) == len(filter_response), "Filter and spectrum must have the same length."
    convolved_value = np.sum(spectrum * filter_response) / np.sum(filter_response)
    return convolved_value

def gaussian_filter(wavelength, mean, sigma): #amplitude = 1
    return np.exp(-((wavelength - mean)**2)/(2 * sigma**2))

wavelengths = spectrum['wavelength']
flux = spectrum['flux_lambda']
print("Wavelength shape:", wavelengths.shape)
print("Flux shape:", flux.shape)
air_wavelength = 10830.3
s = 10**4/air_wavelength
n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)
vacc_wvlt = n * air_wavelength
sig = 1.3/(2* np.sqrt(2*np.log(2)))
# Define filter parameters
filter_center = vacc_wvlt  # Center of the square filter in Å
#filter_center = 10833
filter_width = sig*2   # Full width of the filter in Å
print("Wavelength range in spectrum:", wavelengths.min(), wavelengths.max())
print("Filter range:", filter_center - filter_width / 2, filter_center + filter_width / 2)

# Generate square filter response
filter_response = square_filter(wavelengths, filter_center, filter_width)


# Perform the convolution
convolved_value = convolve_with_filter(wavelengths, flux, filter_response)

# Display the results
print(f"Convolved value: {convolved_value}")

# Gaussian filter
g_filter_response = gaussian_filter(wavelengths, filter_center, filter_width)


#g_filter_response /= np.sum(g_filter_response)

convolved_flux = np.convolve(flux, g_filter_response, mode='same')
g_convolved_value = convolve_with_filter(wavelengths, flux, g_filter_response) #Gaussian filter convolution
print(f"Gaussian convolved value: {g_convolved_value}")
# Plot the spectrum and the filter
plt.figure(figsize=(8, 5))
plt.plot(wavelengths, flux, label='Transmission Spectrum', color='blue')
plt.plot(wavelengths, filter_response * max(flux), label='Square Filter', color='red', linestyle='--')
plt.xlabel("Wavelength (Å)")
plt.ylabel("Transmission")
plt.legend()
plt.title("Spectrum and Filter Convolution")
plt.show()

# Gaussian filter
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, flux, label='Original Transmission Spectrum', color='blue')
plt.plot(wavelengths, g_filter_response * max(flux), label='Gaussian Filter (scaled)', color='red', linestyle='--')
plt.xlabel("Wavelength (Å)")
plt.ylabel("Transmission")
plt.title("Spectrum Convolved with Gaussian Filter")
plt.legend()
plt.show()