import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from p_winds import tools

units = {
    'wavelength': u.angstrom,  # x-axis is wavelength in angstroms
    'flux': u.erg / u.s / u.cm**2 / u.angstrom  # flux unit matches the dataset
}

spectrum = tools.make_spectrum_from_file('HD209458b_spectrum_lambda.dat',
                                    units)


def square_filter(wavelength, center, width):
    """
    Create a square filter response function.
    
    Parameters:
        wavelengths (numpy array): The wavelengths over which the filter is defined.
        center (float): The central wavelength of the square filter (e.g., 10833 Å).
        width (float): The full width of the square filter.
        
    Returns:
        numpy array: Filter response (1 inside the band, 0 outside).
    """
    filter_response = np.where((wavelength >= center - width / 2) &
                               (wavelength <= center + width / 2), 1, 0)
    return filter_response

def convolve_with_filter(wavelength, spectrum, filter_response):
    """
    Convolve a spectrum with a given filter response.
    
    Parameters:
        wavelengths (numpy array): The wavelengths of the spectrum.
        spectrum (numpy array): The transmission spectrum values.
        filter_response (numpy array): The filter response function.
    
    Returns:
        float: Convolved spectrum (single value).
    """
    # Ensure the filter matches the spectrum's resolution
    assert len(wavelength) == len(filter_response), "Filter and spectrum must have the same length."
    
    
    # Perform a weighted average of the spectrum with the filter
    convolved_value = np.sum(spectrum * filter_response) / np.sum(filter_response)
    return convolved_value

# Example dataset (replace with your actual data)

wavelengths = spectrum['wavelength']
flux = spectrum['flux_lambda']
print("Wavelength shape:", wavelengths.shape)
print("Flux shape:", flux.shape)  # spectrum['transmitted_flux']



# Define filter parameters
filter_center = 1083.3  # Center of the square filter in Å
filter_width = 2       # Full width of the filter in Å
print("Wavelength range in spectrum:", wavelengths.min(), wavelengths.max())
print("Filter range:", filter_center - filter_width / 2, filter_center + filter_width / 2)

# Generate square filter response
filter_response = square_filter(wavelengths, filter_center, filter_width)

# Perform the convolution
convolved_value = convolve_with_filter(wavelengths, flux, filter_response)

# Display the results
print(f"Convolved value: {convolved_value}")

# Plot the spectrum and the filter
plt.figure(figsize=(8, 5))
plt.plot(wavelengths, flux, label='Transmission Spectrum', color='blue')
plt.plot(wavelengths, filter_response * max(flux), label='Square Filter', color='red', linestyle='--')
plt.xlabel("Wavelength (Å)")
plt.ylabel("Transmission")
plt.legend()
plt.title("Spectrum and Filter Convolution")
plt.show()
