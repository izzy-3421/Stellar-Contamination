import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter


wavelength_file = '/Users/stephi/Desktop/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'

# Define a list of file paths for different flux FITS files
flux_files = [
    '/Users/stephi/Desktop/lte06000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits',
    '/Users/stephi/Desktop/lte05000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits',
     '/Users/stephi/Desktop/lte05500-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
]

with fits.open(wavelength_file) as hdul_wave:
    wavelength = hdul_wave[0].data  # Wavelength grid in Angstroms

# Define the Gaussian function for fitting
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

# Plot all the spectra with Gaussian fits
plt.figure(figsize=(12, 8))
colors = ['blue', 'green', 'red']  # Define colors for different spectra
temperatures = [6000, 5000, 4000]  # Temperatures for each spectrum in Kelvin

# Desired peak flux values for each Gaussian fit
peak_flux_values = [1.4e15, 0.8e15, 0.5e15]

# Loop over each flux file and perform Gaussian fitting
reference_mu = None
for idx, flux_file in enumerate(flux_files):
    # Load the flux data from the FITS file
    with fits.open(flux_file) as hdul_flux:
        flux = hdul_flux[0].data  # Flux data in erg/s/cm^2/Hz

    # Plot the original spectrum
    plt.plot(wavelength, flux, label=f'Original Spectrum {idx + 1} ({temperatures[idx]}K)', color=colors[idx], alpha=0.5)

    # Smoothing the flux to make it easier for fitting
    flux_smoothed = gaussian_filter(flux, sigma=10)

    # Set initial guesses and bounds for Gaussian parameters
    amp_guess = peak_flux_values[idx]
    mu_guess = np.mean(wavelength) if reference_mu is None else reference_mu
    sigma_guess = 2000 if idx == 0 else 3000  # Increase sigma for broader fits

    # Set initial guesses
    initial_guesses = [amp_guess, mu_guess, sigma_guess]

    # Set bounds to ensure valid parameters for fitting
    bounds = (
        [amp_guess * 0.5, mu_guess - 5000, 1000],  # Lower bounds for [amp, mu, sigma]
        [amp_guess * 1.5, mu_guess + 5000, 10000]  # Upper bounds for [amp, mu, sigma]
    )

    # Fit the Gaussian to the data
    try:
        params, covariance = curve_fit(gaussian, wavelength, flux_smoothed, p0=initial_guesses, bounds=bounds)
    except RuntimeError:
        print(f"Optimal parameters not found for flux file {idx + 1}: fitting failed.")
        continue
    except ValueError as e:
        print(f"ValueError for flux file {idx + 1}: {e}")
        continue

    # Extract the fitted parameters
    amp_fit, mu_fit, sigma_fit = params

    # For the first fit, save the fitted mu value to use for subsequent fits
    if idx == 0:
        reference_mu = mu_fit

    # Generate fitted Gaussian for visualization
    fitted_gaussian = gaussian(wavelength, amp_fit, mu_fit, sigma_fit)

    # Plot the fitted Gaussian on the original data
    plt.plot(wavelength, fitted_gaussian, label=f'Gaussian Fit {idx + 1} ({temperatures[idx]}K)', color=colors[idx], linestyle='--')

    # Annotate the temperature of each spectrum near the peak of each Gaussian
    max_flux_idx = np.argmax(fitted_gaussian)
    plt.text(
        wavelength[max_flux_idx], fitted_gaussian[max_flux_idx] * 1.05,
        f"{temperatures[idx]}K", color=colors[idx], fontsize=10,
        horizontalalignment='center', verticalalignment='bottom'
    )

# Plot settings
plt.xlabel('Wavelength (Å)')
plt.ylabel('Flux (erg/s/cm²/Hz)')
plt.title('Stellar Spectra with Multiple Gaussian Fits (Aligned Centers, Different Peak Values)')
plt.legend()
plt.grid(True)
plt.show()