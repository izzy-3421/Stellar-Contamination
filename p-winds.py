import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import astropy.units as u
import astropy.constants as c
from p_winds import tools, parker, hydrogen, helium, transit, lines
from astropy.io import fits

# HD 209458 b planetary parameters, measured
R_pl = 1.39  # Planetary radius in Jupiter radii
M_pl = 0.73  # Planetary mass in Jupiter masses
impact_parameter = 0.499  # Transit impact parameter

# A few assumptions about the planet's atmosphere
m_dot = 10 ** 10.27  # Total atmospheric escape rate in g / s
T_0 = 9100  # Wind temperature in K
h_fraction = 0.90  # H number fraction
he_fraction = 1 - h_fraction  # He number fraction
he_h_fraction = he_fraction / h_fraction
mean_f_ion = 0.0  # Mean ionization fraction (will be self-consistently calculated later)
mu_0 = (1 + 4 * he_h_fraction) / (1 + he_h_fraction + mean_f_ion)
# mu_0 is the constant mean molecular weight (assumed for now, will be updated later)

units = {'wavelength': u.angstrom, 'flux': u.erg / u.s / u.cm ** 2 / u.angstrom}
spectrum = tools.make_spectrum_from_file('HD209458b_spectrum_lambda.dat',
                                    units)
plt.figure(dpi=300)
plt.loglog(spectrum['wavelength'], spectrum['flux_lambda'])
plt.ylim(1E-5, 1E4)
#plt.xlim(10827,10832)
plt.xlabel(r'Wavelength (${\rm \AA}$)')
plt.ylabel(r'Flux density (erg s$^{-1}$ cm$^{-2}$ ${\rm \AA}^{-1}$)')
plt.title("Spectrum of HD 209458")
plt.show()
#%%
initial_f_ion = 0.0
r = np.logspace(0, np.log10(20), 100)  # Radial distance profile in unit of planetary radii

f_r, mu_bar,rates = hydrogen.ion_fraction(r, R_pl, T_0, h_fraction,
                            m_dot, M_pl, mu_0,
                            spectrum_at_planet=spectrum, exact_phi=True,
                            initial_f_ion=initial_f_ion, relax_solution=True,
                            return_mu=True, return_rates=True)

initial_f_ion = 0.0
r = np.logspace(0, np.log10(20), 100)  # Radial distance profile in unit of planetary radii

f_r, mu_bar,rates = hydrogen.ion_fraction(r, R_pl, T_0, h_fraction,
                            m_dot, M_pl, mu_0,
                            spectrum_at_planet=spectrum, exact_phi=True,
                            initial_f_ion=initial_f_ion, relax_solution=True,
                            return_mu=True, return_rates=True)

f_ion = f_r
f_neutral = 1 - f_r

plt.figure(dpi=300)
plt.plot(r, f_neutral, color='C0', label='f$_\mathrm{neutral}$')
plt.plot(r, f_ion, color='C1', label='f$_\mathrm{ion}$')
plt.xlabel(r'Radius (R$_\mathrm{pl}$)')
plt.ylabel('Number fraction')
plt.xlim(1, 10)
plt.ylim(0, 1)
plt.legend()
plt.show()

ionization_rate = rates['photoionization']
recombination_rate = rates['recombination']
plt.figure(dpi=300)
plt.semilogy(r, ionization_rate, color='C0', label='Photoionization')
plt.semilogy(r, recombination_rate, color='C1', label='Recombination')
plt.xlabel(r'Radius (R$_\mathrm{pl}$)')
plt.ylabel(r'Rate (s$^{-1}$)')
plt.xlim(1, 10)
plt.legend()
plt.show()

vs = parker.sound_speed(T_0, mu_bar)  # Speed of sound (km/s, assumed to be constant)
rs = parker.radius_sonic_point(M_pl, vs)  # Radius at the sonic point (jupiterRad)
rhos = parker.density_sonic_point(m_dot, rs, vs)  # Density at the sonic point (g/cm^3)

r_array = r * R_pl / rs
v_array, rho_array = parker.structure(r_array)

# Convenience arrays for the plots
r_plot = r_array * rs / R_pl
v_plot = v_array * vs
rho_plot = rho_array * rhos

# Finally plot the structure of the upper atmosphere
# The circles mark the velocity and density at the sonic point
plt.figure(dpi=300)
ax1 = plt.subplot()
ax2 = ax1.twinx()
ax1.semilogy(r_plot, v_plot, color='C0')
ax1.plot(rs / R_pl, vs, marker='o', markeredgecolor='w', color='C0')
ax2.semilogy(r_plot, rho_plot, color='C1')
ax2.plot(rs / R_pl, rhos, marker='o', markeredgecolor='w', color='C1')
ax1.set_xlabel(r'Radius (R$_{\rm pl}$)')
ax1.set_ylabel(r'Velocity (km s$^{-1}$)', color='C0')
ax2.set_ylabel(r'Density (g cm$^{-3}$)', color='C1')
ax1.set_xlim(1, 10)
plt.show()
 
# In the initial state, the fraction of singlet and triplet helium
# are, respectively, 1.0 and 0.0
initial_state = np.array([1.0, 0.0])
f_he_1, f_he_3, reaction_rates = helium.population_fraction(
    r, v_array, rho_array, f_ion,
    R_pl, T_0, h_fraction, vs, rs, rhos, spectrum,
    initial_state=initial_state, relax_solution=True, return_rates=True)

labels = reaction_rates.keys()
plt.figure(dpi=300)
for l in labels:
    plt.semilogy(r, reaction_rates[l], label=l)
plt.xlabel(r'Radius (R$_\mathrm{pl}$)')
plt.ylabel(r'Rate (s$^{-1}$)')
plt.xlim(1, 10)
plt.legend(fontsize=10)
plt.show()

# Hydrogen atom mass
m_h = c.m_p.to(u.g).value

# Number density of helium nuclei
he_fraction = 1 - h_fraction
n_he = (rho_array * rhos * he_fraction / (h_fraction + 4 * he_fraction) / m_h)

n_he_1 = f_he_1 * n_he
n_he_3 = f_he_3 * n_he
n_he_ion = (1 - f_he_1 - f_he_3) * n_he

plt.figure(dpi=300)
plt.semilogy(r, n_he_1, color='C0', label='He singlet')
plt.semilogy(r, n_he_3, color='C1', label='He triplet')
plt.semilogy(r, n_he_ion, color='C2', label='He ionized')
plt.xlabel(r'Radius (R$_\mathrm{pl}$)')
plt.ylabel('Number density (cm$^{-3}$)')
plt.xlim(1, 10)
plt.ylim(1E-2, 1E10)
plt.legend()
plt.show()
#%%

def pixelate_image(image, target_size=(100, 100)):
    """
    Reduces the size of the image to the target_size using binning.
    
    Parameters:
        image (ndarray): The original image array.
        target_size (tuple): The desired (height, width) of the pixelated image.
    
    Returns:
        ndarray: The pixelated image.
    """
    # Original image dimensions
    original_height, original_width = image.shape
    target_height, target_width = target_size

    # Compute binning factor
    bin_height = original_height // target_height
    bin_width = original_width // target_width

    # Reshape and bin
    pixelated_image = image.reshape(
        target_height, bin_height, target_width, bin_width
    ).mean(axis=(1, 3))  # Average over blocks
    
    return pixelated_image

# First convert everything to SI units because they make our lives
# much easier.
R_pl_physical = R_pl * 71492000  # Planet radius in m
r_SI = r * R_pl_physical  # Array of altitudes in m
v_SI = v_array * vs * 1000  # Velocity of the outflow in m / s
n_he_3_SI = n_he_3 * 1E6  # Volumetric densities in 1 / m ** 3
planet_to_star_ratio = 69911 / 696340

dimensions = 100

# Set up the ray tracing. We will use a coarse 100-px grid size,
# but we use supersampling to avoid hard pixel edges.
flux_map, t_depth, r_from_planet = transit.draw_transit(
    planet_to_star_ratio,
    planet_physical_radius=R_pl_physical,
    impact_parameter=impact_parameter,
    phase=0.0,
    supersampling=10,
    grid_size=dimensions)

flux_map = flux_map.astype(np.float64)

# And now we plot it just to check how the transit looks
plt.figure(dpi=300)
plt.imshow(flux_map, cmap='gray', origin='lower')
plt.colorbar(label='Intensity')
plt.title("Homogenous Disk")
plt.show()

fits_file = 'chrotel-he_l2_20120522T135120_zic_v1.fits'  # replace with your FITS file name
with fits.open(fits_file) as hdul:
    data1 = hdul[0].data
    header = hdul[0].header

data = {
    10830.30: data1[3, :, :],
    10828.47: data1[5, :, :],
}

# normlaise out of transit flux to be 1 and remove pixels (0 them) at planet position
for wavelength in data: 
    
    # Extract image data
    image_data = data[wavelength].astype(np.float64)
    
    # Remove pixels at object position. (may be susceptible to edge effects)
    y_center = impact_parameter * 1000 + 1000
    x_center = 1000 # 0 phase
    radius = planet_to_star_ratio * 1000 # radius of planet in pixels
    
    # Get the shape of the image
    height, width = image_data.shape
    
    # Create a grid of x and y coordinates
    y, x = np.ogrid[:height, :width]
    
    # Compute the distance from the center for each pixel
    distance_from_center = np.sqrt((x - x_center)**2 + (y - y_center)**2)
    
    # Create a mask for all pixels within the radius
    mask = distance_from_center <= radius
    image_data_planet = image_data.copy() 
    
    # Set all pixels within the mask to zero
    image_data_planet[mask] = 0
    transit_depth = (np.sum(image_data_planet)) / np.sum(image_data)
    
    
    # Pixelate the image to 100x100
   #  pixelated_image_no_planet = pixelate_image(image_data, target_size=(dimensions, dimensions))
    pixelated_image = pixelate_image(image_data_planet, target_size=(dimensions, dimensions))
    
   # total_flux_no_planet = np.sum(pixelated_image_no_planet)
    pixelated_image /= np.sum(pixelated_image)
    pixelated_image *= np.sum(flux_map)
    
    # Store information directly in the data dictionary
    data[wavelength] = {
        'ImageData': pixelated_image,  # Keep the image data after background removal
        'TransitDepth': transit_depth
    }
    
plt.figure(dpi=300)
plt.imshow(data[10830.30]['ImageData'], cmap='gray', origin='lower')
plt.title("10830.30 Å")
plt.colorbar(label='Intensity')
plt.show()

plt.figure(dpi=300)
plt.imshow(data[10828.47]['ImageData'], cmap='gray', origin='lower')
plt.title("10828.47 Å")
plt.colorbar(label='Intensity')
plt.show()

#%%
# Retrieve the properties of the triplet; they were hard-coded
# using the tabulated values of the NIST database
# wX = central wavelength, fX = oscillator strength, a_ij = Einstein coefficient
w0, w1, w2, f0, f1, f2, a_ij = lines.he_3_properties()

m_He = 4 * 1.67262192369e-27  # Helium atomic mass in kg
wl = np.linspace(1.0827, 1.0832, 200) * 1E-6  # Wavelengths in m

# First, let's do the radiative transfer for each line of the triplet
# separately. Check the documentation to understand what are the
# input parameters, as there are many of them.

# Another important thing to have in mind is that the formal calculation
# of the radiative transfer can take a long time. To make it faster,
# there is an option that assumes something about the atmosphere
# and accelerates the modeling. That approximation is triggered by the
# `wind_broadening_method` input parameter set to `'average'`. If you want
# to do the formal calculation, set `wind_broadening_method` to `'formal'`.
# The default is `'average'`.
method = 'average'

spectrum_0 = transit.radiative_transfer_2d(data[10828.47]['ImageData'], r_from_planet,
                                        r_SI, n_he_3_SI, v_SI, w0, f0, a_ij,
                                        wl, T_0, m_He, wind_broadening_method=method)# / 0.98537555

spectrum_1 = transit.radiative_transfer_2d(data[10830.30]['ImageData'], r_from_planet,
                                        r_SI, n_he_3_SI, v_SI, w1, f1, a_ij,
                                        wl, T_0, m_He, wind_broadening_method=method)# / 0.98532933

spectrum_2 = transit.radiative_transfer_2d(data[10830.30]['ImageData'], r_from_planet,
                                        r_SI, n_he_3_SI, v_SI, w2, f2, a_ij,
                                        wl, T_0, m_He, wind_broadening_method=method)#/ 0.98532932

spectrum_0_homogen = transit.radiative_transfer_2d(flux_map, r_from_planet,
                                        r_SI, n_he_3_SI, v_SI, w0, f0, a_ij,
                                        wl, T_0, m_He, wind_broadening_method=method)# / 0.98528348

spectrum_1_homogen = transit.radiative_transfer_2d(flux_map, r_from_planet,
                                        r_SI, n_he_3_SI, v_SI, w1, f1, a_ij,
                                        wl, T_0, m_He, wind_broadening_method=method)# / 0.98528347

spectrum_2_homogen = transit.radiative_transfer_2d(flux_map, r_from_planet,
                                        r_SI, n_he_3_SI, v_SI, w2, f2, a_ij,
                                        wl, T_0, m_He, wind_broadening_method=method)# / 0.98528346

# Finally let's calculate the combined spectrum of all lines in the triplet
# To do that, we combine all the line properties in their respective arrays
w_array = np.array([w0, w1, w2])
f_array = np.array([f0, f1, f2])
a_array = np.array([a_ij, a_ij, a_ij])  # This is the same for all lines in then triplet

w_array_blended = np.array([w1, w2])
f_array_blended = np.array([f1, f2])
a_array_blended = np.array([a_ij, a_ij]) 


spectrum_homogen = transit.radiative_transfer_2d(flux_map, r_from_planet,
                                        r_SI, n_he_3_SI, v_SI, w_array, f_array, a_array,
                                        wl, T_0, m_He, wind_broadening_method=method)

spectrum_imhomogen_blended = transit.radiative_transfer_2d(data[10830.30]['ImageData'], r_from_planet,
                                        r_SI, n_he_3_SI, v_SI, w_array_blended, f_array_blended, a_array_blended,
                                        wl, T_0, m_He, wind_broadening_method=method)

spectrum_imhomogen_third = transit.radiative_transfer_2d(data[10828.47]['ImageData'], r_from_planet,
                                        r_SI, n_he_3_SI, v_SI, w0, f0, a_ij,
                                        wl, T_0, m_He, wind_broadening_method=method)


plt.figure(dpi=300)

# plt.plot(wl * 1E6, spectrum_0, ls='-', label = '10829.09 Å')
# plt.plot(wl * 1E6, spectrum_1, ls='-', label = '10830.25 Å')
# plt.plot(wl * 1E6, spectrum_2, ls='-', label = '10830.34 Å')
# #plt.plot(wl * 1E6, spectrum, color='k', lw=2)

# plt.plot(wl * 1E6, spectrum_0_homogen, ls='--', label = '10829.09 Å')
# plt.plot(wl * 1E6, spectrum_1_homogen, ls='--', label = '10830.25 Å')
# plt.plot(wl * 1E6, spectrum_2_homogen, ls='--', label = '10830.34 Å')
plt.plot(wl * 1E6, spectrum_homogen, color='k', lw=1)
#plt.plot(wl * 1E6, spectrum_imhomogen_blended, color='r', lw=1, ls = '--')
#plt.plot(wl * 1E6, spectrum_imhomogen_third, color='r', lw=1, ls = '--')
#plt.axhline(y=1)

plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('Normalised Flux')
plt.legend(frameon = False)
plt.show()

transit_depth_homogon = (1-spectrum_homogen) * 100
transit_depth_inhomogon_blended = ((1-spectrum_imhomogen_blended) - (1 -0.98971699)) * 100
transit_depth_inhomogon_third= (1-spectrum_imhomogen_third) * 100
transit_depth_inhomogon_combined = transit_depth_inhomogon_blended + transit_depth_inhomogon_third

# plt.figure(dpi=300)
# plt.plot(wl * 1E6, transit_depth_homogon, color='k', lw=1, label = "Homogenous")
# plt.plot(wl * 1E6, transit_depth_inhomogon_combined, color='r', lw=1, ls = '--', label = "Inhomogenous")
#plt.axhline(y=1)

# plt.xlabel('Wavelength ($\mu$m)')
# plt.ylabel('Transit Depth (%) or Excess Absorption (%)?')
# plt.legend(frameon = False)
# plt.show()

# plt.figure(dpi=300)
# plt.plot(wl * 1E6, transit_depth_inhomogon_combined - transit_depth_homogon + 1.02830063 , color='r', lw=1, ls = '--', label = "Difference")
# #plt.axhline(y=1)

# plt.xlabel('Wavelength ($\mu$m)')
# plt.ylabel('Transit Depth (%)')
# plt.legend(frameon = False)
# plt.show()




#plt.plot(wl * 1E6, spectrum_homogen-spectrum_imhomogen_blended - spectrum_imhomogen_third, color='k', lw=1)
#%%
from matplotlib.ticker import ScalarFormatter


plt.figure(dpi=300)

plt.plot(wl * 1E6, 1+((spectrum_0 - spectrum_0_homogen)), ls='-', label = '10829.09 Å')
plt.plot(wl * 1E6, 1+((spectrum_1 - spectrum_1_homogen)), ls='-',label = '10830.25 Å')
plt.plot(wl * 1E6, 1+((spectrum_2 - spectrum_2_homogen)), ls='-', label = '10830.34 Å')
plt.title("Homogenous disk spectra - Contaminated disk spectra")

# plt.plot(wl * 1E6, (1+((spectrum_0 - spectrum_0_homogen))) * (1+((spectrum_1 - spectrum_1_homogen))) * (1+((spectrum_2 - spectrum_2_homogen)))
#          , color='k', lw=2,label = 'Full Absorption')
#plt.plot(wl * 1E6, spectrum_homogen, color='k', lw=2)


# Remove scientific notation on both axes
ax = plt.gca()  # Get current axis

# Use ScalarFormatter to prevent scientific notation
formatter = ScalarFormatter()
formatter.set_scientific(False)  # Disable scientific notation
formatter.set_useOffset(False)   # Disable offset

# Apply the formatter to both axes
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)


plt.legend(frameon = False)
plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel('Normalized flux')
plt.show()


# 3.500000000000725e-05 for p winds
# 5.999999999994898e-05









