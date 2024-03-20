import numpy as np
from matplotlib import pyplot as plt
from astroML.datasets import sdss_corrected_spectra

data = sdss_corrected_spectra.fetch_sdss_corrected_spectra()
spectra = sdss_corrected_spectra.reconstruct_spectra(data)
wavelength = sdss_corrected_spectra.compute_wavelengths(data)

mask_below = (spectra < 0).any(axis=1)

# Create a mask for values above 100
mask_above = (spectra > 150).any(axis=1)

# Combine the two masks using bitwise AND
data_mask = mask_below | mask_above

spectra = spectra[~data_mask]

plt.plot(wavelength, spectra.T, linewidth=0.5)
plt.show()


avg_spectra = np.average(spectra, axis=0)

plt.plot(wavelength, avg_spectra)
plt.show()

rescaled = spectra - avg_spectra
# rescaled = spectra / np.sum(spectra, axis = 0)
# rescaled = rescaled - np.average(rescaled, axis=0)

plt.plot(wavelength, rescaled.T, linewidth=0.5)
plt.show()

print(np.mean(rescaled))

# %%


U, Sigma2, VT = np.linalg.svd(rescaled, full_matrices=False)