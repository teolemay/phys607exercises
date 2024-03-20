import numpy as np
from matplotlib import pyplot as plt
from astroML.datasets import sdss_corrected_spectra
from sklearn.decomposition import PCA

data = sdss_corrected_spectra.fetch_sdss_corrected_spectra()
spectra = sdss_corrected_spectra.reconstruct_spectra(data)
wavelength = sdss_corrected_spectra.compute_wavelengths(data)

mask_below = (spectra < 0).any(axis=1)

# Create a mask for values above 100
mask_above = (spectra > 150).any(axis=1)

# Combine the two masks using bitwise AND
data_mask = mask_below | mask_above

spectra = spectra[~data_mask]

n = np.arange(len(spectra))

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


U, Sigma, VT = np.linalg.svd(rescaled, full_matrices=False)

total_variance = np.sum(Sigma**2)
variance_ratio = Sigma**2 / total_variance
# Cumulative sum of eigenvalues normalized to unity
cumulative_variance_ratio = np.cumsum(variance_ratio)

fig = plt.figure(figsize=(10, 7.5))
fig.subplots_adjust(hspace=0.05, bottom=0.12)

ax = fig.add_subplot(211, xscale='log', yscale='log')
ax.grid()
ax.plot(variance_ratio, c='k')
ax.set_ylabel('Normalized Eigenvalues')
ax.xaxis.set_major_formatter(plt.NullFormatter())
# ax.set_ylim(5E-4, 100)

ax = fig.add_subplot(212)
ax.grid()
ax.semilogx(cumulative_variance_ratio, color='k')
ax.set_xlabel('Eigenvalue Number')
ax.set_ylabel('Cumulative Eigenvalues')
# ax.set_ylim(0.65, 1.00)

plt.show()




n_components = 2
X_projected = np.dot(rescaled, VT[:, :n_components] )

plt.plot(n, X_projected[:,0])
plt.plot(n, X_projected[:,1])
plt.show()

plt.plot(wavelength, VT[:, 0], alpha=0.5, linewidth=0.5)
plt.plot(wavelength, VT[:, 1], alpha=0.5, linewidth=0.5)
plt.show()

plt.scatter(X_projected[:,0], X_projected[:,1], s=0.5)
plt.show()

# %%

pca = PCA()
pca.fit(rescaled)
comp = pca.transform(rescaled)

mean = pca.mean_
components = pca.components_
evals = pca.explained_variance_ratio_
evals_cs = evals.cumsum()

fig = plt.figure(figsize=(10, 7.5))
fig.subplots_adjust(hspace=0.05, bottom=0.12)

ax = fig.add_subplot(211, xscale='log', yscale='log')
ax.grid()
ax.plot(evals, c='k')
ax.set_ylabel('Normalized Eigenvalues')
ax.xaxis.set_major_formatter(plt.NullFormatter())
# ax.set_ylim(5E-4, 100)

ax = fig.add_subplot(212, xscale='log')
ax.grid()
ax.semilogx(evals_cs, color='k')
ax.set_xlabel('Eigenvalue Number')
ax.set_ylabel('Cumulative Eigenvalues')
# ax.set_ylim(0.65, 1.00)

plt.show()