#!pip install astroML

#%%
import numpy as np
from matplotlib import pyplot as plt
from astroML.datasets import sdss_corrected_spectra
from sklearn.decomposition import PCA

#%%

data = sdss_corrected_spectra.fetch_sdss_corrected_spectra()
spectra = sdss_corrected_spectra.reconstruct_spectra(data)
wavelength = sdss_corrected_spectra.compute_wavelengths(data)
cl = data['spec_cln']
star_mask = (cl == 1)
quasar_mask = (cl == 3) | (cl == 4)
redstar_mask = (cl == 6)
unknown_mask = (cl == 0)


n = np.arange(len(spectra))

plt.plot(wavelength, spectra.T, linewidth=0.5)
plt.show()


avg_spectra = np.average(spectra, axis=0)

# plt.plot(wavelength, avg_spectra)
# plt.show()

rescaled = spectra - avg_spectra
# rescaled = spectra / np.sum(spectra, axis = 0)
# rescaled = rescaled - np.average(rescaled, axis=0)

# plt.plot(wavelength, rescaled.T, linewidth=0.5)
# plt.show()

print(np.mean(rescaled))

# %%

def pca(rescaled_data):
    '''
    Principal Components Analysis

    Parameters
    ----------
    rescaled_data : Array [M, F]
        M - number of samples of data. F - number of features.
        In your case it`s [galaxies, wavelengthes]

    Returns
    -------
    ei_vals : Array [min(M, F),]
        Eigenvalues which define the amount of variance contained within each component.
    ei_vecs : Array [min(M, F), min(M, F)]
        Eigenvectors (principal components) - vectors that are aligned with the deriction of maximal variance.
    projected : Array [M, min(M, F)]
        Data projected on eigenvectors.

    '''
    
    U, Sigma, VT = np.linalg.svd(rescaled_data, full_matrices=False)
    ei_vals = Sigma**2
    ei_vecs = VT
    projected = np.dot(rescaled_data, VT[:, :] )
    
    return ei_vals, ei_vecs, projected


ei_vals, ei_vecs, projected = pca(rescaled)

total_variance = np.sum(ei_vals)
variance_ratio = ei_vals / total_variance
# Cumulative sum of eigenvalues normalized to unity
cumulative_variance_ratio = np.cumsum(variance_ratio)

# fig = plt.figure(figsize=(10, 7.5))
# fig.subplots_adjust(hspace=0.05, bottom=0.12)

# ax = fig.add_subplot(211, xscale='log', yscale='log')
# ax.grid()
# ax.plot(variance_ratio, c='k')
# ax.set_ylabel('Normalized Eigenvalues')
# ax.xaxis.set_major_formatter(plt.NullFormatter())
# # ax.set_ylim(5E-4, 100)

# ax = fig.add_subplot(212)
# ax.grid()
# ax.semilogx(cumulative_variance_ratio, color='k')
# ax.set_xlabel('Eigenvalue Number')
# ax.set_ylabel('Cumulative Eigenvalues')
# # ax.set_ylim(0.65, 1.00)

# plt.show()


plt.figure(figsize=(10, 6))
plt.bar(range(1, len(ei_vals) + 1), variance_ratio, alpha=0.7, align='center', label='Individual explained variance')
plt.step(range(1, len(ei_vals) + 1), cumulative_variance_ratio, where='mid', label='Cumulative explained variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
# plt.title('Scree Plot')
plt.xscale('log')
plt.xlim(1,1000)
plt.legend()
plt.grid(True)
plt.show()


plt.plot(n, projected[:,0], linewidth=0.5, alpha=0.5)
plt.plot(n, projected[:,1], linewidth=0.5, alpha=0.5)
plt.show()

plt.plot(wavelength, ei_vecs[:, 0], alpha=0.5, linewidth=0.5)
plt.plot(wavelength, ei_vecs[:, 1], alpha=0.5, linewidth=0.5)
plt.show()

#%% PCP

PC_index_1 = 0
PC_index_2 = 1

plt.scatter(projected[:,PC_index_1][unknown_mask], 
            projected[:,PC_index_2][unknown_mask], s=1, c='black',zorder=10)
plt.scatter(projected[:,PC_index_1][quasar_mask], 
            projected[:,PC_index_2][quasar_mask], s=1, c='orange',zorder=10)
plt.scatter(projected[:,PC_index_1][redstar_mask], 
            projected[:,PC_index_2][redstar_mask], s=1, c='red',zorder=10)
plt.scatter(projected[:,PC_index_1][star_mask], 
            projected[:,PC_index_2][star_mask], s=1, c='purple',zorder=10)
plt.scatter(projected[:,PC_index_1], 
            projected[:,PC_index_2], s=1, zorder=1)
plt.show()

#%%

def reconstruct(PC_coords, eigenvectors, mean):
    '''
    Reconstructs data from projection coordinates.

    Parameters
    ----------
    PCP_coords : Array [M , K]
        Coodinates in PC hyperplane.
        M is number of measurements, K is number of principal components.
        Example: if you want to reconstruct single point on 2D PCP, 
        your PCP_coords are gonna be [[x, y]].
    eigenvectors : Array [K , F]
        Eigenvector array of your PCA.
        K is number of principal components. F is number of features, in your case –
        number of wavelength.
    mean : array [M,]
        Array of average values of your raw data. In your case – average spectra

    Returns
    -------
    reconstructed : Array [M x F]
        Returns ([M , K] ⋅ [K , F]) + [F,] – data reconstructed from K 
        principal components

    '''
    
    
    reconstructed = np.dot(PC_coords, eigenvectors[:n_comp, :])

    reconstructed = reconstructed + mean
    
    return reconstructed
    

interesting = reconstruct([[0, 0]], ei_vecs[:2, :], avg_spectra)

plt.plot(wavelength, interesting.T)
plt.show()

plt.plot(wavelength, avg_spectra)
plt.show()

a = interesting[0] - avg_spectra

#%%
n_comp=1

reconstructed = reconstruct(projected[:, :n_comp], ei_vecs[:n_comp, :], avg_spectra)


plt.plot(wavelength, reconstructed.T, linewidth=0.2, alpha=0.5)
plt.title('reconstructed data; '+ str(n_comp)+' PCs')
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