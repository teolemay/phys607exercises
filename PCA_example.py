#!pip install astroML

#%%
import numpy as np
from matplotlib import pyplot as plt
from astroML.datasets import sdss_corrected_spectra

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


X_projected = np.dot(rescaled, VT[:, :] )

plt.plot(n, X_projected[:,0])
plt.plot(n, X_projected[:,1])
plt.show()

plt.plot(wavelength, VT[:, 0], alpha=0.5, linewidth=0.5)
plt.plot(wavelength, VT[:, 1], alpha=0.5, linewidth=0.5)
plt.show()

#%% PCP

PC_index_1 = 0
PC_index_2 = 1

plt.scatter(X_projected[:,PC_index_1][unknown_mask], 
            X_projected[:,PC_index_2][unknown_mask], s=1, c='black',zorder=10)
plt.scatter(X_projected[:,PC_index_1][quasar_mask], 
            X_projected[:,PC_index_2][quasar_mask], s=1, c='orange',zorder=10)
plt.scatter(X_projected[:,PC_index_1][redstar_mask], 
            X_projected[:,PC_index_2][redstar_mask], s=1, c='red',zorder=10)
plt.scatter(X_projected[:,PC_index_1][star_mask], 
            X_projected[:,PC_index_2][star_mask], s=1, c='purple',zorder=10)
plt.scatter(X_projected[:,PC_index_1], 
            X_projected[:,PC_index_2], s=1, zorder=1)
plt.show()

#%%

def reconstruct(PCP_coords, eigenvectors, mean):
    '''
    Reconstructs data from projection coordinates.

    Parameters
    ----------
    PCP_coords : Array [N , K]
        Coodinates in PCP hyperplane.
        N is number of measurements, K is number of principal components.
        Example: if you want to reconstruct single point on 2D PCP, 
        your PCP_coords are gonna be [[x, y]].
    eigenvectors : Array [K , M]
        Eigenvector array of your PCA.
        K is number of principal components. M is number of features, in your case –
        number of wavelength.
    mean : array [M,]
        Array of average values of your raw data. In your case – average spectra

    Returns
    -------
    reconstructed : Array [N x M]
        Returns ([N , K] ⋅ [K , M]) + [M,] – data reconstructed from K 
        principal components

    '''
    
    
    reconstructed = np.dot(PCP_coords, 
                           VT[:n_comp, :])

    reconstructed = reconstructed + mean
    
    return reconstructed
    

outlier = reconstruct([[5, -5]], VT[:2, :], avg_spectra)

plt.plot(wavelength, outlier.T)
plt.show()



#%%
n_comp=2

reconstructed = np.dot(X_projected[:, :n_comp], 
                       VT[:n_comp, :])

# reconstructed = scaler.inverse_transform(reconstructed)
reconstructed = reconstructed + avg_spectra

plt.plot(wavelength, reconstructed.T, linewidth=0.2)
plt.title('reconstructed data; '+ str(n_comp)+' PCs')
plt.show()

# %%

# pca = PCA()
# pca.fit(rescaled)
# comp = pca.transform(rescaled)

# mean = pca.mean_
# components = pca.components_
# evals = pca.explained_variance_ratio_
# evals_cs = evals.cumsum()

# fig = plt.figure(figsize=(10, 7.5))
# fig.subplots_adjust(hspace=0.05, bottom=0.12)

# ax = fig.add_subplot(211, xscale='log', yscale='log')
# ax.grid()
# ax.plot(evals, c='k')
# ax.set_ylabel('Normalized Eigenvalues')
# ax.xaxis.set_major_formatter(plt.NullFormatter())
# # ax.set_ylim(5E-4, 100)

# ax = fig.add_subplot(212, xscale='log')
# ax.grid()
# ax.semilogx(evals_cs, color='k')
# ax.set_xlabel('Eigenvalue Number')
# ax.set_ylabel('Cumulative Eigenvalues')
# # ax.set_ylim(0.65, 1.00)

# plt.show()