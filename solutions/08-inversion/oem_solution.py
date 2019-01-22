"""Perform an OEM retrieval and plot the results. """
import numpy as np
import matplotlib.pyplot as plt
from typhon.arts import xml
from typhon.plots import (cmap2rgba, profile_z, styles)
from scipy.linalg import inv


styles.use()


def retrieve(y, K, xa, ya, Sx, Sy):
    """Perform an OEM retrieval.

    Parameters:
        y (ndarray): Measuremed brightness temperature [K].
        K (ndarray): Jacobians [K/1].
        xa (ndarray): A priori state [VMR].
        ya (ndarray): Forward simulation of a priori state ``F(xa)`` [K].
        Sx (ndarray): A priori error covariance matrix.
        Sy (ndarray): Measurement covariance matrix

    Returns:
        ndarray: Retrieved atmospheric state.

    """
    # return xa + inv(K.T @ inv(Sy) @ K + inv(Sx)) @ K.T @ inv(Sy) @ (y - K @ (xa - xa) - ya)
    return xa + inv(K.T @ inv(Sy) @ K + inv(Sx)) @ K.T @ inv(Sy) @ (y - ya)


def averaging_kernel_matrix(K, S_a, S_y):
    """Calculate the averaging kernel matrix.

    Parameters:
        K (np.array): Simulated Jacobians.
        S_a (np.array): A priori error covariance matrix.
        S_y (np.array): Measurement covariance matrix.

    Returns:
        np.array: Averaging kernel matrix.
    """
    return inv(inv(S_a) + K.T @ inv(S_y) @ K) @ K.T @ inv(S_y) @ K


# Load a priori information.
f_grid = xml.load('input/f_grid.xml')
x_apriori = xml.load('input/x_apriori.xml').get('abs_species-H2O', keep_dims=False)
y_apriori = xml.load('results/y_apriori.xml')
S_x = xml.load('input/S_x.xml')
S_y = xml.load('input/S_y.xml') * np.eye(f_grid.size)

# Load ARTS results.
z = xml.load('results/z_field.xml')
K = xml.load('results/jacobian.xml')

# Load y measurement.
y_measure = xml.load('input/y_measurement.xml')

# Plot the y measurement alongside the simulated y for the a priori.
fig, ax = plt.subplots()
ax.plot(f_grid / 1e9, y_measure, label='Measurement', color='C0', lw=1)
ax.plot(f_grid / 1e9, y_apriori, label='A priori', color='C1')
ax.set_xlim(f_grid.min() / 1e9, f_grid.max() / 1e9)
ax.legend()
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('$B_\mathrm{T}$ [K]')
fig.savefig('plots/bt_spectrum.pdf')

# Plot the Jacobians.
fig, ax = plt.subplots()
ax.axvline(0, ls='solid', lw=0.8, color='black')
ax.set_prop_cycle(color=cmap2rgba('magma', len(K[::100])))
profile_z(z.ravel(), K[::100].T, ls='solid')
ax.set_xlabel('Jacobian [K/1]')
ax.set_ylim(z.min(), z.max())
fig.savefig('plots/jacobians.pdf')

# Plot the OEM result next to the true atmospheric state and the a priori.
x_oem = retrieve(y_measure, K, x_apriori, y_apriori, S_x, S_y)
x_true = xml.load('input/x_true.xml').get('abs_species-H2O', keep_dims=False)

fig, ax = plt.subplots()
profile_z(z.ravel(), x_true, label='True', ls='dashed', color='black')
profile_z(z.ravel(), x_apriori, label='A priori', color='C1')
profile_z(z.ravel(), x_oem, label='OEM', color='C0')
ax.set_xscale('log')
ax.set_xlabel('Water vapor [VMR]')
ax.set_ylim(z.min(), z.max())
ax.legend()
fig.savefig('plots/water_vapor_profile.pdf')

# Plot the averaging kernels and the measurement response.
A = averaging_kernel_matrix(K, S_x, S_y)

fig, ax = plt.subplots()
ax.axvline(0, ls='solid', lw=0.8, color='black')
ax.set_prop_cycle(color=cmap2rgba('magma', len(A)))
profile_z(z.ravel(), 5 * A, label='Test')
profile_z(z.ravel(), A.sum(axis=1), color='black', ls='dashed')
ax.text(0.2, 90e3, r'The kernels are scaled with a factor of 5', size='small')
ax.set_ylim(z.min(), z.max())
ax.set_xlabel('Averaging kernel')
fig.savefig('plots/averaging_kernels.pdf')

# plt.show()
