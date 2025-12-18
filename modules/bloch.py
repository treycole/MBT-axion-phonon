import numpy as np
from itertools import permutations
from math import factorial
from .model import Model
from .k_mesh import K_mesh
from pythtb import WFArray
from itertools import product
import matplotlib.pyplot as plt


def get_trial_wfs(tf_list, norb, nspin=1):
    """
    Args:
        tf_list: list[int | list[tuple]]
            list of tuples defining the orbital and amplitude of the trial function
            on that orbital. Of the form [ [(orb, amp), ...], ...]. If spin is included,
            then the form is [ [(orb, spin, amp), ...], ...]

    Returns:
        tfs: np.ndarray
            Array of trial functions
    """

    # number of trial functions to define
    num_tf = len(tf_list)

    if nspin == 2:
        tfs = np.zeros([num_tf, norb, 2], dtype=complex)
        for j, tf in enumerate(tf_list):
            assert isinstance(tf, (list, np.ndarray)), "Trial function must be a list of tuples"
            for orb, spin, amp in tf:
                tfs[j, orb, spin] = amp
            tfs[j] /= np.linalg.norm(tfs[j])

    elif nspin == 1:
        # initialize array containing tfs = "trial functions"
        tfs = np.zeros([num_tf, norb], dtype=complex)
        for j, tf in enumerate(tf_list):
            assert isinstance(tf, (list, np.ndarray)), "Trial function must be a list of tuples"
            for site, amp in tf:
                tfs[j, site] = amp
            tfs[j] /= np.linalg.norm(tfs[j])

    return tfs

def get_periodic_H(model, H_flat, k_vals):
    orb_vecs = model.get_orb_vecs()
    orb_vec_diff = orb_vecs[:, None, :] - orb_vecs[None, :, :]
    # orb_phase = np.exp(1j * 2 * np.pi * np.einsum('ijm, ...m->...ij', orb_vec_diff, k_vals))
    orb_phase = np.exp(1j * 2 * np.pi * np.matmul(orb_vec_diff, k_vals.T)).transpose(2,0,1)
    H_per_flat = H_flat * orb_phase
    return H_per_flat


# def vel_op_fin_diff(model, H_flat, k_vals, dk, order_eps=1, mode='central'):
#     """
#     Compute velocity operators using finite differences.

#     Parameters:
#         H_mesh: ndarray of shape (Nk, M, M)
#             The Hamiltonian on the parameter grid.
#         dk: list of float
#             Step sizes in each parameter direction.

#     Returns:
#         v_mu_fd: list of ndarray
#             Velocity operators for each parameter direction.
#     # """

#     # recip_lat_vecs = model.get_recip_lat_vecs()
#     # recip_basis = recip_lat_vecs/ np.linalg.norm(recip_lat_vecs, axis=1, keepdims=True)
#     # g = recip_basis @ recip_basis.T
#     # sqrt_mtrc = np.sqrt(np.linalg.det(g))
#     # g_inv = np.linalg.inv(g)

#     # dk = np.einsum("ij, j -> i", g_inv, dk)

#     # assume only k for now
#     dim_param = model._dim_k # Number of parameters (dimensions)
#     # assume equal number of mesh points along each dimension
#     nks = ( int(H_flat.shape[0]**(1/dim_param)),)*dim_param

#     # Switch to periodic gauge H(k) = H(k+G)
#     H_flat = get_periodic_H(model, H_flat, k_vals)
#     H_mesh = H_flat.reshape(*nks, model._norb, model._norb)
#     v_mu_fd = np.zeros((dim_param, *H_mesh.shape), dtype=complex)

#     # Compute Jacobian
#     recip_lat_vecs = model.get_recip_lat_vecs()
#     inv_recip_lat = np.linalg.inv(recip_lat_vecs)

#     for mu in range(dim_param):
#         coeffs, stencil = finite_diff_coeffs(order_eps=order_eps, mode=mode)

#         derivative_sum = np.zeros_like(H_mesh)

#         for s, c in zip(stencil, coeffs):
#             H_shifted = np.roll(H_mesh, shift=-s, axis=mu)
#             derivative_sum += c * H_shifted

#         v_mu_fd[mu] = derivative_sum / (dk[mu])

#         # Ensure Hermitian symmetry
#         v_mu_fd[mu] = 0.5 * (v_mu_fd[mu] + np.conj(v_mu_fd[mu].swapaxes(-1, -2)))

#     return v_mu_fd


def finite_diff_coeffs(order_eps, derivative_order=1, mode='central'):
    """
    Compute finite difference coefficients using the inverse of the Vandermonde matrix.

    Parameters:
        stencil_points (array-like): The relative positions of the stencil points (e.g., [-2, -1, 0, 1, 2]).
        derivative_order (int): Order of the derivative to approximate (default is first derivative).

    Returns:
        coeffs (numpy array): Finite difference coefficients for the given stencil.
    """
    if mode not in ["central", "forward", "backward"]:
        raise ValueError("Mode must be 'central', 'forward', or 'backward'.")

    num_points = derivative_order + order_eps

    if mode == "central":
        if num_points % 2 == 0:
            num_points += 1
        half_span = num_points//2
        stencil = np.arange(-half_span, half_span + 1)

    elif mode == "forward":
        stencil = np.arange(0, num_points)

    elif mode == "backward":
        stencil = np.arange(-num_points+1, 1)

    A = np.vander(stencil, increasing=True).T  # Vandermonde matrix
    b = np.zeros(num_points)
    b[derivative_order] = factorial(derivative_order) # Right-hand side for the desired derivative

    coeffs = np.linalg.solve(A, b)  # Solve system Ax = b
    return coeffs, stencil


def compute_d4k_and_d2k(delta_k):
    """
    Computes the 4D volume element d^4k and the 2D plaquette areas d^2k for a given set of difference vectors in 4D space.

    Parameters:
    delta_k (numpy.ndarray): A 4x4 matrix where each row is a 4D difference vector.

    Returns:
    tuple: (d4k, plaquette_areas) where
        - d4k is the absolute determinant of delta_k (4D volume element).
        - plaquette_areas is a dictionary with keys (i, j) and values representing d^2k_{ij}.
    """
    # Compute d^4k as the determinant of the 4x4 difference matrix
    d4k = np.abs(np.linalg.det(delta_k))

    # Function to compute 2D plaquette area in 4D space
    def compute_plaquette_area(v1, v2):
        """Compute the 2D plaquette area spanned by two 4D vectors."""
        area_squared = 0.0
        # Sum over all unique (m, n) pairs where m < n
        for m in range(4):
            for n in range(m + 1, 4):
                area_squared += (v1[m] * v2[n] - v1[n] * v2[m]) ** 2
        return np.sqrt(area_squared)

    # Compute all unique plaquette areas
    plaquette_areas = {}
    for i in range(4):
        for j in range(i + 1, 4):
            plaquette_areas[(i, j)] = compute_plaquette_area(delta_k[i], delta_k[j])

    return d4k, plaquette_areas


def levi_civita(n, d):
    """
    Constructs the rank-n Levi-Civita tensor in dimension d.

    Parameters:
    n (int): Rank of the tensor (number of indices).
    d (int): Dimension (number of possible index values).

    Returns:
    np.ndarray: Levi-Civita tensor of shape (d, d, ..., d) with n dimensions.
    """
    shape = (d,) * n
    epsilon = np.zeros(shape, dtype=int)
    # Generate all possible permutations of n indices
    for perm in permutations(range(d), n):
        # Compute the sign of the permutation
        sign = np.linalg.det(np.eye(n)[list(perm)])
        epsilon[perm] = int(np.sign(sign))  # +1 for even, -1 for odd permutations

    return epsilon



class Bloch(WFArray):
    def __init__(self, model: Model, *param_dims):
        """Class for storing and manipulating Bloch like wavefunctions.

        Wavefunctions are defined on a semi-full reciprocal space mesh.
        """
        super().__init__(model, param_dims)
        assert len(param_dims) >= model._dim_k, "Number of dimensions must be >= number of reciprocal space dimensions"

        self.model: Model = model
        # model attributes
        self._n_orb = model.get_num_orbitals()
        self._nspin = self.model._nspin
        self._n_states = self._n_orb * self._nspin

        # reciprocal space dimensions
        self.dim_k = model._dim_k
        self.nks = param_dims[:self.dim_k]
        # set k_mesh
        self.model.set_k_mesh(*self.nks)
        # stores k-points on a uniform mesh, calculates nearest neighbor points given the model lattice
        self.k_mesh: K_mesh = model.k_mesh

        # adiabatic dimension
        self.dim_lam = len(param_dims)- self.dim_k
        self.n_lambda = param_dims[self.dim_k:]

        # Total adiabatic parameter space
        self.dim_param = self.dim_adia = self.dim_k + self.dim_lam
        self.n_param = self.n_adia = (*self.nks, *self.n_lambda)

        # periodic boundary conditions assumed True unless specified
        self.pbc_lam = True



        # axes indexes
        self.k_axes = tuple(range(self.dim_k))
        self.lambda_axes = tuple(range(self.dim_k, self.dim_param))

        if self._nspin == 2:
            self.spin_axis = -1
            self.orb_axis = -2
            self.state_axis = -3
        else:
            self.spin_axis = None
            self.orb_axis = -1
            self.state_axis = -2

        # wavefunction shapes
        if self.dim_lam > 0:
            if self._nspin == 2:
                self._wf_shape = (*self.nks, *self.n_lambda, self._n_states, self._n_orb, self._nspin)
            else:
                self._wf_shape = (*self.nks, *self.n_lambda, self._n_states, self._n_orb)
        else:
            if self._nspin == 2:
                self._wf_shape = (*self.nks, self._n_states, self._n_orb, self._nspin)
            else:
                self._wf_shape = (*self.nks, self._n_states, self._n_orb)

        # self.set_Bloch_ham()

    def get_wf_axes(self):
        dict_axes = {
            "wf shape": self._wf_shape,
            "Number of axes": len(self._wf_shape),
            "k-axes": self.k_axes, "lambda-axes": self.lambda_axes, "spin-axis": self.spin_axis,
            "orbital axis": self.orb_axis, "state axis": self.state_axis
            }
        return dict_axes

    def set_pbc_lam(self):
        self.pbc_lam = True

    def set_Bloch_ham(self, lambda_vals=None, model_fxn=None):
        if lambda_vals is None:
            H_k = self.model.get_ham(k_pts=self.k_mesh.flat_mesh) # [Nk, norb, norb]
            # [nk1, nk2, ..., norb, norb]
            self.H_k = H_k.reshape(*[nk for nk in self.k_mesh.nks], *H_k.shape[1:])
            return

        lambda_keys = list(lambda_vals.keys())
        lambda_ranges = list(lambda_vals.values())
        lambda_shape = tuple(len(vals) for vals in lambda_ranges)
        dim_lambda = len(lambda_keys)

        n_kpts = self.k_mesh.Nk
        n_orb = self._n_orb
        n_spin = self._n_spin
        n_states = n_orb*n_spin

        # Initialize storage for wavefunctions and energies
        if n_spin == 1:
            H_kl = np.zeros((*lambda_shape, n_kpts, n_states, n_states), dtype=complex)
        elif n_spin == 2:
            H_kl = np.zeros((*lambda_shape, n_kpts, n_orb, n_spin, n_orb, n_spin), dtype=complex)

        for idx, param_set in enumerate(np.ndindex(*lambda_shape)):
            # kwargs for model_fxn with specified parameter values
            param_dict = {lambda_keys[i]: lambda_ranges[i][param_set] for i in range(dim_lambda)}

            # Generate the model with modified parameters
            modified_model = model_fxn(**param_dict)

            H_kl[param_set] = modified_model.get_ham(k_pts=self.k_mesh.flat_mesh)

        # Reshape for compatibility with existing Berry curvature methods

        if self._nspin == 1:
            new_axes = (dim_lambda,) + tuple(range(dim_lambda)) + tuple(range(dim_lambda+1, dim_lambda+3))
        else:
            new_axes = (dim_lambda,) + tuple(range(dim_lambda))+tuple(range(dim_lambda+1, dim_lambda+5))
        H_kl = np.transpose(H_kl, axes=new_axes)

        if self._nspin == 1:
            new_shape = (*self.k_mesh.nks, *lambda_shape, n_states, n_states)
        else:
            new_shape = (*self.k_mesh.nks, *lambda_shape, n_states, n_orb, n_spin)
        H_kl = H_kl.reshape(new_shape)

        self.H_k = H_kl


    def solve_model(self, model_fxn=None, lambda_vals=None):
        """
        Solves for the eigenstates of the Bloch Hamiltonian defined by the model over a semi-full
        k-mesh, e.g. in 3D reduced coordinates {k = [kx, ky, kz] | k_i in [0, 1)}.

        Args:
            model_fxn (function, optional):
                A function that returns a model given a set of parameters.
            param_vals (dict, optional):
                Dictionary of parameter values for adiabatic evoltuion. Each key corresponds to
                a varying parameter and the values are arrays
        """

        if lambda_vals is None:
            # compute eigenstates and eigenenergies on full k_mesh
            eigvals, eigvecs = self.model.solve_ham(self.k_mesh.flat_mesh, return_eigvecs=True)
            eigvecs = eigvecs.reshape(*self.k_mesh.nks, *eigvecs.shape[1:])
            eigvals = eigvals.reshape(*self.k_mesh.nks, *eigvals.shape[1:])
            self.set_wfs(eigvecs)
            self.energies = eigvals
            self.is_energy_eigstate = True
            return

        lambda_keys = list(lambda_vals.keys())
        lambda_ranges = list(lambda_vals.values())
        lambda_shape = tuple(len(vals) for vals in lambda_ranges)
        dim_lambda = len(lambda_keys)

        n_kpts = self.k_mesh.Nk
        n_orb = self.model.get_num_orbitals()
        n_spin = self.model.n_spin
        n_states = n_orb*n_spin

        # Initialize storage for wavefunctions and energies
        if n_spin == 1:
            u_wfs = np.zeros((*lambda_shape, n_kpts, n_states, n_states), dtype=complex)
        elif n_spin == 2:
            u_wfs = np.zeros((*lambda_shape, n_kpts, n_states, n_orb, n_spin), dtype=complex)

        energies = np.zeros((*lambda_shape, n_kpts, n_states))

        for idx, param_set in enumerate(np.ndindex(*lambda_shape)):
            param_dict = {lambda_keys[i]: lambda_ranges[i][param_set] for i in range(dim_lambda)}

            # Generate the model with modified parameters
            modified_model = model_fxn(**param_dict)

            # Solve for eigenstates
            eigvals, eigvecs = modified_model.solve_ham(self.k_mesh.flat_mesh, return_eigvecs=True)

            # Store results
            energies[param_set] = eigvals
            u_wfs[param_set] = eigvecs

        # Reshape for compatibility with existing Berry curvature methods
        new_axes = (dim_lambda,) + tuple(range(dim_lambda))+(dim_lambda+1, )
        energies = np.transpose(energies, axes=new_axes)
        if self._nspin == 1:
            new_axes = (dim_lambda,) + tuple(range(dim_lambda))+tuple(range(dim_lambda+1, dim_lambda+3))
        else:
            new_axes = (dim_lambda,) + tuple(range(dim_lambda))+tuple(range(dim_lambda+1, dim_lambda+4))
        u_wfs = np.transpose(u_wfs, axes=new_axes)

        if self._nspin == 1:
            new_shape = (*self.k_mesh.nks, *lambda_shape, n_states, n_states)
        else:
            new_shape = (*self.k_mesh.nks, *lambda_shape, n_states, n_orb, n_spin)
        u_wfs = u_wfs.reshape(new_shape)
        energies = energies.reshape((*self.k_mesh.nks, *lambda_shape, n_states))

        self.set_wfs(u_wfs, cell_periodic=True)
        self.energies = energies
        self.is_energy_eigstate = True


    def solve_on_path(self, k_arr):
        """
        Solves on model passed when initialized. Not suitable for
        adiabatic parameters in the model beyond k.
        """
        eigvals, eigvecs = self.model.solve_ham(k_arr, return_eigvecs=True)
        self.set_wfs(eigvecs)
        self.energies = eigvals


    ###### Retrievers  #######

    def get_states(self, flatten_spin=False):
        """Returns dictionary containing Bloch and cell-periodic eigenstates."""
        assert hasattr(self, "_psi_wfs"), "Need to call `solve_model` or `set_wfs` to initialize Bloch states"
        psi_wfs = self._psi_wfs
        u_wfs = self._u_wfs

        if flatten_spin:
            # shape is [nk1, ..., nkd, [n_lambda,] n_state, n_orb, n_spin], flatten last two axes
            psi_wfs = psi_wfs.reshape((*psi_wfs.shape[:-2], -1))
            u_wfs = u_wfs.reshape((*u_wfs.shape[:-2], -1))

        return {"Bloch": psi_wfs,  "Cell periodic": u_wfs}


    def get_projector(self, return_Q = False):
        assert hasattr(self, "_P"), "Need to call `solve_model` or `set_wfs` to initialize Bloch states"
        if return_Q:
            return self._P, self._Q
        else:
            return self._P

    def get_nbr_projector(self, return_Q = False):
        assert hasattr(self, "_P_nbr"), "Need to call `solve_model` or `set_wfs` to initialize Bloch states"
        if return_Q:
            return self._P_nbr, self._Q_nbr
        else:
            return self._P_nbr

    def get_energies(self):
        assert hasattr(self, "energies"), "Need to call `solve_model` to initialize energies"
        return self.energies

    def get_Bloch_Ham(self):
        """Returns the Bloch Hamiltonian of the model defined over the semi-full k-mesh."""
        if hasattr(self, "H_k"):
            return self.H_k
        else:
            self.set_Bloch_ham()
            return self.H_k

    def get_overlap_mat(self):
        """Returns overlap matrix.

        Overlap matrix defined as M_{n,m,k,b} = <u_{n, k} | u_{m, k+b}>
        """
        assert hasattr(self, "_M"), "Need to call `solve_model` or `set_wfs` to initialize overlap matrix"
        return self._M

    def set_wfs(self, wfs, cell_periodic: bool=True, spin_flattened=False, set_projectors=True):
        """
        Sets the Bloch and cell-periodic eigenstates as class attributes.

        Args:
            wfs (np.ndarray):
                Bloch (or cell-periodic) eigenstates defined on a semi-full k-mesh corresponding
                to nks passed during class instantiation. The mesh is assumed to exlude the
                endpoints, e.g. in reduced coordinates {k = [kx, ky, kz] | k_i in [0, 1)}.
        """
        if spin_flattened and self._nspin == 2:
            self._n_states = wfs.shape[-2]
        else:
            self._n_states = wfs.shape[self.state_axis]

        if self.dim_lam > 0:
            if self._nspin == 2:
                self._wf_shape = (*self.nks, *self.n_lambda, self._n_states, self._n_orb, self._nspin)
            else:
                self._wf_shape = (*self.nks, *self.n_lambda, self._n_states, self._n_orb)
        else:
            if self._nspin == 2:
                self._wf_shape = (*self.nks, self._n_states, self._n_orb, self._nspin)
            else:
                self._wf_shape = (*self.nks, self._n_states, self._n_orb)

        wfs = wfs.reshape(self._wf_shape)

        if cell_periodic:
            self._u_wfs = wfs
            self._psi_wfs = self._apply_phase(wfs)
        else:
            self._psi_wfs = wfs
            self._u_wfs = self._apply_phase(wfs, inverse=True)

        if self.dim_lam == 0 and set_projectors:
            # overlap matrix
            self._M = self._get_self_overlap_mat()
            # band projectors
            self._set_projectors()


    def _get_pbc_wfs(self):

        dim_k = self.k_mesh.dim
        orb_vecs = self.model.get_orb_vecs(Cartesian=False)

        # Initialize the extended array by padding with an extra element along each k-axis
        pbc_uwfs = np.pad(
            self._u_wfs, pad_width=[(0, 1) if i < dim_k else (0, 0) for i in range(self._u_wfs.ndim)], mode="wrap")
        pbc_psiwfs = np.pad(
            self._psi_wfs, pad_width=[(0, 1) if i < dim_k else (0, 0) for i in range(self._psi_wfs.ndim)], mode="wrap")

        # Compute the reciprocal lattice vectors (unit vectors for each dimension)
        G_vectors = list(product([0, 1], repeat=dim_k))
        # Remove the zero vector
        G_vectors = [np.array(vector) for vector in G_vectors if any(vector)]

        for  G in G_vectors:
            phase = np.exp(-1j * 2 * np.pi * (orb_vecs @ G.T)).T[np.newaxis, :]
            slices_new = []
            slices_old = []

            for i, value in enumerate(G):
                if value == 1:
                    slices_new.append(slice(-1, None))  # Take the last element along this axis
                    slices_old.append(slice(0, None))
                else:
                    slices_new.append(slice(None))  # Take all elements along this axis
                    slices_old.append(slice(None))  # Take all elements along this axis

            # Add slices for any remaining dimensions (m, n) if necessary
            slices_new.extend([slice(None)] * (pbc_uwfs.ndim - len(G)))
            slices_old.extend([slice(None)] * (pbc_uwfs.ndim - len(G)))
            pbc_uwfs[tuple(slices_new)] *= phase

        return pbc_psiwfs, pbc_uwfs

    # Works with and without spin and lambda
    def _apply_phase(self, wfs, inverse=False):
        """
        Change between cell periodic and Bloch wfs by multiplying exp(\pm i k . tau)

        Args:
        wfs (pythtb.wf_array): Bloch or cell periodic wfs [k, nband, norb]

        Returns:
        wfsxphase (np.ndarray):
            wfs with orbitals multiplied by phase factor

        """
        lam = -1 if inverse else 1  # overall minus if getting cell periodic from Bloch
        per_dir = list(range(self.k_mesh.flat_mesh.shape[-1]))  # list of periodic dimensions
        # slice second dimension to only keep only periodic dimensions in orb
        per_orb = self.model._orb_vecs[:, per_dir]

        # compute a list of phase factors: exp(pm i k . tau) of shape [k_val, orbital]
        phases = np.exp(lam * 1j * 2 * np.pi * per_orb @ self.k_mesh.flat_mesh.T, dtype=complex).T
        phases = phases.reshape(*self.k_mesh.nks, self._n_orb)

        if hasattr(self, "n_lambda") and self.n_lambda:
            phases = phases[..., np.newaxis, :]

        # if len(self._wf_shape) != len(wfs.shape):
        wfs = wfs.reshape(*self._wf_shape)

        # broadcasting to match dimensions
        if self._nspin == 1:
            # reshape to have each k-dimension as an axis
            # wfs = wfs.reshape(*self.k_mesh.nks, self._n_states, self._n_orb)
            # newaxis along state dimension
            phases = phases[..., np.newaxis, :]
        elif self._nspin == 2:
            # reshape to have each k-dimension as an axis
            # newaxis along state and spin dimension
            phases = phases[..., np.newaxis, :, np.newaxis]

        return wfs * phases

    # TODO: allow for projectors onto subbands
    # TODO: possibly get rid of nbr by storing boundary states
    def _set_projectors(self):
        num_nnbrs = self.k_mesh.num_nnbrs
        nnbr_idx_shell = self.k_mesh.nnbr_idx_shell

        if self._nspin == 2:
            u_wfs = self.get_states(flatten_spin=True)["Cell periodic"]
        else:
            u_wfs = self.get_states()["Cell periodic"]

        # band projectors
        self._P = np.einsum("...ni, ...nj -> ...ij", u_wfs, u_wfs.conj())
        self._Q = np.eye(self._n_orb*self._nspin) - self._P

        # NOTE: lambda friendly
        self._P_nbr = np.zeros((self._P.shape[:-2] + (num_nnbrs,) + self._P.shape[-2:]), dtype=complex)
        self._Q_nbr = np.zeros_like(self._P_nbr)

        # NOTE: not lambda friendly
        # self._P_nbr = np.zeros((*nks, num_nnbrs, self._n_orb*self._nspin, self._n_orb*self._nspin), dtype=complex)
        # self._Q_nbr = np.zeros((*nks, num_nnbrs, self._n_orb*self._nspin, self._n_orb*self._nspin), dtype=complex)

        #TODO need shell to iterate over extra lambda dims also, shift accordingly
        for idx, idx_vec in enumerate(nnbr_idx_shell[0]):  # nearest neighbors
            # accounting for phase across the BZ boundary
            states_pbc = np.roll(
                u_wfs, shift=tuple(-idx_vec), axis=self.k_axes
                ) * self.k_mesh.bc_phase[..., idx, np.newaxis,  :]
            self._P_nbr[..., idx, :, :] = np.einsum(
                    "...ni, ...nj -> ...ij", states_pbc, states_pbc.conj()
                    )
            self._Q_nbr[..., idx, :, :] = np.eye(self._n_orb*self._nspin) - self._P_nbr[..., idx, :, :]

        return

    # TODO: allow for subbands and possible lamda dim
    def _get_self_overlap_mat(self):
        """Compute the overlap matrix of the cell periodic eigenstates.

        Overlap matrix of the form

        M_{m,n}^{k, k+b} = < u_{m, k} | u_{n, k+b} >

        Assumes that the last u_wf along each periodic direction corresponds to the
        next to last k-point in the mesh (excludes endpoints).

        Returns:
            M (np.array):
                Overlap matrix with shape [*nks, num_nnbrs, n_states, n_states]
        """

        # Assumes only one shell for now
        _, idx_shell = self.k_mesh.get_k_shell(N_sh=1, report=False)
        idx_shell = idx_shell[0]
        bc_phase = self.k_mesh.bc_phase

        #TODO: Not lambda friendly
        # overlap matrix
        M = np.zeros(
            (*self.k_mesh.nks, len(idx_shell), self._n_states, self._n_states), dtype=complex
        )

        if self._nspin == 2:
            u_wfs = self.get_states(flatten_spin=True)["Cell periodic"]
        else:
            u_wfs = self.get_states()["Cell periodic"]

        for idx, idx_vec in enumerate(idx_shell):  # nearest neighbors
            # introduce phases to states when k+b is across the BZ boundary
            states_pbc = np.roll(
                u_wfs, shift=tuple(-idx_vec), axis=[i for i in range(self.k_mesh.dim)]
                ) * bc_phase[..., idx, np.newaxis,  :]
            M[..., idx, :, :] = np.einsum("...mj, ...nj -> ...mn", u_wfs.conj(), states_pbc)

        return M

    #TODO: Not working
    def berry_phase(self, dir=0, state_idx=None, evals=False):
        """
        Computes Berry phases for wavefunction arrays defined in parameter space.

        Parameters:
            wfs (np.ndarray):
                Wavefunction array of shape [*param_arr_lens, n_orb, n_orb] where
                axis -2 corresponds to the eigenvalue index and axis -1 corresponds
                to amplitude.
            dir (int):
                The direction (axis) in the parameter space along which to compute the Berry phase.

        Returns:
            phase (np.ndarray):
                Berry phases for the specified parameter space direction.
        """
        wfs = self.get_states()["Cell periodic"]
        if state_idx is not None:
            wfs = np.take(wfs, state_idx, axis=self.state_axis)
        orb_vecs = self.model.get_orb_vecs()
        dim_param = self.k_mesh.dim  # dimensionality of parameter space
        param_axes = np.arange(0, dim_param)  # parameter axes
        param_axes = np.setdiff1d(param_axes, dir)  # remove dir from axes to loop
        lens = [wfs.shape[ax] for ax in param_axes]  # sizes of loop directions
        idxs = np.ndindex(*lens)  # index mesh

        phase = np.zeros((*lens, wfs.shape[dim_param]))

        G = np.zeros(dim_param)
        G[0] = 1
        phase_shift = np.exp(-1j * 2 * np.pi * (orb_vecs @ G.T))
        print(param_axes)
        for idx_set in idxs:
            # print(idx_set)
            # take wfs along loop axis at given idex
            sliced_wf = wfs.copy()
            for ax, idx in enumerate(idx_set):
                # print(param_axes[ax])
                sliced_wf = np.take(sliced_wf, idx, axis=param_axes[ax])

            # print(sliced_wf.shape)
            end_state = sliced_wf[0,...] * phase_shift[np.newaxis, :, np.newaxis]
            sliced_wf = np.append(sliced_wf, end_state[np.newaxis, ...], axis=0)
            phases = self.berry_loop(sliced_wf, evals=evals)
            phase[idx_set] = phases

        return phase

    # works in all cases
    def wilson_loop(self, wfs_loop, evals=False):
        """Compute Wilson loop unitary matrix and its eigenvalues for multiband Berry phases.

        Multiband Berry phases always returns numbers between -pi and pi.

        Args:
            wfs_loop (np.ndarray):
                Has format [loop_idx, band, orbital, spin] and loop has to be one dimensional.
                Assumes that first and last loop-point are the same. Therefore if
                there are n wavefunctions in total, will calculate phase along n-1
                links only!
            berry_evals (bool):
                If berry_evals is True then will compute phases for
                individual states, these corresponds to 1d hybrid Wannier
                function centers. Otherwise just return one number, Berry phase.
        """

        wfs_loop = wfs_loop.reshape(wfs_loop.shape[0], wfs_loop.shape[1], -1)
        ovr_mats = wfs_loop[:-1].conj() @ wfs_loop[1:].swapaxes(-2, -1)
        V, _, Wh = np.linalg.svd(ovr_mats, full_matrices=False)
        U_link = V @ Wh
        U_wilson = U_link[0]
        for i in range(1, len(U_link)):
            U_wilson = U_wilson @ U_link[i]

        # calculate phases of all eigenvalues
        if evals:
            evals = np.linalg.eigvals(U_wilson) # Wilson loop eigenvalues
            eval_pha = -np.angle(evals) # Multiband  Berrry phases
            return U_wilson, eval_pha
        else:
            return U_wilson

    # works in all cases
    def berry_loop(self, wfs_loop, evals=False):
        U_wilson = self.wilson_loop(wfs_loop, evals=evals)

        if evals:
            return U_wilson[1]
        else:
            return -np.angle(np.linalg.det(U_wilson)) # total Berry phase

    # Works in all cases
    def get_links(self, state_idx):
        wfs = self.get_states()["Cell periodic"]

        orb_vecs = self.model._orb_vecs      # Orbtial position vectors (reduced units)
        n_param = self.n_adia                # Number of points in adiabatic mesh
        dim = self.dim_adia                  # Total dimensionality of adiabatic space
        n_spin = getattr(self, "_nspin", 1)  # Number of spin components

        # State selection
        if state_idx is not None:
            wfs = np.take(wfs, state_idx, axis=self.state_axis)
            if isinstance(state_idx, int):
                wfs = np.expand_dims(wfs, self.state_axis)

        n_states = wfs.shape[self.state_axis]

        U_forward = []
        wfs_flat = wfs.reshape(*n_param, n_states, -1)
        for mu in range(dim):
            # print(f"Computing links for direction: mu={mu}")
            wfs_shifted = np.roll(wfs, -1, axis=mu)

            # Apply phase factor e^{-i G.r} to shifted u_nk states at the boundary (now 0th state)
            if mu < self.k_mesh.dim:
                mask = np.zeros(n_param, dtype=bool)
                idx = [slice(None)] * dim
                idx[mu] = n_param[mu] - 1
                mask[tuple(idx)] = True

                G = np.zeros(self.k_mesh.dim)
                G[mu] = 1
                phase = np.exp(-2j * np.pi * G @ orb_vecs.T)

                if n_spin == 1:
                    phase_broadcast = phase[np.newaxis, :]
                    mask_expanded = mask[..., np.newaxis, np.newaxis]
                else:
                    phase_broadcast = phase[np.newaxis, :, np.newaxis]
                    mask_expanded = mask[..., np.newaxis, np.newaxis, np.newaxis]

                wfs_shifted = np.where(mask_expanded, wfs_shifted * phase_broadcast, wfs_shifted)

            # Flatten along spin
            wfs_shifted_flat = wfs_shifted.reshape(*n_param, n_states, -1)
            # <u_nk| u_m k+delta_mu>
            ovr_mu = wfs_flat.conj() @ wfs_shifted_flat.swapaxes(-2, -1)

            U_forward_mu = np.zeros_like(ovr_mu, dtype=complex)
            V, _, Wd = np.linalg.svd(ovr_mu, full_matrices= False)
            U_forward_mu = V @ Wd
            U_forward.append(U_forward_mu)

        return np.array(U_forward)



    def berry_flux_plaq(self, state_idx=None, non_abelian=False):
        """Compute fluxes on a two-dimensional plane of states.

        For a given set of states, returns the band summed Berry curvature
        rank-2 tensor for all combinations of surfaces in reciprocal space.
        By convention, the Berry curvature is reported at the point where the loop
        started, which is the lower left corner of a plaquette.
        """
        n_states = len(state_idx)  # Number of states considered
        n_param = self.n_adia      # Number of points in adiabatic mesh
        dim = self.dim_adia        # Total dimensionality of adiabatic space

        # Initialize Berry flux array
        shape = (dim, dim, *n_param, n_states, n_states) if non_abelian else (dim, dim, *n_param)
        Berry_flux = np.zeros(shape, dtype=complex)

        # Overlaps <u_{nk} | u_{n, k+delta k_mu}>
        U_forward = self.get_links(state_idx=state_idx)
        # Wilson loops W = U_{mu}(k_0) U_{nu}(k_0 + delta_mu) U^{-1}_{mu}(k_0 + delta_mu + delta_nu) U^{-1}_{nu}(k_0)
        for mu in range(dim):
            for nu in range(mu+1, dim):
                print(f"Computing flux in plane: mu={mu}, nu={nu}")
                U_mu = U_forward[mu]
                U_nu = U_forward[nu]

                U_nu_shift_mu = np.roll(U_nu, -1, axis=mu)
                U_mu_shift_nu = np.roll(U_mu, -1, axis=nu)

                U_wilson = np.matmul(
                    np.matmul(
                        np.matmul(U_mu, U_nu_shift_mu), U_mu_shift_nu.conj().swapaxes(-1, -2)
                        ),
                        U_nu.conj().swapaxes(-1, -2)
                        )

                if non_abelian:
                    eigvals, eigvecs = np.linalg.eig(U_wilson)
                    angles = -np.angle(eigvals)
                    angles_diag = np.einsum("...i, ij -> ...ij", angles, np.eye(angles.shape[-1]))
                    eigvecs_inv = np.linalg.inv(eigvecs)
                    phases_plane = np.matmul(np.matmul(eigvecs, angles_diag), eigvecs_inv)
                else:
                    det_U = np.linalg.det(U_wilson)
                    phases_plane = -np.angle(det_U)

                Berry_flux[mu, nu] = phases_plane
                Berry_flux[nu, mu] = -phases_plane

        return Berry_flux


    def berry_curv(
            self, dirs=None, state_idx=None, non_abelian=False, delta_lam=1, return_flux=False, Kubo=False
            ):

        nks = self.nks  # Number of mesh points per direction
        n_lambda = self.n_lambda
        dim_k = len(nks)      # Number of k-space dimensions
        dim_lam = len(n_lambda)  # Number of adiabatic dimensions
        dim_total = dim_k + dim_lam          # Total number of dimensions

        if dim_k < 2:
            raise ValueError("Berry curvature only defined for dim_k >= 2.")

        if Kubo:
            if not self.is_energy_eigstate:
                raise ValueError("Must be energy eigenstate to use Kubo formula.")
            if not hasattr(self, "_u_wfs") or not hasattr(self, "energies"):
                raise ValueError("Must diagonalize model first to set wavefunctions and energies.")
            if state_idx is not None:
                print("Berry curvature in Kubo formula is for all occupied bands. Using half filling for occupied bands.")
            if dim_lam != 0 or delta_lam != 1:
                raise ValueError("Adiabatic dimensions not yet supported for Kubo formula.")
            if return_flux:
                print("Kubo formula doesn't support flux. Will return dimensionful Berry curvature only.")

            u_wfs = self.get_states(flatten_spin=True)["Cell periodic"]
            energies = self.energies
            # flatten k_dims
            u_wfs = u_wfs.reshape(-1, u_wfs.shape[-2], u_wfs.shape[-1])
            energies = energies.reshape(-1, energies.shape[-1])
            n_states = u_wfs.shape[-2]

            if n_states != self.model._nstate:
                raise ValueError("Wavefunctions must be defined for all bands, not just a subset.")

            k_mesh = self.k_mesh.flat_mesh
            occ_idx = np.arange(n_states // 2)
            abelian = not non_abelian
            if dirs is None:
                dirs = 'all'
                b_curv = self.model.berry_curvature(k_mesh, evals=energies, evecs=u_wfs, occ_idxs=occ_idx, abelian=abelian)
                b_curv = b_curv.reshape(*b_curv.shape[:2], *nks, *b_curv.shape[3:])
            else:
                b_curv = self.model.berry_curvature(k_mesh, evals=energies, evecs=u_wfs, occ_idxs=occ_idx, abelian=abelian, dirs=dirs)
                b_curv = b_curv.reshape(*nks, *b_curv.shape[3:])

            return b_curv


        Berry_flux = self.berry_flux_plaq(state_idx=state_idx, non_abelian=non_abelian)
        Berry_curv = np.zeros_like(Berry_flux, dtype=complex)

        dim = Berry_flux.shape[0]  # Number of dimensions in parameter space
        recip_lat_vecs = self.model.get_recip_lat_vecs()  # Expressed in cartesian (x,y,z) coordinates

        dks = np.zeros((dim_total, dim_total))
        dks[:dim_k, :dim_k] = recip_lat_vecs / np.array(self.nks)[:, None]
        if self.dim_lam > 0:
            np.fill_diagonal(dks[dim_k:, dim_k:], delta_lam / np.array(self.n_lambda))

        # Divide by area elements for the (mu, nu)-plane
        for mu in range(dim):
            for nu in range(mu+1, dim):
                A = np.vstack([dks[mu], dks[nu]])
                # area_element = np.prod([np.linalg.norm(dk[i]), np.linalg.norm(dk[j])])
                area_element = np.sqrt(np.linalg.det(A @ A.T))

                # Divide flux by the area element to get approx curvature
                Berry_curv[mu, nu] = Berry_flux[mu, nu] / area_element
                Berry_curv[nu, mu] = Berry_flux[nu, mu] / area_element

        if dirs is not None:
            Berry_curv, Berry_flux = Berry_curv[dirs], Berry_flux[dirs]
        if return_flux:
            return Berry_curv, Berry_flux
        else:
            return Berry_curv



    def chern_num(self, dirs=(0,1), band_idxs=None):
        if band_idxs is None:
            n_occ = int(self._n_states/2)
            band_idxs = np.arange(n_occ) # assume half-filled occupied

        berry_flux = self.berry_flux_plaq(state_idx=band_idxs)
        Chern = np.sum(berry_flux[dirs]/(2*np.pi))

        return Chern

    # TODO allow for subbands
    def trace_metric(self):
        P = self._P
        Q_nbr = self._Q_nbr

        nks = Q_nbr.shape[:-3]
        num_nnbrs = Q_nbr.shape[-3]
        w_b, _, _ = self.k_mesh.get_weights(N_sh=1)

        T_kb = np.zeros((*nks, num_nnbrs), dtype=complex)
        for nbr_idx in range(num_nnbrs):  # nearest neighbors
            T_kb[..., nbr_idx] = np.trace(P[..., :, :] @ Q_nbr[..., nbr_idx, :, :], axis1=-1, axis2=-2)

        return w_b[0] * np.sum(T_kb, axis=-1)

    #TODO allow for subbands
    def omega_til(self):
        M = self._M
        w_b, k_shell, idx_shell = self.k_mesh.get_weights(N_sh=1)
        w_b = w_b[0]
        k_shell = k_shell[0]

        nks = M.shape[:-3]
        Nk = np.prod(nks)
        k_axes = tuple([i for i in range(len(nks))])

        diag_M = np.diagonal(M, axis1=-1, axis2=-2)
        log_diag_M_imag = np.log(diag_M).imag
        abs_diag_M_sq = abs(diag_M) ** 2

        r_n = -(1 / Nk) * w_b * np.sum(log_diag_M_imag, axis=k_axes).T @ k_shell

        Omega_tilde = (1 / Nk) * w_b * (
                np.sum((-log_diag_M_imag - k_shell @ r_n.T)**2) +
                np.sum(abs(M)**2) - np.sum(abs_diag_M_sq)
            )
        return Omega_tilde


    def interp_op(self, O_k, k_path, plaq=False):
        k_mesh = np.copy(self.k_mesh.square_mesh)
        k_idx_arr = self.k_mesh.idx_arr
        nks = self.k_mesh.nks
        dim_k = len(nks)
        Nk = np.prod([nks])

        supercell = list(product(*[range(-int((nk-nk%2)/2), int((nk-nk%2)/2)) for nk in nks]))

        if plaq:
            # shift by half a mesh point to get the center of the plaquette
            k_mesh += np.array([(1/nk)/2 for nk in nks])

        # Fourier transform to real space
        O_R = np.zeros((len(supercell), *O_k.shape[dim_k:]), dtype=complex)
        for idx, pos in enumerate(supercell):
            for k_idx in k_idx_arr:
                R_vec = np.array(pos)
                phase = np.exp(-1j * 2 * np.pi * np.vdot(k_mesh[k_idx], R_vec))
                O_R[idx] += O_k[k_idx] * phase / Nk

        # interpolate to arbitrary k
        O_k_interp = np.zeros((k_path.shape[0], *O_k.shape[dim_k:]), dtype=complex)
        for k_idx, k in enumerate(k_path):
            for idx, pos in enumerate(supercell):
                R_vec = np.array(pos)
                phase = np.exp(1j * 2 * np.pi * np.vdot(k, R_vec))
                O_k_interp[k_idx] += O_R[idx] * phase

        return O_k_interp


    def interp_energy(self, k_path, return_eigvecs=False):
        H_k_proj = self.get_proj_ham()
        H_k_interp = self.interp_op(H_k_proj, k_path)
        if return_eigvecs:
            u_k_interp = self.interp_op(self._u_wfs, k_path)
            eigvals_interp, eigvecs_interp = np.linalg.eigh(H_k_interp)
            eigvecs_interp = np.einsum("...ij, ...ik -> ...jk", u_k_interp, eigvecs_interp)
            eigvecs_interp = np.transpose(eigvecs_interp, axes=[0, 2, 1])
            return eigvals_interp, eigvecs_interp
        else:
            eigvals_interp = np.linalg.eigvalsh(H_k_interp)
            return eigvals_interp

    #TODO allow for subbands
    def get_proj_ham(self):
        if not hasattr(self, "H_k_proj"):
            self.set_Bloch_ham()
        H_k_proj = self._u_wfs.conj() @ self.H_k @ np.swapaxes(self._u_wfs, -1, -2)
        return H_k_proj

    #TODO allow for subbands
    def plot_interp_bands(
        self, k_path, nk=101, k_label=None, red_lat_idx=None,
        fig=None, ax=None, title=None, scat_size=3,
        lw=2, lc='b', ls='solid', cmap="bwr", show=False, cbar=True
        ):
        if fig is None and ax is None:
            fig, ax = plt.subplots()

        (k_vec, k_dist, k_node) = self.model.k_path(k_path, nk, report=False)
        k_vec = np.array(k_vec)

        if red_lat_idx is not None:
            eigvals, eigvecs = self.interp_energy(k_vec, return_eigvecs=True)

            n_eigs = eigvecs.shape[-2]
            wt = abs(eigvecs)**2
            col = np.sum([  wt[..., i] for i in red_lat_idx ], axis=0)

            for n in range(n_eigs):
                scat = ax.scatter(k_dist, eigvals[:, n], c=col[:, n], cmap=cmap, marker='o', s=scat_size, vmin=0, vmax=1, zorder=2)

            if cbar:
                cbar = fig.colorbar(scat, ticks=[1,0])
                cbar.ax.set_yticklabels([r'$\psi_2$', r'$\psi_1$'], size=12)

        else:
            eigvals = self.interp_energy(k_vec)

            # continuous bands
            for n in range(eigvals.shape[1]):
                ax.plot(k_dist, eigvals[:, n], c=lc, lw=lw, ls=ls)

        ax.set_xlim(0, k_node[-1])
        ax.set_xticks(k_node)
        for n in range(len(k_node)):
            ax.axvline(x=k_node[n], linewidth=0.5, color='k', zorder=1)
        if k_label is not None:
            ax.set_xticklabels(k_label, size=12)

        ax.set_title(title)
        ax.set_ylabel(r"Energy $E(\mathbf{{k}})$", size=12)
        ax.yaxis.labelpad = 10

        if show:
            plt.show()

        return fig, ax
