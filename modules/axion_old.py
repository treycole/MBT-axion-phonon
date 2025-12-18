from pythtb import Wannier
from pythtb import Bloch, finite_diff_coeffs, levi_civita

import numpy as np
import tensorflow as tf

def flat_spin_H(model, H_k):
    # flatten spin axes
    if H_k.ndim == 2 * model._nspin + 1:
        # have k points
        new_shape = (H_k.shape[0],) + (
            model._nspin * model._norb,
            model._nspin * model._norb,
        )
    elif H_k.ndim == 2 * model._nspin:
        # must be a finite sample, no k-points
        new_shape = (model._nspin * model._norb, model._nspin * model._norb)
    else:
        raise ValueError("Hamiltonian has wrong shape.")

    ham_use = H_k.reshape(*new_shape)

    if not np.allclose(ham_use, ham_use.swapaxes(-1, -2).conj()):
        raise Exception("Hamiltonian matrix is not hermitian")

    return ham_use

def axion_angle(
        model, 
        tf_list,
        nks: tuple, 
        use_curv=True, 
        return_both=False, 
        order_fd=3,
        use_tf_speedup=True
    ):
        r"""Compute the axion angle from the Berry curvature.

        .. versionadded:: 2.0.0

        The axion angle is a topological invariant in three-dimensional insulators,
        related to the magnetoelectric response. It is defined as 

        .. math::

            \theta = -\frac{1}{4\pi} \epsilon^{\mu\nu\rho} 
            \int d^3k \, \text{Tr} 
            \left[ \mathcal{A}_{\mu} \partial_{\nu} \mathcal{A}_{\rho} 
            - \frac{2i}{3} \mathcal{A}_{\mu} \mathcal{A}_{\nu} \mathcal{A}_{\rho} \right]

        Alternatively, it may be expressed 

        .. math::

            \theta = -\frac{1}{4\pi} \epsilon^{\mu\nu\rho} 
            \int d^3k \, \text{Tr} 
            \left[ \frac{1}{2} \mathcal{A}_{\mu} \hat{\Omega}_{\nu\rho} 
            + \frac{i}{3} \mathcal{A}_{\mu} \mathcal{A}_{\nu} \mathcal{A}_{\rho} \right]

        The latter form has the benefit that errors introduced by finite difference approximations
        of :math:`\partial_{\nu} \mathcal{A}_{\rho}` can be avoided by using the Kubo formula for
        computing the Berry curvature :math:`\hat{\Omega}_{\nu\rho}` directly.

        The axion angle is only gauge-invariant modulo :math:`2\pi`, and its precise value can depend 
        on the choice of gauge. Because of this, we must fix the gauge
        choice by using the projection method, often used in the context of Wannier functions. This
        involves projecting the occupied (and conduction) states onto a set of trial wavefunctions to 
        obtain a smooth gauge. The trial wavefunctions should be chosen to have the same symmetry
        properties as the occupied states, and should be linearly independent to ensure a well-defined
        projection. They should be chosen to capture the essential features of the occupied subspace,
        such as the orbital character and spatial localization.

        Parameters
        ----------
        tf_list : list
            List of trial wavefunctions for projection.
        use_curv : bool, optional
            Whether to use the Berry curvature in the calculation. Default is True.
        return_both : bool, optional
            Whether to return both the Berry curvature and the axion angle. Default is False.
        order_fd : int, optional
            Order of the finite difference used in the calculation. Default is 3.
        use_tf_speedup : bool, optional
            Whether to use TensorFlow for speedup. Default is True.

        Returns
        -------
        theta : float
            The computed axion angle.

        Notes
        ------
        The axion angle is only defined for three-dimensional k-space models. It must be ensured by the user
        that the `WFArray` is defined on a 3D k-space mesh, and the underlying model is also 3D. It must also
        be ensured that the `WFArray` is populated by energy eigenstates on the mesh.

        If the system has a non-trivial :math:`\mathbb{Z}_2` index, there is an obstruction to choosing a smooth
        and periodic gauge choice. In this case, one must pick a set of trial wavefunctions that break time-reversal
        symmetry. 

        """
        
        from pythtb import Wannier, Mesh, WFArray
        from pythtb.utils import levi_civita, fin_diff


        mesh = Mesh(model, dim_k=3, dim_param=0, axis_types=["k", "k", "k"])
        mesh.build_full_grid(shape=nks)
        wfa = WFArray(model, mesh)
        wfa.solve_mesh(use_metal=True)

        if wfa.dim_k != 3:
            raise ValueError("Axion angle is only defined for 3D k-space models.")
        if wfa.dim_lambda != 0:
            raise ValueError("Adiabatic dimensions not yet supported for axion angle.")

        flat_mesh = mesh.flat

        E_nk = wfa.energies
        n_states = wfa.nstates   # Total number of states
        n_occ = n_states // 2     # Number of occupied states

        # velocity operator
        v_k = model.grad_ham(flat_mesh)
        # axes for each k-dimension, expand k-dimensions
        v_k = v_k.reshape(model.dim_r, *nks, n_states, n_states)

        # Energy eigensates (flattened spin and unflattened)
        state_dict_flat = wfa.get_bloch_states(flatten_spin=True)

        u_nk_flat = state_dict_flat["cell"]
        psi_nk_flat = state_dict_flat["bloch"]        

        # Getting spin flattened occupied and conduction states and energies
        psi_occ_flat = psi_nk_flat[..., :n_occ, :]
        psi_con_flat = psi_nk_flat[..., n_occ:, :]

        ### Projection
        WF = Wannier(model, wfa)

        # For tilde (projection) gauge states
        bloch_tilde = WF.tilde_states
        bloch_tilde.nstates = len(tf_list)  # Set number of states to number of trial functions

        twfs = WF.get_trial_wfs(tf_list)
        # Flatten spin axis
        twfs_flat = twfs.reshape((*twfs.shape[:1], -1))

        # Overlap matrix S_nm = <psi_nk| g_m> with occupied bands
        S_occ = np.einsum("...nj, mj -> ...nm", psi_occ_flat.conj(), twfs_flat)
        # Overlap matrix S_nm = <psi_nk| g_m> with conduction bands
        S_con = np.einsum("...nj, mj -> ...nm", psi_con_flat.conj(), twfs_flat)

        if use_tf_speedup:
            from tensorflow import convert_to_tensor
            from tensorflow import einsum as tfeinsum
            from tensorflow import complex64 as tfcomplex64
            from tensorflow.linalg import svd as tfsvd

            S_tf = convert_to_tensor(S_occ, dtype=tfcomplex64)

            # batched SVD on Metal
            D, W, V = tfsvd(S_tf, full_matrices=True)

            # back to NumPy for the rest
            W, D, V = W.numpy(), D.numpy(), V.numpy()
            Vh = V.conj().swapaxes(-1,-2)

        else:
            # Use NumPy SVD
            W, D, Vh = np.linalg.svd(S_occ, full_matrices=True)
            V = Vh.conj().swapaxes(-1, -2)

        D_mat = np.einsum("...i, ij -> ...ij", D, np.eye(D.shape[-1])) # Make a diagonal matrix
        # Unitary part
        U_SVD = W @ Vh
        # Semi-positive definite Hermitian part
        P = V @ D_mat @ Vh

        # Use unitary to rotate occupied bands into tilde basis for smooth gauge
        psi_tilde = np.einsum("...mn, ...mj -> ...nj", U_SVD, psi_occ_flat) # shape: (*nks, states, orbs*n_spin])
        # print(psi_tilde.shape)
        # set wfs in Bloch class
        bloch_tilde.set_wfs(psi_tilde, cell_periodic=False, spin_flattened=True, set_projectors=False)

        occ_idxs = np.arange(n_occ)
        cond_idxs = np.setdiff1d(np.arange(n_states), occ_idxs)  # Identify conduction bands

        # Velocity operator in energy eigenbasis
        evecs_conj_tf = u_nk_flat.conj()
        evecs_T_tf = u_nk_flat.swapaxes(-1,-2)  # (n_kpts, n_beta, n_state, n_state)

        v_k_rot = np.matmul(
                evecs_conj_tf[None, ...],  # (1, n_kpts, n_state, n_state)
                np.matmul(
                    v_k,                       # (dim_k, n_kpts, n_state, n_state)
                    evecs_T_tf[None, ...]  # (1, n_kpts, n_beta, n_state, n_state)
                )
            )

        # Compute energy denominators
        delta_E = E_nk[..., None, :] - E_nk[..., :, None]
        delta_E_occ_cond = np.take(np.take(delta_E, occ_idxs, axis=-2), cond_idxs, axis=-1)
        inv_delta_E_occ_cond_tf = 1 / delta_E_occ_cond

        v_occ_cond = np.take(np.take(v_k_rot, occ_idxs, axis=-2), cond_idxs, axis=-1)
        v_cond_occ = np.take(np.take(v_k_rot, cond_idxs, axis=-2), occ_idxs, axis=-1)
        v_occ_cond = v_occ_cond * -inv_delta_E_occ_cond_tf
        v_cond_occ = v_cond_occ * -inv_delta_E_occ_cond_tf

        # vhat
        vhat = v_occ_cond

        orb_vecs = model.orb_vecs
        r_mu_twfs = 2*np.pi * (orb_vecs.T[:, None, :, None] * twfs).reshape(3, 2, 4)
        # rhat
        rhat = np.einsum("...nj, amj -> a...nm", psi_occ_flat.conj(), r_mu_twfs)

        term =  S_occ.conj().swapaxes(-1,-2) @ (rhat + 1j*vhat@ S_con)
        term = Vh @ term @ Vh.conj().swapaxes(-1,-2)

        for a in range(term.shape[-2]):
            for b in range(term.shape[-1]):
                term[..., a, b] *= (1 / (D[..., a] + D[..., b]))

        term = Vh.conj().swapaxes(-1,-2) @ term @ Vh
        term  = term - term.conj().swapaxes(-1,-2)

        # Berry connection in projection gauge
        A_til = (
            U_SVD.conj().swapaxes(-1,-2) @ (rhat + 1j*vhat @ S_con)
            - term
        ) @ np.linalg.inv(P)

        # subtract one since endpoint is included
        dks = [1/(nk-1) for nk in nks]

        # CS Axion angle
        epsilon = levi_civita(3,3)

        if use_curv or return_both:
            omega_kubo = wfa.berry_curv(non_abelian=True, Kubo=True)
            omega_til = np.swapaxes(U_SVD.conj(), -1,-2) @ omega_kubo @ U_SVD

            if use_tf_speedup:
                A_til_tf = convert_to_tensor(A_til, tfcomplex64)
                Omega_til_tf = convert_to_tensor(omega_til, tfcomplex64)
                AOmega = tfeinsum('i...ab,jk...ba->ijk...', A_til_tf, Omega_til_tf)
                AAA = tfeinsum('i...ab,j...bc,k...ca->ijk...', A_til_tf, A_til_tf, A_til_tf)#.numpy()
                integrand = tfeinsum("ijk, ijk... -> ...", epsilon, (1/2) * AOmega + (1j/3) * AAA).numpy()
            else:
                AOmega = np.einsum('i...ab,jk...ba->ijk...', A_til, omega_til)
                AAA = np.einsum('i...ab,j...bc,k...ca->ijk...', A_til, A_til, A_til)
                integrand = np.einsum("ijk, ijk... -> ...", epsilon, (1/2) * AOmega + (1j/3) * AAA)

            # keep all but last k-point along each k-dimension
            integrand = integrand[:-1, :-1, :-1, ...]

            theta = -(4*np.pi)**(-1) * np.sum(integrand) * np.prod(dks)

            if not return_both:
                return theta.real

        A_til_par = A_til[:, :-1, :-1, :-1]
        # Finite difference of A
        parx_A = fin_diff(A_til_par, mu=1, dk_mu=dks[0], order_eps=order_fd)
        pary_A = fin_diff(A_til_par, mu=2, dk_mu=dks[1], order_eps=order_fd)
        parz_A = fin_diff(A_til_par, mu=3, dk_mu=dks[2], order_eps=order_fd)
        par_A = np.array([parx_A, pary_A, parz_A])

        if use_tf_speedup:
            A_til_tf = convert_to_tensor(A_til_par, tfcomplex64)
            AdA = tfeinsum('i...ab,jk...ba->ijk...', A_til_tf, par_A)#.numpy()
            AAA = tfeinsum('i...ab,j...bc,k...ca->ijk...', A_til_tf, A_til_tf, A_til_tf)#.numpy()
            integrand = tfeinsum("ijk, ijk... -> ...", epsilon, AdA - (2j/3) * AAA).numpy()
        else:
            AdA = np.einsum('i...ab,jk...ba->ijk...', A_til_par, par_A)
            AAA = np.einsum('i...ab,j...bc,k...ca->ijk...', A_til_par, A_til_par, A_til_par)
            integrand = np.einsum("ijk, ijk... -> ...", epsilon, AdA - (2j/3) * AAA)

        theta2 = -(4*np.pi)**(-1) * np.sum(integrand) * np.prod(dks)

        if return_both:
            return theta.real, theta2.real
        else:
            return theta2.real


def get_evec_shape(model, H_k):
     if H_k.ndim == 2 * model._nspin + 1:
        if model._nspin == 1:
            shape_evecs = (H_k.shape[0],) + (model._norb, model._norb)
        elif model._nspin == 2:
            shape_evecs = (H_k.shape[0],) + (
                model._nspin * model._norb,
                model._norb,
                model._nspin,
            )
        elif H_k.ndim == 2 * model._nspin:
            if model._nspin == 1:
                shape_evecs = (model._norb, model._norb)
            elif model._nspin == 2:
                shape_evecs = (model._nspin * model._norb, model._norb, model._nspin)
        else:
            raise ValueError("Hamiltonian has wrong shape.")
        return shape_evecs

def fin_diff(U_k, mu, dk_mu, order_eps, mode='central'):
    coeffs, stencil = finite_diff_coeffs(order_eps=order_eps, mode=mode)

    fd_sum = np.zeros_like(U_k)

    for s, c in zip(stencil, coeffs):
        fd_sum += c * np.roll(U_k, shift=-s, axis=mu)

    v = fd_sum / (dk_mu)
    return v


def get_axion_angle(model, tf_list, *nks, curv=True, both=False, order_fd=3):
    n_spin = model.nspin   # Number of spins
    n_orb = model.norb     # Number of orbitals
    n_states = n_spin * n_orb   # Total number of states
    n_occ = n_states//2         # Number of occupied states

    bloch_states = Bloch(model, *nks)
    bloch_states.set_Bloch_ham()

    # Bloch Hamiltonian, reshape and get eigenvector shapes
    H_k = bloch_states.H_k
    H_k = H_k.reshape(-1, n_orb, n_spin, n_orb, n_spin)
    shape_evecs = get_evec_shape(model, H_k)
    H_k = flat_spin_H(model, H_k)

    # k-mesh flat and square
    flat_mesh = bloch_states.k_mesh.flat_mesh

    H_tf = tf.convert_to_tensor(H_k, dtype=tf.complex64)
    eval, evec = tf.linalg.eigh(H_tf)  # runs on METAL if plugin is installed
    eval = eval.numpy()
    evec = evec.numpy()

    evec = evec.swapaxes(-1, -2)
    evec = evec.reshape(*shape_evecs)

    # Make k-axes square instead of flat
    eigvecs = evec.reshape(*bloch_states.k_mesh.nks, *evec.shape[1:])
    eigvals = eval.reshape(*bloch_states.k_mesh.nks, *eval.shape[1:])

    # Set bloch states energies, wfs, and set energy eigenstate flag to True
    bloch_states.energies = eigvals
    bloch_states.set_wfs(eigvecs, spin_flattened=False, set_projectors=False)
    bloch_states.is_energy_eigstate = True

    # velocity operator
    v_k = model.get_velocity(flat_mesh)
    # axes for each k-dimension, flatten spin
    v_k = v_k.reshape(model.dim_r, *nks, n_states, n_states)

    # Energy eigensates (flattened spin and unflattened)
    u_nk = bloch_states.get_states()["Cell periodic"]
    u_nk_flat = bloch_states.get_states(flatten_spin=True)["Cell periodic"]
    E_nk = bloch_states.energies

    # Getting occupied and conduction states and energies
    n_occ = bloch_states._n_states//2
    u_occ = u_nk[..., :n_occ, :, :]
    u_con = u_nk[..., n_occ:, :, :]

    # Bloch class for occupied bands
    bloch_occ = Bloch(model, *nks)
    bloch_occ.set_wfs(u_occ, cell_periodic=True, set_projectors=False)
    # Bloch class for conduction bands
    bloch_con = Bloch(model, *nks)
    bloch_con.set_wfs(u_con, cell_periodic=True, set_projectors=False)

    # Getting spin flattened occupied and conduction states and energies
    psi_occ_flat = bloch_occ.get_states(flatten_spin=True)["Bloch"]
    psi_con_flat = bloch_con.get_states(flatten_spin=True)["Bloch"]

    ### Projection
    # Just for getting trial wavefunctions
    WF = Wannier(model, bloch_states, *nks)

    # For tilde (projection) gauge states
    bloch_tilde = WF.tilde_states

    twfs = WF.get_trial_wfs(tf_list)
    # Flatten spin axis
    twfs_flat = twfs.reshape((*twfs.shape[:1], -1))

    # Overlap matrix S_nm = <psi_nk| g_m> with occupied bands
    S_occ = np.einsum("...nj, mj -> ...nm", psi_occ_flat.conj(), twfs_flat)
    # Overlap matrix S_nm = <psi_nk| g_m> with conduction bands
    S_con = np.einsum("...nj, mj -> ...nm", psi_con_flat.conj(), twfs_flat)

    S_tf = tf.convert_to_tensor(S_occ, dtype=tf.complex64)

    # batched SVD on Metal
    D, W, V = tf.linalg.svd(S_tf, full_matrices=True)

    # back to NumPy for the rest
    W, D, V = W.numpy(), D.numpy(), V.numpy()
    # print(f"Minimum singular value: {np.amin(D)}")

    D_mat = np.einsum("...i, ij -> ...ij", D, np.eye(D.shape[-1])) # Make a diagonal matrix
    Vh = V.conj().swapaxes(-1,-2)
    # Unitary part
    U_SVD = W @ Vh
    # Semi-positive definite Hermitian part
    P = V @ D_mat @ Vh

    # Use unitary to rotate occupied bands into tilde basis for smooth gauge
    psi_tilde = np.einsum("...mn, ...mj -> ...nj", U_SVD, psi_occ_flat) # shape: (*nks, states, orbs*n_spin])
    # set wfs in Bloch class
    bloch_tilde.set_wfs(psi_tilde, cell_periodic=False, spin_flattened=True, set_projectors=False)

    occ_idxs = np.arange(2)
    cond_idxs = np.setdiff1d(np.arange(u_nk_flat.shape[-2]), occ_idxs)  # Identify conduction bands

    # Velocity operator in energy eigenbasis
    evecs_conj_tf = u_nk_flat.conj()
    evecs_T_tf = u_nk_flat.swapaxes(-1,-2)  # (n_kpts, n_beta, n_state, n_state)

    v_k_rot = np.matmul(
            evecs_conj_tf[None, ...],  # (1, n_kpts, n_state, n_state)
            np.matmul(
                v_k,                       # (dim_k, n_kpts, n_state, n_state)
                evecs_T_tf[None, ...]  # (1, n_kpts, n_beta, n_state, n_state)
            )
        )

    # Compute energy denominators
    delta_E = E_nk[..., None, :] - E_nk[..., :, None]
    delta_E_occ_cond = np.take(np.take(delta_E, occ_idxs, axis=-2), cond_idxs, axis=-1)
    inv_delta_E_occ_cond_tf = 1 / delta_E_occ_cond

    v_occ_cond = np.take(np.take(v_k_rot, occ_idxs, axis=-2), cond_idxs, axis=-1)
    v_cond_occ = np.take(np.take(v_k_rot, cond_idxs, axis=-2), occ_idxs, axis=-1)
    v_occ_cond = v_occ_cond * -inv_delta_E_occ_cond_tf
    v_cond_occ = v_cond_occ * -inv_delta_E_occ_cond_tf

    # vhat
    vhat = v_occ_cond

    orb_vecs = model.orb_vecs
    r_mu_twfs = 2*np.pi * (orb_vecs.T[:, None, :, None] * twfs).reshape(3, 2, 4)
    # rhat
    rhat = np.einsum("...nj, amj -> a...nm", psi_occ_flat.conj(), r_mu_twfs)

    term =  S_occ.conj().swapaxes(-1,-2) @ (rhat + 1j*vhat@ S_con)
    term = Vh @ term @ Vh.conj().swapaxes(-1,-2)

    for a in range(term.shape[-2]):
        for b in range(term.shape[-1]):
            term[..., a, b] *= (1 / (D[..., a] + D[..., b]))

    term = Vh.conj().swapaxes(-1,-2) @ term @ Vh
    term  = term - term.conj().swapaxes(-1,-2)

    # Berry connection in projection gauge
    A_til = (
        U_SVD.conj().swapaxes(-1,-2) @ (rhat + 1j*vhat @ S_con)
        - term
    ) @ np.linalg.inv(P)

    # CS Axion angle
    dks = [1/nk for nk in nks]

    if curv:
        omega_kubo = bloch_states.berry_curv(non_abelian=True, Kubo=True)
        omega_til = np.swapaxes(U_SVD.conj(), -1,-2) @ omega_kubo @ U_SVD

        epsilon = levi_civita(3,3)
        A_til_tf = tf.convert_to_tensor(A_til, tf.complex64)
        Omega_til_tf = tf.convert_to_tensor(omega_til, tf.complex64)
        AOmega = tf.einsum('i...ab,jk...ba->ijk...', A_til_tf, Omega_til_tf)
        AAA = tf.einsum('i...ab,j...bc,k...ca->ijk...', A_til_tf, A_til_tf, A_til_tf)#.numpy()
        integrand = tf.einsum("ijk, ijk... -> ...", epsilon, (1/2) * AOmega + (1j/3) * AAA).numpy()

        # intrace = (
        #     A_til[0] @ omega_til[1,2] + A_til[1] @ omega_til[2,0] + A_til[2] @ omega_til[0, 1]
        #     + 1j * (A_til[0] @ A_til[1] - A_til[1] @ A_til[0] ) @ A_til[2]
        # )
        # integrand = np.trace(intrace, axis1=-1, axis2=-2)
        theta = -(4*np.pi)**(-1) * np.sum(integrand) * np.prod(dks)

        if both:
            # Finite difference of A
            parx_A = fin_diff(A_til, mu=1, dk_mu=dks[0], order_eps=order_fd)
            pary_A = fin_diff(A_til, mu=2, dk_mu=dks[1], order_eps=order_fd)
            parz_A = fin_diff(A_til, mu=3, dk_mu=dks[2], order_eps=order_fd)
            par_A = np.array([parx_A, pary_A, parz_A])

            epsilon = levi_civita(3,3)
            A_til_tf = tf.convert_to_tensor(A_til, tf.complex64)
            AdA = tf.einsum('i...ab,jk...ba->ijk...', A_til_tf, par_A)#.numpy()
            AAA = tf.einsum('i...ab,j...bc,k...ca->ijk...', A_til_tf, A_til_tf, A_til_tf)#.numpy()
            integrand = tf.einsum("ijk, ijk... -> ...", epsilon, AdA - (2j/3) * AAA).numpy()
            theta2 = -(4*np.pi)**(-1) * np.sum(integrand) * np.prod(dks)

        return theta.real, theta2.real
    else:
        # Finite difference of A
        parx_A = fin_diff(A_til, mu=1, dk_mu=dks[0], order_eps=order_fd)
        pary_A = fin_diff(A_til, mu=2, dk_mu=dks[1], order_eps=order_fd)
        parz_A = fin_diff(A_til, mu=3, dk_mu=dks[2], order_eps=order_fd)
        par_A = np.array([parx_A, pary_A, parz_A])

        epsilon = levi_civita(3,3)
        A_til_tf = tf.convert_to_tensor(A_til, tf.complex64)
        AdA = tf.einsum('i...ab,jk...ba->ijk...', A_til_tf, par_A)#.numpy()
        AAA = tf.einsum('i...ab,j...bc,k...ca->ijk...', A_til_tf, A_til_tf, A_til_tf)#.numpy()
        integrand = tf.einsum("ijk, ijk... -> ...", epsilon, AdA - (2j/3) * AAA).numpy()
        theta2 = -(4*np.pi)**(-1) * np.sum(integrand) * np.prod(dks)

        return theta2.real
