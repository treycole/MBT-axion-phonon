from pythtb import Wannier, Mesh, WFArray
from pythtb.utils import levi_civita, finite_diff_coeffs
import numpy as np

def fin_diff(U_k, mu, dk_mu, order_eps, mode='central'):
    coeffs, stencil = finite_diff_coeffs(order=order_eps, mode=mode)

    fd_sum = np.zeros_like(U_k)

    for s, c in zip(stencil, coeffs):
        fd_sum += c * np.roll(U_k, shift=-s, axis=mu)

    v = fd_sum / (dk_mu)
    return v

def axion_angle_3form(
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
        mesh = Mesh(dim_k=3, axis_types=["k", "k", "k"])
        mesh.build_grid(shape=nks)
        wfa = WFArray(model.lattice, mesh, spinful=model.spinful)
        wfa.solve_model(model, use_tensorflow=True)

        flat_mesh = mesh.flat

        if wfa.dim_k != 3:
            raise ValueError("Axion angle is only defined for 3D k-space models.")
        if wfa.dim_lambda != 0:
            raise ValueError("Adiabatic dimensions not yet supported for axion angle.")

        E_nk = wfa.energies
        n_states = wfa.nstates   # Total number of states
        n_occ = n_states // 2    # Number of occupied states
        occ_idxs = np.arange(n_occ) # Identify occupied bands
        cond_idxs = np.setdiff1d(np.arange(n_states), occ_idxs)  # Identify conduction bands

        # Energy eigensates (flattened spin and unflattened)
        u_nk_flat, psi_nk_flat = wfa.states(flatten_spin_axis=True, return_psi=True)  

        # Getting spin flattened occupied and conduction states and energies
        psi_occ_flat = psi_nk_flat[..., :n_occ, :]
        psi_con_flat = psi_nk_flat[..., n_occ:, :]

        # --------- Projection ---------
        WF = Wannier(wfa)
        twfs = WF._get_trial_wfs(tf_list) # trial wavefunctions
        twfs_flat = twfs.reshape((*twfs.shape[:1], -1)) # Flatten spin axis

        # Overlap matrix S_nm = <psi_nk| g_m> with occupied bands
        S_occ = np.einsum("...nj, mj -> ...nm", psi_occ_flat.conj(), twfs_flat)
        # Overlap matrix S_nm = <psi_nk| g_m> with conduction bands
        S_con = np.einsum("...nj, mj -> ...nm", psi_con_flat.conj(), twfs_flat)

        if use_tf_speedup:
            import tensorflow as tf

            S_tf = tf.convert_to_tensor(S_occ, dtype=tf.complex64)

            # batched SVD on Metal
            D, W, V = tf.linalg.svd(S_tf, full_matrices=True)

            # back to NumPy for the rest
            W, D, V = W.numpy(), D.numpy(), V.numpy()
            Vh = V.conj().swapaxes(-1,-2)

        else:
            # Use NumPy SVD
            W, D, Vh = np.linalg.svd(S_occ, full_matrices=True)
            V = Vh.conj().swapaxes(-1, -2)

        # Diagonal matrix Sigma from singular values
        eye_trial = np.eye(V.shape[-1], dtype=complex)
        Sigma = np.einsum("...i,ij->...ij", D, eye_trial)   # (..., n_trial, n_trial)
        print("Min singular value of S_occ:", np.min(D))

        U_SVD = W @ Vh  # Unitary part of SVD
        P = V @ Sigma @ Vh  # Semi-positive definite Hermitian part

        # ----- Build V_mu (Kubo numerator / energy denominator), R_mu -----
        # Velocity operator in ORBITAL basis and rotate to eigenbasis:
        # v_k_rot[mu]_{nm} = <u_n| partial_mu H |u_m>

        # Velocity operator in energy eigenbasis
        evecs_conj = u_nk_flat.conj()
        evecs_T = u_nk_flat.swapaxes(-1,-2)  # (n_kpts, n_beta, n_state, n_state)

        
        # ------- V_mu -------

        # velocity operator
        v_k_flat = model.velocity(flat_mesh, flatten_spin_axis=True)  # shape: (dim_k, n_kpts, n_states, n_states)
        # axes for each k-dimension, expand k-dimensions
        v_k = v_k_flat.reshape(model.dim_r, *nks, n_states, n_states)

        # Rotate velocity operator to energy eigenbasis
        if use_tf_speedup:
            v_k_tf = tf.convert_to_tensor(v_k, dtype=tf.complex64)
            evecs_conj_tf = tf.convert_to_tensor(evecs_conj, dtype=tf.complex64)
            evecs_T_tf = tf.convert_to_tensor(evecs_T, dtype=tf.complex64)

            v_k_rot = tf.matmul(
                evecs_conj_tf[None, ...],  # (1, n_kpts, n_state, n_state)
                tf.matmul(
                    v_k_tf,  # (dim_k, n_kpts, n_state, n_state)
                    evecs_T_tf[None, ...],  # (1, n_kpts, n_state, n_state)
                ),
            ).numpy()  # (dim_k, n_kpts, n_state, n_state)
        else:
            v_k_rot = np.matmul(
                    evecs_conj[None, ...],  # (1, n_kpts, n_state, n_state)
                    np.matmul(
                        v_k,                # (dim_k, n_kpts, n_state, n_state)
                        evecs_T[None, ...]  # (1, n_kpts, n_beta, n_state, n_state)
                    )
                )
            
        # Occupied and conduction energies
        E_occ = np.take(E_nk, occ_idxs, axis=-1)
        E_cond = np.take(E_nk, cond_idxs, axis=-1)
        # Delta_{nm} = E_n - E_m (occ - cond)
        delta_occ_cond = E_occ[..., np.newaxis] - E_cond[..., np.newaxis, :]
        if np.any(np.isclose(delta_occ_cond, 0.0)):
            raise ZeroDivisionError(
                "Degenerate occupied/conduction bands encountered."
            )
        
        # Compute energy denominators
        inv_delta_E_occ_cond = np.divide(1.0, delta_occ_cond)  # (..., n_occ, n_cond)
        inv_delta_E_cond_occ = np.swapaxes(inv_delta_E_occ_cond, -2, -1)  # (..., n_cond, n_occ)

        v_occ_cond = np.take(np.take(v_k_rot, occ_idxs, axis=-2), cond_idxs, axis=-1)
        v_cond_occ = np.take(np.take(v_k_rot, cond_idxs, axis=-2), occ_idxs, axis=-1)

        V_mu = v_occ_cond * inv_delta_E_occ_cond

        # ------- R_mu --------

        orb_vecs = model.orb_vecs
        r_mu_twfs = 2*np.pi * (orb_vecs.T[:, None, :, None] * twfs).reshape(3, 2, 4)
        R_mu = np.einsum("...nj, amj -> a...nm", psi_occ_flat.conj(), r_mu_twfs)

        # ------- X_mu --------

        X_mu = -1j * R_mu + V_mu @ S_con

        # ------- A_til --------

        term = Vh @ S_occ.conj().swapaxes(-1,-2) @ X_mu @ Vh.conj().swapaxes(-1,-2)
        term +=  term.conj().swapaxes(-1,-2)  # h.c.

        for a in range(term.shape[-2]):
            for b in range(term.shape[-1]):
                term[..., a, b] *= (1 / (D[..., a] + D[..., b]))

        # Berry connection in projection gauge
        A_til = 1j * (
            U_SVD.conj().swapaxes(-1,-2) @ X_mu
            -  Vh.conj().swapaxes(-1,-2) @ term @ Vh
        ) @ np.linalg.inv(P)

        # CS Axion angle
        dks = [1/nk for nk in nks]
        epsilon = levi_civita(3, 3)

        if use_curv or return_both:
            Q = np.matmul(
                v_occ_cond[:, None]*inv_delta_E_occ_cond,
                v_cond_occ[None, :]*inv_delta_E_cond_occ
                )
        
            omega_kubo = 1j * (Q - np.swapaxes(Q, -1, -2).conj())
            omega_kubo = omega_kubo.reshape(*omega_kubo.shape[:2], *nks, *omega_kubo.shape[-2:])
            #wfa.berry_curv(non_abelian=True, Kubo=True)
            omega_til = np.swapaxes(U_SVD.conj(), -1,-2) @ omega_kubo @ U_SVD

            if use_tf_speedup:
                A_til_tf = tf.convert_to_tensor(A_til, tf.complex64)
                Omega_til_tf = tf.convert_to_tensor(omega_til, tf.complex64)

                AOmega = tf.einsum('i...ab,jk...ba->ijk...', A_til_tf, Omega_til_tf)
                AAA = tf.einsum('i...ab,j...bc,k...ca->ijk...', A_til_tf, A_til_tf, A_til_tf)
                integrand = tf.einsum("ijk, ijk... -> ...", epsilon, (1/2) * AOmega + (1j/3) * AAA).numpy()
            else:
                AOmega = np.einsum('i...ab,jk...ba->ijk...', A_til, omega_til)
                AAA = np.einsum('i...ab,j...bc,k...ca->ijk...', A_til, A_til, A_til)
                integrand = np.einsum("ijk, ijk... -> ...", epsilon, (1/2) * AOmega + (1j/3) * AAA)

            theta = -(4*np.pi)**(-1) * np.sum(integrand) * np.prod(dks)

            if not return_both:
                return theta.real

        A_til_par = A_til
        # Finite difference of A
        parx_A = fin_diff(A_til_par, mu=1, dk_mu=dks[0], order_eps=order_fd)
        pary_A = fin_diff(A_til_par, mu=2, dk_mu=dks[1], order_eps=order_fd)
        parz_A = fin_diff(A_til_par, mu=3, dk_mu=dks[2], order_eps=order_fd)
        par_A = np.array([parx_A, pary_A, parz_A])

        if use_tf_speedup:
            A_til_tf = tf.convert_to_tensor(A_til_par, tf.complex64)
            AdA = tf.einsum('i...ab,jk...ba->ijk...', A_til_tf, par_A)
            AAA = tf.einsum('i...ab,j...bc,k...ca->ijk...', A_til_tf, A_til_tf, A_til_tf)
            integrand = tf.einsum("ijk, ijk... -> ...", epsilon, AdA - (2j/3) * AAA).numpy()
        else:
            AdA = np.einsum('i...ab,jk...ba->ijk...', A_til_par, par_A)
            AAA = np.einsum('i...ab,j...bc,k...ca->ijk...', A_til_par, A_til_par, A_til_par)
            integrand = np.einsum("ijk, ijk... -> ...", epsilon, AdA - (2j/3) * AAA)

        theta2 = -(4*np.pi)**(-1) * np.sum(integrand) * np.prod(dks)

        if return_both:
            return theta.real, theta2.real
        else:
            return theta2.real

