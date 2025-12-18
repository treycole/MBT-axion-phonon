# Linear Response of the Axion Coupling

**David Vanderbilt**  
*8/26/24*

The Chern-Simons axion coupling $\theta_{\text{CS}}$ plays an important role in the topology and magnetoelectric response of magnetic insulators [1, 2]. 
Over the years, our group has developed a variety of methods for computing $\theta_{\text{CS}}$ [3–9].

Incidentally, one approach which I don’t think has been tried, and which may ultimately be the most practical for first-principles calculations, is to use 
disentangled Wannier interpolation methods. That is, after using the Wannier90 code [10, 11] to construct a Wannier representation of all valence and some conduction 
bands from a first-principles calculation of Bloch functions on a coarse mesh, the multiband Berry connection and its $k$-derivatives can be Wannier interpolated 
cheaply onto a fine mesh to perform the needed $k$-space integral. It might be of interest to implement and test such a calculation, and if successful, contribute 
the capability to the open-source `WANNIERBERRI` postprocessing package [12].

However, to my knowledge, the calculation of first derivatives of $\theta_{\text{CS}}$ with respect to perturbations of the Hamiltonian has not previously been addressed. 
This would be the main goal of the project I have in mind.

---

## Motivation

### Electrodynamics Perspective
The electrodynamics in a medium are unaffected by a spatially and temporally uniform $\theta_{\text{CS}}$, being instead only sensitive to its spatial gradients, 
producing local anomalous Hall conductivity: $\mathbf{J}(\mathbf{r}, t) \propto \mathbf{E} \times \nabla \theta_{\text{CS}}(\mathbf{r}, t),$ 
or to its time derivative, producing a local chiral magnetic effect: $\mathbf{J}(\mathbf{r}, t) \propto \mathbf{B} \frac{\partial \theta_{\text{CS}}(\mathbf{r}, t)}{\partial t}.$ 
These effects will be activated insofar as some parameters in the crystal Hamiltonian are spatially inhomogeneous or time-dependent, and their magnitudes will be controlled by the 
first derivatives at issue here.

### Dynamical Axion Field
Another motivation connects with recent explorations of the possibility that the axion field, usually treated as a static background field depending on the ground-state crystal Hamiltonian, 
might be promoted to a dynamical field in some contexts [14–17]. It is unlikely that the axion field would have its own independent dynamics and quasiparticles; instead, $\theta_{\text{CS}}$ 
likely varies with preexisting (e.g., phonon or charge-density-wave) degrees of freedom of the solid. Still, such situations could be quite interesting, especially in topological materials where 
$\theta_{\text{CS}}$ is comparable to or equal to $\pi$.

---

## Approach

There is reason to think that the computation of _derivatives_ of $\theta_{\text{CS}}$ may actually be more straightforward than the computation of 
$\theta_{\text{CS}}$ itself. After all, $\theta_{\text{CS}}$ is a 3D analog of the 1D electric polarization $\mathbf{P}$, and historically first derivatives 
of $\mathbf{P}$, such as dynamical charges and dielectric constants, were computed well before the modern theory of polarization. This can be understood because 
$\mathbf{P}$ itself is only well-defined modulo a quantum and is expressed as an integral of a gauge-dependent Berry connection, whereas its derivative 
$\partial_\lambda \mathbf{P}$ does not suffer from either problem.

Similarly, $\partial_\lambda \theta_{\text{CS}}$ can be expressed as a 3D $k$-space integral of the gauge-invariant 4-curvature:

$$
F^{(4)} = \frac{1}{16\pi} \epsilon^{ijkl} \text{Tr}(\Omega_{ij} \Omega_{kl}),
$$

in $(k_x, k_y, k_z, \lambda)$-space, where $\Omega_{ij}$ and $\Omega_{kl}$ are 2D Berry curvatures such as $\Omega_{k_x k_y}$ or $\Omega_{k_x \lambda}$.

---

## Goal

The goal of this project is to develop and code practical methods for computing the linear response of $\theta_{\text{CS}}$ to perturbations such as lattice displacements, 
external fields, strain, etc. The implementation will be tested by comparison to finite-difference calculations of the full $\theta_{\text{CS}}$. The main focus will be on 
performing this in the first-principles context using density-functional perturbation theory.

For narrow-gap systems or those close to a topological transition, the implementation may need to be extended to the Wannier interpolation context, involving the `Wannier90` code [10, 11] 
to construct a Wannier representation on a coarse $k$-mesh, followed by interpolation onto a fine $k$-mesh to obtain $k$-integrated quantities accurately. 
The $\lambda$-perturbation will be included using methods from [18].

If successful, this capability could be contributed as a feature of the `WANNIERBERRI` postprocessing package [12].

---

## References

1. X. L. Qi, T. L. Hughes, and S. C. Zhang.  
   *“Topological field theory of time-reversal invariant insulators.”*  
   Phys. Rev. B **78**, 195424 (2008).  
   [DOI:10.1103/PhysRevB.78.195424](http://dx.doi.org/10.1103/PhysRevB.78.195424)

2. A. M. Essin, J. E. Moore, and D. Vanderbilt.  
   *“Magnetoelectric polarizability and axion electrodynamics in crystalline insulators.”*  
   Phys. Rev. Lett. **102**, 146805 (2009).  
   [DOI:10.1103/PhysRevLett.102.146805](http://dx.doi.org/10.1103/PhysRevLett.102.146805)

3. A. M. Essin, A. M. Turner, J. E. Moore, and D. Vanderbilt.  
   *“Orbital magnetoelectric coupling in band insulators.”*  
   Phys. Rev. B **81**, 205104 (2010).  
   [DOI:10.1103/PhysRevB.81.205104](http://dx.doi.org/10.1103/PhysRevB.81.205104)

4. A. Malashevich, I. Souza, S. Coh, and D. Vanderbilt.  
   *“Theory of orbital magnetoelectric response.”*  
   New J. Phys. **12**, 053032 (2010).  
   [DOI:10.1088/1367-2630/12/5/053032](http://stacks.iop.org/1367-2630/12/i=5/a=053032)

5. S. Coh, D. Vanderbilt, A. Malashevich, and I. Souza.  
   *“Chern-Simons orbital magnetoelectric coupling in generic insulators.”*  
   Phys. Rev. B **83**, 085108 (2011).  
   [DOI:10.1103/PhysRevB.83.085108](http://dx.doi.org/10.1103/PhysRevB.83.085108)

6. A. Malashevich, S. Coh, I. Souza, and D. Vanderbilt.  
   *“Full magnetoelectric response of Cr₂O₃ from first principles.”*  
   Phys. Rev. B **86**, 094430 (2012).  
   [DOI:10.1103/PhysRevB.86.094430](http://dx.doi.org/10.1103/PhysRevB.86.094430)

7. S. Coh and D. Vanderbilt.  
   *“Canonical magnetic insulators with isotropic magnetoelectric coupling.”*  
   Phys. Rev. B **88**, 121106 (2013).  
   [DOI:10.1103/PhysRevB.88.121106](http://dx.doi.org/10.1103/PhysRevB.88.121106)

8. T. Olsen, M. Taherinejad, D. Vanderbilt, and I. Souza.  
   *“Surface theorem for the Chern-Simons axion coupling.”*  
   Phys. Rev. B **95**, 075137 (2017).  
   [DOI:10.1103/PhysRevB.95.075137](http://dx.doi.org/10.1103/PhysRevB.95.075137)

9. N. Varnava, I. Souza, and D. Vanderbilt.  
   *“Axion coupling in the hybrid Wannier representation.”*  
   Phys. Rev. B **101**, 155130 (2020).  
   [DOI:10.1103/PhysRevB.101.155130](http://dx.doi.org/10.1103/PhysRevB.101.155130)

10. A. A. Mostofi, J. R. Yates, Y. S. Lee, I. Souza, D. Vanderbilt, and N. Marzari.  
    *“Wannier90: A tool for obtaining maximally-localised Wannier functions.”*  
    Comput. Phys. Commun. **178**, 685 (2008).  
    [DOI:10.1016/j.cpc.2007.11.016](http://dx.doi.org/10.1016/j.cpc.2007.11.016)

11. G. Pizzi, V. Vitale, R. Arita, S. Blügel, F. Freimuth, G. Géranton, M. Gibertini, D. Gresch, C. Johnson,  
    T. Koretsune, J. Ibañez-Azpiroz, H. Lee, J.-M. Lihm, D. Marchand, A. Marrazzo, Y. Mokrousov, J. I.  
    Mustafa, Y. Nohara, Y. Nomura, L. Paulatto, S. Poncé, T. Ponweiser, J. Qiao, F. Thöle, S. S. Tsirkin,  
    M. Wierzbowska, N. Marzari, D. Vanderbilt, I. Souza, A. A. Mostofi, and J. R. Yates.  
    *“Wannier90 as a community code: new features and applications.”*  
    J. Phys.: Condens. Matter **32**, 165902 (2020).  
    [DOI:10.1088/1361-648X/ab51ff](http://dx.doi.org/10.1088/1361-648X/ab51ff)

12. S. S. Tsirkin.  
    *“High performance Wannier interpolation of Berry curvature and related quantities with WannierBerri code.”*  
    npj Comput Mater **7**, 1 (2021).  
    [DOI:10.1038/s41524-021-00498-5](http://dx.doi.org/10.1038/s41524-021-00498-5)

13. N. Armitage, E. Mele, and A. Vishwanath.  
    *“Weyl and Dirac semimetals in three dimensional solids.”*  
    Rev. Mod. Phys. **90**, 015001 (2018).  
    [DOI:10.1103/RevModPhys.90.015001](http://dx.doi.org/10.1103/RevModPhys.90.015001)

14. R. Li, J. Wang, X.-L. Qi, and S.-C. Zhang.  
    *“Dynamical axion field in topological magnetic insulators.”*  
    Nature Physics **6**, 284 (2010).  
    [DOI:10.1038/nphys1534](http://dx.doi.org/10.1038/nphys1534)

15. H. Ooguri and M. Oshikawa.  
    *“Instability in magnetic materials with a dynamical axion field.”*  
    Phys. Rev. Lett. **108**, 161803 (2012).  
    [DOI:10.1103/PhysRevLett.108.161803](http://dx.doi.org/10.1103/PhysRevLett.108.161803)

16. J. Wang, B. Lian, and S.-C. Zhang.  
    *“Dynamical axion field in a magnetic topological insulator superlattice.”*  
    Phys. Rev. B **93**, 045115 (2016).  
    [DOI:10.1103/PhysRevB.93.045115](http://dx.doi.org/10.1103/PhysRevB.93.045115)

17. B. J. Wieder, K.-S. Lin, and B. Bradlyn.  
    *“Axionic band topology in inversion-symmetric Weyl-charge-density waves.”*  
    Phys. Rev. Res. **2**, 042010 (2020).  
    [DOI:10.1103/PhysRevResearch.2.042010](http://dx.doi.org/10.1103/PhysRevResearch.2.042010)

18. J.-M. Lihm and C.-H. Park.  
    *“Wannier function perturbation theory: Localized representation and interpolation of wave function perturbation.”*  
    Phys. Rev. X **11**, 041053 (2021).  
    [DOI:10.1103/PhysRevX.11.041053](http://dx.doi.org/10.1103/PhysRevX.11.041053)
