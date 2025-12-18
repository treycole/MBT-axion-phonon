# Phonon mode calculation

## Packages

### SMODES
- SMODES is a symmetry analysis package for phonon modes
- https://iso.byu.edu/smodes.php
- Can find irreps of phonon modes at different k-points
- Input: crystal structure (lattice vectors, atomic positions, space group)
- Output: list of irreps, degeneracies, and displacement patterns for each mode

#### Input
See SMODES.in.txt for input file. The (a, b, c, alpha, beta, gamma) parameters are obtained from MnBI2Te4_hex.cif. This file is from springer https://materials.springer.com/isp/crystallographic/docs/sd_1940632. The Wyckoff positions are from PhysRevMaterials.3.064202.
For some reason, using Wyckoff from cif file gives different atomic positions than those in nscf output. 

#### Output
See SMODES.out.txt for output file. This contains the list of irreps, degeneracies, and displacement patterns for each mode.

## Axion active phonons

1. Run SMODES with primitive non-magnetic cell
- choose Gamma and Z modes
- TR odd modes will be at Z, since atoms with opposite spin have opposite displacement

2. Find axion active irreps 
- Odd under TR
- Odd under improper rotations
- Check literature
- Check character tables for irreps
- Focus on 1d irreps at first (easier to see) 
    - T2- is our irrep of interest

3. Compute the phonon modes with Quantum Espresso or Phonopy
- Give you list of modes (set of displacements of atoms in Cartesian coordinates) and frequency
- Tells you symmetry of each mode (eigenvectors of force constant matrix)
- Filter axion active irreps

4. Apply distortion to unit cell multiplying by small constant so that we are in linear response regime
- Forward and back (positive and negative amplitude)

## Steps
--------
1. 5e-3 -> 0.02 Angstrom amplitude
2. displace smodes output in cartesian
3. use angstrom cartesian coords in QE input file
4. Set tprnfor = :true:
5. Get total forces on each atom
6. Project forces onto mode

- Dynamical matrix is 4x4 matrix in basis of modes of T2-.
- General DM entry i,j is slope of F_j due to displacing atom i. Index i/j 
  is composite for atom i and direction x/y/z.
- Now in 4 dim space entry i,j mode di (basis mode of T2-)
  slope of the force on atom n comp m wrt di on each QE calc 
  dotted into mode dj -> 4x4 matrix
- Diagonalize, 4 eigenvectors give you linear comb of d's, these
  physical modes, eigenvalues are frequencies 
- Displace atoms again based on physical modes
- Wannier Hamiltonian
---------------------------------
- Can't use non-collinear and Hubbard U with tprnfor = .true..
- Use collinear with Hubbard U, then do non-collinear without U
- Forces should be similar with and without U