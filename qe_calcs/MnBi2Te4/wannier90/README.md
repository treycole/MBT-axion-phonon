All calculations are done with Q=0.01

This file contains the input files for a `quantumespresso` calculation of septuple layers of $\rm Mn Bi_2 Te_4$.

### Wannier windows

- Distentanglement: -7-20 eV
- Frozen: 0-10 eV

### Trial wavefunctions
Number of bands from -7-20 eV = 75 (without SOC)
Number of bands from 0-10 eV ~ 46 (without SOC)

- Mn1 d: 5 (l orbs) x 1 (atoms) x 2 (spins) = 10
- Mn2 d: 5 (l orbs) x 1 (atoms) x 2 (spins) = 10
- Bi p: 3 (l orbs) x 4 (atoms) x 2 (spins) = 24
- Te p: 3 (l orbs) x 8 (atoms) x 2 (spins) = 48
Total number of trial states = 92 (/2 = 46)

Remaining states:
- Mn1 s2: 1 (l orbs) x 1 (atoms) x 2 (spins) = 2
- Mn2 s2: 1 (l orbs) x 1 (atoms) x 2 (spins) = 2
- Bi s: 1 (l orbs) x 4 (atoms) x 2 (spins) = 8
- Te s: 1 (l orbs) x 8 (atoms) x 2 (spins) = 16
