#!/bin/bash
set -x
########################################################################
# SUN Grid Engine job wrapper
########################################################################
#$ -N MnBi2Te4_m2_p01
#$ -pe orte_one 64
#$ -q dko64m4
#$ -j y
#$ -l h_vmem=3.6G
##$ -M bc889@physics.rutgers.edu
##$ -m e
#$ -v WIEN_DMFT_ROOT,WIENROOT,LD_LIBRARY_PATH,PATH
########################################################################
# DON'T remove the following line!
source $TMPDIR/sge_init.sh
########################################################################
export SMPD_OPTION_NO_DYNAMIC_HOSTS=1
export OMP_NUM_THREADS=1
export PATH=.:$PATH
export MODULEPATH=/opt/apps/modulefiles:/opt/intel/modulefiles:/opt/pgi/modulefiles:/opt/gnu/modulefiles:/opt/sw/modulefiles
 
#Loading IB modules
module load intel/2024 intel/ompi 
module load iompi/wann/3.1
module load iompi/qe/7.2

#Run memory check in background
sh memory.sh &

#Run command. Run the compiled executable
#mkdir tmp

# Untar save directory
tar -xzvf tmp.tar.gz

# First set: inside .save, ends with .dat.gz
for f in tmp/MnBi2Te4.save/wfc*.dat.gz; do
  n=$(echo "$f" | sed -E 's/.*wfc([0-9]+)\.dat\.gz/\1/')
  gunzip -c "$f" > "tmp/MnBi2Te4.save/wfc${n}.dat"
done

# Second set: in run dir, ends with just .gz
for f in tmp/MnBi2Te4.wfc*.gz; do
  n=$(echo "$f" | sed -E 's/.*wfc([0-9]+)\.gz/\1/')
  gunzip -c "$f" > "tmp/wfc${n}"
done

# Quantum Espresso
mpirun -np $NSLOTS pw.x -input MnBi2Te4.scf.in > MnBi2Te4.scf.out
# mpirun -np $NSLOTS pw.x -input MnBi2Te4.nscf.in > MnBi2Te4.nscf.out
#mpirun -n 32 taskset -c 0-31 pw.x -input MnBi2Te4.bands.in > MnBi2Te4.bands.out
#mpirun -n 32 taskset -c 0-31 bands.x -input bands.in > bands_out.out

# Wannier90
# mpirun -np $NSLOTS wannier90.x -pp MnBi2Te4 > MnBi2Te4_pp.wout
# mpirun -np $NSLOTS pw2wannier90.x -pd true -input MnBi2Te4_pw2wan.in > MnBi2Te4_pw2wan.out
# mpirun -np $NSLOTS wannier90.x MnBi2Te4 > MnBi2Te4.wout


# Remove heavy temporary files, including wave functions
#rm -r tmp
rm -r UNK00*
