#!/bin/bash
set -x
########################################################################
# SUN Grid Engine job wrapper
########################################################################
#$ -N MnBi2Te4
#$ -pe orte_one 64
#$ -q dko64m4
#$ -j y
#$ -l h_vmem=3.9G
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
module load intel/2024 intel/ompi iompi/wann/3.1 iompi/qe/7.2

#Run command. Run the compiled executable
mkdir tmp

mpirun -n $NSLOTS pw.x -input MnBi2Te4.scf.in > MnBi2Te4.scf.out
mpirun -n $NSLOTS pw.x -input MnBi2Te4.nscf.in > MnBi2Te4.nscf.out

# restart mode QE doc
mv # tmp/charge_density .
mv # tmp/XML .

mpirun -n $NSLOTS pw.x -input MnBi2Te4.bands.in > MnBi2Te4.bands.out
mpirun -n $NSLOTS bands.x -input bands.in > bands.out

#Remove heavy temporary files, including wave functions

rm -r tmp
rm -r UNK00*
