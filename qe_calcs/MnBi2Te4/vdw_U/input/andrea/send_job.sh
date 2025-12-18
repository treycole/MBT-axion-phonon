#!/bin/bash
set -x
########################################################################
# SUN Grid Engine job wrapper
########################################################################
#$ -N MnBi2Te4_test
#$ -pe orte_one 64
#$ -q dko64m4
#$ -j y
#$ -l h_vmem=3.6G
##$ -M au168@physics.rutgers.edu
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
module load iompi/qe/7.2

#Run memory check in background
sh memory.sh &

#Run command. Run the compiled executable
#mkdir tmp_dir
mpirun -n 32 taskset -c 0-31 pw.x -input MnBi2Te4.scf.in > MnBi2Te4.scf.out
mpirun -n 32 taskset -c 0-31 pw.x -input MnBi2Te4.nscf.in > MnBi2Te4.nscf.out
mpirun -n 32 taskset -c 0-31 pw.x -input MnBi2Te4.bands.in > MnBi2Te4.bands.out
mpirun -n 32 taskset -c 0-31 bands.x -input bands.in > bands_out.out

#Stop memory check
ps axu | grep memory.sh | awk '{print $2}' | xargs kill -9

#Remove heavy temporary files, including wave functions
#rm -r tmp_dir
rm -r tmp
rm -r UNK00*
