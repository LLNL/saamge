# 
# some examples of serial runs
# 

# ./spe10 --small --permeability /home/barker29/spe10/spe_perm.dat --refine 0 --solver both --elems-per-agg 100 --theta 0.001 --nu-pro 2 --nu-relax 2

mpirun -np 2 ./spe10 --permeability /home/barker29/spe10/spe_perm.dat --refine 0 --solver cg --elems-per-agg 100 --theta 0.001 --nu-pro 2 --nu-relax 2

