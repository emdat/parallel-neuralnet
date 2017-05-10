import time
from mpi4py import rc
rc.initialize = False
from mpi4py import MPI
import numpy as np

MPI.Init()
comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank   = comm.Get_rank()

wins = []
win2s = []
hi = np.array((1, 5), dtype=int)
if rank == 0:
    print "r0 hi"
    hi.fill(1)#([1, 2, 3, 4, 5])
    print hi
    bye = np.array([1, 2, 3, 4, 5])
    wins.append(MPI.Win.Create(hi, comm=comm))
else:
    wins.append(MPI.Win.Create(None, comm=comm))       
print "done wins"
if rank > 0:
    print "ok"
    print rank
    wins[0].Lock(rank=0)
    print "locked"
    #hi = np.array([5, 4, 3, 2, 1])
    wins[0].Get([hi, MPI.INT], 0)
    time.sleep(5)
    print "gotten"
    print hi
    comm.Barrier()
    wins[0].Unlock(rank=0)
    print "unlocked"
    print hi
