echo "VARY NUM PROCS"
python serial_tests.py 50000 10 5
mpirun -np 1 python parallel_tests.py 50000 10 30
mpirun -np 2 python parallel_tests.py 50000 10 30
mpirun -np 4 python parallel_tests.py 50000 10 30
mpirun -np 8 python parallel_tests.py 50000 10 30
mpirun -np 16 python parallel_tests.py 50000 10 30
mpirun -np 32 python parallel_tests.py 50000 10 30

echo "VARY NUM EXAMPLES"
python serial_tests.py 25000 10 5
mpirun -np 1 python parallel_tests.py 25000 10 30
mpirun -np 2 python parallel_tests.py 25000 10 30
mpirun -np 4 python parallel_tests.py 25000 10 30
mpirun -np 8 python parallel_tests.py 25000 10 30
mpirun -np 16 python parallel_tests.py 25000 10 30
mpirun -np 32 python parallel_tests.py 25000 10 30

python serial_tests.py 5000 10 5
mpirun -np 1 python parallel_tests.py 5000 10 30
mpirun -np 2 python parallel_tests.py 5000 10 30
mpirun -np 4 python parallel_tests.py 5000 10 30
mpirun -np 8 python parallel_tests.py 5000 10 30
mpirun -np 16 python parallel_tests.py 5000 10 30
mpirun -np 32 python parallel_tests.py 5000 10 30

