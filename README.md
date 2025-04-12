Exercises 2 – MPI-based Poisson Solvers (Q1–Q4)

This project solves 1D and 2D Poisson equations using MPI-based Jacobi iteration. The project includes blocking and non-blocking versions, as well as structured decomposition for parallelization.

——————
Directory Structure and Files
——————

poisson1d/   (Q1 – Q3)

- main.c             — Blocking Jacobi for Q1–Q2
- mainnb.c           — Non-blocking Jacobi for Q3
- jacobi.c/h         — Jacobi iteration, error, exact solution
- decomp1d.c/h       — 1D domain decomposition
- poisson1d.h        — Common definitions
- poiss1d            — Executable for Q1–Q2
- poiss1dnb          — Executable for Q3
- log_q1.txt         — Output log for Q1
- log_q2_15.txt      — Output log for Q2 (nx=15)
- log_q2_31.txt      — Output log for Q2 (nx=31)
- q3_solution.txt    — Output from Q3

poisson2d/   (Q4)

- main2d.c           — Main driver for Q4
- jacobi2d.c/h       — 2D Jacobi, communication, error, output
- decomp2d.c/h       — 2D domain decomposition using MPI Cartesian
- poisson2d.h        — Common defines
- Makefile           — Supports build and predefined run commands
- poiss2d            — Executable
- q4_solution.txt    — Output grid from Q4
- log_q4_*.txt       — Logs for different configurations (see below)

——————
Build Commands
——————

make         # in poisson1d or poisson2d directory

——————
Run Instructions
——————

Q1:
mpirun -np 4 ./poiss1d 31 > log_q1.txt

Q2:
mpirun -np 4 ./poiss1d 15 > log_q2_15.txt
mpirun -np 4 ./poiss1d 31 > log_q2_31.txt

Q3:
mpirun -np 4 ./poiss1dnb 31 > q3_solution.txt

Q4:
# Blocking
mpirun -np 4 ./poiss2d 15 0 > log_q4_sendrecv_4_15.txt
mpirun -np 4 ./poiss2d 31 0 > log_q4_sendrecv_4_31.txt
mpirun -np 16 ./poiss2d 15 0 > log_q4_sendrecv_16_15.txt
mpirun -np 16 ./poiss2d 31 0 > log_q4_sendrecv_16_31.txt

# Non-blocking
mpirun -np 4 ./poiss2d 15 1 > log_q4_nonblock_4_15.txt
mpirun -np 4 ./poiss2d 31 1 > log_q4_nonblock_4_31.txt
mpirun -np 16 ./poiss2d 15 1 > log_q4_nonblock_16_15.txt
mpirun -np 16 ./poiss2d 31 1 > log_q4_nonblock_16_31.txt

——————
Output Explanation
——————

- All logs (log_q*.txt) contain timing, error, and convergence info.
- Solution files (q3_solution.txt, q4_solution.txt) contain the numerical results.
- Boundary conditions are printed at the end of Q4 runs for verification.
