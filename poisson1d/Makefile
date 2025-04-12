# --- Compiler and flags ---
CC = mpicc
CFLAGS = -g -Wall
LDFLAGS = -lm

# --- Object files ---
POISSOBJS = decomp1d.o jacobi.o
EXECS = poiss1d poiss1dnb

# --- Default target ---
all: $(EXECS)

# --- Build Poisson solver with blocking MPI (Q2) ---
poiss1d: main.o $(POISSOBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# --- Build Poisson solver with non-blocking MPI (Q3) ---
poiss1dnb: mainnb.o $(POISSOBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# --- Object compilation rules ---
main.o: main.c poisson1d.h jacobi.h decomp1d.h
	$(CC) $(CFLAGS) -c $<

mainnb.o: mainnb.c poisson1d.h jacobi.h decomp1d.h
	$(CC) $(CFLAGS) -c $<

jacobi.o: jacobi.c jacobi.h poisson1d.h
	$(CC) $(CFLAGS) -c $<

decomp1d.o: decomp1d.c decomp1d.h
	$(CC) $(CFLAGS) -c $<

# --- Run tests ---
run_q2_15:
	mpirun -np 4 ./poiss1d 15 > log_q2_15.txt

run_q2_31:
	mpirun -np 4 ./poiss1d 31 > log_q2_31.txt

run_q3_15:
	mpirun -np 4 ./poiss1dnb 15 > log_q3_15.txt

run_q3_31:
	mpirun -np 4 ./poiss1dnb 31 > log_q3_31.txt

# --- Python plotting / validation scripts ---
plot_solution:
	python plot_solution.py

validate_solution:
	python validate_poisson_solution.py

# --- Tags for navigation ---
tags:
	etags *.c *.h

# --- Clean target ---
clean:
	$(RM) *.o $(EXECS) TAGS tags

# --- Declare phony targets ---
.PHONY: all clean tags plot_solution validate_solution \
        run_q2_15 run_q2_31 run_q3_15 run_q3_31
