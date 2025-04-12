# --- Compiler and flags ---
CC = mpicc
CFLAGS = -g -Wall
LDFLAGS = -lm

# --- Object files ---
POISS2DOBJS = decomp2d.o jacobi2d.o

# --- All executables ---
EXECS = poiss2d

# --- Default target ---
all: $(EXECS)

# --- Build 2D Poisson solver (Q4) ---
poiss2d: main2d.o $(POISS2DOBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# --- Object compilation rules ---
main2d.o: main2d.c poisson2d.h jacobi2d.h decomp2d.h
	$(CC) $(CFLAGS) -c $<

jacobi2d.o: jacobi2d.c jacobi2d.h poisson2d.h
	$(CC) $(CFLAGS) -c $<

decomp2d.o: decomp2d.c decomp2d.h
	$(CC) $(CFLAGS) -c $<

# --- Run tests for Q4 (2D decomposition) ---
run_q4_4_15:
	mpirun -np 4 ./poiss2d 15 0 > log_q4_sendrecv_4_15.txt
	mpirun -np 4 ./poiss2d 15 1 > log_q4_nonblock_4_15.txt

run_q4_4_31:
	mpirun -np 4 ./poiss2d 31 0 > log_q4_sendrecv_4_31.txt
	mpirun -np 4 ./poiss2d 31 1 > log_q4_nonblock_4_31.txt

run_q4_16_15:
	mpirun -np 16 ./poiss2d 15 0 > log_q4_sendrecv_16_15.txt
	mpirun -np 16 ./poiss2d 15 1 > log_q4_nonblock_16_15.txt

run_q4_16_31:
	mpirun -np 16 ./poiss2d 31 0 > log_q4_sendrecv_16_31.txt
	mpirun -np 16 ./poiss2d 31 1 > log_q4_nonblock_16_31.txt

# --- Run all Q4 tests ---
run_all: run_q4_4_15 run_q4_4_31 run_q4_16_15 run_q4_16_31

# --- Tags for navigation ---
tags:
	etags *.c *.h

# --- Clean target ---
clean:
	$(RM) *.o $(EXECS) TAGS tags *.txt

# --- Declare phony targets ---
.PHONY: all clean tags run_all run_q4_4_15 run_q4_4_31 run_q4_16_15 run_q4_16_31
