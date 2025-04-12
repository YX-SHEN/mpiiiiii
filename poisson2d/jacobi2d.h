#ifndef JACOBI2D_H
#define JACOBI2D_H
#include "poisson2d.h"
#include <mpi.h>

/* Function prototypes for Jacobi 2D methods */

/* 2D sweep function for Jacobi iteration */
void sweep2d(double a[][maxn], double f[][maxn], int nx, int ny,
             int sx, int ex, int sy, int ey, double b[][maxn]);

/* Ghost exchange using MPI_Sendrecv for 2D decomposition */
void exchange2d_sendrecv(double x[][maxn], int nx, int ny, 
                         int sx, int ex, int sy, int ey, MPI_Comm comm,
                         int nbrleft, int nbrright, int nbrbottom, int nbrtop);

/* Ghost exchange using non-blocking communication for 2D decomposition */
void exchange2d_nonblocking(double x[][maxn], int nx, int ny, 
                           int sx, int ex, int sy, int ey, MPI_Comm comm,
                           int nbrleft, int nbrright, int nbrbottom, int nbrtop);

/* Combined exchange and sweep function for 2D decomposition */
void exchange_and_sweep2d(double u[][maxn], double f[][maxn], int nx, int ny,
                         int sx, int ex, int sy, int ey, double unew[][maxn], MPI_Comm comm,
                         int nbrleft, int nbrright, int nbrbottom, int nbrtop);

/* Calculate grid difference for convergence check */
double griddiff2d(double a[][maxn], double b[][maxn], int nx, int ny,
                 int sx, int ex, int sy, int ey);

/* Write grid to file or stdout */
void write_grid2d(double x[][maxn], int nx, int ny, int sx, int ex, int sy, int ey, 
                 int myid, MPI_Comm comm, char *filename);

/* Gather the distributed grid to processor 0 */
void GatherGrid2D(double x[][maxn], int nx, int ny, 
                 int sx, int ex, int sy, int ey,
                 double gathered[][maxn], MPI_Comm comm);

#endif // JACOBI2D_H
