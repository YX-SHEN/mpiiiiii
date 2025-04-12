#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "poisson2d.h"
#include "jacobi2d.h"

/* 2D sweep function for Jacobi iteration */
void sweep2d(double a[][maxn], double f[][maxn], int nx, int ny,
            int sx, int ex, int sy, int ey, double b[][maxn])
{
    double h;
    int i, j;
    
    h = 1.0/((double)(nx+1));
    
    for (i = sx; i <= ex; i++) {
        for (j = sy; j <= ey; j++) {
            b[i][j] = 0.25 * (a[i-1][j] + a[i+1][j] + a[i][j+1] + a[i][j-1] - h*h*f[i][j]);
        }
    }
}

/* Ghost exchange using MPI_Sendrecv for 2D decomposition */
void exchange2d_sendrecv(double x[][maxn], int nx, int ny, 
                       int sx, int ex, int sy, int ey, MPI_Comm comm,
                       int nbrleft, int nbrright, int nbrbottom, int nbrtop)
{
    /* Exchange in the x-direction (horizontal) */
    if (nbrleft != MPI_PROC_NULL) {
        /* Send leftmost column to left neighbor, receive into left ghost column */
        MPI_Sendrecv(&x[sx][sy], ey-sy+1, MPI_DOUBLE, nbrleft, 0,
                    &x[sx-1][sy], ey-sy+1, MPI_DOUBLE, nbrleft, 1,
                    comm, MPI_STATUS_IGNORE);
    }
    
    if (nbrright != MPI_PROC_NULL) {
        /* Send rightmost column to right neighbor, receive into right ghost column */
        MPI_Sendrecv(&x[ex][sy], ey-sy+1, MPI_DOUBLE, nbrright, 1,
                    &x[ex+1][sy], ey-sy+1, MPI_DOUBLE, nbrright, 0,
                    comm, MPI_STATUS_IGNORE);
    }
    
    /* Create MPI datatype for non-contiguous y-direction (vertical) exchange */
    MPI_Datatype column_type;
    /* The type includes ghost cells in x-direction for completeness */
    MPI_Type_vector(ex-sx+3, 1, maxn, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);
    
    /* Exchange in the y-direction (vertical) */
    if (nbrbottom != MPI_PROC_NULL) {
        /* Send bottom row to bottom neighbor, receive into bottom ghost row */
        MPI_Sendrecv(&x[sx-1][sy], 1, column_type, nbrbottom, 2,
                    &x[sx-1][sy-1], 1, column_type, nbrbottom, 3,
                    comm, MPI_STATUS_IGNORE);
    }
    
    if (nbrtop != MPI_PROC_NULL) {
        /* Send top row to top neighbor, receive into top ghost row */
        MPI_Sendrecv(&x[sx-1][ey], 1, column_type, nbrtop, 3,
                    &x[sx-1][ey+1], 1, column_type, nbrtop, 2,
                    comm, MPI_STATUS_IGNORE);
    }
    
    /* Free the datatype */
    MPI_Type_free(&column_type);
}

/* Ghost exchange using non-blocking communication for 2D decomposition */
void exchange2d_nonblocking(double x[][maxn], int nx, int ny, 
                          int sx, int ex, int sy, int ey, MPI_Comm comm,
                          int nbrleft, int nbrright, int nbrbottom, int nbrtop)
{
    MPI_Request reqs[8];  /* Up to 8 requests (send/recv for each direction) */
    int req_count = 0;
    
    /* Create MPI datatype for non-contiguous vertical data */
    MPI_Datatype column_type;
    MPI_Type_vector(ex-sx+3, 1, maxn, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);
    
    /* Post receives first */
    
    /* Left neighbor */
    if (nbrleft != MPI_PROC_NULL) {
        MPI_Irecv(&x[sx-1][sy], ey-sy+1, MPI_DOUBLE, nbrleft, 1, comm, &reqs[req_count++]);
    }
    
    /* Right neighbor */
    if (nbrright != MPI_PROC_NULL) {
        MPI_Irecv(&x[ex+1][sy], ey-sy+1, MPI_DOUBLE, nbrright, 0, comm, &reqs[req_count++]);
    }
    
    /* Bottom neighbor */
    if (nbrbottom != MPI_PROC_NULL) {
        MPI_Irecv(&x[sx-1][sy-1], 1, column_type, nbrbottom, 3, comm, &reqs[req_count++]);
    }
    
    /* Top neighbor */
    if (nbrtop != MPI_PROC_NULL) {
        MPI_Irecv(&x[sx-1][ey+1], 1, column_type, nbrtop, 2, comm, &reqs[req_count++]);
    }
    
    /* Post sends */
    
    /* To left neighbor */
    if (nbrleft != MPI_PROC_NULL) {
        MPI_Isend(&x[sx][sy], ey-sy+1, MPI_DOUBLE, nbrleft, 0, comm, &reqs[req_count++]);
    }
    
    /* To right neighbor */
    if (nbrright != MPI_PROC_NULL) {
        MPI_Isend(&x[ex][sy], ey-sy+1, MPI_DOUBLE, nbrright, 1, comm, &reqs[req_count++]);
    }
    
    /* To bottom neighbor */
    if (nbrbottom != MPI_PROC_NULL) {
        MPI_Isend(&x[sx-1][sy], 1, column_type, nbrbottom, 2, comm, &reqs[req_count++]);
    }
    
    /* To top neighbor */
    if (nbrtop != MPI_PROC_NULL) {
        MPI_Isend(&x[sx-1][ey], 1, column_type, nbrtop, 3, comm, &reqs[req_count++]);
    }
    
    /* Wait for all communications to complete */
    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
    
    /* Free the datatype */
    MPI_Type_free(&column_type);
}

/* Calculate grid difference for convergence check */
double griddiff2d(double a[][maxn], double b[][maxn], int nx, int ny,
                int sx, int ex, int sy, int ey)
{
    double sum = 0.0;
    double tmp;
    int i, j;
    
    for (i = sx; i <= ex; i++) {
        for (j = sy; j <= ey; j++) {
            tmp = (a[i][j] - b[i][j]);
            sum = sum + tmp*tmp;
        }
    }
    
    return sum;
}

/* Write grid to file or stdout */
void write_grid2d(double x[][maxn], int nx, int ny, int sx, int ex, int sy, int ey, 
                int myid, MPI_Comm comm, char *filename)
{
    int i, j;
    FILE *fp = NULL;
    
    if (filename != NULL && strlen(filename) > 0) {
        fp = fopen(filename, "w");
        if (fp == NULL) {
            fprintf(stderr, "Error opening file %s for writing\n", filename);
            return;
        }
    } else {
        fp = stdout;
    }
    
    /* Write header with grid dimensions */
    fprintf(fp, "%d %d\n", nx, ny);
    
    /* Write grid data in mesh/grid format */
    for (i = 0; i <= nx+1; i++) {
        for (j = 0; j <= ny+1; j++) {
            fprintf(fp, "%lf ", x[i][j]);
        }
        fprintf(fp, "\n");
    }
    
    if (filename != NULL && strlen(filename) > 0) {
        fclose(fp);
    }
}

/* Gather the distributed grid to processor 0 */
void GatherGrid2D(double x[][maxn], int nx, int ny, 
                int sx, int ex, int sy, int ey,
                double gathered[][maxn], MPI_Comm comm)
{
    int myid, nprocs;
    int coords[2], dims[2], periods[2];
    int i, j, p;
    int source_rank, source_coords[2];
    int source_sx, source_ex, source_sy, source_ey;
    double *buffer = NULL;
    int max_size = maxn * maxn;
    MPI_Status status;
    
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &nprocs);
    
    /* Get cartesian topology information */
    MPI_Cart_get(comm, 2, dims, periods, coords);
    
    if (myid == 0) {
        /* Allocate buffer for receiving data */
        buffer = (double *)malloc(max_size * sizeof(double));
        
        /* Copy local data to gathered grid */
        for (i = sx; i <= ex; i++) {
            for (j = sy; j <= ey; j++) {
                gathered[i][j] = x[i][j];
            }
        }
        
        /* Receive data from all other processes */
        for (p = 1; p < nprocs; p++) {
            /* Get coordinates of source process */
            MPI_Cart_coords(comm, p, 2, source_coords);
            
            /* Calculate source grid boundaries */
            MPE_Decomp1d(nx, dims[0], source_coords[0], &source_sx, &source_ex);
            MPE_Decomp1d(ny, dims[1], source_coords[1], &source_sy, &source_ey);
            
            int count = (source_ex - source_sx + 1) * (source_ey - source_sy + 1);
            int position = 0;
            
            /* Receive packed data */
            MPI_Recv(buffer, count, MPI_DOUBLE, p, 0, comm, &status);
            
            /* Unpack data into gathered grid */
            for (i = source_sx; i <= source_ex; i++) {
                for (j = source_sy; j <= source_ey; j++) {
                    gathered[i][j] = buffer[position++];
                }
            }
        }
        
        /* Set boundary values correctly */
        double h = 1.0 / (nx + 1);
        
        /* Bottom boundary: y = 0, u(x, 0) = 0 */
        for (i = 0; i <= nx+1; i++) {
            gathered[i][0] = 0.0;
        }
        
        /* Top boundary: y = 1, u(x, 1) = 1/((1+x)^2+1) */
        for (i = 0; i <= nx+1; i++) {
            double x = i * h;
            gathered[i][ny+1] = 1.0 / ((1.0 + x) * (1.0 + x) + 1.0);
        }
        
        /* Left boundary: x = 0, u(0, y) = y/(1+y^2) */
        for (j = 0; j <= ny+1; j++) {
            double y = j * h;
            gathered[0][j] = y / (1.0 + y * y);
        }
        
        /* Right boundary: x = 1, u(1, y) = y/(4+y^2) */
        for (j = 0; j <= ny+1; j++) {
            double y = j * h;
            gathered[nx+1][j] = y / (4.0 + y * y);
        }
        
        free(buffer);
    } else {
        /* Pack local data (excluding ghost cells) */
        int count = (ex - sx + 1) * (ey - sy + 1);
        buffer = (double *)malloc(count * sizeof(double));
        int position = 0;
        
        for (i = sx; i <= ex; i++) {
            for (j = sy; j <= ey; j++) {
                buffer[position++] = x[i][j];
            }
        }
        
        /* Send packed data to process 0 */
        MPI_Send(buffer, count, MPI_DOUBLE, 0, 0, comm);
        
        free(buffer);
    }
}

double error = compute_error(a, nx, ny, sx, ex, sy, ey);
if (myid == 0)
    printf("Grid size: %d, Error: %.6e\n", nx, error);
