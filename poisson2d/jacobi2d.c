#include <stdlib.h>
#include <stdio.h>
#include <string.h>  // Added for strlen()
#include <mpi.h>
#include <math.h> 
#include "decomp2d.h"    
#include "poisson2d.h"
#include "jacobi2d.h"

/* 2D sweep function for Jacobi iteration */
void sweep2d(double a[][maxn], double f[][maxn], int nx, int ny,
             int sx, int ex, int sy, int ey, double b[][maxn])
{
    double h;
    int i, j;
    
    h = 1.0/((double)(nx+1));
    
    for(i=sx; i<=ex; i++) {
        for(j=sy; j<=ey; j++) {
            b[i][j] = 0.25 * (a[i-1][j] + a[i+1][j] + a[i][j+1] + a[i][j-1] - h*h*f[i][j]);
        }
    }
}

/* Ghost exchange using MPI_Sendrecv for 2D decomposition */
void exchange2d_sendrecv(double x[][maxn], int nx, int ny, 
                         int sx, int ex, int sy, int ey, MPI_Comm comm,
                         int nbrleft, int nbrright, int nbrbottom, int nbrtop)
{
    // Exchange in the x-direction (left-right)
    MPI_Sendrecv(&x[ex][sy], ey-sy+1, MPI_DOUBLE, nbrright, 0, 
                 &x[sx-1][sy], ey-sy+1, MPI_DOUBLE, nbrleft, 0, 
                 comm, MPI_STATUS_IGNORE);
    
    MPI_Sendrecv(&x[sx][sy], ey-sy+1, MPI_DOUBLE, nbrleft, 1, 
                 &x[ex+1][sy], ey-sy+1, MPI_DOUBLE, nbrright, 1, 
                 comm, MPI_STATUS_IGNORE);
    
    // Create MPI datatype for non-contiguous y-direction exchange
    MPI_Datatype column_type;
    int count = ex - sx + 3; // 包含所有行（包括ghost cells）
    MPI_Type_vector(count, 1, maxn, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);
    
    // Exchange in the y-direction (bottom-top)
    MPI_Sendrecv(&x[sx-1][ey], 1, column_type, nbrtop, 2, 
                 &x[sx-1][sy-1], 1, column_type, nbrbottom, 2, 
                 comm, MPI_STATUS_IGNORE);
    
    MPI_Sendrecv(&x[sx-1][sy], 1, column_type, nbrbottom, 3, 
                 &x[sx-1][ey+1], 1, column_type, nbrtop, 3, 
                 comm, MPI_STATUS_IGNORE);
    
    MPI_Type_free(&column_type);
}

/* Ghost exchange using non-blocking communication for 2D decomposition */
void exchange2d_nonblocking(double x[][maxn], int nx, int ny, 
                           int sx, int ex, int sy, int ey, MPI_Comm comm,
                           int nbrleft, int nbrright, int nbrbottom, int nbrtop)
{
    MPI_Request reqs[8];
    MPI_Datatype column_type;
    int req_count = 0;
    
    // Create MPI datatype for non-contiguous column data
    MPI_Type_vector(ex-sx+3, 1, maxn, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);
    
    // Post receives first
    // Left neighbor
    if (nbrleft != MPI_PROC_NULL) {
        MPI_Irecv(&x[sx-1][sy], ey-sy+1, MPI_DOUBLE, nbrleft, 0, comm, &reqs[req_count++]);
    }
    
    // Right neighbor
    if (nbrright != MPI_PROC_NULL) {
        MPI_Irecv(&x[ex+1][sy], ey-sy+1, MPI_DOUBLE, nbrright, 1, comm, &reqs[req_count++]);
    }
    
    // Bottom neighbor
    if (nbrbottom != MPI_PROC_NULL) {
        MPI_Irecv(&x[sx-1][sy-1], 1, column_type, nbrbottom, 2, comm, &reqs[req_count++]);
    }
    
    // Top neighbor
    if (nbrtop != MPI_PROC_NULL) {
        MPI_Irecv(&x[sx-1][ey+1], 1, column_type, nbrtop, 3, comm, &reqs[req_count++]);
    }
    
    // Now post sends
    // To right neighbor
    if (nbrright != MPI_PROC_NULL) {
        MPI_Isend(&x[ex][sy], ey-sy+1, MPI_DOUBLE, nbrright, 0, comm, &reqs[req_count++]);
    }
    
    // To left neighbor
    if (nbrleft != MPI_PROC_NULL) {
        MPI_Isend(&x[sx][sy], ey-sy+1, MPI_DOUBLE, nbrleft, 1, comm, &reqs[req_count++]);
    }
    
    // To top neighbor
    if (nbrtop != MPI_PROC_NULL) {
        MPI_Isend(&x[sx-1][ey], 1, column_type, nbrtop, 2, comm, &reqs[req_count++]);
    }
    
    // To bottom neighbor
    if (nbrbottom != MPI_PROC_NULL) {
        MPI_Isend(&x[sx-1][sy], 1, column_type, nbrbottom, 3, comm, &reqs[req_count++]);
    }
    
    // Wait for all communications to complete
    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
    
    // Free the datatype
    MPI_Type_free(&column_type);
}

/* Combined exchange and sweep function for 2D decomposition */
void exchange_and_sweep2d(double u[][maxn], double f[][maxn], int nx, int ny,
                         int sx, int ex, int sy, int ey, double unew[][maxn], MPI_Comm comm,
                         int nbrleft, int nbrright, int nbrbottom, int nbrtop)
{
    MPI_Request reqs[8];
    MPI_Datatype column_type;
    int req_count = 0;
    double h;
    int i, j;
    
    h = 1.0/((double)(nx+1));
    
    // Create MPI datatype for non-contiguous column data
    MPI_Type_vector(ex-sx+3, 1, maxn, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);
    
    // Post receives
    if (nbrleft != MPI_PROC_NULL) {
        MPI_Irecv(&u[sx-1][sy], ey-sy+1, MPI_DOUBLE, nbrleft, 0, comm, &reqs[req_count++]);
    }
    
    if (nbrright != MPI_PROC_NULL) {
        MPI_Irecv(&u[ex+1][sy], ey-sy+1, MPI_DOUBLE, nbrright, 1, comm, &reqs[req_count++]);
    }
    
    if (nbrbottom != MPI_PROC_NULL) {
        MPI_Irecv(&u[sx-1][sy-1], 1, column_type, nbrbottom, 2, comm, &reqs[req_count++]);
    }
    
    if (nbrtop != MPI_PROC_NULL) {
        MPI_Irecv(&u[sx-1][ey+1], 1, column_type, nbrtop, 3, comm, &reqs[req_count++]);
    }
    
    // Post sends
    if (nbrright != MPI_PROC_NULL) {
        MPI_Isend(&u[ex][sy], ey-sy+1, MPI_DOUBLE, nbrright, 0, comm, &reqs[req_count++]);
    }
    
    if (nbrleft != MPI_PROC_NULL) {
        MPI_Isend(&u[sx][sy], ey-sy+1, MPI_DOUBLE, nbrleft, 1, comm, &reqs[req_count++]);
    }
    
    if (nbrtop != MPI_PROC_NULL) {
        MPI_Isend(&u[sx-1][ey], 1, column_type, nbrtop, 2, comm, &reqs[req_count++]);
    }
    
    if (nbrbottom != MPI_PROC_NULL) {
        MPI_Isend(&u[sx-1][sy], 1, column_type, nbrbottom, 3, comm, &reqs[req_count++]);
    }
    
    // Perform internal updates (not dependent on ghost cells)
    if ((ex-sx+1 > 2) && (ey-sy+1 > 2)) {
        for (i = sx+1; i < ex; i++) {
            for (j = sy+1; j < ey; j++) {
                unew[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j+1] + u[i][j-1] - h*h*f[i][j]);
            }
        }
    }
    
    // Boundary points that need only some ghost cells
    // Top and bottom rows (except corners)
    for (i = sx+1; i < ex; i++) {
        // Bottom
        unew[i][sy] = 0.25 * (u[i-1][sy] + u[i+1][sy] + u[i][sy+1] - h*h*f[i][sy]);
        // Top
        unew[i][ey] = 0.25 * (u[i-1][ey] + u[i+1][ey] + u[i][ey-1] - h*h*f[i][ey]);
    }
    
    // Left and right columns (except corners)
    for (j = sy+1; j < ey; j++) {
        // Left
        unew[sx][j] = 0.25 * (u[sx][j-1] + u[sx][j+1] + u[sx+1][j] - h*h*f[sx][j]);
        // Right
        unew[ex][j] = 0.25 * (u[ex][j-1] + u[ex][j+1] + u[ex-1][j] - h*h*f[ex][j]);
    }
    
    // Wait for all communications to complete
    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
    
    // Now update points that required ghost cell data
    // Corner points
    unew[sx][sy] = 0.25 * (u[sx-1][sy] + u[sx+1][sy] + u[sx][sy-1] + u[sx][sy+1] - h*h*f[sx][sy]);
    unew[ex][sy] = 0.25 * (u[ex-1][sy] + u[ex+1][sy] + u[ex][sy-1] + u[ex][sy+1] - h*h*f[ex][sy]);
    unew[sx][ey] = 0.25 * (u[sx-1][ey] + u[sx+1][ey] + u[sx][ey-1] + u[sx][ey+1] - h*h*f[sx][ey]);
    unew[ex][ey] = 0.25 * (u[ex-1][ey] + u[ex+1][ey] + u[ex][ey-1] + u[ex][ey+1] - h*h*f[ex][ey]);
    
    // Complete the updates for boundary points
    // Left and right columns (except corners) - add ghost cell contributions
    for (j = sy+1; j < ey; j++) {
        unew[sx][j] += 0.25 * u[sx-1][j];
        unew[ex][j] += 0.25 * u[ex+1][j];
    }
    
    // Top and bottom rows (except corners) - add ghost cell contributions
    for (i = sx+1; i < ex; i++) {
        unew[i][sy] += 0.25 * u[i][sy-1];
        unew[i][ey] += 0.25 * u[i][ey+1];
    }
    
    // Free the datatype
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
    
    fprintf(fp, "# Grid data for process %d\n", myid);
    fprintf(fp, "# Local grid boundaries: [%d:%d, %d:%d]\n", sx, ex, sy, ey);
    fprintf(fp, "# Format: i j x[i][j]\n");
    
    for (i = sx; i <= ex; i++) {
        for (j = sy; j <= ey; j++) {
            fprintf(fp, "%d %d %lf\n", i, j, x[i][j]);
        }
    }
    
    if (filename != NULL && strlen(filename) > 0) {
        fclose(fp);
    }
}

/* Gather the distributed grid to processor 0 */
/* Gather the distributed grid to processor 0 */
void GatherGrid2D(double x[][maxn], int nx, int ny,
    int sx, int ex, int sy, int ey,
    double gathered[][maxn], MPI_Comm comm_2d)
{
    int myid, nprocs;
    MPI_Comm_rank(comm_2d, &myid);
    MPI_Comm_size(comm_2d, &nprocs);

    // 1. 获取拓扑信息
    int dims[2], periods[2], coords[2];
    MPI_Cart_get(comm_2d, 2, dims, periods, coords);

    // 2. 计算有效数据区域（不含ghost cells）
    int local_nx = ex - sx + 1;
    int local_ny = ey - sy + 1;
    double *sendbuf = (double*)malloc(local_nx * local_ny * sizeof(double));

    // 3. 展平数据（行优先）
    for(int i = 0; i < local_nx; i++) {
        for(int j = 0; j < local_ny; j++) {
            sendbuf[i*local_ny + j] = x[sx + i][sy + j];
        }
    }

    // 4. 准备收集参数（仅进程0）
    int *recvcounts = NULL, *displs = NULL;
    double *recvbuf = NULL;

    if(myid == 0) {
        recvcounts = (int*)malloc(nprocs * sizeof(int));
        displs = (int*)malloc(nprocs * sizeof(int));
        recvbuf = (double*)malloc(nx * ny * sizeof(double));

        // 计算每个进程的数据量和位移
        int offset = 0;
        for(int rank = 0; rank < nprocs; rank++) {
            int rank_coords[2];
            MPI_Cart_coords(comm_2d, rank, 2, rank_coords);

            // 计算该进程的局部尺寸（考虑非整除）
            int lx = nx / dims[0] + ((rank_coords[0] < (nx % dims[0])) ? 1 : 0);
            int ly = ny / dims[1] + ((rank_coords[1] < (ny % dims[1])) ? 1 : 0);
            recvcounts[rank] = lx * ly;
            displs[rank] = offset;
            offset += recvcounts[rank];
        }
    }

    // 5. 收集各进程数据量（用于验证）
    int sendcount = local_nx * local_ny;
    MPI_Gather(&sendcount, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, comm_2d);

    // 6. 收集数据
    MPI_Gatherv(sendbuf, sendcount, MPI_DOUBLE,
                recvbuf, recvcounts, displs, MPI_DOUBLE,
                0, comm_2d);

    // 7. 进程0重组数据到全局网格
    if(myid == 0) {
        // 初始化边界条件（根据物理坐标）
        double h = 1.0 / (nx + 1);
        for(int i = 0; i <= nx+1; i++) {
            double x = i * h;
            gathered[i][0] = 0.0;                     // u(x,0) = 0
            gathered[i][ny+1] = 1.0 / (pow(1.0 + x, 2) + 1.0); // u(x,1)
        }
        for(int j = 0; j <= ny+1; j++) {
            double y = j * h;
            gathered[0][j] = y / (1.0 + y*y);          // u(0,y)
            gathered[nx+1][j] = y / (4.0 + y*y);       // u(1,y)
        }

        // 填充内部点
        int offset = 0;
        for(int rank = 0; rank < nprocs; rank++) {
            int rank_coords[2];
            MPI_Cart_coords(comm_2d, rank, 2, rank_coords);

            // 计算全局起始索引（考虑非整除）
            int gx_start = 0, gy_start = 0;
            for(int i = 0; i < rank_coords[0]; i++) {
                gx_start += (nx / dims[0]) + ((i < (nx % dims[0])) ? 1 : 0);
            }
            for(int j = 0; j < rank_coords[1]; j++) {
                gy_start += (ny / dims[1]) + ((j < (ny % dims[1])) ? 1 : 0);
            }

            // 填充数据（+1跳过ghost cells）
            int lx = nx / dims[0] + ((rank_coords[0] < (nx % dims[0])) ? 1 : 0);
            int ly = ny / dims[1] + ((rank_coords[1] < (ny % dims[1])) ? 1 : 0);
            for(int i = 0; i < lx; i++) {
                for(int j = 0; j < ly; j++) {
                    gathered[gx_start + i + 1][gy_start + j + 1] = recvbuf[offset++];
                }
            }
        }
        free(recvbuf);
        free(recvcounts);
        free(displs);
    }
    free(sendbuf);
}
