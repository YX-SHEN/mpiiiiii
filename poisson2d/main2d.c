#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>
#include "poisson2d.h"
#include "jacobi2d.h"
#include "decomp2d.h"

// 定义 max 和 min 宏（在文件顶部添加）
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

#define maxit 2000  //  固定迭代 2000 次

/* 函数原型声明 */
void init_full_grid(double g[][maxn]);
void init_full_grids(double a[][maxn], double b[][maxn], double f[][maxn]);
void twod_init(double a[][maxn], double b[][maxn], double f[][maxn], 
               int nx, int ny, int sx, int ex, int sy, int ey);
void print_full_grid(double x[][maxn]);
void print_in_order(double x[][maxn], MPI_Comm comm);
void print_grid_to_file(char *fname, double x[][maxn], int nx, int ny);
double compute_error(double u_numeric[][maxn], int nx, int ny, int sx, int ex, int sy, int ey);

int main(int argc, char **argv) {
    double a[maxn][maxn], b[maxn][maxn], f[maxn][maxn];
    double gathered_grid[maxn][maxn];
    int nx, ny;
    int myid, nprocs;
    int nbrleft, nbrright, nbrbottom, nbrtop;
    int sx, ex, sy, ey;
    int it;
    double glob_diff;
    double ldiff;
    double t1, t2;
    int use_nonblocking = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (myid == 0) {
        if (argc > 3) {
            fprintf(stderr, "Usage: mpirun -np <nproc> %s <nx> [use_nonblocking]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        nx = (argc >= 2) ? atoi(argv[1]) : 31;
        if (nx > maxn-2) {
            fprintf(stderr, "Grid size too large (max: %d)\n", maxn-2);
            exit(1);
        }
        use_nonblocking = (argc == 3) ? atoi(argv[2]) : 0;
    }

    MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&use_nonblocking, 1, MPI_INT, 0, MPI_COMM_WORLD);
    ny = nx;

    if (myid == 0) {
        printf("Grid size: %d x %d\n", nx, ny);
        printf("Using %s communication\n", use_nonblocking ? "non-blocking" : "sendrecv");
    }

    init_full_grids(a, b, f);

    int dims[2] = {0, 0}, periods[2] = {0, 0};
    MPE_Compute2dDims(nprocs, nx, ny, dims);
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);
    MPI_Comm_rank(cart_comm, &myid);
    int coords[2];
    MPI_Cart_coords(cart_comm, myid, 2, coords);
    MPI_Cart_shift(cart_comm, 0, 1, &nbrleft, &nbrright);
    MPI_Cart_shift(cart_comm, 1, 1, &nbrbottom, &nbrtop);
    MPE_Decomp2d(nx, ny, dims, coords, &sx, &ex, &sy, &ey);

    printf("Rank %d: Coords = (%d,%d), Domain = [%d:%d, %d:%d], Neighbors = L:%d R:%d B:%d T:%d\n",
           myid, coords[0], coords[1], sx, ex, sy, ey, nbrleft, nbrright, nbrbottom, nbrtop);

    twod_init(a, b, f, nx, ny, sx, ex, sy, ey);
    MPI_Barrier(cart_comm);
    t1 = MPI_Wtime();

    if (myid == 0) {
        printf("\n======> Running with %s communication (fixed 2000 iterations)\n\n", 
               use_nonblocking ? "non-blocking" : "sendrecv");
    }

    glob_diff = 1000;
    for (it = 0; it < maxit; it++) {  // 严格运行 2000 次迭代
        if (use_nonblocking) {
            exchange2d_nonblocking(a, nx, ny, sx, ex, sy, ey, cart_comm, 
                                  nbrleft, nbrright, nbrbottom, nbrtop);
        } else {
            exchange2d_sendrecv(a, nx, ny, sx, ex, sy, ey, cart_comm, 
                               nbrleft, nbrright, nbrbottom, nbrtop);
        }

        sweep2d(a, f, nx, ny, sx, ex, sy, ey, b);

        if (use_nonblocking) {
            exchange2d_nonblocking(b, nx, ny, sx, ex, sy, ey, cart_comm, 
                                  nbrleft, nbrright, nbrbottom, nbrtop);
        } else {
            exchange2d_sendrecv(b, nx, ny, sx, ex, sy, ey, cart_comm, 
                               nbrleft, nbrright, nbrbottom, nbrtop);
        }

        sweep2d(b, f, nx, ny, sx, ex, sy, ey, a);

        ldiff = griddiff2d(a, b, nx, ny, sx, ex, sy, ey);
        MPI_Allreduce(&ldiff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM, cart_comm);

        if (myid == 0 && it % 100 == 0) {  //  每 100 次迭代输出一次进度
            printf("(myid %d) iteration: %d, glob_diff: %le\n", myid, it, glob_diff);
        }
    }  // 结束 2000 次迭代

    t2 = MPI_Wtime();

    if (myid == 0) {
        printf("DONE! (iterations: %d)\n", maxit);  // 明确显示固定迭代次数
        printf("Run took %.6lf s\n", t2 - t1);
    }

    /* 修改后的误差计算部分（替换原代码） */

// 计算局部平方和
double local_sqsum = compute_error(a, nx, ny, sx, ex, sy, ey);

// 计算本进程处理的内部点数
int local_points = (min(ex, nx) - max(sx, 1) + 1);     // X方向点数
local_points *= (min(ey, ny) - max(sy, 1) + 1);        // Y方向点数（总点数）

// 全局归约
double global_sqsum;
int global_points;
MPI_Reduce(&local_sqsum, &global_sqsum, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
MPI_Reduce(&local_points, &global_points, 1, MPI_INT, MPI_SUM, 0, cart_comm);

// 根进程输出
if (myid == 0) {
    // 计算归一化 L2 误差
    double l2_error = sqrt(global_sqsum / (double)global_points);
    
    printf("L2 squared error (sum): %.12le\n", global_sqsum);
    printf("Total internal points: %d\n", global_points);
    printf("Normalized L2 error: %.12le\n", l2_error);  //  正确命名
}

    GatherGrid2D(a, nx, ny, sx, ex, sy, ey, gathered_grid, cart_comm);

    if (myid == 0) {
        write_grid2d(gathered_grid, nx, ny, 1, nx, 1, ny, 0, cart_comm, "q4_solution.txt");
        printf("Solution written to q4_solution.txt\n");

        double h = 1.0 / (nx + 1);
        int mid_j = (int)(0.5 / h);
        int mid_i = (int)(0.5 / h);

        printf("\nBoundary verification:\n");
        printf("u(0,0.5) = %.6f (should be %.6f)\n", 
               gathered_grid[0][mid_j], 0.5 / (1.0 + pow(0.5, 2)));
        printf("u(1,0.5) = %.6f (should be %.6f)\n", 
               gathered_grid[nx+1][mid_j], 0.5 / (4.0 + pow(0.5, 2)));
        printf("u(0.5,1) = %.6f (should be %.6f)\n",
               gathered_grid[mid_i][ny+1], 1.0 / (pow(1.0 + 0.5, 2) + 1.0));
    }

    MPI_Finalize();
    return 0;
}


void twod_init(double a[][maxn], double b[][maxn], double f[][maxn],
    int nx, int ny, int sx, int ex, int sy, int ey)
{
int i, j;
double h = 1.0 / (nx + 1);

// 初始化所有点（包括ghost cells）为0
for (i = sx-1; i <= ex+1; i++) {
for (j = sy-1; j <= ey+1; j++) {
 a[i][j] = 0.0;
 b[i][j] = 0.0;
 f[i][j] = 0.0;
}
}

// 精确设置边界条件（基于物理坐标）

// 下边界 (y=0): u(x,0) = 0
if (sy == 0) { // 检查是否包含下边界
for (i = sx; i <= ex; i++) {
 a[i][0] = 0.0;
 b[i][0] = 0.0;
}
}

// 上边界 (y=1): u(x,1) = 1/((1+x)^2 + 1)
if (ey == ny) {
for (i = sx; i <= ex; i++) {
 double x = i * h;
 double val = 1.0 / (pow(1.0 + x, 2) + 1.0);
 a[i][ny+1] = val;
 b[i][ny+1] = val;
}
}

// 左边界 (x=0): u(0,y) = y/(1 + y^2)
if (sx == 0) { // 检查是否包含左边界
for (j = sy; j <= ey; j++) {
 double y = j * h;
 double val = y / (1.0 + y*y);
 a[0][j] = val;
 b[0][j] = val;
}
}

// 右边界 (x=1): u(1,y) = y/(4 + y^2)
if (ex == nx) {
for (j = sy; j <= ey; j++) {
 double y = j * h;
 double val = y / (4.0 + y*y);
 a[nx+1][j] = val;
 b[nx+1][j] = val;
}
}
}

void init_full_grid(double g[][maxn])
{
  int i, j;
  const double junkval = -5;
  
  for (i = 0; i < maxn; i++) {
    for (j = 0; j < maxn; j++) {
      g[i][j] = junkval;
    }
  }
}

void init_full_grids(double a[][maxn], double b[][maxn], double f[][maxn])
{
  int i, j;
  const double junkval = -5;
  
  for (i = 0; i < maxn; i++) {
    for (j = 0; j < maxn; j++) {
      a[i][j] = junkval;
      b[i][j] = junkval;
      f[i][j] = junkval;
    }
  }
}

void print_full_grid(double x[][maxn])
{
  int i, j;
  for (j = maxn-1; j >= 0; j--) {
    for (i = 0; i < maxn; i++) {
      if (x[i][j] < 10000.0) {
        printf("|%2.6lf| ", x[i][j]);
      } else {
        printf("%9.2lf ", x[i][j]);
      }
    }
    printf("\n");
  }
}

void print_in_order(double x[][maxn], MPI_Comm comm)
{
  int myid, size;
  int i;
  
  MPI_Comm_rank(comm, &myid);
  MPI_Comm_size(comm, &size);
  MPI_Barrier(comm);
  printf("Attempting to print in order\n");
  sleep(1);
  MPI_Barrier(comm);
  
  for (i = 0; i < size; i++) {
    if (i == myid) {
      printf("proc %d\n", myid);
      print_full_grid(x);
    }
    fflush(stdout);
    usleep(500);
    MPI_Barrier(comm);
  }
}

void print_grid_to_file(char *fname, double x[][maxn], int nx, int ny)
{
  FILE *fp;
  int i, j;
  
  fp = fopen(fname, "w");
  if (!fp) {
    fprintf(stderr, "Error: can't open file %s\n", fname);
    exit(4);
  }
  
  for (j = ny+1; j >= 0; j--) {
    for (i = 0; i < nx+2; i++) {
      fprintf(fp, "%lf ", x[i][j]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

double compute_error(double u_numeric[][maxn], int nx, int ny, int sx, int ex, int sy, int ey) {
    double h = 1.0 / (nx + 1);
    double sum = 0.0;
    
    // 仅计算内部点（i,j 从 1 到 nx/ny）
    for (int i = max(sx, 1); i <= min(ex, nx); i++) {      // i ∈ [1, nx]
        for (int j = max(sy, 1); j <= min(ey, ny); j++) {  // j ∈ [1, ny]
            double x = i * h;
            double y = j * h;
            double exact = y / ((1.0 + x) * (1.0 + x) + y * y);
            double diff = u_numeric[i][j] - exact;
            sum += diff * diff;
        }
    }
    return sum;
}
