#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>

/* Include header files in correct order */
#include "poisson1d.h"
#include "jacobi.h"
#include "decomp1d.h"

#define maxit 2000

/* Function prototypes */
void init_full_grid(double g[][maxn]);
void init_full_grids(double a[][maxn], double b[][maxn], double f[][maxn]);
void onedinit_basic(double a[][maxn], double b[][maxn], double f[][maxn], int nx, int ny, int s, int e);
void print_full_grid(double x[][maxn]);
void print_in_order(double x[][maxn], MPI_Comm comm);
void print_grid_to_file(char *fname, double x[][maxn], int nx, int ny);
void GatherGrid(double a[][maxn], int nx, int ny, MPI_Comm comm);
void write_grid(const char *fname, double g[][maxn], int nx, int ny);
double compute_error(double u_numeric[][maxn], int nx, int ny, int s, int e);

int main(int argc, char **argv)
{
  double a[maxn][maxn], b[maxn][maxn], f[maxn][maxn];
  int nx, ny;
  int myid, nprocs;
  int nbrleft, nbrright, s, e, it;
  double glob_diff;
  double ldiff;
  double t1, t2;
  double tol=1.0E-11;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (myid == 0) {
    /* Set the size of the problem */
    if (argc > 2) {
      fprintf(stderr, "---->Usage: mpirun -np <nproc> %s <nx>\n", argv[0]);
      fprintf(stderr, "---->(for this code nx=ny)\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (argc == 2) {
      nx = atoi(argv[1]);
    } else {
      nx = 31; /* Default size if not specified */
    }

    if (nx > maxn-2) {
      fprintf(stderr, "Grid size too large (max: %d)\n", maxn-2);
      exit(1);
    }
  }

  MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  printf("(myid: %d) nx = %d\n", myid, nx);
  ny = nx;

  init_full_grids(a, b, f);

  /* Create Cartesian topology */
  int dims[1] = {0};  
  MPI_Dims_create(nprocs, 1, dims);   
  int periods[1] = {0};
  int reorder = 0;
  MPI_Comm cart_comm;

  MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, &cart_comm);
  MPI_Cart_shift(cart_comm, 0, 1, &nbrleft, &nbrright);

  MPE_Decomp1d(nx, nprocs, myid, &s, &e);

  printf("(myid: %d) nx: %d s: %d; e: %d; nbrleft: %d; nbrright: %d\n", 
         myid, nx, s, e, nbrleft, nbrright);
  
  MPI_Barrier(cart_comm);

  onedinit_basic(a, b, f, nx, ny, s, e);

  t1 = MPI_Wtime();

  if (myid == 0) {
    printf("\n======> NB VERSION\n\n");
  }

  glob_diff = 1000;
  for (it = 0; it < maxit; it++) {
    nbxchange_and_sweep(a, f, nx, ny, s, e, b, cart_comm, nbrleft, nbrright);
    nbxchange_and_sweep(b, f, nx, ny, s, e, a, cart_comm, nbrleft, nbrright);

    ldiff = griddiff(a, b, nx, s, e);
    MPI_Allreduce(&ldiff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
    if (myid == 0 && it % 10 == 0) {
      printf("(myid %d) locdiff: %lf; glob_diff: %lf\n", myid, ldiff, glob_diff);
    }
    if (glob_diff < tol) {
      if (myid == 0) {
        printf("Iterative solve converged at iteration %d\n", it);
      }
      break;
    }
  }
  
  t2 = MPI_Wtime();
  
  printf("DONE! (it: %d)\n", it);

  if (myid == 0) {
    if (it == maxit) {
      fprintf(stderr, "Failed to converge\n");
    }
    printf("Run took %lf s\n", t2-t1);
  }

  /* Calculate error compared to analytical solution */
  double local_error = compute_error(a, nx, ny, s, e);
  double global_error;
  MPI_Reduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
  
  if (myid == 0) {
    printf("L2 squared error: %.12lf\n", global_error);
    printf("L2 error: %.12lf\n", sqrt(global_error));
  }

  /* Gather the solution to rank 0 and write to file */
  GatherGrid(a, nx, ny, cart_comm);
  
  if (myid == 0) {
    write_grid("q3_solution.txt", a, nx, ny);
    printf("Solution written to q3_solution.txt\n");
  }

  MPI_Finalize();
  return 0;
}

void onedinit_basic(double a[][maxn], double b[][maxn], double f[][maxn],
                   int nx, int ny, int s, int e)
{
  int i, j;
  double h = 1.0 / (nx + 1);
  double x, y;

  /* Set everything to 0 first */
  for (i = s-1; i <= e+1; i++) {
    for (j = 0; j <= nx+1; j++) {
      a[i][j] = 0.0;
      b[i][j] = 0.0;
      f[i][j] = 0.0;
    }
  }

  /* Deal with boundaries */
  /* Bottom boundary: y = 0, u(x, 0) = 0 */
  for (i = s; i <= e; i++) {
    a[i][0] = 0.0;
    b[i][0] = 0.0;
  }

  /* Top boundary: y = 1, u(x, 1) = 1 / ((1 + x)^2 + 1) */
  for (i = s; i <= e; i++) {
    x = i * h;
    double val = 1.0 / ((1.0 + x) * (1.0 + x) + 1.0);
    a[i][ny+1] = val;
    b[i][ny+1] = val;
  }

  /* Left boundary: x = 0, u(0, y) = y / (1 + y^2) */
  if (s == 1) {
    for (j = 1; j <= ny; j++) {
      y = j * h;
      double val = y / (1.0 + y * y);
      a[0][j] = val;
      b[0][j] = val;
    }
  }

  /* Right boundary: x = 1, u(1, y) = y / (4 + y^2) */
  if (e == nx) {
    for (j = 1; j <= ny; j++) {
      y = j * h;
      double val = y / (4.0 + y * y);
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

/* Set global a,b,f to initial arbitrarily chosen junk value */
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

/* Prints to stdout in GRID view */
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

double compute_error(double u_numeric[][maxn], int nx, int ny, int s, int e) {
  double h = 1.0 / (nx + 1);
  double x, y, exact, diff, sum = 0.0;
  int i, j;

  for (i = s; i <= e; i++) {
    x = i * h;
    for (j = 1; j <= ny; j++) {
      y = j * h;
      exact = y / ((1.0 + x) * (1.0 + x) + y * y);
      diff = u_numeric[i][j] - exact;
      sum += diff * diff;
    }
  }

  return sum;
}

void GatherGrid(double a[][maxn], int nx, int ny, MPI_Comm comm) {
  int myid, nprocs;
  MPI_Comm_rank(comm, &myid);
  MPI_Comm_size(comm, &nprocs);

  double *sendbuf = NULL, *recvbuf = NULL;
  int *recvcounts = NULL, *displs = NULL;
  int s, e;
    
  /* Get local grid boundaries */
  MPE_Decomp1d(nx, nprocs, myid, &s, &e);
  int local_cols = e - s + 1;
    
  /* Pack local data (excluding ghost cells) */
  sendbuf = (double *)malloc(local_cols * ny * sizeof(double));
  int idx = 0;
  for (int i = 0; i < local_cols; i++) {
    for (int j = 1; j <= ny; j++) {
      sendbuf[idx++] = a[s+i][j];
    }
  }
    
  /* Prepare for gathering */
  if (myid == 0) {
    recvcounts = (int *)malloc(nprocs * sizeof(int));
    displs = (int *)malloc(nprocs * sizeof(int));
    recvbuf = (double *)malloc(nx * ny * sizeof(double));
  }
    
  /* Share local sizes */
  int sendcount = local_cols * ny;
  MPI_Gather(&sendcount, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, comm);
    
  /* Calculate displacements */
  if (myid == 0) {
    displs[0] = 0;
    for (int i = 1; i < nprocs; i++) {
      displs[i] = displs[i-1] + recvcounts[i-1];
    }
  }
    
  /* Gather all data to root */
  MPI_Gatherv(sendbuf, sendcount, MPI_DOUBLE, 
              recvbuf, recvcounts, displs, MPI_DOUBLE, 
              0, comm);
    
  /* Unpack the data on rank 0 */
  if (myid == 0) {
    int pos = 0;
    for (int p = 0; p < nprocs; p++) {
      int proc_s, proc_e;
      MPE_Decomp1d(nx, nprocs, p, &proc_s, &proc_e);
      int proc_cols = proc_e - proc_s + 1;
            
      for (int i = 0; i < proc_cols; i++) {
        for (int j = 0; j < ny; j++) {
          a[proc_s+i][j+1] = recvbuf[pos++];
        }
      }
    }
        
    /* Set boundary conditions correctly after gathering */
    double h = 1.0 / (nx + 1);
        
    /* Bottom boundary: y = 0, u(x, 0) = 0 */
    for (int i = 0; i <= nx+1; i++) {
      a[i][0] = 0.0;
    }
        
    /* Top boundary: y = 1, u(x, 1) = 1/((1+x)^2+1) */
    for (int i = 0; i <= nx+1; i++) {
      double x = i * h;
      a[i][ny+1] = 1.0 / ((1.0 + x) * (1.0 + x) + 1.0);
    }
        
    /* Left boundary: x = 0, u(0, y) = y/(1+y^2) */
    for (int j = 0; j <= ny+1; j++) {
      double y = j * h;
      a[0][j] = y / (1.0 + y * y);
    }
        
    /* Right boundary: x = 1, u(1, y) = y/(4+y^2) */
    for (int j = 0; j <= ny+1; j++) {
      double y = j * h;
      a[nx+1][j] = y / (4.0 + y * y);
    }
  }
    
  /* Clean up */
  free(sendbuf);
  if (myid == 0) {
    free(recvcounts);
    free(displs);
    free(recvbuf);
  }
}

void write_grid(const char *fname, double g[][maxn], int nx, int ny) {
  FILE *fp = fopen(fname, "w");
  if (!fp) {
    fprintf(stderr, "Error: can't open file %s\n", fname);
    return;
  }

  /* Write header with dimensions for potential visualization */
  fprintf(fp, "%d %d\n", nx, ny);
    
  /* Write grid data in indexed order */
  for (int i = 0; i <= nx+1; i++) {
    for (int j = 0; j <= ny+1; j++) {
      fprintf(fp, "%lf ", g[i][j]);
    }
    fprintf(fp, "\n");
  }

  fclose(fp);
}
