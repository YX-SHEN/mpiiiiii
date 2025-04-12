#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>

#include <mpi.h>

#include "poisson1d.h"
#include "jacobi.h"

#define maxit 2000

#include "decomp1d.h"

void init_full_grid(double g[][maxn]);
void init_full_grids(double a[][maxn], double b[][maxn] ,double f[][maxn]);

void onedinit_basic(double a[][maxn], double b[][maxn], double f[][maxn],
		    int nx, int ny, int s, int e);

void print_full_grid(double x[][maxn]);
void print_in_order(double x[][maxn], MPI_Comm comm);
void print_grid_to_file(char *fname, double x[][maxn], int nx, int ny);
double compute_error(double u_numeric[][maxn], int nx, int ny, int s, int e);
int main(int argc, char **argv)
{
  double a[maxn][maxn], b[maxn][maxn], f[maxn][maxn];
  int nx, ny;
  int myid, nprocs;
  /* MPI_Status status; */
  int nbrleft, nbrright, s, e, it;
  double glob_diff;
  double ldiff;
  double t1, t2;
  double tol=1.0E-11;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (myid == 0) {
    /* set the size of the problem */
    if (argc > 2) {
        fprintf(stderr,"---->Usage: mpirun -np <nproc> %s <nx>\n",argv[0]);
        fprintf(stderr,"---->(for this code nx=ny)\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (argc == 2) {
        nx = atoi(argv[1]);
    } else {
        nx = 31;  // 默认网格大小
    }

    if (nx > maxn - 2) {
        fprintf(stderr,"Grid size too large (max: %d)\n", maxn - 2);
        exit(1);
    }
}
  MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  printf("(myid: %d) nx = %d\n",myid,nx);
  ny = nx;

  init_full_grids(a, b, f);

  int dims[1] = {0};  
  MPI_Dims_create(nprocs, 1, dims);   
  int periods[1] = {0};
  int reorder = 0;
  MPI_Comm cart_comm;

  MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, &cart_comm);
  MPI_Comm_rank(cart_comm, &myid);
  MPI_Cart_shift(cart_comm, 0, 1, &nbrleft, &nbrright);
  printf("Rank %d: Left = %d, Right = %d\n", myid, nbrleft, nbrright);


  MPE_Decomp1d(nx, nprocs, myid, &s, &e );

  printf("(myid: %d) nx: %d s: %d; e: %d; nbrleft: %d; nbrright: %d\n",myid, nx , s, e,
  	 nbrleft, nbrright);
  
  MPI_Barrier(MPI_COMM_WORLD);
  //MPI_Abort(MPI_COMM_WORLD, 1);

  onedinit_basic(a, b, f, nx, ny, s, e);

  t1 = MPI_Wtime();

  glob_diff = 1000;
  for(it=0; it<maxit; it++){

    exchang1(a, ny, s, e, MPI_COMM_WORLD, nbrleft, nbrright);
    sweep1d(a, f, nx, s, e, b);


    exchang1(b, nx, s, e, MPI_COMM_WORLD, nbrleft, nbrright);
    sweep1d(b, f, nx, s, e, a);

    ldiff = griddiff(a, b, nx, s, e);
    MPI_Allreduce(&ldiff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(myid==0 && it%10==0){
      printf("(myid %d) locdiff: %lf; glob_diff: %lf\n",myid, ldiff, glob_diff);
    }
    if( glob_diff < tol ){
      if(myid==0){
  	printf("iterative solve converged\n");
      }
      break;
    }

  }
  
  t2=MPI_Wtime();
  
  printf("DONE! (it: %d)\n",it);

  if( myid == 0 ){
    if( it == maxit ){
      fprintf(stderr,"Failed to converge\n");
    }
    printf("Run took %lf s\n",t2-t1);
  }

  print_in_order(a, MPI_COMM_WORLD);
  if( nprocs == 1  ){
    print_grid_to_file("grid", a,  nx, ny);
    print_full_grid(a);
  }

  double local_error = compute_error(a, nx, ny, s, e);
  double global_error;
  MPI_Reduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
  if (myid == 0) {
    printf("L2 squared error: %.12lf\n", global_error);
  }

  MPI_Finalize();
  return 0;
}

void onedinit_basic(double a[][maxn], double b[][maxn], double f[][maxn],
		    int nx, int ny, int s, int e)
{
  int i,j;
  double h = 1.0 / (nx + 1);
  double x, y;

  /* set everything to 0 first */
  for(i=s-1; i<=e+1; i++){
    for(j=0; j <= nx+1; j++){
      a[i][j] = 0.0;
      b[i][j] = 0.0;
      f[i][j] = 0.0;
    }
  }

  /* deal with boundaries */
  // 设置下边界：y = 0, u(x, 0) = 0
  for(i = s; i <= e; i++){
    a[i][0] = 0.0;
    b[i][0] = 0.0;
  }

  // 设置上边界：y = 1, u(x, 1) = 1 / ((1 + x)^2 + 1)
  for(i = s; i <= e; i++){
    x = i * h;
    double val = 1.0 / (pow(1 + x , 2) + 1);
    a[i][ny + 1] = val;
    b[i][ny + 1] = val;
  }

  // 设置左边界：x = 0, u(0, y) = y / (1 + y^2)
  if (s == 1) {
    for(j = 1; j <= ny; j++){
      y = j * h;
      double val = y / (1 + y * y);
      a[0][j] = val;
      b[0][j] = val;
    }
  }

  // 设置右边界：x = 1, u(1, y) = y / (4 + y^2)
  if (e == nx) {
    for(j = 1; j <= ny; j++){
      y = j * h;
      double val = y / (4 + y * y);
      a[nx + 1][j] = val;
      b[nx + 1][j] = val;
    }
  }

}

void init_full_grid(double g[][maxn])
{
  int i,j;
  const double junkval = -5;

  for(i=0; i < maxn; i++){
    for(j=0; j<maxn; j++){
      g[i][j] = junkval;
    }
  }
}

/* set global a,b,f to initial arbitrarily chosen junk value */
void init_full_grids(double a[][maxn], double b[][maxn] ,double f[][maxn])
{
  int i,j;
  const double junkval = -5;

  for(i=0; i < maxn; i++){
    for(j=0; j<maxn; j++){
      a[i][j] = junkval;
      b[i][j] = junkval;
      f[i][j] = junkval;
    }
  }

}

/* prints to stdout in GRID view */
void print_full_grid(double x[][maxn])
{
  int i,j;
  for(j=maxn-1; j>=0; j--){
    for(i=0; i<maxn; i++){
      if(x[i][j] < 10000.0){
	printf("|%2.6lf| ",x[i][j]);
      } else {
	printf("%9.2lf ",x[i][j]);
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

  for(i=0; i<size; i++){
    if( i == myid ){
      printf("proc %d\n",myid);
      print_full_grid(x);
    }
    fflush(stdout);
    usleep(500);	
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void print_grid_to_file(char *fname, double x[][maxn], int nx, int ny)
{
  FILE *fp;
  int i,j;

  fp = fopen(fname, "w");
  if( !fp ){
    fprintf(stderr, "Error: can't open file %s\n",fname);
    exit(4);
  }

  for(j=ny+1; j>=0; j--){
    for(i=0; i<nx+2; i++){
      fprintf(fp, "%lf ",x[i][j]);
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
      exact = y / (pow(1 + x, 2) + y * y);
      diff = u_numeric[i][j] - exact;
      sum += diff * diff;
    }
  }

  return sum;
}
