#ifndef DECOMP2D_H
#define DECOMP2D_H

int MPE_Decomp1d(int n, int size, int rank, int *s, int *e);
int MPE_Decomp2d(int nx, int ny, int dims[2], int coords[2], int *s1, int *e1, int *s2, int *e2);
int MPE_Compute2dDims(int size, int nx, int ny, int dims[2]);


#endif // DECOMP2D_H

