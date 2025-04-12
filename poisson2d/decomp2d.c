/*
 This file contains routines for producing decompositions for 2D grids
 when given a number of processors. It includes both 1D and 2D decomposition functions.
 The values returned assume a "global" domain in [1:n]
 */

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

/*@
 MPE_Decomp1d - Compute a balanced decomposition of a 1-D array
 Input Parameters:
+ n - Length of the array
. size - Number of processors in decomposition
- rank - Rank of this processor in the decomposition (0 <= rank < size)
 Output Parameters:
. s,e - Array indices are s:e, with the original array considered as 1:n.
@*/
int MPE_Decomp1d(int n, int size, int rank, int *s, int *e)
{
    int nlocal, deficit;
    nlocal = n / size;
    *s = rank * nlocal + 1;
    deficit = n % size;
    *s = *s + ((rank < deficit) ? rank : deficit);
    if (rank < deficit) nlocal++;
    *e = *s + nlocal - 1;
    if (*e > n || rank == size-1) *e = n;
    return MPI_SUCCESS;
}

/*@
 MPE_Decomp2d - Compute a balanced 2D decomposition for a 2D grid
 Input Parameters:
+ nx, ny - Dimensions of the 2D grid
. dims[2] - Number of processors in each dimension
- coords[2] - Coordinates of this processor in the 2D grid
 Output Parameters:
. s1,e1 - Array indices for first dimension (s1:e1)
. s2,e2 - Array indices for second dimension (s2:e2)
@*/
int MPE_Decomp2d(int nx, int ny, int dims[2], int coords[2], 
                int *s1, int *e1, int *s2, int *e2)
{
    // Decompose in x direction (first dimension)
    MPE_Decomp1d(nx, dims[0], coords[0], s1, e1);
    
    // Decompose in y direction (second dimension)
    MPE_Decomp1d(ny, dims[1], coords[1], s2, e2);
    
    return MPI_SUCCESS;
}

/*@
 MPE_Compute2dDims - Compute a balanced distribution of processors in a 2D grid
 Input Parameters:
+ size - Total number of processors
- nx, ny - Grid dimensions (can be used to influence the aspect ratio)
 Output Parameters:
. dims[2] - Number of processors in each dimension
@*/
int MPE_Compute2dDims(int size, int nx, int ny, int dims[2])
{
    // Use MPI's dimension creation utility
    dims[0] = 0;
    dims[1] = 0;
    
    // Let MPI compute a balanced decomposition
    MPI_Dims_create(size, 2, dims);
    
    // If grid dimensions are very different, we might want to
    // adjust the processor grid to have a similar aspect ratio
    if (nx > ny && dims[0] < dims[1]) {
        // Swap dims if the aspect ratios don't match
        int temp = dims[0];
        dims[0] = dims[1];
        dims[1] = temp;
    }
    
    return MPI_SUCCESS;
}
