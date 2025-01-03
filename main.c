#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>

double** generateRandomSquareMatrix(int n, double min_val, double max_val);
double** generatePositiveDefiniteMatrix(double** G, int n);
void printMatrix(double** mat, int n);
double** copySquareMatrix(double** mat, int n);
void freeSquareMatrix(double** mat, int n);
void sequentialCholeskyDecomposition(double** A, double** L, int n);
void choleskyDecomposition(double** A, double** L, int n);
void computeLLT(double** L, double** LLT, int n);
double frobeniusNorm(double** A, double** LLT, int n);

int calculateDecompositionSequentially(double** A, int n);
int calculateDecompositionUsingMPI(double** A, int n);
int calculateDecompositionUsingOMP(double** A, int n);


int main(void)
{
    //for one thread computing arrays this size should be around 1sec
    //my tests've shown that we need to aim for 10k matrix to receive 2mins computation time
    //remember that print function and generation of content also takes time
    int n = 1000;
    printf("Generating matrix %d X %d ...\n", n, n);
    double** G = generateRandomSquareMatrix(n, -10, 10);
    double** A = generatePositiveDefiniteMatrix(G, n);
    freeSquareMatrix(G, n);

    calculateDecompositionSequentially(A, n);
    calculateDecompositionUsingMPI(A, n);
    calculateDecompositionUsingOMP(A, n);


    freeSquareMatrix(A, n);


    return 0;
}

int calculateDecompositionSequentially(double** A, int n)
{
    printf("Sequential Cholesky Decomposition ...\n");
    double** L = (double**)malloc(n * sizeof(double*));
    double** LLT = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++)
    {
        L[i] = (double*)calloc(n, sizeof(double));
        LLT[i] = (double*)calloc(n, sizeof(double));
    }
    double** A_cpy = copySquareMatrix(A, n);
    double start = omp_get_wtime();
    sequentialCholeskyDecomposition(A_cpy, L, n);
    double end = omp_get_wtime();
    printf("Computation time: %8.6f s\n", end - start);
    computeLLT(L, LLT, n);
    double norm = frobeniusNorm(A, LLT, n);
    printf("Frobenius Norm: %8.6f\n", norm);
    freeSquareMatrix(A_cpy, n);
    freeSquareMatrix(L, n);
    freeSquareMatrix(LLT, n);
    return 0;
}


int calculateDecompositionUsingMPI(double** A, int n)
{
    MPI_Init(0, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    double** L = (double**)malloc(n * sizeof(double*));
    double** LLT = (double**)malloc(n * sizeof(double*));

    for (int i = 0; i < n; i++)
    {
        L[i] = (double*)calloc(n, sizeof(double));
        LLT[i] = (double*)calloc(n, sizeof(double));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    printf("Parallel Cholesky Decomposition using MPI...\n");

    double** A_cpy = copySquareMatrix(A, n);
    double start = MPI_Wtime();
    choleskyDecomposition(A_cpy, L, n);
    double end = MPI_Wtime();

    printf("Computation time MPI: %8.6f s\n", end - start);
    computeLLT(L, LLT, n);
    double norm = frobeniusNorm(A, LLT, n);
    printf("Frobenius Norm MPI: %8.6f\n", norm);

    freeSquareMatrix(A_cpy, n);
    freeSquareMatrix(L, n);
    freeSquareMatrix(LLT, n);

    MPI_Finalize();
    return 0;
}

int calculateDecompositionUsingOMP(double** A, int n)
{

    double** L = (double**)malloc(n * sizeof(double*));
    double** LLT = (double**)malloc(n * sizeof(double*));

    for (int i = 0; i < n; i++)
    {
        L[i] = (double*)calloc(n, sizeof(double));
        LLT[i] = (double*)calloc(n, sizeof(double));
    }

    printf("Parallel Cholesky Decomposition using OMP...\n");

    double** A_cpy = copySquareMatrix(A, n);
    double start = omp_get_wtime();
    choleskyDecomposition(A_cpy, L, n);
    double end = omp_get_wtime();
    printf("Computation time OMP: %8.6f s\n", end - start);
    computeLLT(L, LLT, n);
    double norm = frobeniusNorm(A, LLT, n);
    printf("Frobenius Norm OMP: %8.6f\n", norm);

    freeSquareMatrix(A_cpy, n);
    freeSquareMatrix(L, n);
    freeSquareMatrix(LLT, n);

    return 0;
}


double** generateRandomSquareMatrix(int n, double min_val, double max_val)
{
    srand(time(NULL)); // init pseudo random generator
    double range = max_val - min_val;
    double divider = RAND_MAX / range;
    double** G = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++)
    {
        G[i] = (double*)malloc(n * sizeof(double));
        for (int j = 0; j < n; j++)
        {
            G[i][j] = min_val + rand() / divider;
        }
    }
    return G;
}

// Function to generate a symmetric positive definite matrix
double** generatePositiveDefiniteMatrix(double** G, int n)
{
    double** A= (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++)
    {
        A[i] = (double*)malloc(n * sizeof(double));
        for (int j = 0; j < n; j++)
        {
            A[i][j] = 0;
            for (int k = 0; k < n; k++)
            {
                A[i][j] += G[i][k] * G[j][k];
            }
        }
    }
    return A;
}

void printMatrix(double** mat, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%7.4f ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

double** copySquareMatrix(double** mat, int n)
{
    double** cpy = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++)
    {
        cpy[i] = (double*)malloc(n * sizeof(double));
        for (int j = 0; j < n; j++)
        {
            cpy[i][j] = mat[i][j];
        }
    }
    return cpy;
}

void freeSquareMatrix(double** mat, int n)
{
    for (int i = 0; i < n; i++)
    {
        free(mat[i]);
    }
    free(mat);
}

void sequentialCholeskyDecomposition(double** A, double** L, int n)
{
    for (int k = 0; k < n; k++)
    {
        L[k][k] = sqrt(A[k][k]);
        for (int i = k + 1; i < n; i++)
        {
            L[i][k] = A[i][k] / L[k][k];
        }
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j <= i; j++)
            {
                A[i][j] -= L[i][k] * L[j][k];
            }
        }
    }
}

void choleskyDecomposition(double** A, double** L, int n)
{
    for (int k = 0; k < n; k++)
    {
        L[k][k] = sqrt(A[k][k]);
#pragma omp parallel for
        for (int i = k + 1; i < n; i++)
        {
            L[i][k] = A[i][k] / L[k][k];
        }
#pragma omp parallel for collapse(2)
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j <= i; j++)
            {
                A[i][j] -= L[i][k] * L[j][k];
            }
        }
    }
}

// Compute LL^T from L
void computeLLT(double** L, double** LLT, int n)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            LLT[i][j] = 0.0;
            for (int k = 0; k <= ((i < j) ? i : j); k++)
            {
                LLT[i][j] += L[i][k] * L[j][k];
            }
        }
    }
}

// Calculate Frobenius Norm
double frobeniusNorm(double** A, double** LLT, int n)
{
    double norm = 0.0;
#pragma omp parallel for reduction(+:norm)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double diff = A[i][j] - LLT[i][j];
            norm += diff * diff;
        }
    }
    return sqrt(norm);
}
