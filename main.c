#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

double** generatePositiveDefiniteMatrix(int n);
void printMatrix(double** mat, int n);
void choleskyDecomposition(double** A, double** L, int n);
void computeLLT(double** L, double** LLT, int n);
double frobeniusNorm(double** A, double** LLT, int n);

int main() {
    //not required, I was just testing CPU threads performance
    omp_set_num_threads(3);

    //for one thread computing arrays this size should be around 1sec
    //my tests've shown that we need to aim for 10k matrix to receive 2mins computation time
    //remember that print function and generation of content also takes time
    int n = 4;
    double** A = generatePositiveDefiniteMatrix(n);
    printMatrix(A, n);

    double** L = (double**)malloc(n * sizeof(double*));
    double** LLT = (double**)malloc(n * sizeof(double*));

    for (int i = 0; i < n; i++) {
        L[i] = (double*)calloc(n, sizeof(double));
        LLT[i] = (double*)calloc(n, sizeof(double));
    }

    double** A_cpy = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        A_cpy[i] = (double*)malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) {
            A_cpy[i][j] = A[i][j];
        }
    }

    double start = omp_get_wtime();
    choleskyDecomposition(A_cpy, L, n);
    double end = omp_get_wtime();
    printMatrix(A, n);
    printMatrix(L, n);
    printf("Computation time: %8.6f s\n", end - start);

    computeLLT(L, LLT, n);
    printMatrix(LLT, n);

    double norm = frobeniusNorm(A, LLT, n);
    printf("Frobenius Norm: %8.6f\n", norm);

    for (int i = 0; i < n; i++) {
        free(A[i]);
        free(L[i]);
        free(LLT[i]);
        free(A_cpy[i]);
    }
    free(A);
    free(L);
    free(LLT);
    free(A_cpy);

    return 0;
}

// Function to generate a symmetric positive definite matrix
double** generatePositiveDefiniteMatrix(int n) {
    srand(time(NULL)); // init pseudo random generator
    double** G = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        G[i] = (double*)malloc(n * sizeof(double));
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            G[i][j] = (rand() % 10) + 1;
        }
    }

    double** A = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        A[i] = (double*)malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) {
            A[i][j] = 0;
            for (int k = 0; k < n; k++) {
                A[i][j] += G[i][k] * G[j][k];
            }
        }
    }

    for (int i = 0; i < n; i++) {
        free(G[i]);
    }
    free(G);

    return A;
}

void printMatrix(double** mat, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%7.4f ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void choleskyDecomposition(double** A, double** L, int n) {
    for (int k = 0; k < n; k++) {
        L[k][k] = sqrt(A[k][k]);
        #pragma omp parallel for
        for (int i = k + 1; i < n; i++) {
            L[i][k] = A[i][k] / L[k][k];
        }
        #pragma omp parallel for collapse(2)
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j <= i; j++) {
                A[i][j] -= L[i][k] * L[j][k];
            }
        }
    }
}



// Compute LL^T from L
void computeLLT(double** L, double** LLT, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            LLT[i][j] = 0.0;
            for (int k = 0; k <= ((i < j) ? i : j); k++) {
                LLT[i][j] += L[i][k] * L[j][k];
            }
        }
    }
}

// Calculate Frobenius Norm
double frobeniusNorm(double** A, double** LLT, int n) {
    double norm = 0.0;
    #pragma omp parallel for reduction(+:norm)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double diff = A[i][j] - LLT[i][j];
            norm += diff * diff;
        }
    }
    return sqrt(norm);
}