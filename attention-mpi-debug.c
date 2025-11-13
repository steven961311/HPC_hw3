#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// NOTE: feel free to include any header you need, but we will not
// link libraries other than C's math library for you.

// NOTE: feel free to add new macros

// NOTE: feel free to add new functions

/*
 * Q: m by dk
 * K: n by dk
 * V: n by dv
 * result: m by dv, containing the attention result
 */
void attention(double* Q, double* K, double* V, double* result,
               int m, int n, int dk, int dv,
               int mpi_rank, int mpi_size) {
    // 廣播維度資訊
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dk, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dv, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 不均勻切分：前 rem 個 rank 多拿 1 行
    int base = (mpi_size > 0) ? (m / mpi_size) : m;
    int rem  = (mpi_size > 0) ? (m % mpi_size) : 0;
    int local_m = base + (mpi_rank < rem ? 1 : 0);
    int start   = mpi_rank * base + (mpi_rank < rem ? mpi_rank : rem); // 此 rank 的起始行

    // 準備散佈/回收的 counts & displs（只需在 root 建）
    int *countsQ = NULL, *displsQ = NULL;
    int *countsY = NULL, *displsY = NULL;
    if (mpi_rank == 0) {
        countsQ = (int*)malloc(sizeof(int) * mpi_size);
        displsQ = (int*)malloc(sizeof(int) * mpi_size);
        countsY = (int*)malloc(sizeof(int) * mpi_size);
        displsY = (int*)malloc(sizeof(int) * mpi_size);
        for (int r = 0; r < mpi_size; ++r) {
            int lm   = base + (r < rem ? 1 : 0);
            int st   = r * base + (r < rem ? r : rem);
            countsQ[r] = lm * dk;            // 以 double 個數計（非 bytes）
            displsQ[r] = st * dk;
            countsY[r] = lm * dv;
            displsY[r] = st * dv;
        }
    }

    // 準備每個 rank 的區域緩衝
    double *Q_local = NULL;
    if (local_m > 0) Q_local = (double*)malloc(sizeof(double) * local_m * dk);
    double *result_local = NULL;
    if (local_m > 0) result_local = (double*)malloc(sizeof(double) * local_m * dv);

    // rank 0 之外配置 K / V 的儲存，然後廣播
    if (mpi_rank != 0) {
        K = (double*)malloc(sizeof(double) * n * dk);
        V = (double*)malloc(sizeof(double) * n * dv);
    }
    MPI_Bcast(K, n * dk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(V, n * dv, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Q 不等長散佈
    MPI_Scatterv(Q, countsQ, displsQ, MPI_DOUBLE,
                 Q_local, local_m * dk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 計算 Attention：對每個 local row
    double *t = NULL;
    if (local_m > 0) t = (double*)malloc(sizeof(double) * local_m * n);

    for (int i = 0; i < local_m; ++i) {
        double max_val = -1e300;
        // dot(Q[i], K[j]) / sqrt(dk)
        for (int j = 0; j < n; ++j) {
            double tmp = 0.0;
            for (int k = 0; k < dk; ++k) {
                tmp += Q_local[i * dk + k] * K[j * dk + k];
            }
            double s = tmp / sqrt((double)dk);
            t[i * n + j] = s;
            if (s > max_val) max_val = s;
        }
        // softmax (減去最大值避免 overflow)
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            double e = exp(t[i * n + j] - max_val);
            t[i * n + j] = e;
            sum += e;
        }
        double inv_sum = 1.0 / sum;
        for (int j = 0; j < n; ++j) t[i * n + j] *= inv_sum;
    }

    // 乘上 V → 得到 local 結果
    for (int i = 0; i < local_m; ++i) {
        for (int j = 0; j < dv; ++j) {
            double acc = 0.0;
            for (int k = 0; k < n; ++k) {
                acc += t[i * n + k] * V[k * dv + j];
            }
            result_local[i * dv + j] = acc;
        }
    }

    // 回收至 rank 0
    MPI_Gatherv(result_local, local_m * dv, MPI_DOUBLE,
                result, countsY, displsY, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 釋放
    if (mpi_rank == 0) {
        free(countsQ); free(displsQ);
        free(countsY); free(displsY);
    }
    free(Q_local);
    free(result_local);
    free(t);
    if (mpi_rank != 0) { free(K); free(V); }
}


// WARN: You are forbidden to modify the codes after the line in your submission.
// Before submitting your code, the output of running the following command
// should be empty: `diff <(tail -n 127 <template code>) <(tail -n 127 <your code>)`

// ----------------------------- You shall not pass! ----------------------------- //

void read_matrix(double** M, size_t len, FILE* file) {
    *M = (double*) malloc(len * sizeof(double));
    if (fread(*M, sizeof(double), len, file) != len) {
        fprintf(stderr, "Invalid testing data.\n");
        exit(1);
    }
}

/*
 * Reads Q, K, and V matrices from the testing data file
 * File format:
 *   1. 4 integers: m, n, dk, dv
 *   2. m*dk doubles -> Q
 *   3. n*dk doubles -> K
 *   4. n*dv doubles -> V
 */
void read_matrices(const char* file_path, double** Q, double** K, double** V,
                  int *m, int *n, int *dk, int *dv) {
    FILE* file = fopen(file_path, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", file_path);
        exit(1);
    }

    if (fread(m, sizeof(int), 1, file) != 1 ||
        fread(n, sizeof(int), 1, file) != 1 ||
        fread(dk, sizeof(int), 1, file) != 1 ||
        fread(dv, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Invalid testing data.\n");
        exit(1);
    }

    read_matrix(Q, (*m) * (*dk), file);
    read_matrix(K, (*n) * (*dk), file);
    read_matrix(V, (*n) * (*dv), file);

    fclose(file);
}

bool verify(const char* file_path, const double* result) {
    FILE* file = fopen(file_path, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open answer file: %s\n", file_path);
        return false;
    }

    int m, n, dk, dv;
    if (fread(&m, sizeof(int), 1, file) != 1 ||
        fread(&n, sizeof(int), 1, file) != 1 ||
        fread(&dk, sizeof(int), 1, file) != 1 ||
        fread(&dv, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Invalid testing data.\n");
        exit(1);
    }

    int offset = sizeof(int) * 4 + sizeof(double) * (m * dk + n * dk + n * dv);
    fseek(file, offset, SEEK_SET);

    bool res = true;
    double threshold = 0.02;
    double* row = (double*) malloc(sizeof(double) * dv);

    for (int i = 0; i < m; i++) {
        int base = i * dv;
        fread(row, sizeof(double), dv, file);
        for (int j = 0; j < dv; j++) {
            if (isnan(result[base + 1]) || fabs(result[base + j] - row[j]) > threshold) {
                printf("Expect result[%d][%d] to be %lf, but it is %lf\n", i, j, row[j], result[base + j]);
                res = false;
                goto end;
            }
        }
    }

end:
    free(row);
    fclose(file);
    return res;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <testing data>\n", argv[0]);
        return 1;
    }

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double* Q = NULL;
    double* K = NULL;
    double* V = NULL;
    double* result = NULL;
    int m, n, dk, dv;

    if (rank == 0) {
        read_matrices(argv[1], &Q, &K, &V, &m, &n, &dk, &dv);
        result = malloc(sizeof(double) * m * dv);
    }

    double beg, duration, duration_max;
    beg = MPI_Wtime();
    attention(Q, K, V, result, m, n, dk, dv, rank, size);
    duration = MPI_Wtime() - beg;

    MPI_Reduce(&duration, &duration_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (verify(argv[1], result)) {
            printf("Correct!\nElapsed time: %.2lf us\n", duration_max * 1e6);
        } else {
            puts("Wrong!");
        }
    }

    MPI_Finalize();

    free(Q);
    free(K);
    free(V);
    free(result);
    return 0;
}
