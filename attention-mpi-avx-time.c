#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// NOTE: feel free to include any header you need, but we will not
// link libraries other than C's math library for you.
#include <immintrin.h>

// NOTE: feel free to add new macros

// NOTE: feel free to add new functions
void transpose_V(double* V, double* V_T, int n, int dv) {
    int block = 64 / sizeof(double);
    for (int i = 0; i < n; i += block) {
        for (int j = 0; j < dv; j += block) {
            int i_max = (i + block < n) ? (i + block) : n;
            int j_max = (j + block < dv) ? (j + block) : dv;
            for (int ii = i; ii < i_max; ++ii) {
                for (int jj = j; jj < j_max; ++jj) {
                    V_T[jj * n + ii] = V[ii * dv + jj];
                }
            }
        }
    }
}

/*
 * Q: m by dk
 * K: n by dk
 * V: n by dv
 * result: m by dv, containing the attention result
 */
void attention(double* Q, double* K, double* V, double* result,
               int m, int n, int dk, int dv,
               int mpi_rank, int mpi_size) {
    // TODO: your Open MPI attention implementation
    // NOTE:
    //  - you should not read testing data from the file by yourself
    //  - only rank 0 has the input data (including m, n, dk, dv),
    //    you should figure out a way to distribute the data.
    //  - you only need to provide your answer in rank 0's result buffer
    
    //row-wise
    //communicateif (mpi_rank == 0)
    double t_comm_start, t_comm_end, t_comp_start, t_comp_end;
    double comm_time = 0.0, comp_time = 0.0;
    t_comm_start = MPI_Wtime();

    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dk, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dv, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int m_local = m / mpi_size;
    double* Q_local = malloc(m_local * dk * sizeof(double));
    double* result_local = malloc(m_local * dv * sizeof(double));
    if(mpi_rank != 0) {
        K = malloc(n * dk * sizeof(double));
        V = malloc(n * dv * sizeof(double));
    }
    MPI_Bcast(K, n * dk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(V, n * dv, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(Q, m_local * dk, MPI_DOUBLE, Q_local, m_local * dk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    t_comm_end = MPI_Wtime();
    comm_time += (t_comm_end - t_comm_start);
    t_comp_start = MPI_Wtime();
    
    //calculate
    double* t = malloc(sizeof(double) * m_local * n);
    for(int i = 0; i < m_local; i++) {
        double max_val = -1e9;
        for(int j = 0; j < n; j++) {
            __m256d vsum = _mm256_setzero_pd();
            int k;
            for(k = 0; k + 3 < dk; k += 4) {
                __m256d vq = _mm256_loadu_pd(&Q_local[i * dk + k]);
                __m256d vk = _mm256_loadu_pd(&K[j * dk + k]);
                __m256d vmul = _mm256_mul_pd(vq, vk);
                vsum = _mm256_add_pd(vsum, vmul);
            }
            double tmp_v[4];
            _mm256_storeu_pd(tmp_v, vsum);
            double tmp = tmp_v[0] + tmp_v[1] + tmp_v[2] + tmp_v[3];
            for(; k < dk; k++) {
                tmp += Q_local[i * dk + k] * K[j * dk + k];
            }
            t[i * n + j] = tmp / sqrt((double)dk);
            max_val = t[i * n + j] > max_val? t[i * n + j] : max_val;
        }
        /*
        double sum = 0;
        for(int j = 0; j + 3 < n; j += 4) {
            __mm256d vt = _mm256_loadu_pd(&t[i * n + j]);
            vt = _mm256_exp_pd(_mm256_sub_pd(vt, _mm256_set1_pd(max_val)));
            _mm256_storeu_pd(&t[i * n + j], vt);
            sum += t[i * n + j] + t[i * n + j + 1] + t[i * n + j + 2] + t[i * n + j + 3];
        }*/
        double sum = 0;
        for(int j = 0; j < n; j++) {
            t[i * n + j] = exp(t[i * n + j] - max_val);
            sum += t[i * n + j];
        }
        
        for(int j = 0; j + 3 < n; j += 4) {
            __m256d vt = _mm256_loadu_pd(&t[i * n + j]);
            vt = _mm256_div_pd(vt, _mm256_set1_pd(sum));
            _mm256_storeu_pd(&t[i * n + j], vt);
        }
        int j_tail = (n / 4) * 4;
        for (int j = j_tail; j < n; j++) {
            t[i * n + j] /= sum;
        }
    }
   
    double* VT = malloc(n * dv * sizeof(double));
    transpose_V(V, VT, n, dv);

    /*
    for(int i = 0; i < m_local; i++) {
        for(int j = 0; j < dv; j++) {
            result_local[i * dv + j] = 0.0;
            for(int k = 0; k < n; k++) {
                //result_local[i * dv + j] += t[i * n + k] * VT[j * n + k];
                result_local[i * dv + j] += t[i * n + k] * V[k * dv + j];
            }
        }
    }*/
    
    for(int i = 0; i < m_local; i++) {
        for(int j = 0; j < dv; j++) {
            __m256d vsum = _mm256_setzero_pd();
            int k;
            for(k = 0; k + 3 < n; k += 4) {
                __m256d vt = _mm256_loadu_pd(&t[i * n + k]);
                __m256d vV = _mm256_loadu_pd(&VT[j * n + k]);
                __m256d vmul = _mm256_mul_pd(vt, vV);
                vsum = _mm256_add_pd(vsum, vmul);
            }
            double tmp_v[4];
            _mm256_storeu_pd(tmp_v, vsum);
            double sum = tmp_v[0] + tmp_v[1] + tmp_v[2] + tmp_v[3];
            for(; k < n; k++) {
                sum += t[i * n + k] * VT[j * n + k];
            }
            result_local[i * dv + j] = sum;
        }
    }

    t_comp_end = MPI_Wtime();
    comp_time += (t_comp_end - t_comp_start);

    //gather
    t_comm_start = MPI_Wtime();

    MPI_Gather(result_local, m_local * dv, MPI_DOUBLE, result, m_local * dv, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    t_comm_end = MPI_Wtime();
    comm_time += (t_comm_end - t_comm_start);
    free(Q_local);
    free(result_local);
    free(t);
    free(VT);

    double comm_time_max, comp_time_max;
    double comm_time_us = comm_time * 1e6;
    double comp_time_us = comp_time * 1e6;
    MPI_Reduce(&comm_time, &comm_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time, &comp_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    printf("[Rank %d] Comm: %.2lf us | Comp: %.2lf us\n",
           mpi_rank, comm_time_us, comp_time_us);

    MPI_Barrier(MPI_COMM_WORLD);
    if (mpi_rank == 0) {
        printf("Communication time: %.2lf us\n", comm_time_max * 1e6);
        printf("Computation time:   %.2lf us\n", comp_time_max * 1e6);
    }
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
