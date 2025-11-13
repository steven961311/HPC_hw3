#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void attention(double* Q, double* K, double* V, double* result,
               int m, int n, int dk, int dv,
               int mpi_rank, int mpi_size) {
    // ========================= 基本通訊（與你前版一致） =========================
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dk, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dv, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const double inv_sqrt_dk = 1.0 / sqrt((double)dk);

    int *rows_per_rank = NULL, *q_counts = NULL, *q_displs = NULL;
    int *r_counts = NULL, *r_displs = NULL;

    if (mpi_rank == 0) {
        rows_per_rank = (int*) malloc(sizeof(int) * mpi_size);
        int base = m / mpi_size, rem = m % mpi_size;
        for (int r = 0; r < mpi_size; ++r) rows_per_rank[r] = base + (r < rem ? 1 : 0);

        q_counts = (int*) malloc(sizeof(int) * mpi_size);
        q_displs = (int*) malloc(sizeof(int) * mpi_size);
        r_counts = (int*) malloc(sizeof(int) * mpi_size);
        r_displs = (int*) malloc(sizeof(int) * mpi_size);

        int q_disp = 0, r_disp = 0;
        for (int r = 0; r < mpi_size; ++r) {
            q_counts[r] = rows_per_rank[r] * dk;
            r_counts[r] = rows_per_rank[r] * dv;
            q_displs[r] = q_disp; q_disp += q_counts[r];
            r_displs[r] = r_disp; r_disp += r_counts[r];
        }
    } else {
        rows_per_rank = (int*) malloc(sizeof(int) * mpi_size);
    }

    MPI_Bcast(rows_per_rank, mpi_size, MPI_INT, 0, MPI_COMM_WORLD);
    int my_rows = rows_per_rank[mpi_rank];

    double *Q_local = (double*) malloc((size_t)my_rows * dk * sizeof(double));
    double *result_local = (double*) malloc((size_t)my_rows * dv * sizeof(double));

    if (mpi_rank != 0) {
        K = (double*) malloc((size_t)n * dk * sizeof(double));
        V = (double*) malloc((size_t)n * dv * sizeof(double));
    }
    MPI_Bcast(K, n * dk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(V, n * dv, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        MPI_Scatterv(Q, q_counts, q_displs, MPI_DOUBLE,
                     Q_local, my_rows * dk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE,
                     Q_local, my_rows * dk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // ========================= K/V 的加速重點 =========================
    // [KV-OPT] 0) 參數：k/dv 分塊大小（可依實機調整）
    const int BK  = 256;  // 沿 k（也就是 n）做分塊，降低 V 的工作集壓力
    const int BDV = 256;  // 沿 dv 做分塊，讓 out 工作集留在快取

    // [KV-OPT] 1) 選擇性：把 K、V 打包到對齊的緩衝區（可略過這步，仍然正確）
    // 這裡為了簡單，使用 unaligned load（與你現有旗標相容）；若要極限壓榨再做對齊打包。
    double *w_blk = (double*) malloc((size_t)BK * sizeof(double));  // 暫存 logits or exp

    for (int i = 0; i < my_rows; ++i) {
        const double *Qi = &Q_local[(size_t)i * dk];

        // [KV-OPT] 2) 線上 softmax 狀態（跨 k-區塊累計）
        double m_run = -INFINITY;   // 當前累計的最大 logit
        double Z_run = 0.0;         // 當前累計的分母（以 m_run 為基準）
        // out 先清 0，之後可能會被 scale（遇到更大的 m_blk）
        double *out = &result_local[(size_t)i * dv];
        for (int j = 0; j < dv; ++j) out[j] = 0.0;

        // 逐 k-區塊
        for (int kb = 0; kb < n; kb += BK) {
            const int kend = (kb + BK < n) ? (kb + BK) : n;
            const int blkN = kend - kb;

            // [KV-OPT] 3) 先算這個區塊的 logits（同時找區塊最大值）
            double m_blk = -INFINITY;
            for (int t = 0; t < blkN; ++t) {
                const double *Kj = &K[(size_t)(kb + t) * dk];
                double dot = 0.0;
                #ifdef USE_AVX2
                int k = 0;
                __m256d acc = _mm256_setzero_pd();
                for (; k + 4 <= dk; k += 4) {
                    __m256d qv = _mm256_loadu_pd(Qi + k);
                    __m256d kv = _mm256_loadu_pd(Kj + k);
                    acc = _mm256_fmadd_pd(qv, kv, acc);
                }
                double tmp[4];
                _mm256_storeu_pd(tmp, acc);
                dot = tmp[0] + tmp[1] + tmp[2] + tmp[3];
                for (; k < dk; ++k) dot += Qi[k] * Kj[k];
                #else
                for (int k = 0; k < dk; ++k) dot += Qi[k] * Kj[k];
                #endif
                double logit = dot * inv_sqrt_dk;
                w_blk[t] = logit;
                if (logit > m_blk) m_blk = logit;
            }

            // [KV-OPT] 4) 若區塊最大值更大，將先前累積縮放到新基準
            if (m_blk > m_run) {
                const double scale = exp(m_run - m_blk);
                Z_run *= scale;
                // 把 out 也縮放（因 out 以同一基準計）
                // 這裡搭配 dv-blocking，減少 out 的快取壓力
                for (int d0 = 0; d0 < dv; d0 += BDV) {
                    int d1 = (d0 + BDV < dv) ? (d0 + BDV) : dv;
                    for (int d = d0; d < d1; ++d) out[d] *= scale;
                }
                m_run = m_blk;
            }

            // [KV-OPT] 5) 將本區塊的 exp(logit - m_run) 累加到 Z 與 out
            // 先把 exponent 存回 w_blk，順便把區塊 Z 做出來
            double Z_blk = 0.0;
            for (int t = 0; t < blkN; ++t) {
                double e = exp(w_blk[t] - m_run);
                w_blk[t] = e;
                Z_blk += e;
            }
            Z_run += Z_blk;

            // out += Σ e * V[k,:]（以 dv-blocking 進行 AXPY）
            for (int d0 = 0; d0 < dv; d0 += BDV) {
                int d1 = (d0 + BDV < dv) ? (d0 + BDV) : dv;
                for (int t = 0; t < blkN; ++t) {
                    const double alpha = w_blk[t];
                    const double *Vk = &V[(size_t)(kb + t) * dv + d0];
                    #ifdef USE_AVX2
                    int j = 0;
                    __m256d av = _mm256_set1_pd(alpha);
                    for (; j + 4 <= (d1 - d0); j += 4) {
                        __m256d outv = _mm256_loadu_pd(out + d0 + j);
                        __m256d vkv  = _mm256_loadu_pd(Vk + j);
                        outv = _mm256_fmadd_pd(av, vkv, outv);
                        _mm256_storeu_pd(out + d0 + j, outv);
                    }
                    for (; j < (d1 - d0); ++j) out[d0 + j] += alpha * Vk[j];
                    #else
                    for (int j = d0; j < d1; ++j) {
                        out[j] += alpha * V[(size_t)(kb + t) * dv + j];
                    }
                    #endif
                }
            }

            // （可選）[KV-OPT] 6) Prefetch 下一塊的 V
            // if (kend < n) _mm_prefetch((const char*)(&V[(size_t)kend * dv]), _MM_HINT_T0);
        } // end for kb

        // [KV-OPT] 7) 正規化：out /= Z_run
        const double invZ = 1.0 / Z_run;
        for (int j = 0; j < dv; ++j) out[j] *= invZ;
    }

    free(w_blk);

    // 回收結果
    if (mpi_rank == 0) {
        MPI_Gatherv(result_local, my_rows * dv, MPI_DOUBLE,
                    result, r_counts, r_displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(result_local, my_rows * dv, MPI_DOUBLE,
                    NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    free(Q_local);
    free(result_local);
    if (mpi_rank != 0) { free(K); free(V); }

    if (mpi_rank == 0) {
        free(rows_per_rank);
        free(q_counts); free(q_displs);
        free(r_counts); free(r_displs);
    } else {
        free(rows_per_rank);
    }
}

// void attention(double* Q, double* K, double* V, double* result,
//                int m, int n, int dk, int dv,
//                int mpi_rank, int mpi_size) {
//     // =========================================================================
//     // 高效版本注意事項：
//     // 1) 處理 m 不能整除 mpi_size：使用 Scatterv/Gatherv (row-wise)
//     // 2) 只保留每列一個長度為 n 的臨時緩衝，不建立 m_local*n 的大矩陣
//     // 3) softmax subtract-max，兩趟：max -> sum -> streaming AXPY 累加輸出
//     // 4) K、V 由 rank 0 廣播到所有 rank（每個 rank 都要用到）
//     // 5) 提供可選 AVX2 加速（-DUSE_AVX2 編譯旗標）
//     // =========================================================================

//     // --- 廣播維度（rank 0 有正確值，其餘 rank 在 Bcast 後獲得） ---
//     MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&dk, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&dv, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     const double inv_sqrt_dk = 1.0 / sqrt((double)dk);

//     // --- 在 rank 0 建立 Scatterv/Gatherv 的 rows, counts, displs ---
//     int *rows_per_rank = NULL, *q_counts = NULL, *q_displs = NULL;
//     int *r_counts = NULL, *r_displs = NULL;

//     if (mpi_rank == 0) {
//         rows_per_rank = (int*) malloc(sizeof(int) * mpi_size);
//         int base = m / mpi_size;
//         int rem  = m % mpi_size;
//         for (int r = 0; r < mpi_size; ++r) rows_per_rank[r] = base + (r < rem ? 1 : 0);

//         q_counts = (int*) malloc(sizeof(int) * mpi_size);
//         q_displs = (int*) malloc(sizeof(int) * mpi_size);
//         r_counts = (int*) malloc(sizeof(int) * mpi_size);
//         r_displs = (int*) malloc(sizeof(int) * mpi_size);

//         int q_disp = 0, r_disp = 0;
//         for (int r = 0; r < mpi_size; ++r) {
//             q_counts[r] = rows_per_rank[r] * dk;
//             r_counts[r] = rows_per_rank[r] * dv;
//             q_displs[r] = q_disp;
//             r_displs[r] = r_disp;
//             q_disp += q_counts[r];
//             r_disp += r_counts[r];
//         }
//     }

//     // --- 每個 rank 先取得自己要處理的 row 數（Broadcast rows）---
//     int my_rows = 0;
//     if (mpi_rank == 0) {
//         my_rows = rows_per_rank[0];
//     }
//     // 為了簡化，不重複算分配表；由 rank 0 廣播每個 rank 的 rows
//     //（也可以 MPI_Scatter rows_per_rank，但一個 int broadcast 就好）
//     MPI_Bcast(&my_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     if (mpi_rank != 0) {
//         // 不是 0 的 rank 還是需要知道“自己”的 rows。最簡單是單獨發一輪：
//         // 用一個迴圈逐個廣播會很慢，所以這裡改用一次性廣播整個陣列。
//         // 為避免重複通訊，我們在下面做一次 rows_per_rank 的全域廣播。
//         rows_per_rank = (int*) malloc(sizeof(int) * mpi_size);
//     }
//     // 這裡做一次 rows_per_rank 的全域廣播，讓所有 rank 都知道自己的 rows
//     MPI_Bcast(rows_per_rank, mpi_size, MPI_INT, 0, MPI_COMM_WORLD);
//     my_rows = rows_per_rank[mpi_rank];

//     // --- 分配本地 Q 與結果緩衝 ---
//     double *Q_local = (double*) malloc((size_t)my_rows * dk * sizeof(double));
//     double *result_local = (double*) malloc((size_t)my_rows * dv * sizeof(double));

//     // --- 廣播 K、V：非 root 需先配置 ---
//     if (mpi_rank != 0) {
//         K = (double*) malloc((size_t)n * dk * sizeof(double));
//         V = (double*) malloc((size_t)n * dv * sizeof(double));
//     }
//     MPI_Bcast(K, n * dk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//     MPI_Bcast(V, n * dv, MPI_DOUBLE, 0, MPI_COMM_WORLD);

//     // --- Scatterv Q 到每個 rank（row-wise）---
//     if (mpi_rank == 0) {
//         MPI_Scatterv(Q, q_counts, q_displs, MPI_DOUBLE,
//                      Q_local, my_rows * dk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//     } else {
//         MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE,
//                      Q_local, my_rows * dk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//     }

//     // --- 計算 ---
//     // 緩衝：softmax 權重（或先放 exp(logits - max)）
//     double *w = (double*) malloc(n * sizeof(double));

//     // 小工具：AVX2 輔助
//     #ifdef USE_AVX2
//     #include <immintrin.h>
//     #endif

//     for (int i = 0; i < my_rows; ++i) {
//         const double *Qi = &Q_local[(size_t)i * dk];

//         // 1) 計算 logits_j = (Qi · Kj) / sqrt(dk)，同時找 max
//         double max_val = -INFINITY;

//         for (int j = 0; j < n; ++j) {
//             const double *Kj = &K[(size_t)j * dk];

//             // dot(Qi, Kj) over dk
//             double dot = 0.0;
//             #ifdef USE_AVX2
//             // AVX2 版本：一次處理 4 doubles (256-bit)
//             int k = 0;
//             __m256d acc = _mm256_setzero_pd();
//             for (; k + 4 <= dk; k += 4) {
//                 __m256d qv = _mm256_loadu_pd(Qi + k);
//                 __m256d kv = _mm256_loadu_pd(Kj + k);
//                 acc = _mm256_fmadd_pd(qv, kv, acc);
//             }
//             double tmp[4];
//             _mm256_storeu_pd(tmp, acc);
//             dot = tmp[0] + tmp[1] + tmp[2] + tmp[3];
//             for (; k < dk; ++k) dot += Qi[k] * Kj[k];
//             #else
//             for (int k = 0; k < dk; ++k) dot += Qi[k] * Kj[k];
//             #endif

//             double logit = dot * inv_sqrt_dk;
//             w[j] = logit;                 // 先暫存 logits（下一步會就地覆蓋為 exp）
//             if (logit > max_val) max_val = logit;
//         }

//         // 2) 第二趟：w[j] = exp(logit - max)，並累加分母
//         double denom = 0.0;
//         for (int j = 0; j < n; ++j) {
//             double e = exp(w[j] - max_val);
//             w[j] = e;
//             denom += e;
//         }
//         const double inv_denom = 1.0 / denom;

//         // 3) 輸出 out_i = sum_k softmax_i[k] * V[k,:]
//         //    寫成 streaming AXPY：out[:] = Σ (w[k]/denom) * V[k,:]
//         double *out = &result_local[(size_t)i * dv];

//         // 先清零
//         for (int j = 0; j < dv; ++j) out[j] = 0.0;

//         for (int k = 0; k < n; ++k) {
//             const double alpha = w[k] * inv_denom;
//             const double *Vk = &V[(size_t)k * dv];

//             #ifdef USE_AVX2
//             int j = 0;
//             for (; j + 4 <= dv; j += 4) {
//                 __m256d outv = _mm256_loadu_pd(out + j);
//                 __m256d vkv  = _mm256_loadu_pd(Vk + j);
//                 __m256d av   = _mm256_set1_pd(alpha);
//                 outv = _mm256_fmadd_pd(av, vkv, outv); // out += alpha * Vk
//                 _mm256_storeu_pd(out + j, outv);
//             }
//             for (; j < dv; ++j) out[j] += alpha * Vk[j];
//             #else
//             for (int j = 0; j < dv; ++j) out[j] += alpha * Vk[j];
//             #endif
//         }
//     }

//     free(w);

//     // --- Gatherv 回 rank 0 ---
//     if (mpi_rank == 0) {
//         MPI_Gatherv(result_local, my_rows * dv, MPI_DOUBLE,
//                     result, r_counts, r_displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//     } else {
//         MPI_Gatherv(result_local, my_rows * dv, MPI_DOUBLE,
//                     NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//     }

//     // --- 清理 ---
//     free(Q_local);
//     free(result_local);

//     if (mpi_rank != 0) {
//         free(K);
//         free(V);
//     }

//     if (mpi_rank == 0) {
//         free(rows_per_rank);
//         free(q_counts);
//         free(q_displs);
//         free(r_counts);
//         free(r_displs);
//     } else {
//         free(rows_per_rank);
//     }
// }
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
