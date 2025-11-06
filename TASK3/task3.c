#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <sys/stat.h>

/* Последовательное умножение матриц (на одном процессе) — возвращает время выполнения */
static double seq_matmul_time(int N, double *A, double *B, double *C) {
    for (int i = 0; i < N * N; ++i) C[i] = 0.0;

    double t0 = MPI_Wtime();
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            double a_ik = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
    double t1 = MPI_Wtime();
    return t1 - t0;
}

static int file_exists(const char *path) {
    struct stat buf;
    return (stat(path, &buf) == 0);
}

static void append_result_csv(const char *fname, int N, int P, double T_par, double T_seq) {
    int exists = file_exists(fname);
    FILE *f = fopen(fname, "a");
    if (!f) {
        fprintf(stderr, "Cannot open %s for appending\n", fname);
        return;
    }
    if (!exists) {
        fprintf(f, "N,P,T_par,T_seq,Speedup,Efficiency\n");
    }
    double speedup = (T_par > 0.0) ? (T_seq / T_par) : 0.0;
    double efficiency = (P > 0) ? (speedup / (double)P) : 0.0;
    fprintf(f, "%d,%d,%.9f,%.9f,%.6f,%.6f\n", N, P, T_par, T_seq, speedup, efficiency);
    fclose(f);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N;
    if (argc > 1) N = atoi(argv[1]);
    if (N <= 0) N = 500;

    /* Требуем, чтобы количество процессов было квадратом целого */
    int q = (int)floor(sqrt((double)size) + 0.5);
    if (q * q != size) {
        if (rank == 0)
            fprintf(stderr, "Error: number of processes (%d) must be a perfect square.\n", size);
        MPI_Finalize();
        return 1;
    }

    /* Проверяем, что N делится на q (чтобы блоки были целыми) */
    if (N % q != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: N=%d is not divisible by q=%d (block size must be integer).\n", N, q);
        MPI_Finalize();
        return 1;
    }

    int n = N / q; /* размер локального блока n x n */
    /* Выделяем локальные буферы для блоков A, B, C */
    double *A_blk = (double *)malloc((size_t)n * n * sizeof(double));
    double *B_blk = (double *)malloc((size_t)n * n * sizeof(double));
    double *C_blk = (double *)calloc((size_t)n * n, sizeof(double)); 
    if (!A_blk || !B_blk || !C_blk) {
        fprintf(stderr, "Rank %d: allocation failed for local blocks\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* На rank 0 создаём полные матрицы A_full и B_full, заполняем значениями,
       выполняем последовательное умножение для замера T_seq, затем раздаём блоки через MPI_Scatter */
    double T_seq = 0.0;
    double *A_full = NULL;
    double *B_full = NULL;
    if (rank == 0) {
        A_full = (double *)malloc((size_t)N * N * sizeof(double));
        B_full = (double *)malloc((size_t)N * N * sizeof(double));
        if (!A_full || !B_full) {
            fprintf(stderr, "Rank 0: failed to allocate full matrices\n");
            free(A_full); free(B_full);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        /* Заполнение матриц небольшими числами (как и в исходнике) */
        for (int i = 0; i < N * N; ++i) {
            A_full[i] = (double)(i % 10);
            B_full[i] = (double)((i + 1) % 10);
        }

        /* Выделяем временный буфер для результата последовательного умножения */
        double *C_seq = (double *)malloc((size_t)N * N * sizeof(double));
        if (!C_seq) {
            fprintf(stderr, "Rank 0: cannot allocate C_seq\n");
            free(A_full); free(B_full);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        T_seq = seq_matmul_time(N, A_full, B_full, C_seq);
        free(C_seq);
    }

    /* Раздаём блоки A и B всем процессам */
    MPI_Scatter(A_full, n * n, MPI_DOUBLE, A_blk, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(B_full, n * n, MPI_DOUBLE, B_blk, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(A_full);
        free(B_full);
    }

    /* Определяем координаты процесса в q x q сетке (row, col) */
    int row = rank / q;
    int col = rank % q;

    /* 
       - сдвиг A влево на row позиций (каждый ряд) (делаем циклические Sendrecv_replace)
       - сдвиг B вверх на col позиций (каждый столбец)
    */
    for (int s = 0; s < row; ++s) {
        int left = row * q + (col - 1 + q) % q;
        int right = row * q + (col + 1) % q;
        MPI_Sendrecv_replace(A_blk, n * n, MPI_DOUBLE,
                             right, 101,
                             left, 101,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    /* Сдвиг B вверх col раз */
    for (int s = 0; s < col; ++s) {
        int up = ((row - 1 + q) % q) * q + col;
        int down = ((row + 1) % q) * q + col;
        MPI_Sendrecv_replace(B_blk, n * n, MPI_DOUBLE,
                             down, 102,
                             up, 102,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* Основной цикл Кэннона: p итераций (q == sqrt(size)) */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int step = 0; step < q; ++step) {
        /* Локальное умножение блока: C_blk += A_blk * B_blk */
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < n; ++k) {
                double a_ik = A_blk[i * n + k];
                int base = i * n;
                int offset = k * n;
                for (int j = 0; j < n; ++j) {
                    C_blk[base + j] += a_ik * B_blk[offset + j];
                }
            }
        }

        /* Циклические сдвиги: A влево на 1, B вверх на 1 */
        int left = row * q + (col - 1 + q) % q;
        int right = row * q + (col + 1) % q;
        MPI_Sendrecv_replace(A_blk, n * n, MPI_DOUBLE,
                             right, 201,
                             left, 201,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int up = ((row - 1 + q) % q) * q + col;
        int down = ((row + 1) % q) * q + col;
        MPI_Sendrecv_replace(B_blk, n * n, MPI_DOUBLE,
                             down, 202,
                             up, 202,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double local_elapsed = t1 - t0;

    /* Собираем максимальное время выполнения среди процессов (для корректного замера) */
    double T_par = 0.0;
    MPI_Reduce(&local_elapsed, &T_par, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        append_result_csv("results.csv", N, size, T_par, T_seq);
    }

    free(A_blk);
    free(B_blk);
    free(C_blk);

    MPI_Finalize();
    return 0;
}
