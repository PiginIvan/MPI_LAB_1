#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned long long ull;

/* Вспомогательная функция: выделение памяти для массива double */
double* alloc_d(size_t n) {
    double* p = (double*)malloc(n * sizeof(double));
    if (!p) {
        fprintf(stderr, "Allocation failed for %zu elements\n", n);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return p;
}

/* Заполнение массива случайными числами в диапазоне [-1,1] */
void fill_rand(double* a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        a[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
    }
}

/* Умножение матрицы на вектор: построчное распределение */
double matvec_rows(int n, double* A_root, double* v_root, double* y_root, int rank, int size) {
    int base = n / size;
    int rem = n % size;
    int* rows_counts = (int*)malloc(size * sizeof(int));
    int* rows_displs = (int*)malloc(size * sizeof(int));
    int offset = 0;
    for (int p = 0; p < size; ++p) {
        rows_counts[p] = base + (p < rem ? 1 : 0);
        rows_displs[p] = offset;
        offset += rows_counts[p];
    }

    int local_rows = rows_counts[rank];
    double* A_local = alloc_d((size_t)local_rows * n);
    int* sendcounts = (int*)malloc(size * sizeof(int));
    int* sdispls = (int*)malloc(size * sizeof(int));
    for (int p = 0; p < size; ++p) {
        sendcounts[p] = rows_counts[p] * n;
        sdispls[p] = rows_displs[p] * n;
    }

    MPI_Bcast(v_root, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(A_root, sendcounts, sdispls, MPI_DOUBLE,
        A_local, local_rows * n, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    double* y_local = alloc_d(local_rows);
    for (int i = 0; i < local_rows; ++i) {
        double sum = 0.0;
        double* row = A_local + (size_t)i * n;
        for (int j = 0; j < n; ++j) sum += row[j] * v_root[j];
        y_local[i] = sum;
    }

    double t1 = MPI_Wtime();
    double local_time = t1 - t0;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int* recvcounts = rows_counts;
    int* rdispls = rows_displs;
    MPI_Gatherv(y_local, local_rows, MPI_DOUBLE,
        y_root, recvcounts, rdispls, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    free(A_local);
    free(y_local);
    free(sendcounts);
    free(sdispls);
    free(rows_counts);
    free(rows_displs);

    return local_time;
}

/* Умножение матрицы на вектор: поколоночное распределение */
double matvec_cols(int n, double* A_root, double* v_root, double* y_root, int rank, int size) {
    int base = n / size;
    int rem = n % size;
    int* cols_counts = (int*)malloc(size * sizeof(int));
    int* cols_displs = (int*)malloc(size * sizeof(int));
    int offset = 0;
    for (int p = 0; p < size; ++p) {
        cols_counts[p] = base + (p < rem ? 1 : 0);
        cols_displs[p] = offset;
        offset += cols_counts[p];
    }

    int local_cols = cols_counts[rank];
    double* A_local = alloc_d((size_t)n * local_cols);
    double* v_local = alloc_d(local_cols);

    if (rank == 0) {
        for (int col = 0; col < local_cols; ++col) {
            int cidx = cols_displs[0] + col;
            for (int i = 0; i < n; ++i) {
                A_local[(size_t)col * n + i] = A_root[(size_t)i * n + cidx];
            }
            v_local[col] = v_root[cols_displs[0] + col];
        }
        for (int p = 1; p < size; ++p) {
            int lc = cols_counts[p];
            int cstart = cols_displs[p];
            double* buf = alloc_d((size_t)n * lc);
            for (int col = 0; col < lc; ++col) {
                int cidx = cstart + col;
                for (int i = 0; i < n; ++i) {
                    buf[(size_t)col * n + i] = A_root[(size_t)i * n + cidx];
                }
            }
            MPI_Send(buf, n * lc, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            MPI_Send(v_root + cstart, lc, MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
            free(buf);
        }
    }
    else {
        MPI_Recv(A_local, n * local_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(v_local, local_cols, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    double* y_partial = alloc_d(n);
    for (int i = 0; i < n; ++i) y_partial[i] = 0.0;
    for (int col = 0; col < local_cols; ++col) {
        double vv = v_local[col];
        double* acol = A_local + (size_t)col * n;
        for (int i = 0; i < n; ++i) y_partial[i] += acol[i] * vv;
    }

    double t1 = MPI_Wtime();
    double local_time = t1 - t0;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(y_partial, y_root, n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    free(A_local);
    free(v_local);
    free(y_partial);
    free(cols_counts);
    free(cols_displs);

    return local_time;
}

/* Умножение матрицы на вектор: блочное распределение (2D) */
double matvec_blocks(int n, double* A_root, double* v_root, double* y_root, int rank, int size) {
    int prows = 1, pcols = size;
    for (int d = 1; d * d <= size; ++d) {
        if (size % d == 0) prows = d;
    }
    pcols = size / prows;

    int dims[2] = { prows, pcols };
    int periods[2] = { 0, 0 };
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);

    int coords[2];
    MPI_Comm_rank(grid_comm, &rank);
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    int prow = coords[0], pcol = coords[1];

    int* rows_counts = (int*)malloc(dims[0] * sizeof(int));
    int* rows_displs = (int*)malloc(dims[0] * sizeof(int));
    int base_r = n / dims[0];
    int rem_r = n % dims[0];
    int off = 0;
    for (int i = 0; i < dims[0]; ++i) {
        rows_counts[i] = base_r + (i < rem_r ? 1 : 0);
        rows_displs[i] = off;
        off += rows_counts[i];
    }

    int* cols_counts = (int*)malloc(dims[1] * sizeof(int));
    int* cols_displs = (int*)malloc(dims[1] * sizeof(int));
    int base_c = n / dims[1];
    int rem_c = n % dims[1];
    off = 0;
    for (int j = 0; j < dims[1]; ++j) {
        cols_counts[j] = base_c + (j < rem_c ? 1 : 0);
        cols_displs[j] = off;
        off += cols_counts[j];
    }

    int my_rows = rows_counts[prow];
    int my_cols = cols_counts[pcol];
    double* A_block = alloc_d((size_t)my_rows * my_cols);

    MPI_Comm col_comm, row_comm;
    MPI_Comm_split(grid_comm, pcol, prow, &col_comm);
    MPI_Comm_split(grid_comm, prow, pcol, &row_comm);

    if (rank == 0) {
        for (int i = 0; i < dims[0]; ++i) {
            for (int j = 0; j < dims[1]; ++j) {
                int dest_coords[2] = { i, j };
                int dest_rank;
                MPI_Cart_rank(grid_comm, dest_coords, &dest_rank);
                int rcount = rows_counts[i];
                int ccount = cols_counts[j];
                double* buf = alloc_d((size_t)rcount * ccount);
                for (int rr = 0; rr < rcount; ++rr) {
                    int global_r = rows_displs[i] + rr;
                    for (int cc = 0; cc < ccount; ++cc) {
                        int global_c = cols_displs[j] + cc;
                        buf[(size_t)rr * ccount + cc] = A_root[(size_t)global_r * n + global_c];
                    }
                }
                if (dest_rank == 0) {
                    memcpy(A_block, buf, (size_t)rcount * ccount * sizeof(double));
                }
                else {
                    MPI_Send(buf, rcount * ccount, MPI_DOUBLE, dest_rank, 10, grid_comm);
                }
                free(buf);
            }
        }
        for (int j = 0; j < dims[1]; ++j) {
            int ccount = cols_counts[j];
            int cstart = cols_displs[j];
            int dest_coords[2] = { 0, j };
            int dest_rank;
            MPI_Cart_rank(grid_comm, dest_coords, &dest_rank);
            if (dest_rank == 0) {
            }
            else {
                MPI_Send(v_root + cstart, ccount, MPI_DOUBLE, dest_rank, 20, grid_comm);
            }
        }
    }
    else {
        int recv_count = my_rows * my_cols;
        MPI_Recv(A_block, recv_count, MPI_DOUBLE, 0, 10, grid_comm, MPI_STATUS_IGNORE);
    }

    double* v_sub = alloc_d(my_cols);
    if (prow == 0) {
        int cstart = cols_displs[pcol];
        if (rank == 0) {
            for (int cc = 0; cc < my_cols; ++cc) v_sub[cc] = v_root[cstart + cc];
        }
        else {
            MPI_Recv(v_sub, my_cols, MPI_DOUBLE, 0, 20, grid_comm, MPI_STATUS_IGNORE);
        }
    }
    MPI_Bcast(v_sub, my_cols, MPI_DOUBLE, 0, col_comm);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    double* y_part = alloc_d(my_rows);
    for (int i = 0; i < my_rows; ++i) {
        double s = 0.0;
        double* row = A_block + (size_t)i * my_cols;
        for (int j = 0; j < my_cols; ++j) s += row[j] * v_sub[j];
        y_part[i] = s;
    }

    double* y_rowroot = NULL;
    if (pcol == 0) {
        y_rowroot = alloc_d(my_rows);
    }
    MPI_Reduce(y_part, y_rowroot, my_rows, MPI_DOUBLE, MPI_SUM, 0, row_comm);

    double t1 = MPI_Wtime();
    double local_time = t1 - t0;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (pcol == 0) {
        int row_global_start = rows_displs[prow];
        int row_root_coords[2] = { prow, 0 };
        int row_root_rank;
        MPI_Cart_rank(grid_comm, row_root_coords, &row_root_rank);
        if (rank == row_root_rank) {
            if (row_root_rank == 0) {
                for (int i = 0; i < my_rows; ++i) y_root[row_global_start + i] = y_rowroot[i];
                for (int irow = 1; irow < dims[0]; ++irow) {
                    int sender_coords[2] = { irow, 0 };
                    int sender_rank;
                    MPI_Cart_rank(grid_comm, sender_coords, &sender_rank);
                    int recv_rows = rows_counts[irow];
                    MPI_Recv(y_root + rows_displs[irow], recv_rows, MPI_DOUBLE, sender_rank, 30, grid_comm, MPI_STATUS_IGNORE);
                }
            }
            else {
                MPI_Send(y_rowroot, my_rows, MPI_DOUBLE, 0, 30, grid_comm);
            }
        }
        free(y_rowroot);
    }

    free(A_block);
    free(v_sub);
    free(y_part);
    free(rows_counts);
    free(rows_displs);
    free(cols_counts);
    free(cols_displs);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&grid_comm);

    return local_time;
}

/* Функция для записи максимального времени в CSV */
void write_max_time_to_csv(const char* algorithm, int world_size, int n, double max_time, int world_rank) {
    if (world_rank != 0) return; // Òîëüêî ïðîöåññ 0 çàïèñûâàåò â ôàéë

    FILE* fp;
    errno_t err;

    err = fopen_s(&fp, "result.csv", "a");

    if (err == 0 && fp != NULL) {
        fprintf(fp, "%s,%d,%d,%.9f\n", algorithm, world_size, n, max_time);
        fclose(fp);
    }
}

/* Функция для запуска алгоритма и записи максимального времени */
void run_algorithm_max_time(const char* name, double (*matvec_func)(int, double*, double*, double*, int, int),
    int n, double* A_root, double* v_root, double* y_root,
    int world_rank, int world_size) {

    double local_time = matvec_func(n, A_root, v_root, y_root, world_rank, world_size);

    // Находим максимальное время среди всех процессов
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Процесс 0 записывает максимальное время в CSV
    if (world_rank == 0) {
        write_max_time_to_csv(name, world_size, n, max_time, world_rank);
        printf("Algorithm: %s, NP: %d, N: %d, Max Time: %.9f sec\n",
            name, world_size, n, max_time);
        fflush(stdout);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc < 3) {
        if (world_rank == 0) {
            fprintf(stderr, "Usage: %s <n> <algo>\n", argv[0]);
            fprintf(stderr, "  <algo> = rows | cols | blocks | all\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int n = atoi(argv[1]);
    char* algo = argv[2];

    srand((unsigned int)time(NULL) + (unsigned int)world_rank * 1013);

    double* A_root = NULL;
    double* v_root = NULL;
    double* y_root = NULL;

    if (world_rank == 0) {
        A_root = alloc_d((size_t)n * n);
        v_root = alloc_d(n);
        y_root = alloc_d(n);
        fill_rand(A_root, (size_t)n * n);
        fill_rand(v_root, n);

        printf("Matrix-Vector Multiplication Benchmark\n");
        printf("Matrix size: %dx%d\n", n, n);
        printf("Algorithm: %s\n\n", algo);
        fflush(stdout);
    }
    else {
        v_root = alloc_d(n);
    }

    // Запуск выбранного алгоритма
    if (strcmp(algo, "rows") == 0) {
        run_algorithm_max_time("rows", matvec_rows, n, A_root, v_root, y_root, world_rank, world_size);
    }
    else if (strcmp(algo, "cols") == 0) {
        run_algorithm_max_time("cols", matvec_cols, n, A_root, v_root, y_root, world_rank, world_size);
    }
    else if (strcmp(algo, "blocks") == 0) {
        run_algorithm_max_time("blocks", matvec_blocks, n, A_root, v_root, y_root, world_rank, world_size);
    }
    else if (strcmp(algo, "all") == 0) {
        if (world_rank == 0) {
            printf("Running all algorithms sequentially:\n\n");
            fflush(stdout);
        }
        run_algorithm_max_time("rows", matvec_rows, n, A_root, v_root, y_root, world_rank, world_size);
        run_algorithm_max_time("cols", matvec_cols, n, A_root, v_root, y_root, world_rank, world_size);
        run_algorithm_max_time("blocks", matvec_blocks, n, A_root, v_root, y_root, world_rank, world_size);
    }
    else {
        if (world_rank == 0) fprintf(stderr, "Unknown algo: %s\n", algo);
    }

    if (world_rank == 0) {
        printf("\nBenchmark completed successfully!\n");
        printf("Results saved to: result.csv\n");
        fflush(stdout);
    }

    if (A_root) free(A_root);
    if (v_root) free(v_root);
    if (y_root) free(y_root);

    MPI_Finalize();
    return 0;

}
