#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#define ROOT_RANK 0
#define TAG_A 100
#define TAG_B 200
#define TAG_C 300

typedef struct {
    int world_rank;
    int world_size;
    int grid_size;
    int block_size;
    int matrix_size;
    MPI_Comm grid_comm;
    int grid_coords[2];
    int grid_rank;
} process_context_t;

typedef struct {
    double *A_block;
    double *B_block;
    double *C_block;
    size_t block_elements;
} matrix_blocks_t;

/* Функция инициализации контекста процесса */
int initialize_context(process_context_t *ctx, int matrix_size) {
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx->world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx->world_size);
    ctx->matrix_size = matrix_size;

    /* Проверяем, что количество процессов - точный квадрат */
    ctx->grid_size = (int)sqrt(ctx->world_size);
    if (ctx->grid_size * ctx->grid_size != ctx->world_size) {
        if (ctx->world_rank == ROOT_RANK) {
            fprintf(stderr, "Wrong input");
        }
        return -1;
    }

    /* Создаем декартову топологию */
    int dims[2] = {ctx->grid_size, ctx->grid_size};
    int periods[2] = {1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &ctx->grid_comm);
    
    MPI_Comm_rank(ctx->grid_comm, &ctx->grid_rank);
    MPI_Cart_coords(ctx->grid_comm, ctx->grid_rank, 2, ctx->grid_coords);

    /* Вычисляем размер блока с дополнением */
    ctx->block_size = (matrix_size + ctx->grid_size - 1) / ctx->grid_size;
    
    return 0;
}

/* Функция создания блоков матриц */
matrix_blocks_t create_matrix_blocks(int block_size) {
    matrix_blocks_t blocks;
    blocks.block_elements = (size_t)block_size * block_size;
    blocks.A_block = (double*)calloc(blocks.block_elements, sizeof(double));
    blocks.B_block = (double*)calloc(blocks.block_elements, sizeof(double));
    blocks.C_block = (double*)calloc(blocks.block_elements, sizeof(double));
    
    if (!blocks.A_block || !blocks.B_block || !blocks.C_block) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    return blocks;
}

/* Функция освобождения блоков матриц */
void free_matrix_blocks(matrix_blocks_t *blocks) {
    free(blocks->A_block);
    free(blocks->B_block);
    free(blocks->C_block);
    blocks->A_block = blocks->B_block = blocks->C_block = NULL;
}

/* Генерация случайной матрицы на корневом процессе */
double* generate_matrix(int size, unsigned int seed) {
    double *matrix = (double*)malloc((size_t)size * size * sizeof(double));
    if (!matrix) return NULL;
    
    srand(seed);
    for (int i = 0; i < size * size; i++) {
        matrix[i] = (double)rand() / RAND_MAX * 10.0;
    }
    return matrix;
}

/* Распределение матриц по процессам с использованием асинхронных операций */
void distribute_matrices(const process_context_t *ctx, matrix_blocks_t *blocks, 
                        double *global_A, double *global_B) {
    MPI_Request *requests = NULL;
    int request_count = 0;
    
    if (ctx->world_rank == ROOT_RANK) {
        requests = (MPI_Request*)malloc(2 * (ctx->world_size - 1) * sizeof(MPI_Request));
    }

    /* Корневой процесс рассылает блоки, остальные получают */
    if (ctx->world_rank == ROOT_RANK) {
        for (int proc = 0; proc < ctx->world_size; proc++) {
            if (proc == ROOT_RANK) continue;
            
            int coords[2];
            MPI_Cart_coords(ctx->grid_comm, proc, 2, coords);
            
            int row_start = coords[0] * ctx->block_size;
            int col_start = coords[1] * ctx->block_size;
            
            /* Упаковываем блок для отправки */
            double *A_buffer = (double*)malloc(blocks->block_elements * sizeof(double));
            double *B_buffer = (double*)malloc(blocks->block_elements * sizeof(double));
            
            for (int i = 0; i < ctx->block_size; i++) {
                for (int j = 0; j < ctx->block_size; j++) {
                    int global_i = row_start + i;
                    int global_j = col_start + j;
                    int buf_idx = i * ctx->block_size + j;
                    
                    if (global_i < ctx->matrix_size && global_j < ctx->matrix_size) {
                        A_buffer[buf_idx] = global_A[global_i * ctx->matrix_size + global_j];
                        B_buffer[buf_idx] = global_B[global_i * ctx->matrix_size + global_j];
                    } else {
                        A_buffer[buf_idx] = 0.0;
                        B_buffer[buf_idx] = 0.0;
                    }
                }
            }
            
            MPI_Isend(A_buffer, (int)blocks->block_elements, MPI_DOUBLE, proc, 
                     TAG_A, MPI_COMM_WORLD, &requests[request_count++]);
            MPI_Isend(B_buffer, (int)blocks->block_elements, MPI_DOUBLE, proc, 
                     TAG_B, MPI_COMM_WORLD, &requests[request_count++]);
            
            /* Буферы будут освобождены после завершения отправки */
            MPI_Request_free(&requests[request_count-2]);
            MPI_Request_free(&requests[request_count-1]);
        }
        
        /* Корневой процесс копирует свой блок */
        int row_start = ctx->grid_coords[0] * ctx->block_size;
        int col_start = ctx->grid_coords[1] * ctx->block_size;
        
        for (int i = 0; i < ctx->block_size; i++) {
            for (int j = 0; j < ctx->block_size; j++) {
                int global_i = row_start + i;
                int global_j = col_start + j;
                int buf_idx = i * ctx->block_size + j;
                
                if (global_i < ctx->matrix_size && global_j < ctx->matrix_size) {
                    blocks->A_block[buf_idx] = global_A[global_i * ctx->matrix_size + global_j];
                    blocks->B_block[buf_idx] = global_B[global_i * ctx->matrix_size + global_j];
                }
            }
        }
    } else {
        /* Не-корневые процессы получают свои блоки */
        MPI_Recv(blocks->A_block, (int)blocks->block_elements, MPI_DOUBLE, 
                ROOT_RANK, TAG_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(blocks->B_block, (int)blocks->block_elements, MPI_DOUBLE, 
                ROOT_RANK, TAG_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    if (ctx->world_rank == ROOT_RANK) {
        free(requests);
    }
}

/* Локальное умножение блоков: C += A * B */
void local_matrix_multiply(int block_size, const double *A, const double *B, double *C) {
    for (int i = 0; i < block_size; i++) {
        for (int k = 0; k < block_size; k++) {
            double a_val = A[i * block_size + k];
            for (int j = 0; j < block_size; j++) {
                C[i * block_size + j] += a_val * B[k * block_size + j];
            }
        }
    }
}

/* Выполнение алгоритма Кэннона */
void cannon_algorithm(const process_context_t *ctx, matrix_blocks_t *blocks) {
    int left_rank, right_rank, up_rank, down_rank;
    
    /* Определяем соседей для сдвигов */
    MPI_Cart_shift(ctx->grid_comm, 1, -1, &right_rank, &left_rank);  /* Горизонтальные сдвиги */
    MPI_Cart_shift(ctx->grid_comm, 0, -1, &down_rank, &up_rank);     /* Вертикальные сдвиги */
    
    /* Начальные сдвиги (выравнивание) */
    MPI_Sendrecv_replace(blocks->A_block, (int)blocks->block_elements, MPI_DOUBLE,
                        left_rank, 0, right_rank, 0, ctx->grid_comm, MPI_STATUS_IGNORE);
    
    MPI_Sendrecv_replace(blocks->B_block, (int)blocks->block_elements, MPI_DOUBLE,
                        up_rank, 0, down_rank, 0, ctx->grid_comm, MPI_STATUS_IGNORE);
    
    /* Основной цикл умножения */
    for (int step = 0; step < ctx->grid_size; step++) {
        /* Локальное умножение */
        local_matrix_multiply(ctx->block_size, blocks->A_block, blocks->B_block, blocks->C_block);
        
        /* Сдвиги блоков для следующей итерации */
        if (step < ctx->grid_size - 1) {
            MPI_Sendrecv_replace(blocks->A_block, (int)blocks->block_elements, MPI_DOUBLE,
                                left_rank, step + 1, right_rank, step + 1, 
                                ctx->grid_comm, MPI_STATUS_IGNORE);
            
            MPI_Sendrecv_replace(blocks->B_block, (int)blocks->block_elements, MPI_DOUBLE,
                                up_rank, step + 1, down_rank, step + 1, 
                                ctx->grid_comm, MPI_STATUS_IGNORE);
        }
    }
}

/* Сбор результатов на корневом процессе */
void gather_results(const process_context_t *ctx, const matrix_blocks_t *blocks, double *global_C) {
    if (ctx->world_rank == ROOT_RANK) {
        /* Корневой процесс копирует свой блок */
        int row_start = ctx->grid_coords[0] * ctx->block_size;
        int col_start = ctx->grid_coords[1] * ctx->block_size;
        
        for (int i = 0; i < ctx->block_size; i++) {
            for (int j = 0; j < ctx->block_size; j++) {
                int global_i = row_start + i;
                int global_j = col_start + j;
                
                if (global_i < ctx->matrix_size && global_j < ctx->matrix_size) {
                    global_C[global_i * ctx->matrix_size + global_j] = 
                        blocks->C_block[i * ctx->block_size + j];
                }
            }
        }
        
        /* Получаем блоки от других процессов */
        for (int proc = 1; proc < ctx->world_size; proc++) {
            double *recv_buffer = (double*)malloc(blocks->block_elements * sizeof(double));
            MPI_Recv(recv_buffer, (int)blocks->block_elements, MPI_DOUBLE, 
                    proc, TAG_C, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            int coords[2];
            MPI_Cart_coords(ctx->grid_comm, proc, 2, coords);
            
            int row_start = coords[0] * ctx->block_size;
            int col_start = coords[1] * ctx->block_size;
            
            for (int i = 0; i < ctx->block_size; i++) {
                for (int j = 0; j < ctx->block_size; j++) {
                    int global_i = row_start + i;
                    int global_j = col_start + j;
                    
                    if (global_i < ctx->matrix_size && global_j < ctx->matrix_size) {
                        global_C[global_i * ctx->matrix_size + global_j] = 
                            recv_buffer[i * ctx->block_size + j];
                    }
                }
            }
            
            free(recv_buffer);
        }
    } else {
        /* Не-корневые процессы отправляют свои результаты */
        MPI_Send(blocks->C_block, (int)blocks->block_elements, MPI_DOUBLE, 
                ROOT_RANK, TAG_C, MPI_COMM_WORLD);
    }
}

/* Проверка корректности результата */
void verify_result(const process_context_t *ctx, double *A, double *B, double *C) {
    if (ctx->matrix_size > 8 || ctx->world_rank != ROOT_RANK) return;
    
    double *C_seq = (double*)calloc((size_t)ctx->matrix_size * ctx->matrix_size, sizeof(double));
    
    for (int i = 0; i < ctx->matrix_size; i++) {
        for (int k = 0; k < ctx->matrix_size; k++) {
            for (int j = 0; j < ctx->matrix_size; j++) {
                C_seq[i * ctx->matrix_size + j] += A[i * ctx->matrix_size + k] * B[k * ctx->matrix_size + j];
            }
        }
    }
    
    /* Сравнение результатов */
    double max_error = 0.0;
    for (int i = 0; i < ctx->matrix_size * ctx->matrix_size; i++) {
        double error = fabs(C[i] - C_seq[i]);
        if (error > max_error) max_error = error;
    }
    
    printf("Max error: %e\n", max_error);
    free(C_seq);
}

/* Основная функция выполнения умножения */
double run_cannon_multiplication(int matrix_size, const char *output_prefix) {
    process_context_t ctx;
    matrix_blocks_t blocks;
    double *global_A = NULL, *global_B = NULL, *global_C = NULL;
    double execution_time = 0.0;
    
    /* Инициализация контекста */
    if (initialize_context(&ctx, matrix_size) != 0) {
        return -1.0;
    }
    
    /* Создание блоков матриц */
    blocks = create_matrix_blocks(ctx.block_size);
    
    /* Генерация матриц на корневом процессе */
    if (ctx.world_rank == ROOT_RANK) {
        global_A = generate_matrix(ctx.matrix_size, (unsigned int)time(NULL));
        global_B = generate_matrix(ctx.matrix_size, (unsigned int)time(NULL) + 1);
        global_C = (double*)calloc((size_t)ctx.matrix_size * ctx.matrix_size, sizeof(double));
        
        if (!global_A || !global_B || !global_C) {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    
    /* Синхронизация и замер времени */
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    /* Распределение данных */
    distribute_matrices(&ctx, &blocks, global_A, global_B);
    
    /* Выполнение алгоритма Кэннона */
    cannon_algorithm(&ctx, &blocks);
    
    /* Сбор результатов */
    gather_results(&ctx, &blocks, global_C);
    
    double end_time = MPI_Wtime();
    execution_time = end_time - start_time;
    
    if (ctx.world_rank == ROOT_RANK) {
        verify_result(&ctx, global_A, global_B, global_C);
        
        if (output_prefix) {
            
            FILE *f = fopen("results.csv", "a");
            if (f) {
                fprintf(f, "cannon,%d,%d,%.6f\n", ctx.world_size, matrix_size, execution_time);
                fclose(f);
            }
        }
    }
    
    free_matrix_blocks(&blocks);
    if (ctx.world_rank == ROOT_RANK) {
        free(global_A);
        free(global_B);
        free(global_C);
    }
    
    MPI_Comm_free(&ctx.grid_comm);
    
    return execution_time;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    if (argc < 2) {
        if (world_rank == ROOT_RANK) {
            fprintf(stderr, "Wrong input");
        }
        MPI_Finalize();
        return 1;
    }
    
    int matrix_size = atoi(argv[1]);
    const char *output_prefix = (argc >= 3) ? argv[2] : "cannon";
    
    if (matrix_size <= 0) {
        if (world_rank == ROOT_RANK) {
            fprintf(stderr, "Wrong input");
        }
        MPI_Finalize();
        return 1;
    }
    
    double time = run_cannon_multiplication(matrix_size, output_prefix);
    
    MPI_Finalize();
    return 0;
}
