#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);  // общее число процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);        // номер текущего процесса

    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <total_tosses>\n", argv[0]);
            fprintf(stderr, "Example: mpiexec -np 4 %s 10000000\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    char *endptr = NULL;
    long long total_tosses = strtoll(argv[1], &endptr, 10);

    if (endptr == argv[1] || total_tosses <= 0) {
        if (rank == 0)
            fprintf(stderr, "Invalid total_tosses: %s\n", argv[1]);

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Распределение количества бросков по процессам 
    // base = сколько бросков получит каждый процесс минимум
    long long base = total_tosses / world_size;
    long long rem  = total_tosses % world_size;
    // первые <rem> процессов получают на 1 бросок больше
    long long local_tosses = base + (rank < rem ? 1 : 0);

    MPI_Barrier(MPI_COMM_WORLD);    
    double t_start = MPI_Wtime();      

    srand((unsigned int)(rank * 100003));

    long long local_hits = 0;
    float radius = 1.0;  // радиус единичного круга

    // Монте-Карло цикл
    for (long long i = 0; i < local_tosses; ++i) {
        // генерируем точку в квадрате [0..1]
        double rx = (double)rand() / (double)RAND_MAX;
        double ry = (double)rand() / (double)RAND_MAX;

        // переносим точку в квадрат [-1..1]
        double x = 2.0 * rx - 1.0;
        double y = 2.0 * ry - 1.0;

        // проверяем, попала ли в круг
        if (x * x + y * y <= radius) {
            local_hits++;
        }
    }

    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;

    // Суммируем попадания всех процессов
    long long global_hits = 0;
    MPI_Reduce(&local_hits, &global_hits, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);

    // Вычисляем максимальное время среди всех процессов
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    // Только процесс 0 выводит результаты
    if (rank == 0) {
        // оценка числа PI = 4 * (количество попаданий / общее число точек)
        double pi_estimate = 4.0 * (double)global_hits / (double)total_tosses;

        FILE *test = fopen("results.csv", "r");

        FILE *fp = fopen("results.csv", "a");
        if (!fp) {
            fprintf(stderr, "Failed to open results.csv for writing\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        fprintf(fp, "%d,%lld,%.9f,%.12f\n",
                world_size, (long long)total_tosses, max_time, pi_estimate);

        fclose(fp);

        printf("%d,%lld,%.9f,%.12f\n",
               world_size, (long long)total_tosses, max_time, pi_estimate);
        fflush(stdout);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
