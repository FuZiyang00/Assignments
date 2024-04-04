#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <string.h> 
#include <mpi.h>
#include "pgm.h"
#include <omp.h>
#include <pthread.h>

#define mb_t unsigned short
#define TAG_TASK_REQUEST 1
#define TAG_TASK_DATA 2
#define TAG_TASK_ROW 3
#define TAG_MATRIX_ROW 4

// Recursive Mandelbrot function
mb_t mandelbrot_func(double complex z, double complex c, int n, int Imax) {
    if (cabs(z) >= 2.0) { return (mb_t)n; }
    
    if (n >= Imax) { return (mb_t)0; }

    return mandelbrot_func(z * z + c, c, n + 1, Imax);
}

// Compute Mandelbrot set for a single row
void compute_row(mb_t *row, int nx, double xL, double xR, double y, int Imax) {
    double dx = (double)(xR - xL) / (double)(nx - 1);
    double x;
    double complex c;

    #pragma omp parallel for shared(row, nx, xL, dx, y, Imax) private(x, c)
    for (int j = 0; j < nx; j++) {
        x = xL + j * dx;
        c = x + I * y;
        row[j] = mandelbrot_func(0 * I, c, 0, Imax);
    }
}

void slave_process(int nx, int ny, double xL, double yL, double xR, double yR, int Imax, int rank) {
    while (1) {
        // signal the process availability
        MPI_Send(&rank, 1, MPI_INT, 0, TAG_TASK_REQUEST, MPI_COMM_WORLD);

        // receive the row index
        int row_index;
        MPI_Recv(&row_index, 1, MPI_INT, 0, TAG_TASK_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Check if termination signal received
        if (row_index == -1)
        {
            printf("Termination signal received. Process %d exiting.\n", rank);
            break;
        }

        mb_t *row = (mb_t *)malloc(nx * sizeof(mb_t));
        double dy = (yR - yL) / (double)(ny - 1);
        double y = yL + row_index * dy;
        compute_row(row, nx, xL, xR, y, Imax);

        MPI_Send(&row_index, 1, MPI_INT, 0, TAG_TASK_ROW, MPI_COMM_WORLD);
        MPI_Send(row, nx, MPI_UNSIGNED_SHORT, 0, TAG_MATRIX_ROW, MPI_COMM_WORLD);

        // Free allocated memory
        free(row);
    }
}

mb_t *mandelbrot_matrix_rr(int nx, int ny, int size) {
    // allocate space for the global matrix
    mb_t *Mandelbrot = (mb_t *)malloc(sizeof(mb_t) * nx * ny);
    if (Mandelbrot == NULL) {
        // Handle memory allocation failure
        printf("Error: Memory allocation failed\n");
        exit(1);
    }
    int next_row = 0;

    // Declare a mutex variable
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

    while (next_row < ny) {
        // Acquire the mutex
        pthread_mutex_lock(&mutex);

        // Find the first available process
        int available_p;
        MPI_Recv(&available_p, 1, MPI_INT, MPI_ANY_SOURCE, TAG_TASK_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d is available \n", available_p);

        // Send data to the available slave
        MPI_Send(&next_row, 1, MPI_INT, available_p, TAG_TASK_DATA, MPI_COMM_WORLD);
        printf("Sent row %d to process %d\n", next_row, available_p);

        // Release the mutex
        pthread_mutex_unlock(&mutex);
        next_row++;
    }

    // Allocate memory for the receive_matrix buffer
    mb_t *row = (mb_t *)malloc(nx * sizeof(mb_t));

    for (int i = 0; i < ny; i++) {

        MPI_Status status;
        int row_index;
        MPI_Recv(&row_index, 1, MPI_INT, MPI_ANY_SOURCE, TAG_TASK_ROW, MPI_COMM_WORLD, &status);
        printf("Received row %d\n", row_index);

        // Receive the row data
        MPI_Recv(row, nx, MPI_UNSIGNED_SHORT, MPI_ANY_SOURCE, TAG_MATRIX_ROW, MPI_COMM_WORLD, &status);
        printf("processed row %d \n", row_index);

        // mapping the row to the global matrix
        memcpy(&Mandelbrot[row_index * nx], row, nx * sizeof(mb_t));
        printf("Copied row %d to global_matrix.\n", row_index);
    }
    

    // Send termination signal to all slaves
    for (int i = 1; i < size; i++) {
        int termination_signal = -1;
        MPI_Send(&termination_signal, 1, MPI_INT, i, TAG_TASK_DATA, MPI_COMM_WORLD);
        printf("Sent termination signal to slave %d\n", i);
    }

    return Mandelbrot;
}

int main(int argc, char **argv) {

    int mpi_thread_init;
    int rank, size;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_thread_init);

    if (mpi_thread_init < MPI_THREAD_FUNNELED) {
        printf("Error: could not initialize MPI with MPI_THREAD_FUNNELED\n");
        MPI_Finalize();
        exit(1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 9) {
        if (rank == 0) {
            printf("Usage: %s nx ny xL yL xR yR Imax image_name\n", argv[0]);
        }
        MPI_Finalize();
        exit(1);
    }

    if (size < 2) { 
        printf("Error: Number of processes must be greater than 2\n");
        MPI_Finalize();
        exit(1);
    }

    int nx, ny, Imax;
    double xL, yL, xR, yR;
    char *image_name;
    mb_t *Mandelbrot = NULL;

    nx = atoi(argv[1]);
    ny = atoi(argv[2]);
    xL = atof(argv[3]);
    yL = atof(argv[4]);
    xR = atof(argv[5]);
    yR = atof(argv[6]);
    Imax = atoi(argv[7]);
    image_name = argv[8];

    if (rank != 0) {
        slave_process(nx, ny, xL, yL, xR, yR, Imax, rank);
    }
    
    else {
        Mandelbrot = mandelbrot_matrix_rr(nx, ny, size); 
        printf("Writing image\n");
        write_pgm_image(Mandelbrot, Imax, nx, ny, image_name);
        free(Mandelbrot);
    }

    MPI_Finalize();
    return 0;
}
