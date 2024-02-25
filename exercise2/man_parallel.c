#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <string.h> 
#include <mpi.h>
#include "pgm.h"
#include <omp.h>

#define mb_t char

// Recursive Mandelbrot function 
mb_t mandelbrot_func(double complex z, double complex c, int n, int Imax)
{
    if (cabs(z) >= 2.0) {
    return (mb_t) n;
    }
    if (n >= Imax) {
    return (mb_t) 0;
    }
    return mandelbrot_func(z * z + c, c, n + 1, Imax);
}

// Mandel Brot Image Matrix
mb_t* mandelbrot_matrix(int nx, int ny, double xL, double yL, double xR, double yR, int Imax, int my_rank, int num_procs)
{
    double dx, dy, x, y;
    double complex c;
    mb_t* matrix;

    dx = (double) (xR - xL) / (double) (nx - 1);
    dy = (double) (yR - yL) / (double) (ny - 1);

    // Determine the number of rows each process will compute
    int rows_per_process = ny / num_procs;
    int start_row = my_rank * rows_per_process;
    int end_row = (my_rank == num_procs - 1) ? ny : start_row + rows_per_process;

    // Allocate memory for the local matrix
    int local_ny = end_row - start_row;
    matrix = (mb_t*) malloc(sizeof(mb_t) * nx * local_ny);

    // Compute Mandelbrot set for the assigned rows
    #pragma omp parallel
    {
        int num_threads;
        #pragma omp master
        {
            int num_threads = omp_get_num_threads();
            printf("Number of threads: %d\n", num_threads);
        }

        #pragma omp for private(y, c) schedule(static) collapse(2)
        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < nx; j++) {
                y = yL + i * dy;
                x = xL + j * dx;
                c = x + I * y;
                matrix[(i - start_row) * nx + j] = mandelbrot_func(0 * I, c, 0, Imax);
            }
        }
    }
    return matrix;
}

// Root process 
void Master(int num_procs, int nx, int ny, double xL, double yL, double xR, double yR, int Imax, char* image_name) {
    
    MPI_Status status;
    mb_t* global_matrix = (mb_t*) malloc(sizeof(mb_t) * nx * ny);
    printf("init master\n");

    int rows_per_process = ny / num_procs;
    int local_ny = rows_per_process;

    // Compute Mandelbrot set for the portion of the image handled by the master
    int start_row = 0;
    int end_row = rows_per_process;
    mb_t* master_matrix = mandelbrot_matrix(nx, ny, xL, yL, xR, yR, Imax, 0, num_procs);

    // Copy the master's computed portion to the global matrix
    memcpy(global_matrix, master_matrix, sizeof(mb_t) * nx * local_ny);
    free(master_matrix);

    // Receive data from each slave process and integrate it into the global matrix
    for(int i = 1; i < num_procs; i++) {
        int start = i * rows_per_process;
        int end = start + rows_per_process;
        int diff = end - start;
        // Calculate the size of data to be received
        int size = nx * diff * sizeof(mb_t);

        // Allocate memory to receive data from the current slave process
        mb_t* recv_matrix = (mb_t*)malloc(size);

        // Receive data from the current slave process (blocking call)
        MPI_Recv(recv_matrix, size, MPI_CHAR, i, 1, MPI_COMM_WORLD, &status);
        int source = status.MPI_SOURCE - 1;
        printf("Received from %d\n", source);

        // Determine the starting row in the global matrix where this data should be placed
        int start_row = i * rows_per_process;

        // Copy received data into the appropriate location in the global matrix
        memcpy(&global_matrix[start_row * nx], recv_matrix, size);

        // Free the memory allocated for receiving data from this slave process
        free(recv_matrix);
    }

    // Write PGM image only on rank 0
    write_pgm_image(global_matrix, 127, nx, ny, image_name);
    free(global_matrix);
}



// Non-root processes 
void Slave (int nx, int ny, double xL, double yL, double xR, double yR, int Imax, int my_rank, int num_procs) {

    mb_t* matrix = mandelbrot_matrix(nx, ny, xL, yL, xR, yR, Imax, my_rank, num_procs);
    
    int rows_per_process = ny / num_procs;
    int start_row = my_rank * rows_per_process;
    int end_row = (my_rank == num_procs - 1) ? ny : start_row + rows_per_process;

    int local_ny = end_row - start_row;
    int size = nx * local_ny * sizeof(mb_t);
    
    MPI_Send(matrix, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int num_procs, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (argc < 9) {
        if (my_rank == 0) {
            printf("Usage: %s nx ny xL yL xR yR Imax image_name\n", argv[0]);
        }
        MPI_Finalize();
        exit(1);
    }

    printf("number of processes: %d \n", num_procs);

    int nx, ny, Imax;
    double xL, yL, xR, yR;
    char* image_name;

    nx = atoi(argv[1]);
    ny = atoi(argv[2]);
    xL = atof(argv[3]);
    yL = atof(argv[4]);
    xR = atof(argv[5]);
    yR = atof(argv[6]);
    Imax = atoi(argv[7]);
    image_name = argv[8];
    
    if(my_rank == 0){
        Master(num_procs, nx, ny, xL, yL, xR, yR, Imax, image_name);
    }else{
        Slave(nx, ny, xL, yL, xR, yR, Imax, my_rank, num_procs);
    }

    MPI_Finalize();
    return 0;
}


