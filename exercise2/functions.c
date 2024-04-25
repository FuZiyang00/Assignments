#include "functions.h"

// Recursive Mandelbrot function
mb_t mandelbrot_func(double complex z, double complex c, int n, int Imax) {
    if (cabs(z) >= 2.0) { return (mb_t)n; }
    
    if (n >= Imax) { return (mb_t)0; }

    return mandelbrot_func(z * z + c, c, n + 1, Imax);
}

// Functions for the FCFS (first come first serve) scheduling

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

// Compute Mandelbrot set with a single process
mb_t *serial_mandelbrot(int nx, int ny, double xL, double yL, double xR, double yR, int Imax) {
    
    mb_t *Mandelbrot = (mb_t *)malloc(sizeof(mb_t) * nx * ny);
    if (Mandelbrot == NULL) {
        // Handle memory allocation failure
        printf("Error: Memory allocation failed\n");
        exit(1);
    }

    double dx = (double)(xR - xL) / (double)(nx - 1);
    double dy = (double)(yR - yL) / (double)(ny - 1);
    double x, y;
    double complex c;

    #pragma omp parallel for shared(Mandelbrot, nx, ny, xL, dx, yL, dy, Imax) private(x, y, c)
    for (int i = 0; i < ny; i++) {
        y = yL + i * dy;
        for (int j = 0; j < nx; j++) {
            x = xL + j * dx;
            c = x + I * y;
            Mandelbrot[i * nx + j] = mandelbrot_func(0 * I, c, 0, Imax);
        }
    }

    return Mandelbrot;
}

// Compute Mandelbrot set with multiple processes
mb_t *parallel_mandelbrot(int nx, int ny, double xL, double yL, double xR, double yR, int Imax) {
    
    mb_t *Mandelbrot = (mb_t *)malloc(sizeof(mb_t) * nx * ny);
    if (Mandelbrot == NULL) {
        // Handle memory allocation failure
        printf("Error: Memory allocation failed\n");
        exit(1);
    }

    double dx = (double)(xR - xL) / (double)(nx - 1);
    double dy = (double)(yR - yL) / (double)(ny - 1);
    int next_row = 0;
    double x, y;
    double complex c;
    int i;

    #pragma omp parallel private (x, y, c, i)
    {   
        int my_thread_id;
        while (1) {
            // Updates the value of a variable while capturing the original or final value of the variable atomically.
            #pragma omp atomic capture 
            i = next_row++;
            my_thread_id = omp_get_thread_num();

            if (i >= ny) {
                break;
            }
            printf("Threads %d on row %d\n", my_thread_id, i);
            y = yL + i * dy;
            for (int j = 0; j < nx; j++) {
                x = xL + j * dx;
                c = x + I * y;
                Mandelbrot[i * nx + j] = mandelbrot_func(0 * I, c, 0, Imax);
            }
        }
    }

    return Mandelbrot;
}