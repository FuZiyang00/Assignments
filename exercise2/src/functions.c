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
    int row_index;
    // initialize the big array
    int size = 0 ;
    mb_t **big_array = NULL;
    int *big_index_arrays = (int *)malloc(sizeof(int));
    int received_rows = 0;
    
    while (1) {
        // signal availability
        MPI_Send(&rank, 1, MPI_INT, 0, TAG_TASK_REQUEST, MPI_COMM_WORLD);
        // Receive row index
        MPI_Recv(&row_index, 1, MPI_INT, 0, TAG_TASK_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // terminate the reception is the termination signal is received
        if (row_index == -1) { break;}

        mb_t *row = (mb_t *)malloc(nx * sizeof(mb_t));
        double dy = (yR - yL) / (double)(ny - 1);
        double y = yL + row_index * dy;
        compute_row(row, nx, xL, xR, y, Imax);

        // Increase the size by 1
        size++;
        big_array = (mb_t **)realloc(big_array, size * sizeof(mb_t *));
        if (big_array == NULL) {
            // Handle memory allocation failure
            printf("Error: Memory allocation failed\n");
            exit(1);}
        
        // Allocate memory for the new row
        big_index_arrays = (int *)realloc(big_index_arrays, size * sizeof(int)); // Reallocate memory for big_index_arrays
        big_index_arrays[size -1] = row_index;
        big_array[size - 1] = row;
        printf("Computed row %d\n by process %d", row_index, rank);
        received_rows++;
        }
    
    // Send the rows to the master process
    for (int i = 0; i < received_rows; i++) {
        MPI_Send(&big_index_arrays[i], 1, MPI_INT, 0, TAG_TASK_ROW, MPI_COMM_WORLD);
        MPI_Send(big_array[i], nx, MPI_UNSIGNED_SHORT, 0, TAG_MATRIX_ROW, MPI_COMM_WORLD);
        printf("Sent row %d to master process\n", big_index_arrays[i]);
    }
    
    free(big_index_arrays);
    for (int i = 0; i < received_rows; i++) {
        free(big_array[i]);
    }
    free(big_array);
    return;
}


mb_t *mandelbrot_matrix_rr(int nx, int ny, int size) {
    int next_row = 0; 
    while (next_row < ny) {
        int available_p;
        MPI_Recv(&available_p, 1, MPI_INT, MPI_ANY_SOURCE, TAG_TASK_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d is available \n", available_p);
        // Send data to the available slave
        MPI_Send(&next_row, 1, MPI_INT, available_p, TAG_TASK_DATA, MPI_COMM_WORLD);
        printf("Sent row %d to process %d\n", next_row, available_p);
        // Release the mutex
        next_row++;
    }

    // Send termination signal to all slaves
    for (int i = 1; i < size; i++) {
        int termination_signal = -1;
        MPI_Send(&termination_signal, 1, MPI_INT, i, TAG_TASK_DATA, MPI_COMM_WORLD);
        printf("Sent termination signal to slave %d\n", i);
    }

    MPI_Status status;
    // Allocate memory for the 1D array
    mb_t *Mandelbrot_1D = (mb_t *)malloc(nx * ny * sizeof(mb_t));
    if (Mandelbrot_1D == NULL) {
        // Handle memory allocation failure
        printf("Error: Memory allocation failed\n");
        exit(1);
    }
    
    for (int i = 0; i < ny; i++) {
        int row_index;
        MPI_Recv(&row_index, 1, MPI_INT, MPI_ANY_SOURCE, TAG_TASK_ROW, MPI_COMM_WORLD, &status);
        if (row_index < 0 || row_index >= ny) {
            printf("Error: Invalid row index received\n");
            exit(1);
        }
        mb_t *row = (mb_t *)malloc(nx * sizeof(mb_t));
        // Use status.MPI_SOURCE as the source for this MPI_Recv call
        MPI_Recv(row, nx, MPI_UNSIGNED_SHORT, status.MPI_SOURCE, TAG_MATRIX_ROW, MPI_COMM_WORLD, &status);
        // mapping the row to the global matrix
        memcpy(&Mandelbrot_1D[row_index * nx], row, nx * sizeof(mb_t));
        printf("Received and mapped row %d\n", row_index);
    }

    return Mandelbrot_1D;
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

    #pragma omp parallel private (x, y, c, i) shared (next_row, Mandelbrot, nx, ny, xL, dx, yL, dy, Imax)
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
            printf("Thread %d on row %d\n", my_thread_id, i);
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