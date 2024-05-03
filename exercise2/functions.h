#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <string.h> 
#include <mpi.h>
#include <omp.h>
#include <pthread.h>

#define mb_t unsigned short
#define TAG_TASK_REQUEST 1
#define TAG_TASK_DATA 2
#define TAG_TASK_ROW 3
#define TAG_MATRIX_ROW 4

// recursive mandelbrot function
mb_t mandelbrot_func(double complex z, double complex c, int n, int Imax);

// for MPI scaling
void compute_row(mb_t *row, int nx, double xL, double xR, double y, int Imax);
void slave_process(int nx, int ny, double xL, double yL, double xR, double yR, int Imax, int rank);
mb_t *mandelbrot_matrix_rr(int nx, int ny, int size);

// for OMP scaling
mb_t *parallel_mandelbrot(int nx, int ny, double xL, double yL, double xR, double yR, int Imax);

#endif /* FUNCTIONS_H */