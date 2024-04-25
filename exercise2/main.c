#include "functions.h"
#include "pgm.h"

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

    if (size < 2) { 
        if (size > 0) {
            printf("Computing Mandelbrot set with a single process\n");
            Mandelbrot = parallel_mandelbrot(nx, ny, xL, yL, xR, yR, Imax);
            printf("Writing image\n");
            write_pgm_image(Mandelbrot, Imax, nx, ny, image_name);
            free(Mandelbrot);
            printf("Execution completed \n");
            MPI_Finalize();
            return 0;

        } else {
            printf("Error: Number of processes must be greater than 0\n");
            exit(1);
        }
    }

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
