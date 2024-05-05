#!/bin/bash
mpicc -o mandel src/pgm.c src/functions.c main.c -lm -fopenmp
mkdir -p local_output
output_file="local_output/scaling_omp.txt"

# Fixed number of MPI processes
mpi_processes=1

# Loop through different numbers of OpenMP threads
for threads in {2..8}
do
    echo "OpenMP threads: $threads" >> $output_file
    echo "=========================================================" >> $output_file
    # Set the number of OpenMP threads
    export OMP_NUM_THREADS=$threads
    export OMP_PLACES=threads
    # Redirect the stdout of time command to the file
    { time mpirun --mca mca_base_component_show_load_errors 0 --map-by socket -n $mpi_processes ./mandel 4096 4096 -2.5 -1.5 1.5 1.5 6000 image.pgm; } 2>> $output_file
    echo "=========================================================" >> $output_file

done