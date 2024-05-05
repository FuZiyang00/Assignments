#!/bin/bash
mpicc -o mandel src/pgm.c src/functions.c main.c -lm -fopenmp
export OMP_NUM_THREADS=1
mkdir -p local_output
output_file = "local_output/mpi_output.txt"

# Loop through different inputs
for input in {2..4}
do
    echo "MPI processes: $input" >> $output_file
    echo "=============================" >> $output_file
    # Redirect the stdout of time command to the file
    { time mpirun --mca mca_base_component_show_load_errors 0 --map-by core -n $input ./mandel 8096 8096 -2.5 -1.5 1.5 1.5 8000 image.pgm; } 2>> $output_file
    echo "=============================" >> $output_file
done
