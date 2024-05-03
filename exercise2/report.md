# Exercise 2C Hybrid MPI + OpenMP
# The Mandelbrot Set
Fu Ziyang SM3800011 UniTS <br>
MSc in Data Science & Artificial Intelligence <br>
Foundations of Artificial Intelligence and Machine Learning

## Introduction 
The purpose of this assignment is to compute and visualize the Mandelbrot set over a specified region of the complex plane. By discretizing the plane into a grid of pixels, it is possible to iterate the function for each pixel and to
determine whether it belongs to the Mandelbrot set based on a predefined condition. <br>
This process allows us to generate an image where each pixel represents a point in the complex plane, colored according to whether it belongs to the Mandelbrot set or not. 

This computation can be efficiently made in parallel, since each point can be computed independently of each other, and the exercise requires to employ a combination of **MPI (Message Passing Interface)** and **OpenMP (Open Multi-Processing)**. <br> 
This combination enables to ultimately determine:

- the **OMP scaling**: running the computation with a single MPI task and increasing the number of OMP threads;

- the **MPI scaling**: running the conmputation with a single OMP thread per MPI task and increasing the number of MPI tasks. 

## First Come First Serve 
In order to distribute the workload among the different computational units, I employed a **First come First Served - Row Based Partition Scheme**. <br>

Although it does not solve for the imbalance problem (inner points are computationally more demanding than the outer points), it mitigates it by assigning dynamically un-computed rows to the first available units, resulting in faster execution times compared with distributing chunks of rows in static and in a predetermined fashion.

The chosen partition scheme shares some similarity with schedulers in operating systems, with a master node sending work based on the availability. <br>
In the scope of the assignment, the designed partition scheme is characterized by a **single master MPI process** and **MPI worker processes**. 

The first is responsible for: 
- the dynamic assignment of the indexes of rows to computed to the available workers;
- receiving the indexes of the computed rows and map the latter to their position in the global Mandelbrot matrix;
- sending a termination signal to each worker process, once the global matrix has been computed in its entirety.

The workers processes receive indexes, computes the assigned rows and lastly send the processed rows. <br>
Each MPI worker process may spawn up to two hardware threads in order to parallelize at row level the computation of the columns.

The main challenge encountered with implementing the proposed solution was to ensure proper MPI communications between master and workers and a correct logical flow of them. 

Since different types of data are comunicated between workers and the master, it was necessary to define four different communication tags: 

``` {C}
# define TAG_TASK_REQUEST 1
```
used to signal the worker's availability and its rank;
```{C}
# define TAG_TASK_DATA 2 
```
used to communicate the indexes of the rows to be computed and the termination signal; 
```{C}
# define TAG_TASK_ROW 3
```
used to communicate from the worker to the master the index of the computed row; 
```{C}
# define TAG_MATRIX_ROW 4
```
used to identify the sending of the computed row. 


## Code implementation 
Designing the **master process function** was the most crucial step, since it is responsible for both satisfying the rules of the First Come First Serve partition scheme and mapping back the computed rows to the global Mandelbrot matrix.

Firstly it needs to allocate the memory for the global Mandelbrot matrix: 
``` {C}
mb_t *master_process(int nx, int ny, int size) {
    mb_t *Mandelbrot = (mb_t *)malloc(sizeof(mb_t) * nx * ny);
``` 

next it needs to enforce the first come first serve policy:  
``` {C}
    int next_row = 0;
    int completed_rows = 0;
    while (completed_rows < ny) {
        int available_p;
        MPI_Recv(&available_p, 1, MPI_INT, MPI_ANY_SOURCE, 
        TAG_TASK_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        if (next_row < ny) {
            // Send data to the available slave
            MPI_Send(&next_row, 1, MPI_INT, available_p, 
            TAG_TASK_DATA, MPI_COMM_WORLD);
            next_row++;
        }

        MPI_Status status;
        int row_index;
        mb_t *row = (mb_t *)malloc(nx * sizeof(mb_t));
        MPI_Recv(&row_index, 1, MPI_INT, MPI_ANY_SOURCE, 
        TAG_TASK_ROW, MPI_COMM_WORLD, &status);
    
        MPI_Recv(row, nx, MPI_UNSIGNED_SHORT, MPI_ANY_SOURCE, 
        TAG_MATRIX_ROW, MPI_COMM_WORLD, &status);

        memcpy(&Mandelbrot[row_index * nx], row, nx * sizeof
        (mb_t));
        completed_rows++;
        free(row);
    } 
```
The execution flows in the following way: 
1. the master process waits to receive any availability signal from the workers (**available_p represents the available worker's rank**); 
2. once received the signal, it performs a simple check and send to the available worker the index of the row to be processed; 
3. the master prepares to receive back the computed rows: stating **MPI_ANY_SOURCE** in the two blocking receive allows **to not wait for the row that has been just sent**, but to move on and map to global matrix the first row that has been processed; 


Once plugged every row to its position, the master process has one final duty: signaling to the workers the termination of the tasks: 
``` {C}
    for (int i = 1; i < size; i++) {
        int termination_signal = -1;
        MPI_Send(&termination_signal, 1, MPI_INT, i, 
        TAG_TASK_DATA, MPI_COMM_WORLD);
    }
    return Mandelbrot;
}
```

We can now turn our gaze toward the **workers function**, that, as described before, has to carry out three tasks: 

- signaling the availability of a worker:
```{C}
void slave_process(int nx, int ny, double xL, double yL,    double xR, double yR, int Imax, int rank) {

    while (1) {
        MPI_Send(&rank, 1, MPI_INT, 0, TAG_TASK_REQUEST,  
        MPI_COMM_WORLD);
```
- receiving the row index or the termination signal:
```{C}
        int row_index;
        MPI_Recv(&row_index, 1, MPI_INT, 0, TAG_TASK_DATA, 
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (row_index == -1) { break;} 
```
- applying the Mandelbrot function to the row and sending back the result and the received index to the master: 
```{C}
        mb_t *row = (mb_t *)malloc(nx * sizeof(mb_t));

        double dy = (yR - yL) / (double)(ny - 1);
        double y = yL + row_index * dy;
        compute_row(row, nx, xL, xR, y, Imax);

        MPI_Send(&row_index, 1, MPI_INT, 0, TAG_TASK_ROW, 
        MPI_COMM_WORLD);
        MPI_Send(row, nx, MPI_UNSIGNED_SHORT, 0, TAG_MATRIX_ROW, 
        MPI_COMM_WORLD);

        free(row);
    }
```
Each **MPI_Send** has its corresponding **MPI_Recv**, with the tasks tag helping to identify the type of data involved in the communication. 

## MPI Scaling


