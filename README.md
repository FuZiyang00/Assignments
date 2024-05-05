# Exercise1 
The .job are bash files used in ORFEO to retrieve the needed data, that were than examined and used for modelling and plotting in the jupyternotebooks, each located in the dedicated folder.

### Folder structure
```
exercise1/
│
├── Barrier/
│ ├── barrier.ipynb
│ └── barrier.txt
│
├── Broadcast/
│ ├── binary.ipynb
│ └── btree_bcast.txt
| ├── linear.ipynb
│ └── linear_bcast.txt
|
│── barrier.job
├── binary_tree_bcast.job
├── cpus_latency.job
├── linear_bcast.job
└── requirements.txt
```
The results can be seen inside the single notebooks.

# Exercise2
The .job are bash files used in ORFEO to perform MPI and OPM scalings, whose data were than examined  and plotted in the plot jupyternotebook. 

### Folder structure
```
exercise1/
│
├── images/
|
├── results/
│ ├── mpi_scaling.txt
│ └── omp_scaling.txt
|
├── src/
│ ├── functions.c (functions implementing FCFS)
│ └── functions.h
| ├── pgm.c (function for writing the image)
│ └── pgm.h
|
│── Fu_Report.md (report in markdown)
├── Fu_Report.pdf (report in PDF)
├── main.c
├── plot.ipynb
└── scaling_mpi.job
└── scaling_omp.job
```

### Python depencies
The creation of a Python environment is recommended for running the jupyter notebooks. <br>
All dependencies can be installed with 
```
(venv) pip install -r requirements.txt
```