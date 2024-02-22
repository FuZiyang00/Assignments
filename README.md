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

### Python depencies
The creation of a Python environment is recommended and all dependencies can be installed with 
```
(venv) pip install -r requirements.txt
```
