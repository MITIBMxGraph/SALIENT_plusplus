# Setup Instructions

> **NOTE:** All steps should be performed on machines with GPUs.

### 0. Clone this repository

```bash
git clone https://github.com/MITIBMxGraph/SALIENT_plusplus.git
cd SALIENT_plusplus
```

All subsequent steps assume that the working directory is the cloned directory (`SALIENT_plusplus`).

### 1. Install Conda

Follow instructions on the [Conda user guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). For example, to install Miniconda on an x86 Linux machine:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh
bash Miniconda3-py39_23.3.1-0-Linux-x86_64.sh
```

SALIENT++ has been tested on Python 3.9.5.

It is highly recommended to create a new environment and do the subsequent
steps therein. Example with a new environment called `salient_plusplus`:

```bash
conda create -n salient_plusplus python=3.9.5 -y
conda activate salient_plusplus
```

### 2. Install PyTorch

Follow instructions on the [PyTorch homepage](https://pytorch.org). For example, to install on a linux machine with CUDA 11.7:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

SALIENT++ has been tested on PyTorch 2.0.0.

### 3. Install PyTorch-Geometric (PyG) and PyTorch-Sparse

Follow the instructions on the [PyG Github page](https://github.com/pyg-team/pytorch_geometric). For example, with PyTorch 2.0.0 and CUDA 11.7, it suffices to do the following:

```bash
pip install torch_geometric
pip install torch_sparse torch_scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

SALIENT++ has been tested on PyG 2.3.0 and PyTorch-Sparse 0.6.17.

> **NOTE:** For a sanity check if PyTorch-Sparse has been installed with GPU support, start python, import `torch_sparse`, type `torch_sparse.__file__`, obtain the location of the module, and check if there are `*_cuda.so` files inside. If yes, then GPU support is guaranteed.

### 4. Install OGB

```bash
pip install ogb
```

SALIENT++ has been tested on OGB 1.3.6.

### 5. Install SALIENT++'s fast_sampler

Go to the folder `fast_sampler` and install:

```bash
cd fast_sampler
python setup.py install
cd ..
```

To check that `fast_sampler` is properly installed, start python and run:

```python
>>> import torch
>>> import fast_sampler
>>> help(fast_sampler)
```

You should see information of the package.

> **NOTE:** Compilation requires a C++ compiler that supports C++17 (e.g., gcc >= 7).

### 6. Install Other Dependencies

```bash
conda install -c conda-forge nvtx
conda install -c conda-forge matplotlib
conda install -c conda-forge prettytable
conda install -c anaconda colorama
```

### 7. Prepare Datasets

SALIENT++ trains GNNs by using multiple GPUs, each handling a partition of the graph. To use SALIENT++, a graph partitioning must be conducted. There are two ways to get started playing with SALIENT++: (a) use a pre-partitioned dataset, or (b) install the partitioner and partition graphs on your own.

#### 7.1 Option (a): Use a Pre-Partitioned Dataset

Four pre-partitioned OGB datasets are available: `ogbn-arxiv`, `ogbn-products`, `ogbn-papers100M`, and `MAG240`, for several partitioning conditions (e.g., 2, 4, 8, and 16 partitions). Note that `MAG240` is the homogeneous papers-to-papers component of `MAG240M`.

To download these datasets, first go to the folder `utils` and invoke `configure_for_environment.py`, which determines which datasets are feasible for download, constrained on the available disk space. For example, if you have 50GB of disk space and desire to download the datasets up to 2 partitions, invoke the following commands:

```bash
cd utils
python configure_for_environment.py --max_num_parts 2 --force_diskspace_limit 50
```

A configuration file `feasible_datasets.cfg` will be generated under the folder `utils/configuration_files`.

Next, download the data by invoking `download_datasets_fast.py`. This script downloads rather large files when disk space is ample and may ask for confirmation before each download. Use `--skip_confirmation` to disable the tedious confirmation.

```bash
python download_datasets_fast.py --skip_confirmation
cd ..
```

The downloaded data are stored under the folder `dataset`.

#### 7.2 Option (b): Install a Partitioner and Partition Graphs On Your Own

Install METIS from source using our provided repository. METIS needs to be built with 64-bit types, which precludes the use of many METIS libraries currently distributed.

```bash
git clone https://github.com/MITIBMxGraph/METIS-GKlib.git
cd METIS-GKlib
make config shared=1 cc=gcc prefix=$(realpath ../pkgs) i64=1 r64=1 gklib_path=GKlib/
make install
cd ..
```

Then, install `torch-metis`, which provides a Python module named `torch_metis` with
METIS bindings that accept PyTorch tensors.

```bash
git clone https://github.com/MITIBMxGraph/torch-metis.git
cd torch-metis
python setup.py install
cd ..
```

> **NOTE:** The `torch_metis` module normally requires some configuration of environment variables in order to function properly. In the use of SALIENT++, we ensure that the relevant scripts that use METIS set these variables internally. Outside SALIENT++, `import torch_metis` would _not_ work without setting the necessary environment variables.

After installing METIS and `torch-metis`, download dataset(s) from OGB. This can be done inside python by importing ogb and loading the dataset for the first time:

```python
>>> name = # type dataset name here, such as 'ogbn-products'
>>> root = # type dataset root here, such as '/home/username/SALIENT_plusplus/dataset2'
>>> from ogb.nodeproppred import PygNodePropPredDataset
>>> dataset = PygNodePropPredDataset(name=name, root=root)
```

The specified `root` is the same as the `--dataset_dir` used in subsequent steps. To avoid conflicts with the folder `dataset` used in Option (a), we call the folder name `dataset2` here.

After downloading the dataset(s), perform partitioning. For example, to partition the `ogbn-products` graph in 4 parts and store the partition labels under `dataset2/partition-labels`:

```bash
python -m partitioners.run_4constraint_partition --dataset_name ogbn-products --dataset_dir dataset2 --output_directory dataset2/partition-labels --num_parts 4
```

The name of the partition result file is `ogbn-products-4.pt`.

> **NOTE:** Partitioning of large graphs may be very time-consuming and memory-hungry. For `ogbn-papers100M`, it can take 2-4 hours and several hundred gigabytes of memory (500GB as a safe estimate) to perform 8-way partitioning.

After partitioning, reorder the nodes and generate the final dataset. The reordering is used for caching (see paper for details). For example, for the `ogbn-products` dataset downloaded to the folder `dataset2` and partitioned with results stored in `dataset2/partition-labels/ogbn-products-4.pt`, if you reuse the output path `dataset2`, then running:

```bash
python -m partitioners.reorder_data --dataset_name ogbn-products --dataset_dir dataset2 --path_to_part dataset2/partition-labels/ogbn-products-4.pt --output_path dataset2
```

will produce the final, reordered dataset under {OUTPUT_PATH}/metis-reordered-k{NUM_PARTS}/{DATASET_NAME} (in this case, `dataset2/metis-reordered-k4/ogbn-products`).

Note that reordering (see the VIP analysis in the paper) is dependent on the training fanout. By default, the fanout is set to [15,10,5]. If using a different fanout, you need to supply the `--fanouts` argument (which contains a sequence of space separated numbers).

In the following, we assume the dataset root is `dataset`, rather than `dataset2`.

### 8. Try an Example

Congratulations! SALIENT++ has been installed and datasets are prepared. You may run the driver `exp_driver.py` under the folder `utils` to start an experiment.

To run on a local (single) machine, which may contain one or multiple GPUs, invoke the `--run_local` option. For example, with one GPU (in which case no graph partitioning is needed):

```bash
cd utils
python exp_driver.py --dataset_name ogbn-products --dataset_dir ../dataset --job_name test-job --num_machines 1 --num_gpus_per_machine 1 --gpu_percent 0.999 --replication_factor 15 --run_local
```

Or, with two GPUs, invoke additionally the `--distribute_data` option:

```bash
cd utils
python exp_driver.py --dataset_name ogbn-products --dataset_dir ../dataset --job_name test-job --num_machines 1 --num_gpus_per_machine 2 --gpu_percent 0.999 --replication_factor 15 --distribute_data --run_local
```

Running the driver will create a folder {JOB_ROOT}/{JOB_NAME}_{TIMESTAMP}, where JOB_ROOT is specified by the `--job_root` argument (by default `experiments`), JOB_NAME is specified by the `--job_name` argument (in this example `test-job`), and TIMESTAMP is the time when the folder is created. An actual job script `run.sh` is generated under this folder and it is run locally. Experiment results, besides screen outputs, are also stored in this folder. In particular, under the subfolder `outputs` of this folder, the best trained model is stored.

On the other hand, on a SLURM cluster, one should not invoke the `--run_local` option and **may need to edit the SLURM_CONFIG variable at the top of the driver file for the particular cluster**. For example, with two compute nodes and two GPUs per node, run the following **on the login node**:

```bash
cd utils
# Note: edit SLURM_CONFIG at the top of exp_driver.py before running it!
python exp_driver.py --dataset_name ogbn-products --dataset_dir ../dataset --job_name test-job --num_machines 2 --num_gpus_per_machine 2 --gpu_percent 0.999 --replication_factor 15 --distribute_data
```

Running the driver will create a folder {JOB_ROOT}/{JOB_NAME}_{TIMESTAMP} (explained above) and generate a job script `run.sh` under this folder. This job is automatically submitted to the cluster. The screen prints a SBATCH COMMAND and a TAIL COMMAND. From the SBATCH COMMAND one can see the path of `run.sh`. The TAIL COMMAND can be used to view the output results. The screen will also print RUNNING/COMPLETED indications when the job is being executed. One does not need to wait and may kill the interactive session safely by using Ctrl-C.

#### 8.1 Tips on Command Line Arguments

For a complete list of arguments, see `utils/exp_driver.py`. This experiment driver will call the main driver `driver/main.py`. The file `driver/parser.py` specifies all arguments (even more than those used by exp_driver.py), together with key explanations, used by the main driver.
