# Setup Instructions on the [Satori Cluster](https://mit-satori.github.io)

The Satori cluster uses ppc64 CPUs, which pose certain restrictions on the availability of pre-built python packages.

### 0. Preparation

#### 0.1 Log in

We will use RHEL8 nodes to setup SALIENT++ and run experiments. Log in `satori-login-001.mit.edu`.

#### 0.2 Log in an Interactive GPU Node

Request a GPU compute node (RHEL8) without exclusive access:

```bash
srun --gres=gpu:2 -N 1 --mem=1T --time 8:00:00 -p sched_system_all_8 --pty /bin/bash
```

#### 0.3 Clone this repository

```bash
git clone https://github.com/MITIBMxGraph/SALIENT_plusplus.git
cd SALIENT_plusplus
```

All subsequent steps assume that the working directory is the cloned directory (`SALIENT_plusplus`).

### 1. Install Conda

Follow instructions on the [Satori user documentation](https://mit-satori.github.io/satori-ai-frameworks.html#install-anaconda) (step 1 therein) to install Conda.

Then, create a Conda environment (for example, call it `salient_plusplus`):

```bash
conda create -n salient_plusplus python=3.9.5 -y
conda activate salient_plusplus
```

Check that Conda has the following channels:

```bash
$ conda config --show channels
channels:
  - https://opence.mit.edu
  - https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
  - defaults
  - conda-forge
```

If some channels are missing:

```bash
conda config --prepend channels [missing_channel]
```

### 2. Install PyTorch

```bash
conda install pytorch-base==1.12.1
```

PyTorch 1.12 is the highest version supported on Satori (at the time of this instruction document is written). We install `pytorch-base` rather than `pytorch` so that the PyTorch package comes from the channel https://opence.mit.edu with GPU support. Through the above install command, `cudatoolkit 11.4` is installed together. Watch out this information during conda install, and particularly check that the full package name for `pytorch-base` contains `cuda` rather than `cpu`. It is recommended that after the install, start python, import `torch`, type `torch.cuda.is_available()`, and check that indeed GPUs are supported.

### 3. Install PyTorch-Geometric (PyG)

The channel https://opence.mit.edu contains pre-built PyG and PyTorch-Sparse:

```bash
conda install pytorch_geometric=2.0.3
```

### 4. Reinstall PyTorch

Installing PyG above will install a dependency `pytorch`, which is separate from but coexists with `pytorch-base` (you may invoke `conda list` to verify so). The trouble of this is that the cpu-supported `pytorch` has some key files that overwrite those of the cuda-supported `pytorch-base` (for example, `version.py`). Such will cause a trouble that `torch.cuda.is_available()` returns False. The resolution is to overwrite back by reinstalling `pytorch-base`.

```bash
conda install pytorch-base=1.12.1 --force-reinstall
```

### 5. Install PyTorch-Sparse

Through installing PyG, `pytorch-sparse` (0.6.10) and `pytorch-scatter` (2.0.8) are installed as well. However, the installed `pytorch-sparse` is an old version that does not support the `trust_data` keyword that our team contributed to the package for speeding up sparse tensor computations. Therefore, we will need to upgrade `pytorch-sparse` and its dependency `pytorch-scatter`:

```bash
# Find $CUDA_HOME through `module load cuda` but not actually loading it. The cuda module is in conflict with pytorch-base that we just installed.
export CUDA_HOME=/software/cuda/11.4
export FORCE_CUDA=1
pip install git+https://github.com/rusty1s/pytorch_scatter.git --no-cache-dir
pip install git+https://github.com/rusty1s/pytorch_sparse.git --no-cache-dir
```

This will build `pytorch-scatter` and `pytorch-sparse` from source and it may take a while. After install, it is recommended to check that they are indeed installed with GPU supports. To do so, start python, import `torch_sparse`, type `torch_sparse.__file__`, obtain the location of the module, and check if there are `*_cuda.so` files inside. If yes, then GPU support is guaranteed.

### 6. Install OGB

```bash
pip install ogb
```

### 7. Install SALIENT++'s fast_sampler

Go to the folder `fast_sampler` and install:

```bash
cd fast_sampler
sed -i 's/-march/-mcpu/g' setup.py
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

### 8. Install Other Dependencies

```bash
conda install -c conda-forge nvtx
conda install -c conda-forge matplotlib
conda install -c conda-forge prettytable
conda install -c anaconda colorama
```

### 9. Prepare Datasets

SALIENT++ trains GNNs by using multiple GPUs, each handling a partition of the graph. To use SALIENT++, a graph partitioning must be conducted. There are two ways to get started playing with SALIENT++: (a) use a pre-partitioned dataset, or (b) install the partitioner and partition graphs on your own.

#### 9.1 Option (a): Use a Pre-Partitioned Dataset

Four pre-partitioned OGB datasets are available: `ogbn-arxiv`, `ogbn-products`, `ogbn-papers100M`, and `MAG240`, for several partitioning conditions (e.g., 2, 4, 8, and 16 partitions). Note that `MAG240` is the homogeneous papers-to-papers component of `MAG240M`.

To download these datasets, first go to the folder `utils` and invoke `configure_for_environment.py`, which determines which datasets are feasible for download, constrained on the available disk space. For example, if you have 50GB of disk space and desire to download the datasets up to 4 partitions, invoke the following commands:

```bash
cd utils
python configure_for_environment.py --max_num_parts 4 --force_diskspace_limit 50
```

A configuration file `feasible_datasets.cfg` will be generated under the folder `utils/configuration_files`.

Next, download the data by invoking `download_datasets_fast.py`. This script downloads rather large files when disk space is ample and may ask for confirmation before each download. Use `--skip_confirmation` to disable the tedious confirmation.

```bash
python download_datasets_fast.py --skip_confirmation
cd ..
```

The downloaded data are stored under the folder `dataset`.

#### 9.2 Option (b): Install a Partitioner and Partition Graphs On Your Own

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
>>> root = # type dataset root here, such as '/nobackup/users/username/dataset2'
>>> from ogb.nodeproppred import PygNodePropPredDataset
>>> dataset = PygNodePropPredDataset(name=name, root=root)
```

The specified `root` is the same as the `--dataset_dir` used in subsequent steps. To avoid conflicts with the folder `dataset` used in Option (a), we call the folder name `dataset2` here. Note that besides the different names, the folders are placed under different directories (the full paths are `/home/username/SALIENT_plusplus/dataset` and `/nobackup/users/username/dataset2`, respectively).

After downloading the dataset(s), perform partitioning. For example, to partition the `ogbn-products` graph in 4 parts and store the partition labels under `dataset2/partition-labels`:

```bash
python -m partitioners.run_4constraint_partition --dataset_name ogbn-products --dataset_dir /nobackup/users/username/dataset2 --output_directory /nobackup/users/username/dataset2/partition-labels --num_parts 4
```

The name of the partition result file is `ogbn-products-4.pt`.

> **NOTE:** Partitioning of large graphs may be very time-consuming and memory-hungry. For `ogbn-papers100M`, it can take 2-4 hours and several hundred gigabytes of memory (500GB as a safe estimate) to perform 8-way partitioning.

After partitioning, reorder the nodes and generate the final dataset. The reordering is used for caching (see paper for details). For example, for the `ogbn-products` dataset downloaded to the folder `dataset2` and partitioned with results stored in `dataset2/partition-labels/ogbn-products-4.pt`, if you reuse the output path `dataset2`, then running:

```bash
python -m partitioners.reorder_data --dataset_name ogbn-products --dataset_dir /nobackup/users/username/dataset2 --path_to_part /nobackup/users/username/dataset2/partition-labels/ogbn-products-4.pt --output_path /nobackup/users/username/dataset2
```

will produce the final, reordered dataset under {OUTPUT_PATH}/metis-reordered-k{NUM_PARTS}/{DATASET_NAME} (in this case, `dataset2/metis-reordered-k4/ogbn-products`).

Note that reordering (see the VIP analysis in the paper) is dependent on the training fanout. By default, the fanout is set to [15,10,5]. If using a different fanout, you need to supply the `--fanouts` argument (which contains a sequence of space separated numbers).

In the following, we assume the dataset root is `dataset`, rather than `dataset2`.

### 10. Try an Example

Congratulations! SALIENT++ has been installed and datasets are prepared. You may run the driver `exp_driver.py` under the folder `utils` to start an experiment.

> **NOTE:** The interactive mode on Satori is very slow for some reason. All the following examples assume running in the batch mode. That is, (automatically) submit the job through `sbatch` and do not invoke the `--run_local` option. The following examples are all run **on the login node**. Recall that we assume being at the working directory before `cd utils`.

Example: To run on a single node with a single GPU (in which case no graph partitioning is needed):

```bash
cd utils
# Note: Run on the login node (not a GPU compute node)!
python exp_driver.py --dataset_name ogbn-products --dataset_dir ../dataset --job_name test-job --num_machines 1 --num_gpus_per_machine 1 --gpu_percent 0.999 --replication_factor 15
```

Example: To run on a single node with two GPUs (in which case invoke the `--distribute_data` option):

```bash
cd utils
# Note: Run on the login node (not a GPU compute node)!
python exp_driver.py --dataset_name ogbn-products --dataset_dir ../dataset --job_name test-job --num_machines 1 --num_gpus_per_machine 2 --gpu_percent 0.999 --replication_factor 15 --distribute_data
```

Example: To run on two nodes with two GPUs per node:

```bash
cd utils
# Note: Run on the login node (not a GPU compute node)!
python exp_driver.py --dataset_name ogbn-products --dataset_dir ../dataset --job_name test-job --num_machines 2 --num_gpus_per_machine 2 --gpu_percent 0.999 --replication_factor 15 --distribute_data
```

In any of these examples, running the exp_driver will create a folder {JOB_ROOT}/{JOB_NAME}_{TIMESTAMP}, where JOB_ROOT is specified by the `--job_root` argument (by default `experiments`), JOB_NAME is specified by the `--job_name` argument (in this example `test-job`), and TIMESTAMP is the time when the folder is created. A job script `run.sh` is generated under this folder and it is automatically submitted to the cluster through `sbatch`. The screen prints the SBATCH COMMAND and a TAIL COMMAND. From the SBATCH COMMAND one can see the path of `run.sh`. The TAIL COMMAND can be used to view the output results. The screen will also print RUNNING/COMPLETED indications when the job is being executed. One does not need to wait and may kill the interactive session safely by using Ctrl-C.

#### 10.1 Tips on Command Line Arguments

For a complete list of arguments, see `utils/exp_driver.py`. This experiment driver will call the main driver `driver/main.py`. The file `driver/parser.py` specifies all arguments (even more than those used by exp_driver.py), together with key explanations, used by the main driver.
