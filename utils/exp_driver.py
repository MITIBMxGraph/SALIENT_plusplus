import os
import time
import datetime
import argparse
from pathlib import Path
import subprocess


SLURM_CONFIG="""#SBATCH -J sb_{JOB_NAME}
#SBATCH -o {OUTPUT_FILE_DIRECTORY}/%x-%A.%a.out
#SBATCH -e {OUTPUT_FILE_DIRECTORY}/%x-%A.%a.err
#SBATCH --nodes=1
#SBATCH --array=1-{NUM_MACHINES}
#SBATCH --gres=gpu:{NUM_GPUS_PER_MACHINE}
#SBATCH --time 00:05:00
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH -p sched_system_all_8

HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=salient_plusplus
source $HOME2/anaconda3/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

cd {SCRIPT_DIR}
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run a SALIENT++ experiment")
    parser.add_argument("--dataset_name", help="name of dataset", type=str, required=True)
    parser.add_argument("--dataset_dir", help="directory to (partitioned) dataset", type=str, required=True)
    parser.add_argument("--job_name", help="short name of job (no spaces)", type=str, required=True)
    parser.add_argument("--num_machines", help="number of machines", type=int, required=True)
    parser.add_argument("--num_gpus_per_machine", help="number of GPUs per machine", type=int, required=True)
    parser.add_argument("--gpu_percent", help="percent of data to put on GPU", type=float, required=True)
    parser.add_argument("--replication_factor", help="percentage of data to replicate. Integer between 0 and 100", type=int, required=True)
    parser.add_argument("--train_batch_size", help="size of training batch", type=int, default=1024)
    parser.add_argument("--test_batch_size", help="size of validation/test batch", type=int, default=1024)
    parser.add_argument("--train_fanouts", help="training fanouts", type=int, default=[15, 10, 5], nargs="*")
    parser.add_argument("--test_fanouts", help="validation/test fanouts", type=int, default=[20, 20, 20], nargs="*")
    parser.add_argument("--test_epoch_frequency", help="frequency to perform testing/validation", type=int, default=1)
    parser.add_argument("--num_epochs", help="number of epochs", type=int, default=25)
    parser.add_argument("--num_hidden", help="hidden dimension", type=int, default=256)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=0.001)
    parser.add_argument("--model_name", help="name of model", type=str, default="SAGE")
    parser.add_argument("--num_trials", help="numbe of random repetitions", type=int, default=1)
    parser.add_argument("--num_samplers", help="num sampling workers", type=int, default=15)
    parser.add_argument("--train_max_num_batches", help="maximum number of concurrent sampling batches", type=int, default=48)
    parser.add_argument("--job_root", help="root dir for job outputs", type=str, default="../experiments")
    parser.add_argument("--pipeline_disabled", help="disable the pipeline", action="store_true")
    parser.add_argument("--distribute_data", help="whether use partitioned data", action="store_true")
    parser.add_argument("--run_local", help="run on a single machine", action="store_true")
    parser.add_argument("--make_deterministic", help="make training/inference deterministic", action="store_true")
    parser.add_argument("--do_test_run", help="only run inference on the test set", action="store_true")
    parser.add_argument("--do_test_run_filename", help="model filename with path", type=str, default=f"NONE", nargs='*')
    
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    partitioned_dataset = os.path.realpath(args.dataset_dir)
    job_name = args.job_name
    job_root_dir = Path(args.job_root)
    
    def get_job_directory(job_name):
        time_string = str(datetime.datetime.now()).replace(" ", "_")
        return job_name + "_" + time_string
    
    script_dir = Path(script_dir)
    job_dir = job_root_dir / Path(get_job_directory(job_name))

    log_dir_name = "logs"
    output_root_dir_name = "outputs"
    nodelist_dir_name = "nodelist"
    sbatch_file_name = "run.sh"
    
    log_dir = job_dir / Path(log_dir_name)
    output_root_dir = job_dir / Path(output_root_dir_name)
    nodelist_dir = job_dir / Path(nodelist_dir_name)
    sbatch_file = job_dir / Path(sbatch_file_name)

    job_root_dir.mkdir(exist_ok=True) 
    job_dir.mkdir()
    log_dir.mkdir()
    nodelist_dir.mkdir()
    output_root_dir.mkdir(exist_ok=True)
    
    VARS = dict()
    VARS["DATASET_NAME"] = args.dataset_name
    VARS["JOB_NAME"] = args.job_name
    VARS["NUM_MACHINES"] = args.num_machines
    VARS["NUM_GPUS_PER_MACHINE"] = args.num_gpus_per_machine
    VARS["GPU_PERCENT"] = args.gpu_percent
    VARS["REPLICATION_FACTOR"] = args.replication_factor
    VARS["TRAIN_BATCH_SIZE"] = args.train_batch_size
    VARS["TEST_BATCH_SIZE"] = args.test_batch_size
    VARS["TRAIN_FANOUTS"] = " ".join([str(x) for x in args.train_fanouts])
    VARS["TEST_FANOUTS"] =  " ".join([str(x) for x in args.test_fanouts])
    VARS["TEST_EPOCH_FREQUENCY"] = args.test_epoch_frequency
    VARS["EPOCHS"] = args.num_epochs
    VARS["HIDDEN_SIZE"] = args.num_hidden
    VARS["LEARNING_RATE"] = args.learning_rate
    VARS["MODEL_NAME"] = args.model_name
    VARS["NUM_TRIALS"] = args.num_trials
    VARS["NUM_SAMPLING_WORKERS"] = args.num_samplers
    VARS["TRAIN_MAX_NUM_BATCHES"] = args.train_max_num_batches
    
    VARS["NODELIST_DIR_NAME"] = nodelist_dir_name
    VARS["OUTPUT_ROOT_DIR_NAME"] = output_root_dir_name
    VARS["NUM_LAYERS"] = str(len(args.train_fanouts))
    VARS["CACHE_CREATION_EPOCHS"] = 2
    VARS["EXECUTION_MODE"] = "computation"
    VARS["COMPUTATION_MODE"] = "frequency_cache"
    VARS["LOAD_BALANCE_SCHEME"] = "federated"
    
    VARS["PIPELINE_DISABLED"] = "--pipeline_disabled" if args.pipeline_disabled else ""
    VARS["DISTRIBUTE_DATA"] = "--distribute_data" if args.distribute_data else ""
    VARS["MAKE_DETERMINISTIC"] = "--make_deterministic" if args.make_deterministic else ""
    VARS["DO_TEST_RUN"] = "--do_test_run" if args.do_test_run else ""
    VARS["DO_TEST_RUN_FILENAME"] = args.do_test_run_filename

    num_gpus = args.num_machines * args.num_gpus_per_machine
    if args.distribute_data:
        VARS["PARTITIONED_FEATURE_DATASET_ROOT"] = args.dataset_dir + "/metis-reordered-k" + str(num_gpus)
    else:
        VARS["PARTITIONED_FEATURE_DATASET_ROOT"] = args.dataset_dir
    VARS["OUTPUT_FILE_DIRECTORY"] = str(log_dir)
    VARS["SCRIPT_DIR"] = str(script_dir) + "/"
    VARS["PYTHONPATH"] = str(script_dir) + "/../"
    VARS["OUTPUT_ROOT"] = str(job_dir)
    
    VARS["ONE_NODE_DDP"] = ""
    preVARS = dict()
    preVARS["SLURM_CONFIG"] = SLURM_CONFIG
    if args.run_local:
        VARS["SLURM_CONFIG"] = ""
        VARS["ONE_NODE_DDP"] = "--one_node_ddp"
        VARS["NUM_MACHINES"] = 1
    else:
        if args.num_machines == 1 and num_gpus > 1:
            VARS["ONE_NODE_DDP"] = "--one_node_ddp"
        for x in VARS.keys():
            VARS[x] = str(VARS[x])
        preVARS["SLURM_CONFIG"] = preVARS["SLURM_CONFIG"].format(**VARS)
        VARS["SLURM_CONFIG"] = preVARS["SLURM_CONFIG"]
    for x in VARS.keys():
        VARS[x] = str(VARS[x])
        
    text = """#!/bin/bash
{SLURM_CONFIG}
export SLURMD_NODENAME=`hostname`
export PYTHONPATH={PYTHONPATH}
touch {OUTPUT_ROOT}/{NODELIST_DIR_NAME}/$SLURMD_NODENAME

NCCL_NSOCKS_PERTHREAD=1 NCCL_SOCKET_NTHREADS=1 PYTHONOPTIMIZE=1 OMP_NUM_THREADS=1 python -m driver.main \\
    {DATASET_NAME} \\
    {OUTPUT_ROOT_DIR_NAME} \\
    --dataset_root {PARTITIONED_FEATURE_DATASET_ROOT} \\
    --total_num_nodes {NUM_MACHINES} \\
    --max_num_devices_per_node {NUM_GPUS_PER_MACHINE} \\
    --gpu_percent {GPU_PERCENT} \\
    --cache_creation_epochs {CACHE_CREATION_EPOCHS} \\
    --cache_size {REPLICATION_FACTOR} \\
    --train_batch_size {TRAIN_BATCH_SIZE} \\
    --test_batch_size {TEST_BATCH_SIZE} \\
    --final_test_batchsize {TEST_BATCH_SIZE} \\
    --train_fanouts {TRAIN_FANOUTS} \\
    --batchwise_test_fanouts {TEST_FANOUTS} \\
    --final_test_fanouts {TEST_FANOUTS} \\
    --test_epoch_frequency {TEST_EPOCH_FREQUENCY} \\
    --epochs {EPOCHS} \\
    --hidden_features {HIDDEN_SIZE} \\
    --lr {LEARNING_RATE} \\
    --model_name {MODEL_NAME} \\
    --trials {NUM_TRIALS} \\
    --num_workers {NUM_SAMPLING_WORKERS} \\
    --train_max_num_batches {TRAIN_MAX_NUM_BATCHES} \\
    --output_root {OUTPUT_ROOT} \\
    --ddp_dir {OUTPUT_ROOT}/{NODELIST_DIR_NAME}/ \\
    {PIPELINE_DISABLED} \\
    {DISTRIBUTE_DATA} \\
    {MAKE_DETERMINISTIC} \\
    {ONE_NODE_DDP} \\
    {DO_TEST_RUN} \\
    --do_test_run_filename {DO_TEST_RUN_FILENAME} \\
    --num_layers {NUM_LAYERS} \\
    --overwrite_job_dir \\
    --performance_stats \\
    --train_type serial \\
    --execution_mode {EXECUTION_MODE} \\
    --computation_mode {COMPUTATION_MODE} \\
    --load_balance_scheme {LOAD_BALANCE_SCHEME}

"""
    text = text.format(**VARS)
    
    open(str(sbatch_file), 'w+').write(text)
    if args.run_local:
        print("LOCAL JOB COMMAND: bash " + str(sbatch_file))
        output = subprocess.run(["bash", str(sbatch_file)], check=True, capture_output=False)
        time_string = str(datetime.datetime.now()).replace(" ", "_")
        job_id = time_string  
    else:
        print("SBATCH COMMAND: sbatch " + str(sbatch_file))
        print("TAIL COMMAND: tail -f " + str(log_dir) + "/*.1.*")
        output = subprocess.run(["sbatch", str(sbatch_file)], check=True, capture_output=True)
        print(output)
        job_id = int(str(output.stdout).strip().split(" ")[-1].replace("'", "").replace("\\n", "").strip())

    while True and not args.run_local:
        time.sleep(5)
        output = subprocess.run(["squeue", "-j", str(job_id), "--format=\"%T\"", "--states=all"], check=True, capture_output=True)
        lines = str(output.stdout).replace("\\n", "\n").split("\n")
        lines = lines[1:-1]
        is_done = len(lines) > 0
        for x in lines:
            if x.find("COMPLETED") == -1:
                is_done = False
        print(lines)
        if is_done:
            print("Done with job")
            break
    # Wait for it to get into waiting state
