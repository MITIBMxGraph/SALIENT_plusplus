import logging
import subprocess
import sys
import os
import time
import csv
import shutil
import pathlib
from pathlib import Path
import argparse
import colorama
from colorama import Fore, Back, Style
import pprint
def run_command(cmd, asyn=False):
    proc = subprocess.Popen(
        [cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not asyn:
        out, err = proc.communicate()
        return out, err
    else:
        return ""


def get_n_cpus():
    return len(get_cpu_ordering())


def get_cpu_ordering():
    if sys.platform == "darwin":
        # TODO: Replace with something that analyzes CPU configuration on Darwin
        out, err = run_command("sysctl -n hw.physicalcpu_max")
        return [(0, p) for p in range(0, int(str(out, 'utf-8')))]
    else:
        out, err = run_command("lscpu --parse")

    out = str(out, 'utf-8')

    avail_cpus = []
    for l in out.splitlines():
        if l.startswith('#'):
            continue
        items = l.strip().split(',')
        cpu_id = int(items[0])
        core_id = int(items[1])
        socket_id = int(items[2])
        avail_cpus.append((socket_id, core_id, cpu_id))

    avail_cpus = sorted(avail_cpus)
    ret = []
    added_cores = dict()
    for x in avail_cpus:
        if x[1] not in added_cores:
            added_cores[x[1]] = True
        else:
            continue
        ret.append((x[2], x[0]))
    return ret


dir_path = os.path.dirname(os.path.realpath(__file__))


def adapt_config_files():

    message = """\
###############################################################################
######## Modifying experiment configuration files to adapt to hardware ########
###############################################################################
"""
    print(message)

    # determine number of physical CPUs
    try:
        cpu_ordering = get_cpu_ordering()
        num_physical_cpus = len(cpu_ordering)
        hyperthreads_per_core = len(cpu_ordering[0])

        desired_workers = 30

        print("Configuration of number of sampling worker threads")
        print("\t[INFO] Detected " + str(num_physical_cpus) +
              " physical CPUs each with " + str(hyperthreads_per_core) + " hardware threads.")
        if num_physical_cpus >= 30 and hyperthreads_per_core >= 2:
            desired_workers = 30
        elif num_physical_cpus >= 20 and hyperthreads_per_core >= 2:
            desired_workers = num_physical_cpus + num_physical_cpus/2
        elif num_physical_cpus >= 20 and hyperthreads_per_core == 1:
            desired_workers = num_physical_cpus-1
            #print("[INFO] ")
        elif hyperthreads_per_core >= 2:
            desired_workers = num_physical_cpus + num_physical_cpus/2
            print("\t[WARNING] Detected fewer than 20 physical CPUs.")
        elif hyperthreads_per_core == 1:
            desired_workers = num_physical_cpus
            print(
                "\t[WARNING] Detected fewer than 20 physical cpus and no hyperthreading. Performance likely to be suboptimal.")
        desired_workers = int(desired_workers)
        print("\t[INFO] Setting desired workers to " +
              str(desired_workers) + ".")
    except:
        print("Error when analyzing hardware system to adapt config files")
        exit(1)

    update_configurations = ['performance_breakdown_config.cfg']

    for cfg in update_configurations:
        lines = open(
            dir_path + '/performance_breakdown_config.cfg').readlines()
        new_lines = []
        for l in lines:
            l = l.strip()
            if l.startswith('--num_workers'):
                new_lines.append('--num_workers ' + str(desired_workers))
            else:
                new_lines.append(l)
        open(dir_path + '/performance_breakdown_config.cfg',
             'w').write("\n".join(new_lines))
        print("*Updated configuration for file " + cfg)

    open(dir_path + '/.ran_config', 'w+').write("1")


def dataset_feasibility(free_gb):
    if args.max_num_parts >= 16:
        multiplier = 5
    elif args.max_num_parts >= 8:
        multiplier = 4
    elif args.max_num_parts >= 4:
        multiplier = 3
    elif args.max_num_parts >= 2:
        multiplier = 2
    else:
        print("[Error] you specified --max_num_parts N where N < 2. These experiments are designed for testing a distributed training system among multiple nodes.")
        exit(1)

    dataset_list = []
    dataset = ('ogbn-arxiv', 0.5 * multiplier)
    dataset_list.append(dataset)
    dataset = ('ogbn-products', 2.0 * multiplier)
    dataset_list.append(dataset)
    dataset = ('ogbn-papers100M', 100.0 * multiplier)
    dataset_list.append(dataset)
    dataset = ('MAG240', 300.0 * multiplier)
    dataset_list.append(dataset)
    infeasible = []
    feasible = []
    for x in dataset_list:
        if x[1] <= free_gb:
            feasible.append(x)  # x[0] + " ("+str(x[1])+" GB needed)")
        else:
            infeasible.append(x)  # x[0] + " ("+str(x[1])+" GB needed)")
    return feasible, infeasible


def determine_viable_datasets(args):
    import shutil

    message = """\
###############################################################################
######## Checking disk space to decide datasets to use for experiments ########
###############################################################################
"""
    print()
    print(message)

    total, used, free = shutil.disk_usage(__file__)
    free_gb = int((1.0*free)/1e+9)
    if args.force_diskspace_limit > 0 and free_gb > args.force_diskspace_limit:
        print(Fore.YELLOW + f'\t[INFO] Specified value for --force_diskspace_limit of ({args.force_diskspace_limit} GB) is less than the detected available disk space ({free_gb} GB), This script will use the value provided by --force_diskspace_limit' + Style.RESET_ALL)
        free_gb = args.force_diskspace_limit
    elif args.force_diskspace_limit > 0:
        print(Fore.YELLOW + f'\t[INFO] User provided diskspace hint via --force_diskspace_limit suggesting that they had ({args.force_diskspace_limit} GB) available space. This script, however, detected only {free_gb} GB available space. This script will use the smaller of the two limits.' + Style.RESET_ALL)
    print(f'\t[INFO] Using a limit of: {free_gb} GB space')

    feasible, infeasible = dataset_feasibility(free_gb)
    print("\t[INFO] Checking disk space available in directory " + dir_path)
    print("\t[INFO] Available disk space detected: " + str(free_gb) + " GB")
    if len(feasible) == 0:
        print("\t[Warning] **Extremely** low disk space available. You may not be able to download any datasets. Please free space before continuing.")
    if len(feasible) == 1:
        print("\t[Warning] Very low disk space. It is strongly recommended to free additional space. Otherwise you may only run on the smallest datasets.")
    if len(feasible) == 2:
        print("\t[Warning] Somewhat low disk space. You can run on small/medium sized datasets. Free space to run on larger datasets.")
    print("\t[Info] Sufficient space for datasets: " +
          "; ".join([x[0] + " ("+str(x[1])+" GB needed)" for x in feasible]))
    print("\t[Info] Insufficient space for datasets: " +
          "; ".join([x[0] + " ("+str(x[1])+" GB needed)" for x in infeasible]))

    dataset_dir = dir_path+"/dataset"
    for x in infeasible:
        if os.path.exists(dataset_dir + "/" + x[0]):
            print("\t[Info] Dataset " + x[0] +
                  " exists at " + dataset_dir + "/" + x[0])
            print(
                "\t\tWe thought this dataset might be too big, but it seems you've already downloaded it.")
            feasible.append(x)
    for x in feasible:
        if x in infeasible:
            infeasible.remove(x)
        print("\t[Info] Dataset " + x[0] + " will be recorded as feasible")


    config_dir = Path('./configuration_files/')
    feasible_datasets_file = config_dir / 'feasible_datasets' 
    infeasible_datasets_file = config_dir / 'infeasible_datasets' 

    #open(feasible_datasets_file,
    #     "w").write("\n".join([str(x[0]) for x in feasible]))
    #open(infeasible_datasets_file,
    #     "w").write("\n".join([str(x[0]) for x in infeasible]))



    config = dict()
    config['feasible_datasets'] = [x[0] for x in feasible]
    config['infeasible_datasets'] = [x[0] for x in infeasible]
    config['max_num_parts'] = args.max_num_parts


    config_file = config_dir / 'feasible_datasets.cfg'
    open(config_file, "w").write(str(config))
    print(f"\t[Info] Updated config at `{config_file}` with content:\n\t\t{config}")
    quit()
    feasible_list_str = "\n".join(
        ["\t\t-" + x[0] + " ("+str(x[1])+" GB needed)" for x in feasible])
    infeasible_list_str = "\n".join(
        ["\t\t-" + x[0] + " ("+str(x[1])+" GB needed)" for x in infeasible])

    print(f"""
    You can run on the datasets:
    {feasible_list_str}

    You cannot run on the datasets:
    {infeasible_list_str}


    To download the datasets we detected as feasible for your available disk space

        Run: python download_datasets_fast.py

    If you are running via the experiments/initial_setup.sh script, then datasets will be downloaded after this script.

    """)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configure SALIENT++ distributed experiments.")
    parser.add_argument("--force_diskspace_limit", help="Override normal behavior which will analyze disk space using filesystem commands. The script will attempt, but not guarantee, to select a subset of datasets that will fit within the specified disk-space constraints. Specify an integer representing a soft limit on the number of GB used for storage.", type=int, required=False, default=-1)
    parser.add_argument("--max_num_parts", help="Specify the maximum number of nodes (GPUs) you will use for experiments. This determines which preprocessed partitioned datasets are downloaded --- e.g., 8-way partitioned datasets will not be downloaded if you specify a value here of less than 8.", type=int, required=True)
    try:
        args = parser.parse_args()
    except:
        parser.print_help(sys.stderr)
        exit(1)

    #adapt_config_files()
    determine_viable_datasets(args)
    
    
    print("Done")
    exit(0)
