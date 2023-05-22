import ogb
from ogb.utils.url import decide_download, download_url, extract_zip
import os
import pathlib
from pathlib import Path
import shutil
import argparse
parser = argparse.ArgumentParser(description="run distributed experiment")
parser.add_argument("--skip_confirmation",
                    help="Whether to skip confirmations for downloading large datasets.",
                    action="store_true")
parser.add_argument("--simulation_downloads_only",
                    help="Only download the files required for simulations --- e.g., only download partition-labels and base unpartitioned datasets. This skips downloading the pre-processed partitioned copies of the datasets.",
                    action="store_true")
args = parser.parse_args()

directory = os.path.dirname(os.path.abspath(__file__))
dataset_dir = directory + "/../dataset"

dataset_dir_path = Path(dataset_dir)
dataset_dir_path.mkdir(exist_ok=True)

dataset_base_url = "https://salient-datasets-ae.s3.amazonaws.com/"
partitioned_dataset_base_url = "https://salientplus-datasets-ae.s3.amazonaws.com/"

#dataset_list = ["https://salient-datasets-ae.s3.amazonaws.com/ogbn-arxiv.zip","https://salient-datasets-ae.s3.amazonaws.com/ogbn-products.zip"]


config_dir = Path(directory) / 'configuration_files'
if not Path.exists(config_dir) or not Path.exists(config_dir / 'feasible_datasets.cfg'):
    print("[Error] Did not detect configuration file. Please run experiments/configure_for_environment.py before trying to download datasets.")
    quit()

config_file = config_dir / 'feasible_datasets.cfg'
config_info = eval(open(config_file).read())

#print("Okay to proceed. " + str(config_info))
#if not os.path.exists(directory + "/.feasible_datasets"):
#    print("[Error] Did not detect list of feasible datasets. Please run experiments/configure_for_environment.py before trying to download datasets.")
#    quit()


#datasets = open(directory + "/.feasible_datasets").readlines()
dataset_list = config_info['feasible_datasets']#[x.strip() for x in datasets]

#infeasible_datasets = open(directory + "/.infeasible_datasets").readlines()
infeasible_dataset_list = config_info['infeasible_datasets']#[x.strip() for x in infeasible_datasets]

message = """\
#################################################################################################
######## 1. Downloading preprocessed OGB datasets for artifact evaluation (non-partitioned) #####
#################################################################################################
"""


print()
print(message)

print()
print("[Info] Will try to download: " + str(",".join(dataset_list)))
print("[Info] Will *not* try to download: " +
      str(",".join(infeasible_dataset_list)))
print()

for dataset in dataset_list:
    print()
    print("Trying to download " + dataset)
    dataset_name = dataset
    #print (dataset_dir + "/"+dataset_name)
    if os.path.exists(dataset_dir + "/" + dataset_name):
        print("Skip " + dataset + " because it already exists.")
    else:
        dataset_url = dataset_base_url + dataset + ".zip"
        if args.skip_confirmation or decide_download(dataset_url):
            path = download_url(dataset_url, dataset_dir)
            extract_zip(path, dataset_dir)
            os.unlink(path)
    print()

message = """\
#################################################################################################
######## 2. Downloading pre-generated partition labels for OGB datasets                  ########
#################################################################################################
"""
print()
print(message)
print()

partition_labels_dir = dataset_dir_path / 'partition-labels'
partition_labels_tmpdir = dataset_dir_path / 'partition-data'
if partition_labels_dir.exists() or partition_labels_tmpdir.exists():
    if partition_labels_dir.exists():
        print(f"[Info] Skipping downloads of partition-labels because the directory already exists at {partition_labels_dir}")
    if partition_labels_tmpdir.exists():
        print(f"[Info] Skipping downloads of partition-labels because the temporary directory used for downloading this dataset already exists at {partition_labels_tmpdir}")
else:
    dataset_url = partitioned_dataset_base_url + 'partition-data.zip'
    if args.skip_confirmation or decide_download(dataset_url):
        path = download_url(dataset_url, dataset_dir_path)
        extract_zip(path, dataset_dir)
        os.unlink(path)
        print(f"Moving partition labels to {partition_labels_dir}")
        partition_labels_tmpdir.rename(partition_labels_dir)


message = """\
#################################################################################################
######## 3. Downloading preprocessed OGB datasets for artifact evaluation (partitioned)  ########
#################################################################################################
"""
print()
print(message)
print()

max_num_parts = config_info['max_num_parts']
print(f'Will download partitioned versions of feasible datasets for # partitions <= max_num_parts.')
print(f'Obtained max_num_parts = {max_num_parts} from configuration.')

available_parted_data = {'ogbn-papers100M': [2,4,8,16], 'ogbn-products': [2,4,8,16], 'MAG240': [4,8,16]}

desired_datasets = []
for x in dataset_list:
    if x not in available_parted_data:
        continue
    for k in available_parted_data[x]:
        if k <= max_num_parts:
            desired_datasets.append(f'k{k}_{x}')
print(desired_datasets)



if args.simulation_downloads_only:
    print()
    print("Skipping downloads for partitioned and reordered datasets, due to user specifying --simulation_downloads_only.")
    exit(1)


for dataset in desired_datasets:
    print()
    print(f"Trying to download {dataset}")
    dataset_name = dataset


    partition_count = int(dataset_name.split('_')[0].replace('k',''))
    base_dataset_name = dataset_name.split('_')[1]
    if partition_count not in available_parted_data[base_dataset_name]:
        print("[Error] Unexpected file naming conventions. Quitting for safety.")
        exit(1)
    expected_extraction_path = Path(dataset_dir) / 'dataset-4constraint' / f'metis-reordered-k{partition_count}' / base_dataset_name
    expected_extraction_path_base = Path(dataset_dir) / 'dataset-4constraint' / f'metis-reordered-k{partition_count}'
    expected_extraction_path_base2 = Path(dataset_dir) / 'dataset-4constraint'
    print(expected_extraction_path)
    desired_extraction_path = Path(dataset_dir) / f'metis-reordered-k{partition_count}' / base_dataset_name
    desired_extraction_path_base = Path(dataset_dir) / f'metis-reordered-k{partition_count}'

    #if os.path.exists(dataset_dir + "/" + dataset_name):
    if desired_extraction_path.exists():
        print("Skip " + dataset + " because it already exists.")
    else:
        dataset_url = partitioned_dataset_base_url + dataset + ".zip"
        if args.skip_confirmation or decide_download(dataset_url):
            path = download_url(dataset_url, dataset_dir)
            extract_zip(path, dataset_dir)
            os.unlink(path)

            if not expected_extraction_path.exists() or not expected_extraction_path.is_dir():
                print("[Error] Extracted file is not at the expected location. Quitting for safety.")
                exit(1)
            if desired_extraction_path.exists():
                print("[Error] Extracted file destination location already exists... Quitting for safety.")
                exit(1)
            desired_extraction_path_base.mkdir(parents=True, exist_ok=True)
            expected_extraction_path.rename(desired_extraction_path)
            expected_extraction_path_base.rmdir() 
            expected_extraction_path_base2.rmdir()
            #expected_extraction_path.rename()
            #shutil.move(Path(dataset_dir) / 'dataset-4constraint/' / dirs[0], Path(dataset_dir))
            #os.rmdir(Path(dataset_dir) / 'dataset-4constraint/')

            #print(os.listdir(Path(dataset_dir) / 'dataset-4constraint'))
    print()
   









