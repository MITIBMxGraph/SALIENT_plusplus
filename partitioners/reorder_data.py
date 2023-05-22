from driver.dataset import FastDataset, DisjointPartFeatReorderedDataset
from driver.main import get_dataset
import torch
import torch_sparse
import os
import sys
import torch_geometric
import argparse
from pathlib import Path
from partitioners.partition import metis_partition
from partitioners.eval_quality import evaluate_quality, generate_cache, refine_partition, get_frequency_tensors

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generate a reordered dataset from OGB dataset, partition file, and frequency analysis.")
    parser.add_argument("--dataset_name", help="Name of the ogb dataset", type=str, required=True)
    parser.add_argument("--path_to_part", help="Path to the file containing the partitions", required=True)
    parser.add_argument("--output_path", help="Location to save.", type=str, required=True)
    parser.add_argument("--dataset_dir", help="The dataset directory", type=str, required=True)
    parser.add_argument("--fanouts",
                        help="Training fanouts",
                        type=int, default=[15, 10, 5], nargs="*", required=False)
    parser.add_argument("--disable_vip",
                        help="Disables the use of vertex-inclusion probabilities to order vertices within each partition.",
                        action="store_true")
    
    args = parser.parse_args()
    print(args)

    assert args.dataset_name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M', 'MAG240']


    FANOUTS = args.fanouts #[15,10,5]
    print("Using fanouts  " + str(FANOUTS))
    dataset = get_dataset(args.dataset_name, args.dataset_dir)
    rowptr,col,_ = dataset.adj_t().csr()
    parts = torch.load(args.path_to_part)
    num_parts = int(parts.max()+1)
    final_dataset_output_path = Path(args.output_path) / f"metis-reordered-k{num_parts}" / args.dataset_name
    if final_dataset_output_path.exists():
        print(f"[Error] A partitioned dataset already exists at path {final_dataset_output_path}. We will not proceed further for safety. Choose a different path.")
        exit(1)

    if not args.disable_vip:
        print("[INFO] Using VIP order within partitions")
        frequency_tensor = get_frequency_tensors(dataset, FANOUTS, torch.tensor(parts))
    else:
        print("[INFO] Not using VIP order within partitions because you passed flag --disable_vip.")
        frequency_tensor = None


    DisjointPartFeatReorderedDataset.reorder_and_save(dataset, parts, frequency_tensor, Path(args.output_path))

    print(frequency_tensor)
    quit()
