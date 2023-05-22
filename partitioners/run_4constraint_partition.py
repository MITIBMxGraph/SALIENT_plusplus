import argparse


from driver.dataset import FastDataset
from driver.main import get_dataset
import torch
import torch_sparse
import os
import sys
import torch_geometric
from pathlib import Path
from partitioners.partition import metis_partition
from partitioners.eval_quality import evaluate_quality, generate_cache, refine_partition


def compute_node_weights(dataset, id_map, num_vertices):
    weights = torch.zeros([num_vertices], dtype=torch.long)
    
    for i in range(0, len(id_map)):
        weights[id_map[i]] += 1
    return weights


def get_4d_node_weights(dataset, num_vertices, rowptr):
    torch.set_printoptions(edgeitems=10)
    train_idx = dataset.split_idx['train']
    w1 = torch.zeros([num_vertices], dtype=torch.long)
    w1[train_idx] = torch.ones([train_idx.size()[0]], dtype=torch.long)
    w2 = torch.zeros([num_vertices], dtype=torch.long)
    w2[dataset.split_idx['valid']] = torch.ones([dataset.split_idx['valid'].size()[0]], dtype=torch.long)
    w3 = torch.ones([num_vertices], dtype=torch.long)
    w3[dataset.split_idx['valid']] = torch.zeros([dataset.split_idx['valid'].size()[0]], dtype=torch.long)
    w3[dataset.split_idx['train']] = torch.zeros([dataset.split_idx['train'].size()[0]], dtype=torch.long)

    w4 = rowptr[1:] - rowptr[:-1]
    
    return torch.cat([w1.reshape(w2.size()[0],1),w2.reshape(w1.size()[0],1), w3.reshape(w1.size()[0], 1), w4.reshape(w1.size()[0],1)], dim=1).view(-1).to(torch.long).contiguous()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generate a partitioning. Will create file args.output_directory / dataset_name.pt file. ")
    parser.add_argument("--dataset_name", help="Name of the ogb dataset", type=str, required=True)
    parser.add_argument("--output_directory", help="Directory to save.", type=str, required=True)
    parser.add_argument("--dataset_dir", help="The dataset directory", type=str, required=True)
    parser.add_argument("--num_parts", help="Number of partitions to generate", type=int, required=True)
    
    args = parser.parse_args()
    print(args)



    output_dir = Path(args.output_directory)
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / Path(args.dataset_name + "-" + str(args.num_parts) + ".pt")



    if output_file.exists():
        print("Error, output part path exists. Not overwriting it for safety.")
        quit()

    assert not output_file.exists()

    assert args.dataset_name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M', 'MAG240']


    NUM_PARTS = args.num_parts
    dataset = get_dataset(args.dataset_name, args.dataset_dir, skip_features=True)

    rowptr,col,_ = dataset.adj_t().csr()

    torch.set_printoptions(edgeitems=10)
    edge_weights = torch.ones_like(col, dtype=torch.long, memory_format=torch.legacy_contiguous_format).share_memory_()

    node_weights = get_4d_node_weights(dataset, rowptr.size()[0]-1, rowptr)
    nodew_dim=4


    print(rowptr.dtype)
    print(col.dtype)
    print(node_weights.dtype)
    print(edge_weights.dtype)

    parts = metis_partition(rowptr, col, node_weights, edge_weights, nodew_dim=nodew_dim, num_parts=NUM_PARTS)

    torch.save(parts, str(output_file))
    print("saved")
    #parts = torch.load("products-"+str(NUM_PARTS)+".pt")
    quit()
