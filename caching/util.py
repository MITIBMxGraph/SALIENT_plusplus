"""
This module contains utility functions for the VIP caching and
communication volume simulation experiments.
"""


import torch


def vertex_indegrees(graph_rowptr):
    """Calculate vertex in-degrees."""
    degrees = graph_rowptr[1:] - graph_rowptr[:-1]
    degrees = degrees.to(torch.float)
    return degrees


def partitionwise_masks(partition_id_per_vertex):
    """Vertex-wise partition IDs -> partition-wise vertex masks."""
    num_parts = int(partition_id_per_vertex.max()+1)
    num_vertices = partition_id_per_vertex.size()[0]
    mask_per_partition = []
    for part in range(num_parts):
        mask = torch.zeros(num_vertices, dtype=torch.bool)
        mask[partition_id_per_vertex == part] = True
        mask_per_partition.append(mask)
    return mask_per_partition


def partitionwise_mask2idx(mask_per_partition):
    """Partition-wise vertex masks -> partition-wise vertex indices."""
    num_parts = len(mask_per_partition)
    num_vertices = mask_per_partition[0].size()[0]
    idx_per_partition = []
    for part in range(num_parts):
        idx = torch.masked_select(torch.arange(0, num_vertices),
                                  mask_per_partition[part])
        idx_per_partition.append(idx)
    return idx_per_partition


def partition_mask(idx_part, num_vertices):
    """Partition vertex indices -> partition vertex mask."""
    mask_part = torch.zeros(num_vertices, dtype=torch.bool)
    mask_part[idx_part] = True
    return mask_part


def partitionwise_train_idx(partition_ids, train_idx):
    """Get list of partition-wise training vertex IDs."""
    num_parts = int(partition_ids.max() + 1)
    num_vertices = partition_ids.size()[0]
    num_train = train_idx.size()[0]

    train_partition_ids = partition_ids[train_idx]
    perm_sort_train_partition_ids = train_partition_ids.argsort()

    num_train_per_partition = \
        torch.bincount(train_partition_ids, minlength=num_parts)
    offset_train_per_partition = \
        torch.cat([torch.tensor([0]), num_train_per_partition.cumsum(dim=0)])

    train_idx_per_partition = []
    for part in range(num_parts):
        idx = torch.arange(0, num_train)[perm_sort_train_partition_ids][
            offset_train_per_partition[part] : offset_train_per_partition[part+1]
        ]
        idx[:] = idx[torch.randperm(idx.size()[0])]
        train_idx_per_partition.append(train_idx[idx])
        assert (partition_ids[train_idx_per_partition[part]] != part).sum() == 0

    return train_idx_per_partition


def batch_sizes(n_total, n_batch):
    """Return [b0 b1 ... bk] s.t. bi >= n_batch & sum_i(bi) = n_total."""
    if n_batch >= n_total:
        return [n_total]
    num_batches = n_total // n_batch
    batches = [n_batch] * num_batches
    rem = n_total % n_batch
    inc = rem // num_batches
    remrem = rem % num_batches
    for i in range(0, remrem):
        batches[i] += inc + 1
    for i in range(remrem, num_batches):
        batches[i] += inc
    assert sum(batches) == n_total
    return batches


def print_if(flag, string, level=0):
    """Conditionally print indented string."""
    if not flag:
        return
    prefix = "" if (level <= 0) else (("  " * level) + "- ")
    print(prefix + string)
