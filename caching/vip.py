"""
This module contains functions for caching frequently-accessed remote vertices
in distributed GNN computations with node-wise sampling.  These include
functions for estimating vertex-wise inclusion probability (VIP) weights via
different schemes, and for determining partition-wise indices of cached remote
vertices.
"""


import torch
import importlib

import caching.util as util

from torch_scatter import segment_csr, gather_csr
from itertools import accumulate

if importlib.util.find_spec("torch_geometric.loader") is not None:
    import torch_geometric.loader
    if hasattr(torch_geometric.loader, "NeighborSampler"):
        from torch_geometric.loader import NeighborSampler
    else:
        from torch_geometric.data import NeighborSampler
else:
    from torch_geometric.data import NeighborSampler


# ==================================================
# COLLECT VERTEX ACCESS FREQUENCY STATISTICS
# ==================================================


def simulate_vertex_accesses(
        dataset,
        partition_ids,
        fanouts,
        minibatch_size,
        num_epochs,
        num_workers_sampler,
        verbose = True
):
    """Simulate distributed GNN training and return vertex communication stats."""
    util.print_if(verbose, f"Simulating vertex accesses ({num_epochs} epochs)", 0)

    num_parts = int(partition_ids.max() + 1)
    num_vertices = partition_ids.size()[0]
    rowptr, col, _ = dataset.adj_t().csr()
    train_idx = dataset.split_idx['train']
    train_idx_per_partition = util.partitionwise_train_idx(partition_ids, train_idx)

    vertex_accesses_per_partition = [torch.zeros(num_vertices, dtype=torch.long)
                                     for _ in range(num_parts)]

    for iter in range(num_epochs):
        util.print_if(verbose, f"Epoch {iter}", 1)

        for part in range(num_parts):
            util.print_if(verbose, f"Partition {part}", 2)

            perm_shuffle_train_idx = torch.randperm(train_idx_per_partition[part].size()[0])
            sampler = NeighborSampler(
                dataset.adj_t(),
                node_idx=train_idx_per_partition[part][perm_shuffle_train_idx],
                batch_size=minibatch_size,
                sizes=fanouts,
                num_workers=num_workers_sampler,
                pin_memory=True,
                return_e_id=False
            )

            for mfg in sampler:
                _, n_ids, _ = mfg
                vertex_accesses_per_partition[part][n_ids] += 1

    return vertex_accesses_per_partition


# ==================================================
# EVALUATE COMMUNICATION VOLUME
# ==================================================


def evaluate_communication_volume(
        vertex_accesses_per_partition,
        partition_ids,
        cache_idx_per_partition
):
    """Evaluate partition-wise communication volume with vertex caching."""
    num_parts = int(partition_ids.max() + 1)
    num_vertices = partition_ids.size()[0]
    mask_per_partition = util.partitionwise_masks(partition_ids)

    comm_vol_dict = {'total': 0, 'internal': 0, 'cross': 0, 'cache_hits': 0}

    for part in range(num_parts):
        if cache_idx_per_partition is not None:
            cache_mask = util.partition_mask(cache_idx_per_partition[part], num_vertices)
        else:
            cache_mask = torch.zeros(num_vertices, dtype=torch.bool)

        comm_vol_dict['total'] += \
            int(vertex_accesses_per_partition[part].sum())
        comm_vol_dict['internal'] += \
            int(vertex_accesses_per_partition[part][mask_per_partition[part]].sum())
        comm_vol_dict['cross'] += \
            int(vertex_accesses_per_partition[part][~mask_per_partition[part] & ~cache_mask].sum())
        comm_vol_dict['cache_hits'] += \
            int(vertex_accesses_per_partition[part][~mask_per_partition[part] & cache_mask].sum())

        assert comm_vol_dict['total'] \
            == (comm_vol_dict['internal'] + comm_vol_dict['cross'] + comm_vol_dict['cache_hits'])

    return comm_vol_dict


# ==================================================
# VIP WEIGHT ESTIMATION FUNCTIONS
# ==================================================


# ---------- ANALYTICAL VIP MODEL --------------------

def vip_analytical(
        graph_rowptr,
        graph_col,
        train_idx_per_partition,
        minibatch_size,
        fanouts,
        verbose = True
):
    """Node-wise sampling VIP weights: analytical model.

    Returns a partition-wise list of torch tensors which contain vertex-wise
    inclusion probabilities (VIP).  The list length is the same as
    `train_idx_per_partition`, and the order of vertex-wise VIP values in each
    tensor matches the vertex IDs in `graph_rowptr` and `graph_col`.

    The analytical VIP model for neighborhood expansion with node-wise sampling
    is described in:

    T. Kaler, A.S. Iliopoulos, P. Murzynowski, T.B. Schardl, C.E. Leiserson,
    J. Chen.  "Communication-efficient graph neural networks with probabilistic
    neighborhood expansion analysis and caching", MLSys 2023.

    """
    util.print_if(verbose, "Calculating VIP weights: analytical model", 0)

    num_parts = len(train_idx_per_partition)
    num_vertices = graph_rowptr.size()[-1] - 1
    degrees = util.vertex_indegrees(graph_rowptr)
    p_total = []

    for part in range(num_parts):
        util.print_if(verbose, f"Partition {part}", 1)

        p_hop = torch.zeros(num_vertices, dtype=torch.float)
        p_hop[train_idx_per_partition[part]] = \
            (minibatch_size * 1.0) / train_idx_per_partition[part].size()[0]

        p_total.append(torch.ones_like(p_hop))

        for h, fanout in enumerate(fanouts):
            util.print_if(verbose, f"Hop {h}, fanout={fanout}", 2)

            # edge/neighbor-wise probability of NOT being sampled at hop h
            transition_weights = torch.minimum(torch.ones_like(degrees),
                                               fanout / degrees)
            p_col_not_sampled = 1 - transition_weights[graph_col] * p_hop[graph_col]

            # vertex-wise probability of being sampled at hop h
            p_hop[:] = 1 - torch.exp(segment_csr(torch.log(p_col_not_sampled),
                                                 graph_rowptr, reduce='add'))

            # update multi-hop probability of NOT being sampled
            p_total[part][:] = p_total[part] * (1 - p_hop)

        # convert to probability of being sampled
        p_total[part][:] = 1 - p_total[part]

    return p_total


# ---------- ANALYTICAL VIP MODEL (GPU) --------------------

def vip_analytical_gpu(
        graph_rowptr,
        graph_col,
        train_idx_per_partition,
        minibatch_size,
        fanouts,
        gpu_device = 'cuda:0',
        gpu_batch_size = 2**22,
        verbose = True
):
    """Node-wise sampling VIP weights: analytical model (GPU computation)."""
    util.print_if(verbose, "Calculating VIP weights: analytical model (GPU)", 0)

    num_parts = len(train_idx_per_partition)
    num_vertices = graph_rowptr.size()[-1] - 1
    p_total = []

    gpu_batches = util.batch_sizes(num_vertices, gpu_batch_size)
    gpu_batches_cumsum = [0] + list(accumulate(gpu_batches))
    num_batches = len(gpu_batches)

    DEFAULT_STREAM = torch.cuda.default_stream(gpu_device)
    DATA_STREAM = torch.cuda.Stream(gpu_device)

    for part in range(num_parts):
        util.print_if(verbose, f"Partition {part}", 1)
        degrees = util.vertex_indegrees(graph_rowptr).to(device=gpu_device)

        p_hop = torch.zeros(num_vertices, dtype=torch.float, device=gpu_device)
        p_hop[train_idx_per_partition[part]] = \
            (minibatch_size * 1.0) / train_idx_per_partition[part].size()[0]

        p_total.append(torch.ones_like(p_hop))

        rowptr_batches = [None] * num_batches
        col_batches = [None] * num_batches
        for b in range(num_batches):
            rowptr_batches[b] = \
                graph_rowptr[gpu_batches_cumsum[b]:gpu_batches_cumsum[b+1]+1].clone()
            col_batches[b] = graph_col[rowptr_batches[b][0]:rowptr_batches[b][-1]]

        for h, fanout in enumerate(fanouts):
            util.print_if(verbose, f"Hop {h}, fanout={fanout}", 2)

            transition_weights = torch.minimum(torch.ones_like(degrees),
                                               fanout / degrees)

            with torch.cuda.stream(DATA_STREAM): # first batch
                if h != 0:
                    next_rowptr_b.record_stream(DEFAULT_STREAM)
                    next_col_b.record_stream(DEFAULT_STREAM)
                next_rowptr_b = rowptr_batches[0].to(
                    device=gpu_device, non_blocking=True).clone()
                next_col_b = col_batches[0].to(
                    device=gpu_device, non_blocking=True)

            for b in range(num_batches):
                DEFAULT_STREAM.wait_stream(DATA_STREAM)
                rowptr_b = next_rowptr_b
                col_b = next_col_b

                if b < num_batches - 1: # prefetch next batch
                    with torch.cuda.stream(DATA_STREAM):
                        next_rowptr_b.record_stream(DEFAULT_STREAM)
                        next_col_b.record_stream(DEFAULT_STREAM)
                        next_rowptr_b = rowptr_batches[b+1].to(
                            device=gpu_device, non_blocking=True).clone()
                        next_col_b = col_batches[b+1].to(
                            device=gpu_device, non_blocking=True)

                rowptr_b -= rowptr_b[0].clone()

                p_col_not_sampled_b = 1 - transition_weights[col_b] * p_hop[col_b]
                p_hop_b = 1 - torch.exp(segment_csr(torch.log(p_col_not_sampled_b),
                                                    rowptr_b, reduce='add'))

                p_hop[gpu_batches_cumsum[b]:gpu_batches_cumsum[b+1]] = p_hop_b
                # END loop over batches

            p_total[part][:] = p_total[part][:] * (1 - p_hop)
            # END loop over hops

        p_total[part][:] = 1 - p_total[part]
        p_total[part] = p_total[part].cpu()
        # END loop over partitions

    return p_total


# ---------- SIMULATION-BASED VIP --------------------

def vip_simulation(
        dataset,
        partition_ids,
        minibatch_size,
        fanouts,
        num_epochs,
        num_workers_sampler,
        verbose = True
):
    """Node-wise sampling VIP weights: empirical simulation."""
    util.print_if(verbose, "Calculating VIP weights: empirical simulation", 0)
    return simulate_vertex_accesses(dataset, partition_ids, fanouts,
                                    minibatch_size, num_epochs,
                                    num_workers_sampler, verbose)


# ---------- REACHABLE DEGREE --------------------

def vip_proxy_degree_reachable(
        graph_rowptr,
        graph_col,
        train_idx_per_partition,
        num_hops,
        verbose = True
):
    """VIP proxy weights for node-wise sampling: reachable degree heuristic."""
    util.print_if(verbose, "Calculating VIP proxy weights: degree-reachable heuristic", 0)

    num_vertices = graph_rowptr.size()[-1] - 1
    num_parts = len(train_idx_per_partition)
    degrees = util.vertex_indegrees(graph_rowptr);

    degrees_reachable = []

    for part in range(num_parts):
        util.print_if(verbose, f"Partition {part}", 1)

        reachable = torch.zeros(num_vertices, dtype=torch.int)
        reachable[train_idx_per_partition[part]] = 1

        for h in range(num_hops):
            util.print_if(verbose, f"Hop {h}", 2)
            reachable[:] = segment_csr(torch.ones_like(graph_col) * reachable[graph_col],
                                       graph_rowptr, reduce='add')
            reachable[reachable != 0] = 1

        degrees_reachable.append(reachable * degrees)

    return degrees_reachable


# ---------- NUMBER OF REACHABLE PATHS --------------------

def vip_proxy_num_paths_reachable(
        graph_rowptr,
        graph_col,
        train_idx_per_partition,
        num_hops,
        verbose = True
):
    """Node-wise sampling VIP weights: number of reachable paths."""
    util.print_if(verbose, "Calculating VIP weights: number of reachable paths", 0)

    num_vertices = graph_rowptr.size()[-1] - 1
    num_parts = len(train_idx_per_partition)

    num_paths_reachable = []

    for part in range(num_parts):
        util.print_if(verbose, f"Partition {part}", 1)

        num_paths = torch.zeros(num_vertices, dtype=torch.int)
        num_paths[train_idx_per_partition[part]] = 1

        for h in range(num_hops):
            util.print_if(verbose, f"Hop {h}", 2)
            edges_next_hop = torch.ones_like(graph_col) * num_paths[graph_col]
            num_paths[:] += segment_csr(edges_next_hop, graph_rowptr, reduce='add')

        num_paths_reachable.append(num_paths)

    return num_paths_reachable


# ---------- 1-HOP HALO --------------------

def vip_proxy_degree_1hop(
        graph_rowptr,
        graph_col,
        train_idx_per_partition,
        verbose = True
):
    """VIP proxy indicators for node-wise sampling: 1-hop halo heuristic."""
    util.print_if(verbose, "Calculating VIP proxy indicators: 1-hop halo heuristic", 0)

    num_vertices = graph_rowptr.size()[-1] - 1
    mask_1hop = []

    for part in range(len(train_idx_per_partition)):
        util.print_if(verbose, f"Partition {part}", 1)

        mask_train_part = util.partition_mask(train_idx_per_partition[part], num_vertices)
        idx_1hop = torch.masked_select(
            graph_col,
            gather_csr(mask_train_part.to(torch.int), graph_rowptr).to(torch.bool))
        idx_1hop = idx_1hop.unique()
        # idx_1hop = torch.masked_select(idx_1hop, ~mask_part[idx_1hop])

        mask_1hop.append(torch.zeros(num_vertices, dtype=torch.bool))
        mask_1hop[part][idx_1hop] = True

    return mask_1hop


# ---------- RANDOM WALK --------------------

def vip_randomwalk(
        graph_rowptr,
        graph_col,
        train_idx_per_partition,
        fanouts,
        verbose = True
):
    """Node-wise sampling VIP weights: random walk."""
    util.print_if(verbose, "Calculating VIP weights: random walk", 0)

    num_vertices = graph_rowptr.size()[-1] - 1
    degrees = util.vertex_indegrees(graph_rowptr)

    transition_weights = 1 / degrees
    transition_weights[torch.isinf(transition_weights)] = 0

    p_total = []

    for part in range(len(train_idx_per_partition)):
        util.print_if(verbose, f"Partition {part}", 1)

        p_buf = torch.zeros(num_vertices, dtype=torch.float)

        # probability of being sampled in minibatch
        p_buf[train_idx_per_partition[part]] = \
            1 / train_idx_per_partition[part].size()[0]

        # accumulate probability of being sampled at hop 1, ..., L
        for h in range(len(fanouts)):
            util.print_if(verbose, f"Hop {h}", 2)
            col_weighted = transition_weights[graph_col] * p_buf[graph_col]
            p_buf[:] = p_buf + segment_csr(col_weighted, graph_rowptr, reduce='add')

        p_total.append(p_buf)

    return p_total


# ==================================================
# VIP-BASED ORDERING & CACHING
# ==================================================


# ---------- PARTITION-WISE VERTEX REORDERING --------------------

def argsort_vip(vip_weights_per_partition):
    """Return partition-wise vertex IDs in descending VIP order."""
    num_parts = len(vip_weights_per_partition)
    num_vertices = vip_weights_per_partition[0].size()[0]
    perm_per_partition = []

    for part in range(num_parts):
        perm_per_partition.append(
            vip_weights_per_partition[part].argsort(descending=True)
        )

    return perm_per_partition


# ---------- CACHE-GENERATION FUNCTION --------------------

def get_lambda_vip_cache(
        dataset,
        partition_ids,
        fanouts,
        minibatch_size,
        scheme = 'vip-analytical',
        actual_vertex_accesses = None,
        num_iter_simulation = 2,
        num_workers_sampler = 20,
        verbose = True,
):
    """Return lambda: replication factor -> partition-wise cached vertex IDs."""
    num_parts = int(partition_ids.max() + 1)
    num_vertices = partition_ids.size()[0]
    rowptr, col, _ = dataset.adj_t().csr()
    train_idx = dataset.split_idx['train']

    train_idx_per_partition = util.partitionwise_train_idx(partition_ids, train_idx)

    if scheme == 'vip-analytical':
        vip_weights_per_partition = vip_analytical(
            rowptr, col, train_idx_per_partition,
            minibatch_size, fanouts, verbose
        )
    elif scheme == 'vip-analytical-gpu':
        vip_weights_per_partition = vip_analytical_gpu(
            rowptr, col, train_idx_per_partition,
            minibatch_size, fanouts,
            verbose=verbose
        )
    elif scheme == 'vip-simulation':
        vip_weights_per_partition = vip_simulation(
            dataset, partition_ids, minibatch_size, fanouts,
            num_iter_simulation, num_workers_sampler, verbose
        )
    elif scheme == 'degree-reachable':
        vip_weights_per_partition = vip_proxy_degree_reachable(
            rowptr, col, train_idx_per_partition, len(fanouts), verbose
        )
    elif scheme == "num-paths-reachable":
        vip_weights_per_partition = vip_proxy_num_paths_reachable(
            rowptr, col, train_idx_per_partition, len(fanouts), verbose
        )
    elif scheme == 'halo-1hop':
        vip_weights_per_partition = vip_proxy_degree_1hop(
            rowptr, col, train_idx_per_partition, verbose
        )
    elif scheme == 'random-walk':
        vip_weights_per_partition = vip_randomwalk(
            rowptr, col, train_idx_per_partition, fanouts, verbose
        )
    elif scheme == 'oracle':
        assert actual_vertex_accesses is not None
        util.print_if(verbose, "Oracle cache: using actual vertex accesses", 0)
        vip_weights_per_partition = [x.clone() for x in actual_vertex_accesses]
    else:
        raise ValueError(f"Invalid caching scheme: {scheme}")

    # ignore vertices in the same partition as training/seed vertices
    mask_per_partition = util.partitionwise_masks(partition_ids)
    for part in range(num_parts):
        vip_weights_per_partition[part][mask_per_partition[part]] = 0

    idx_vertex_descending_vip = argsort_vip(vip_weights_per_partition)

    def partitionwise_cached_idx(replication_factor):
        """Partition-wise lists of cached vertex indices."""
        cached_idx_per_partition = []
        size_cache = int(num_vertices * replication_factor / num_parts)
        for part in range(num_parts):
            cached_idx_per_partition.append(
                idx_vertex_descending_vip[part][:size_cache]
            )
        return cached_idx_per_partition

    return lambda replication_factor : partitionwise_cached_idx(replication_factor)
