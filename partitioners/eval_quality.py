from driver.dataset import FastDataset
from driver.main import get_dataset
import time
import torch
import torch_sparse
import os
import sys
from torch_scatter import segment_csr, gather_csr
from tqdm import tqdm

import torch.nn.functional as F

import importlib
if importlib.util.find_spec("torch_geometric.loader") is not None:
    import torch_geometric.loader
    if hasattr(torch_geometric.loader, "NeighborSampler"):
        from torch_geometric.loader import NeighborSampler
    else:
        from torch_geometric.data import NeighborSampler
else:
    from torch_geometric.data import NeighborSampler

from pathlib import Path

def get_frequency_tensors_fast(dataset, fanouts, partition_tensor, device):
    BATCH_SIZE_VIP = 2**22;
    BATCH_SIZE_TRAIN = 1024
    num_parts = int(partition_tensor.max()+1)
    rowptr, col, value = dataset.adj_t().csr()

    train_idx = dataset.split_idx['train']
    perm = partition_tensor[train_idx].argsort()
    train_parts = partition_tensor[train_idx]

    num_vertices = rowptr.size()[-1] - 1

    bincounts = torch.bincount(train_parts, minlength=num_parts)
    # print(bincounts)
    bincounts_sum = torch.cat([torch.tensor([0]),bincounts.cumsum(dim=0)])
    # print(bincounts_sum)

    indices_for_partitions = []
    for i in range(num_parts):
        indices = torch.arange(0, train_idx.size()[0])[perm][bincounts_sum[i]:bincounts_sum[i+1]]
        indices = indices[torch.randperm(indices.size()[0])]
        indices_for_partitions.append(dataset.split_idx['train'][indices])
        # assert((partition_tensor[indices_for_partitions[i]] != i).sum() == 0)


    DATA_STREAM = torch.cuda.Stream(device)
    DEFAULT_STREAM = torch.cuda.default_stream(device)
    probability_vectors = []
    for i in range(num_parts):
        node_degrees = rowptr[1:] - rowptr[:-1]
        node_degrees = node_degrees.to(torch.float64).to(device=device)
        indices_for_partitions[i] = indices_for_partitions[i].to(device=device)
        
        p_hopwise_list = [torch.zeros(num_vertices, dtype=torch.float, device=device)]

        # initialize p_hopwise for the training vertices in partition i
        p_hopwise_list[0][indices_for_partitions[i]] = (BATCH_SIZE_TRAIN * 1.0) / indices_for_partitions[i].size()[0]

        batch_sizes = [BATCH_SIZE_VIP]*(num_vertices//BATCH_SIZE_VIP)
        total_done = 0
        for batch_sz in batch_sizes:
            total_done += batch_sz

        while total_done != num_vertices:
            assert(total_done <= num_vertices)
            extra_to_allocate = min(num_vertices - total_done,len(batch_sizes))
            for _i in range(extra_to_allocate):
                batch_sizes[_i] += 1
                total_done += 1


        rowptr_batches = [None]*len(batch_sizes)
        col_batches = [None]*len(batch_sizes)

        batch_sizes_prefixsum = [0]*(len(batch_sizes)+1)
        for _i in range(len(batch_sizes)):
           batch_sizes_prefixsum[_i+1] = batch_sizes[_i]+batch_sizes_prefixsum[_i]

        for _i in range(len(batch_sizes)):
            rowptr_batches[_i] = rowptr[batch_sizes_prefixsum[_i]:batch_sizes_prefixsum[_i+1]+1].clone()
            col_batches[_i] = col[rowptr_batches[_i][0]:rowptr_batches[_i][-1]]

        for h,fanout in enumerate(fanouts):
            res = torch.zeros(num_vertices, dtype=torch.float64, device=device)
            with torch.cuda.stream(DATA_STREAM):
                if h > 0:
                    next_rowptr_b.record_stream(DEFAULT_STREAM)
                    next_col_b.record_stream(DEFAULT_STREAM)
                next_rowptr_b = rowptr_batches[0].to(device=device, non_blocking=True).clone()
                next_col_b = col_batches[0].to(device=device, non_blocking=True)

            print("Fanout " + str(h) + " is " + str(fanout))
            for B in range(len(batch_sizes)):
                DEFAULT_STREAM.wait_stream(DATA_STREAM)
                rowptr_b = next_rowptr_b
                col_b = next_col_b
                
                if B < len(batch_sizes)-1:
                    # prefetch next batch.
                    with torch.cuda.stream(DATA_STREAM):
                        next_rowptr_b.record_stream(DEFAULT_STREAM)
                        next_col_b.record_stream(DEFAULT_STREAM)
                        next_rowptr_b = rowptr_batches[B+1].to(device=device, non_blocking=True).clone()
                        next_col_b = col_batches[B+1].to(device=device, non_blocking=True)

                rowptr_b -= rowptr_b[0].clone()
                
                node_transition_weights = torch.minimum(torch.ones_like(node_degrees), fanout / node_degrees)

                weighted_col_b = node_transition_weights[col_b]*p_hopwise_list[h][col_b]
                res_b = torch.log(1 - weighted_col_b)
                res_b = segment_csr(res_b, rowptr_b, reduce='add')
                #print("res_b size" + str(res_b.size()))
                res[batch_sizes_prefixsum[B]:batch_sizes_prefixsum[B+1]] = 1 - torch.exp(res_b)
            p_hopwise_list.append(res)
        total_probs = torch.ones(num_vertices, dtype=torch.float64)
        for h in range(len(fanouts)):
            total_probs = total_probs * (1-p_hopwise_list[h+1]).cpu()
        total_probs = 1-total_probs
        print("Num nonzero elements " + str(torch.count_nonzero(total_probs)))
        return total_probs
        #quit()
        probability_vectors.append(total_probs.cpu())
    return torch.stack(probability_vectors, dim=0)




def get_frequency_tensors(dataset, fanouts, partition_tensor):
    BATCH_SIZE = 1024
    num_parts = int(partition_tensor.max()+1)
    rowptr, col, value  = dataset.adj_t().csr()

    train_idx = dataset.split_idx['train']
    perm = partition_tensor[train_idx].argsort()
    train_parts = partition_tensor[train_idx]

    num_vertices = rowptr.size()[-1] - 1

    bincounts = torch.bincount(train_parts, minlength=num_parts)
    print(bincounts)
    bincounts_sum = torch.cat([torch.tensor([0]),bincounts.cumsum(dim=0)])
    print(bincounts_sum) 

    indices_for_partitions = []
    for i in range(num_parts):
        indices = torch.arange(0, train_idx.size()[0])[perm][bincounts_sum[i]:bincounts_sum[i+1]]
        indices = indices[torch.randperm(indices.size()[0])]
        indices_for_partitions.append(dataset.split_idx['train'][indices])
        assert((partition_tensor[indices_for_partitions[i]] != i).sum() == 0)




    probability_vectors = []
    for i in range(num_parts):
        node_degrees = rowptr[1:] - rowptr[:-1]
        node_degrees.to(torch.float64)
        
        #node_transition_weights = 
        p_hopwise_list = [torch.zeros(num_vertices, dtype=torch.float64)]

        # initialize p_hopwise for the training vertices in partition i
        p_hopwise_list[0][indices_for_partitions[i]] = (BATCH_SIZE * 1.0) / indices_for_partitions[i].size()[0]
       
        for h, fanout in enumerate(fanouts):
            node_transition_weights = torch.minimum(torch.ones_like(node_degrees), fanout / node_degrees)
            weighted_col = node_transition_weights[col] * p_hopwise_list[h][col]
            res = torch.log(1 - weighted_col)
            res = segment_csr(res, rowptr, reduce='add')
            res = 1 - torch.exp(res)
            p_hopwise_list.append(res)

        total_probs = torch.ones(num_vertices, dtype=torch.float64)
        for h in range(len(fanouts)):
            total_probs = total_probs * (1 - p_hopwise_list[h+1])
        total_probs = 1 - total_probs
        return total_probs
        probability_vectors.append(total_probs)
        print("Num nonzeros " + str(torch.count_nonzero(total_probs)))
    return torch.stack(probability_vectors, dim=0)




# Compute neighborhood sketches for each node in the
#   dataset graph.
def generate_cache(dataset, fanouts, partition_tensor, replication_factor,
                   cache_strategy='frequency', cache_filename = None,
                   empirical_frequency_iters=2):
    assert cache_strategy in ['frequency', 'frequency_analytic', 'frequency_analytic_notaylor',
                              'frequency_analytic_cascade', 'frequency_analytic_cascade_notaylor',
                              'frequency_analytic_linear_approx',
                              'degree', 'degree-reachable', 'numpaths-reachable',
                              'random-walk', 'shuffle', 'random-multiwalk', '1-hop']

    num_parts = int(partition_tensor.max()+1)
    if cache_filename != None:
        num_vertices = partition_tensor.size()[0] 
        # Check if the file already exists.
        my_file = Path(cache_filename+'_'+str(num_parts-1)+".pt")
        if my_file.is_file():
            print("Trying to use existing cache file")
            if cache_strategy == 'frequency_analytic':
                most_frequent_vertices = [None]*num_parts
                for i in range(num_parts):
                    most_frequent_vertices[i] = torch.load(cache_filename+"_"+str(i)+".pt")
                def get_cache_list(replication_factor):
                    cache_list = []
                    print("Cache sizes below")
                    for i in range(num_parts):
                        cache_list.append(most_frequent_vertices[i][:int(num_vertices*(replication_factor)/num_parts)])
                        print(str(cache_list[i].size()) + " / " + str(num_vertices))
                    return cache_list

                return lambda replication_factor : get_cache_list(replication_factor) 
            else:
                assert False, "Loading cache from file only supported for frequency analytic right now."
            
        

    rowptr, col, value  = dataset.adj_t().csr()

    train_idx = dataset.split_idx['train']
    perm = partition_tensor[train_idx].argsort()
    train_parts = partition_tensor[train_idx]

    num_vertices = rowptr.size()[-1] - 1

    bincounts = torch.bincount(train_parts, minlength=num_parts)
    print(bincounts)
    bincounts_sum = torch.cat([torch.tensor([0]),bincounts.cumsum(dim=0)])
    print(bincounts_sum) 

    indices_for_partitions = []
    for i in range(num_parts):
        indices = torch.arange(0, train_idx.size()[0])[perm][bincounts_sum[i]:bincounts_sum[i+1]]
        indices = indices[torch.randperm(indices.size()[0])]
        indices_for_partitions.append(dataset.split_idx['train'][indices])
        assert((partition_tensor[indices_for_partitions[i]] != i).sum() == 0)


    BATCH_SIZE = 1024
    if cache_strategy in ['frequency_analytic', 'frequency_analytic_notaylor',
                          'frequency_analytic_cascade', 'frequency_analytic_cascade_notaylor',
                          'random-multiwalk', 'frequency_analytic_linear_approx']:
        print("cache_strategy = " + cache_strategy)

        BATCH_SIZE = 1024

        most_frequent_vertices = []
        for i in range(num_parts):

            node_degrees = rowptr[1:] - rowptr[:-1]
            node_degrees = node_degrees.to(torch.float64)
            
            p_hopwise_list = [torch.zeros(num_vertices, dtype=torch.float64)]

            # initialize p_hopwise for the training vertices in partition i
            if cache_strategy == 'random-multiwalk':
                p_hopwise_list[0][indices_for_partitions[i]] = 1.0 / indices_for_partitions[i].size()[0]
            else:
                p_hopwise_list[0][indices_for_partitions[i]] = (BATCH_SIZE * 1.0) / indices_for_partitions[i].size()[0]
           
            print("Fanouts (not reversed) = " + str(fanouts))
            for h,fanout in enumerate(fanouts):
                node_transition_weights = torch.minimum(torch.ones_like(node_degrees), fanout / node_degrees)
                weighted_col = node_transition_weights[col] * p_hopwise_list[h][col]

                if cache_strategy in ['frequency_analytic', 'frequency_analytic_cascade', 'frequency_analytic_linear_approx']:
                    res = segment_csr(weighted_col, rowptr, reduce='add')
                    if cache_strategy == 'frequency_analytic_cascade':
                        res = res + p_hopwise_list[-1]
                    if cache_strategy != 'random-multiwalk':
                        res = 1 - torch.exp(-res)
                elif cache_strategy in ['frequency_analytic_notaylor', 'frequency_analytic_cascade_notaylor', 'random-multiwalk']:
                    res = torch.log(1 - weighted_col)
                    res = segment_csr(res, rowptr, reduce='add')
                    if cache_strategy == 'frequency_analytic_cascade_notaylor':
                        res = res + torch.log(1 - p_hopwise_list[-1])
                    res = 1 - torch.exp(res)
                elif cache_strategy == 'frequency_analytic_linear_approx':
                    res = segment

                p_hopwise_list.append(res)

            total_probs = torch.ones(num_vertices, dtype=torch.float64)
            for h in range(len(fanouts)):
                total_probs = total_probs * (1-p_hopwise_list[h+1])
            total_probs = 1-total_probs
            print("Number of external nonzero frequencies is " + torch.count_nonzero(torch.masked_select(total_probs, partition_tensor != i)), flush=True)
            most_frequent = torch.arange(0,num_vertices)[total_probs.argsort(descending=True)] 
            most_frequent_not_i = torch.masked_select(most_frequent, partition_tensor[most_frequent] != i)
            most_frequent_vertices.append(most_frequent_not_i)
            assert((partition_tensor[most_frequent_vertices[i]] == i).sum() == 0)

        if cache_filename != None:
            print("Saving the cached results to " + str(cache_filename))
            # save the results
            for i in range(num_parts):
                torch.save(most_frequent_vertices[i], cache_filename+"_"+str(i)+".pt")


        def get_cache_list(replication_factor):
            cache_list = []
            print("Cache sizes below")
            for i in range(num_parts):
                cache_list.append(most_frequent_vertices[i][:int(num_vertices*(replication_factor)/num_parts)])
                print(str(cache_list[i].size()) + " / " + str(num_vertices))
            return cache_list

        return lambda replication_factor : get_cache_list(replication_factor) 
    # end frequency_analytic

    if cache_strategy == 'random-walk':
        print("Caching strategy: random-walk")

        node_degrees = rowptr[1:] - rowptr[:-1]
        most_frequent_vertices = []

        node_transition_weights = 1 / node_degrees
        node_transition_weights[torch.isinf(node_transition_weights)] = 0

        for i in range(num_parts):
            print("Doing random walk for part " + str(i))

            p_upto_hops = torch.zeros(num_vertices, dtype=torch.float)
            p_upto_hops[indices_for_partitions[i]] = 1 / indices_for_partitions[i].size()[0]

            for _ in range(len(fanouts)):
                weighted_col = node_transition_weights[col] * p_upto_hops[col]
                p_upto_hops = p_upto_hops + segment_csr(weighted_col, rowptr, reduce='add')

            most_frequent = torch.arange(0, num_vertices)[p_upto_hops.argsort(descending=True)]
            most_frequent_not_i = torch.masked_select(most_frequent, partition_tensor[most_frequent] != i)
            most_frequent_vertices.append(most_frequent_not_i)
            assert((partition_tensor[most_frequent_vertices[i]] == i).sum() == 0)

        def get_cache_list(replication_factor):
            cache_list = []
            print("Cache sizes below")
            for i in range(num_parts):
                cache_list.append(most_frequent_vertices[i][:int(num_vertices*(replication_factor)/num_parts)])
                print(str(cache_list[i].size()) + " / " + str(num_vertices))
            return cache_list

        return lambda replication_factor : get_cache_list(replication_factor)

    if cache_strategy == 'numpaths-reachable':
        print("Caching strategy: numpaths-reachable")

        most_numpath_reachable_vertices = []

        for i in range(num_parts):
            print("Doing number of reachable paths for part " + str(i))
            numpaths = torch.zeros(num_vertices, dtype=torch.int)
            numpaths[indices_for_partitions[i]] = 1
            for _ in range(len(fanouts)):
                numpaths = numpaths + segment_csr(torch.ones_like(col) * numpaths[col],
                                                  rowptr, reduce='add')
            most_numpaths = torch.arange(0, num_vertices)[numpaths.argsort(descending=True)]
            most_numpaths_not_i = torch.masked_select(most_numpaths,
                                                      partition_tensor[most_numpaths] != i)
            most_numpath_reachable_vertices.append(most_numpaths_not_i)
            assert((partition_tensor[most_numpath_reachable_vertices[i]] == i).sum() == 0)

        def get_cache_list(replication_factor):
            cache_list = []
            for i in range(num_parts):
                cache_list.append(most_numpath_reachable_vertices[i][:int(num_vertices*(replication_factor)/num_parts)])
                print(str(cache_list[i].size()) + " / " + str(num_vertices))
            return cache_list

        return lambda replication_factor : get_cache_list(replication_factor)

    if cache_strategy=='degree-reachable':
        print("Caching strategy: degree-reachable")

        node_degrees = rowptr[1:] - rowptr[:-1]

        top_degree_reachable_vertices = []
        for i in range(num_parts):
            print("Doing degree reachable for part " + str(i))
            reachable = torch.zeros(num_vertices, dtype=torch.int)
            reachable[indices_for_partitions[i]] = 1
            for h in range(len(fanouts)):
                reachable = segment_csr(torch.ones_like(col) * reachable[col],
                                        rowptr, reduce='add')
                reachable[reachable != 0] = 1
            reachable_degrees = reachable * node_degrees
            top_degree_reachable = torch.arange(0, num_vertices)[reachable_degrees.argsort(descending=True)]
            top_degree_reachable_not_i = torch.masked_select(top_degree_reachable,
                                                             partition_tensor[top_degree_reachable] != i)
            top_degree_reachable_vertices.append(top_degree_reachable_not_i)
            assert((partition_tensor[top_degree_reachable_vertices[i]] == i).sum() == 0)

        def get_cache_list(replication_factor):
            cache_list = []
            for i in range(num_parts):
                cache_list.append(top_degree_reachable_vertices[i][:int(num_vertices*(replication_factor)/num_parts)])
                print(str(cache_list[i].size()) + " / " + str(num_vertices))
            return cache_list

        return lambda replication_factor : get_cache_list(replication_factor)
    # end degree-reachable


    if cache_strategy=='degree':
        node_degrees = rowptr[1:] - rowptr[:-1]
        print("Node degrees below.")
        print(node_degrees)
        #print(node_degrees.to(torch.float))
        top_5percent = torch.arange(0,num_vertices)[node_degrees.argsort(descending=True)[:int(replication_factor * num_vertices * 1.0 / num_parts)]]
        cache_list = []
        print("Cache sizes below")
        def get_cache_list(replication_factor):
            cache_list = []
            for i in range(num_parts):
                external_node_degrees = torch.masked_select(node_degrees, partition_tensor != i)        
                external_node_ids = torch.masked_select(torch.arange(0, num_vertices), partition_tensor != i)
                cache_list.append(external_node_ids[external_node_degrees.argsort(descending=True)[:int(replication_factor * num_vertices * 1.0 / num_parts)]]) 
                assert((partition_tensor[cache_list[i]] == i).sum() == 0)
                print(cache_list[i].size())
        
            return cache_list

        return lambda replication_factor : get_cache_list(replication_factor)
    # end degree

    if cache_strategy == 'shuffle':
        print("Cache strategy: shuffle")
        shuffled_vertices = []
        for i in range(num_parts):
            shuffled = torch.randperm(num_vertices)
            shuffled_not_i = torch.masked_select(shuffled, partition_tensor[shuffled] != i)
            shuffled_vertices.append(shuffled_not_i)
            assert((partition_tensor[shuffled_vertices[i]] == i).sum() == 0)

        def get_cache_list(replication_factor):
            cache_list = []
            print("Cache sizes below")
            for i in range(num_parts):
                cache_list.append(shuffled_vertices[i][:int(num_vertices*(replication_factor)/num_parts)])
                print(str(cache_list[i].size()) + " / " + str(num_vertices))
            return cache_list

        return lambda replication_factor : get_cache_list(replication_factor)


    if cache_strategy=='1-hop':
        print("Begin cache sizes for 1-hop cache")
        # begin 1-hop
        cache_list = []
        for i in range(num_parts):
            # obtain the subset of the vertices in col that are adjacent to vertices in partition i

            # Obtains the entries of 'col' that represent vertices adjacent to partition i.
            masked_col = torch.masked_select(col, gather_csr((partition_tensor == i).to(torch.int), rowptr).to(torch.bool))

            # next, deduplicate them.
            masked_col = masked_col.unique()

            # now remove the entries that are already in partition-i.
            masked_col = torch.masked_select(masked_col, partition_tensor[masked_col] != i)
            cache_list.append(masked_col)
            print(cache_list[i].size())

        return cache_list
    #end 1-hop
















    frequency_info_list = []
    for i in range(num_parts):
        frequency_info_list.append(torch.zeros_like(partition_tensor))

    sampler_list = []
    for ITER in range(empirical_frequency_iters):
        print("Iter: " + str(ITER))
        tic = time.perf_counter()
        for i in range(num_parts):
            sampler = NeighborSampler(dataset.adj_t(), node_idx=indices_for_partitions[i][torch.randperm(indices_for_partitions[i].size()[0])],
                                      batch_size=1024, sizes=fanouts, num_workers=20, pin_memory=True,
                                      return_e_id = False)
            for x in sampler:
                batch_size, n_ids, adjs = x

                mask = torch.masked_select(n_ids, partition_tensor[n_ids] != i)
                if mask.size()[0] == 0:
                    continue
                frequency_info_list[i][mask] += torch.ones_like(mask)
        toc = time.perf_counter()
        print("- time = " + str(toc - tic))

    for i in range(num_parts):
        print(frequency_info_list[i])

    def get_cache_list(replication_factor):
        cache_list = []
        print("Cache sizes below")
        for i in range(num_parts):
            #cache_list.append(torch.arange(0, num_vertices)[frequency_info_list[i].argsort()][int(bincounts[i]*(1-replication_factor)):])
            cache_list.append(torch.arange(0, num_vertices)[frequency_info_list[i].argsort(descending=True)][:int(num_vertices*(replication_factor)/num_parts)])
            print(frequency_info_list[i][frequency_info_list[i].argsort(descending=True)])
            print(str(cache_list[i].size()) + " / " + str(num_vertices))
        return cache_list


    return lambda replication_factor : get_cache_list(replication_factor) 

    return cache_list









# Compute neighborhood sketches for each node in the
#   dataset graph.
def evaluate_quality(dataset, fanouts, _partition_tensor, cache_list=None):
    rowptr, col, value  = dataset.adj_t().csr()

    num_parts = int(_partition_tensor.max()+1)
    original_partition_tensor = _partition_tensor.clone()

    real_num_parts = num_parts
    if cache_list != None:
        num_parts += 1

    train_idx = dataset.split_idx['train']
    perm = original_partition_tensor[train_idx].argsort()
    train_parts = original_partition_tensor[train_idx]

    num_vertices = rowptr.size()[-1] - 1

    bincounts = torch.bincount(train_parts, minlength=num_parts)
    print(bincounts)
    bincounts_sum = torch.cat([torch.tensor([0]),bincounts.cumsum(dim=0)])
    print(bincounts_sum) 


    indices_for_partitions = []


    for i in range(real_num_parts):
        indices = torch.arange(0, train_idx.size()[0])[perm][bincounts_sum[i]:bincounts_sum[i+1]]
        indices = indices[torch.randperm(indices.size()[0])]
        indices_for_partitions.append(dataset.split_idx['train'][indices])
        assert((original_partition_tensor[indices_for_partitions[i]] != i).sum() == 0)
        #print(partition_tensor[torch.arange(0, num_vertices)[perm]][bincounts_sum[i]:bincounts_sum[i+1]])




    cross_partition_communication = torch.zeros_like(bincounts)
    cross_partition_communication_uncached = torch.zeros_like(bincounts)
    total_partition_communication = torch.zeros_like(bincounts)
    internal_partition_communication = torch.zeros_like(bincounts)
    cache_partition_usage = torch.zeros_like(bincounts)

    sampler_list = []
    for i in range(real_num_parts):
        partition_tensor = _partition_tensor.clone()
        if cache_list != None:
            partition_tensor[cache_list[i]] = real_num_parts
            cache_mask = torch.ones_like(torch.arange(0, num_vertices))
            cache_mask[cache_list[i]] = torch.zeros([cache_list[i].size()[0]], dtype=torch.long)

        #TODO(TFK): Need to shuffle.
        sampler = NeighborSampler(dataset.adj_t(), node_idx=indices_for_partitions[i][torch.randperm(indices_for_partitions[i].size()[0])],
                                  batch_size=1024, sizes=fanouts, num_workers=20, pin_memory=True,
                                  return_e_id = False)
        #print("Partition " + str(i))
        for x in sampler:
            batch_size, n_ids, adjs = x
            counts = torch.bincount(partition_tensor[n_ids], minlength=num_parts)
            #print(counts)
            counts_uncached = torch.bincount(original_partition_tensor[n_ids], minlength=num_parts)
            #print("counts = " + str(counts))
            total_partition_communication[i] += n_ids.size()[0]#counts[j]
            #print("n_ids_size = " + str(n_ids.size()[0]))
            for j in range(0, num_parts):
                #print("part " + str(j) + " : " + str(counts[j]))
                #print("counts["+str(j)+"] = " + str(counts[j]))
                if j == i:
                    continue
                cross_partition_communication_uncached[i] += counts_uncached[j]
                if j == num_parts-1 and cache_list != None:
                    continue
                #print("Adding to cross_partition communication " + str(counts[j]))
                cross_partition_communication[i] += counts[j]
            #print("Adding to internal communication counts[i] = " + str(counts[i]) + " and counts[num_parts-1] = " + str(counts[num_parts-1]))
            if cache_list != None:
                internal_partition_communication[i] += counts[i] + counts[num_parts-1]
            else:
                internal_partition_communication[i] += counts[i]
            #print(counts_uncached)
            #print(counts)
            if cache_list != None:
                cache_partition_usage[i] += counts_uncached.sum() - counts_uncached[i] - (counts.sum() - counts[i] - counts[num_parts-1])
            else:
                cache_partition_usage[i] += counts_uncached.sum() - counts_uncached[i] - (counts.sum() - counts[i])
    print("Total neighborhood size sums:")

    total = int(total_partition_communication.sum())

    cross = int(cross_partition_communication.sum())
    internal = int(internal_partition_communication.sum())
    cache_hits = int(cache_partition_usage.sum())
    print(total_partition_communication)
    print("Cross-partition neighborhood size sums:")
    print(cross_partition_communication)
    print(str(100*(1.0*cross)/(1.0*total)) + "% miss-rate")
    print(str(100*(1.0*internal)/(1.0*total)) + "% hit-rate")
    print(str(100*(1.0*cache_hits)/(1.0*total)) + "% cache hit-rate")


    result_dict = dict()
    result_dict["total"] = total
    result_dict["internal"] = internal
    result_dict["cross"] = cross
    result_dict["cache_hits"] = cache_hits
    return result_dict


# Compute neighborhood sketches for each node in the
#   dataset graph.
def refine_partition(dataset, fanouts, partition_tensor, replication_factor):
    rowptr, col, value  = dataset.adj_t().csr()

    num_parts = int(partition_tensor.max()+1)
    train_idx = dataset.split_idx['train']
    perm = partition_tensor[train_idx].argsort()
    train_parts = partition_tensor[train_idx]

    num_vertices = rowptr.size()[-1] - 1

    bincounts = torch.bincount(train_parts, minlength=num_parts)
    print(bincounts)
    bincounts_sum = torch.cat([torch.tensor([0]),bincounts.cumsum(dim=0)])
    print(bincounts_sum) 

    indices_for_partitions = []
    for i in range(num_parts):
        indices = torch.arange(0, train_idx.size()[0])[perm][bincounts_sum[i]:bincounts_sum[i+1]]
        indices = indices[torch.randperm(indices.size()[0])]
        indices_for_partitions.append(train_idx[indices])
        assert((partition_tensor[indices_for_partitions[i]] != i).sum() == 0)

    neighborhood_dist = torch.zeros([num_vertices, num_parts], dtype=torch.long)
    sampler_list = []
    for ITER in range(1):
        print("Iter: " + str(ITER))
        for i in range(num_parts):
            print("Part " + str(i))
            sampler = NeighborSampler(dataset.adj_t(), node_idx=indices_for_partitions[i][torch.randperm(indices_for_partitions[i].size()[0])],
                                      batch_size=1024, sizes=fanouts, num_workers=20, pin_memory=True,
                                      return_e_id = False)
            print(partition_tensor[indices_for_partitions[i]])
            pids_1hot = F.one_hot(partition_tensor, num_classes=num_parts)
            for x in sampler:
                batch_size, n_ids, adjs = x
                #print(partition_tensor[n_ids[:batch_size]])
                pids = partition_tensor[n_ids]
                end_size = adjs[-1][-1][1]

                pids_1hot_list = []
                pids_1hot_list.append(F.one_hot(pids, num_classes=num_parts))

                for j, (edge_index, _, size) in enumerate(adjs):
                    pids_next = torch.zeros([n_ids.size()[0], num_parts], dtype=torch.long)

                    rowptr, col, vals = edge_index.csr()
                    sampled_pids = pids_1hot_list[j][col]
                    reduced_values = segment_csr(sampled_pids, rowptr, reduce='add')
                    
                    pids_next[:size[1]] = reduced_values #torch.minimum(reduced_values, sliced_samples[:size[1]])
                    pids_1hot_list.append(pids_next)

                pids_1hot[n_ids[:end_size]] = pids_1hot_list[-1][:end_size]


            info = pids_1hot[indices_for_partitions[i]]

            #print(info)
            values,inx=torch.max(info, 1)
            #print(values)
            #print(info[:,i])
            delta = values-info[:,i]

            change_mask = torch.masked_select(indices_for_partitions[i], (delta > 0))
            new_pids = torch.masked_select(inx, (delta > 0))
            
            partition_tensor[change_mask] = new_pids
    return partition_tensor

    def get_cache_list(replication_factor):
        cache_list = []
        print("Cache sizes below")
        for i in range(num_parts):
            #cache_list.append(torch.arange(0, num_vertices)[frequency_info_list[i].argsort()][int(bincounts[i]*(1-replication_factor)):])
            cache_list.append(torch.arange(0, num_vertices)[frequency_info_list[i].argsort(descending=True)][:int(num_vertices*(replication_factor)/num_parts)])
            print(frequency_info_list[i][frequency_info_list[i].argsort(descending=True)])
            print(str(cache_list[i].size()) + " / " + str(num_vertices))
        return cache_list


    return lambda replication_factor : get_cache_list(replication_factor) 

    return cache_list






