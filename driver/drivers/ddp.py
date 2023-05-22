import os
import time
from typing import NamedTuple
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import gc
from . import BaseDriver
from fast_trainer.shufflers import DistributedShuffler, FederatedDistributedShuffler
from fast_trainer.utils import Timer

from ..dataset import DisjointPartFeatReorderedDataset

from fast_sampler import RangePartitionBook, Cache
#from torch._C._distributed_c10d import ProcessGroupNCCL


import torch_sparse
from torch_scatter import segment_csr, gather_csr


@torch.no_grad()
def get_frequency_tensors_fast32(dataset, fanouts, partition_tensor, device, my_rank, _batch_size):
    BATCH_SIZE = _batch_size
    num_parts = int(partition_tensor.max()+1)
    rowptr, col, value  = dataset.adj_t().csr()

    num_vertices = rowptr.size()[-1] - 1

    indices_for_partitions = []
    for i in range(num_parts):
        indices = dataset.split_idx_parts[dist.get_rank()]['train'] 
        indices_for_partitions.append(indices)
        assert((partition_tensor[indices_for_partitions[i]] != i).sum() == 0)

    DATA_STREAM = torch.cuda.Stream(device)
    DEFAULT_STREAM = torch.cuda.default_stream(device)
    probability_vectors = []
    for i in [my_rank]:
        node_degrees = rowptr[1:] - rowptr[:-1]
        node_degrees = node_degrees.to(torch.float64).to(device=device)
        indices_for_partitions[i] = indices_for_partitions[i].to(device=device)

        #node_transition_weights =
        p_hopwise_list = [torch.zeros(num_vertices, dtype=torch.float64, device=device)]

        # initialize p_hopwise for the training vertices in partition i
        p_hopwise_list[0][indices_for_partitions[i]] = (BATCH_SIZE * 1.0) / indices_for_partitions[i].size()[0]


        reversed_fanouts = reversed(fanouts)

        VIP_BATCH_SIZE = min(4096*200, num_vertices)

        batch_sizes = [VIP_BATCH_SIZE]*(num_vertices//(VIP_BATCH_SIZE))
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

        time_start = time.perf_counter()
        for h,fanout in enumerate(fanouts):
            print("Fanout : " + str(fanout), flush=True)
            res = torch.zeros(num_vertices, dtype=torch.float64, device=device)
            with torch.cuda.stream(DATA_STREAM):
                if h > 0:
                    next_rowptr_b.record_stream(DEFAULT_STREAM)
                    next_col_b.record_stream(DEFAULT_STREAM)
                next_rowptr_b = rowptr_batches[0].to(device=device, non_blocking=True).clone()
                next_col_b = col_batches[0].to(device=device, non_blocking=True)

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

                #rowptr_b -= rowptr_b[0].clone()
                #node_transition_weights = torch.minimum(torch.ones_like(node_degrees), fanout / node_degrees)
                #weighted_col_b = node_transition_weights[col_b]*p_hopwise_list[h][col_b]
                #res_b = segment_csr(weighted_col_b, rowptr_b, reduce='add')

                #res[batch_sizes_prefixsum[B]:batch_sizes_prefixsum[B+1]] = 1 - torch.exp(-res_b)

                rowptr_b -= rowptr_b[0].clone()
                node_transition_weights = torch.minimum(torch.ones_like(node_degrees), fanout / node_degrees)
                weighted_col_b = node_transition_weights[col_b]*p_hopwise_list[h][col_b]
                res_b = torch.log(1 - weighted_col_b)
                res_b = segment_csr(res_b, rowptr_b, reduce='add')
                res[batch_sizes_prefixsum[B]:batch_sizes_prefixsum[B+1]] = 1 - torch.exp(res_b)

                del rowptr_b, col_b
            p_hopwise_list.append(res)
        total_probs = torch.ones(num_vertices, dtype=torch.float64)
        for h in range(len(fanouts)):
            total_probs = total_probs * (1-p_hopwise_list[h+1]).cpu()



        total_probs = 1-total_probs
        probability_vectors.append(total_probs.cpu())

        time_end = time.perf_counter()
        print("total time is " + str(time_end-time_start))
    return probability_vectors[0]


@torch.no_grad()
def get_frequency_tensors_fast(dataset, fanouts, partition_tensor, device, my_rank, _batch_size):
    BATCH_SIZE = _batch_size
    num_parts = int(partition_tensor.max()+1)
    rowptr, col, value  = dataset.adj_t().csr()

    num_vertices = rowptr.size()[-1] - 1

    indices_for_partitions = []
    for i in range(num_parts):
        indices = dataset.split_idx_parts[dist.get_rank()]['train'] 
        indices_for_partitions.append(indices)
        assert((partition_tensor[indices_for_partitions[i]] != i).sum() == 0)

    DATA_STREAM = torch.cuda.Stream(device)
    DEFAULT_STREAM = torch.cuda.default_stream(device)
    probability_vectors = []
    for i in [my_rank]:
        node_degrees = rowptr[1:] - rowptr[:-1]
        node_degrees = node_degrees.to(torch.float64).to(device=device)
        indices_for_partitions[i] = indices_for_partitions[i].to(device=device)

        p_hopwise_list = [torch.zeros(num_vertices, dtype=torch.float64, device=device)]

        # initialize p_hopwise for the training vertices in partition i
        p_hopwise_list[0][indices_for_partitions[i]] = (BATCH_SIZE * 1.0) / indices_for_partitions[i].size()[0]

        reversed_fanouts = reversed(fanouts)

        VIP_BATCH_SIZE = min(4096*100, num_vertices)
        batch_sizes = [VIP_BATCH_SIZE]*(num_vertices//(VIP_BATCH_SIZE))
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

        time_start = time.perf_counter()
        for h,fanout in enumerate(fanouts):
            res = torch.zeros(num_vertices, dtype=torch.float64, device=device)
            with torch.cuda.stream(DATA_STREAM):
                if h > 0:
                    next_rowptr_b.record_stream(DEFAULT_STREAM)
                    next_col_b.record_stream(DEFAULT_STREAM)
                next_rowptr_b = rowptr_batches[0].to(device=device, non_blocking=True).clone()
                next_col_b = col_batches[0].to(device=device, non_blocking=True)

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

                # Non-approximate. 
                #rowptr_b -= rowptr_b[0].clone()
                #node_transition_weights = torch.minimum(torch.ones_like(node_degrees), fanout / node_degrees)
                #weighted_col_b = node_transition_weights[col_b]*p_hopwise_list[h][col_b]
                #res_b = torch.log(1 - weighted_col_b)
                #res_b = segment_csr(res_b, rowptr_b, reduce='add')
                #res[batch_sizes_prefixsum[B]:batch_sizes_prefixsum[B+1]] = 1 - torch.exp(res_b)

                # Taylor approximation. There is not a good reason, as far as we know, for using the taylor approximation.
                #    and we'll likely switch back to using the commented out variant above.
                rowptr_b -= rowptr_b[0].clone()
                node_transition_weights = torch.minimum(torch.ones_like(node_degrees), fanout / node_degrees)
                weighted_col_b = node_transition_weights[col_b]*p_hopwise_list[h][col_b]
                res_b = segment_csr(weighted_col_b, rowptr_b, reduce='add')

                res[batch_sizes_prefixsum[B]:batch_sizes_prefixsum[B+1]] = 1 - torch.exp(-res_b)



                del rowptr_b, col_b
            p_hopwise_list.append(res)
        total_probs = torch.ones(num_vertices, dtype=torch.float64)
        for h in range(len(fanouts)):
            total_probs = total_probs * (1-p_hopwise_list[h+1]).cpu()

        total_probs = 1-total_probs
        probability_vectors.append(total_probs.cpu())

        time_end = time.perf_counter()
        print("total time is " + str(time_end-time_start))
    return probability_vectors[0]

def get_partitioned_dataset(dataset_name, root, rank):
    assert dataset_name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']
    return DisjointPartFeatReorderedDataset.from_path(root, dataset_name, rank)

def set_master(addr: str, port=1884):
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = str(port)


class DDPConfig(NamedTuple):
    node_num: int
    num_devices_per_node: int
    total_num_nodes: int

    @property
    def world_size(self):
        return self.total_num_nodes * self.num_devices_per_node


def get_ddp_config(job_dir: str, total_num_nodes: int,
                   num_devices_per_node: int):
    assert total_num_nodes > 0
    assert num_devices_per_node > 0

    my_node_name = str(os.environ['SLURMD_NODENAME'])
    print(f'{my_node_name} starting', flush=True)

    print("DDP dir is " + str(job_dir), flush=True)
    while True:
        node_list = os.listdir(job_dir)
        print("Node list is " + str(node_list), flush=True)
        print("Length is " + str(len(node_list)) + " and waiting for "
              + str(total_num_nodes), flush=True)
        if len(node_list) == total_num_nodes:
            break
        time.sleep(1)

    node_list = sorted(node_list)

    for i in range(len(node_list)):
        if my_node_name == node_list[i]:
            node_num = i
            break
    else:
        raise ValueError(f'Unable to find {my_node_name} in {node_list}')

    set_master(node_list[0])

    return DDPConfig(node_num=node_num,
                     num_devices_per_node=num_devices_per_node,
                     total_num_nodes=len(node_list))


class DDPDriver(BaseDriver):
    global_rank: int
    ddp_cfg: DDPConfig

    def __init__(self, args, device, model_type, dataset, ddp_cfg: DDPConfig):
        assert args.train_type == 'serial'

        #pg_options = ProcessGroupNCCL.Options()
        pg_options = torch.distributed.ProcessGroupNCCL.Options()
        pg_options.is_high_priority_stream = True

        self.ddp_cfg = ddp_cfg
        self.global_rank = (
            ddp_cfg.node_num * ddp_cfg.num_devices_per_node + device.index)
        dist.init_process_group(
            'nccl', rank=self.global_rank, world_size=ddp_cfg.world_size, pg_options=pg_options)
        self.orig_model = None

        if args.distribute_data:
           print("Loading dataset", flush=True)
           with Timer('Done loading partitioned dataset'):
               dataset = get_partitioned_dataset(args.dataset_name, args.dataset_root, dist.get_rank())

        super().__init__(args, [device], dataset, model_type)

        if args.distribute_data:
            if args.load_balance_scheme == "fully_random":
                print('Using DistributedShuffler', flush=True)
                self.train_shuffler = DistributedShuffler(
                    dataset.split_idx['train'], ddp_cfg.world_size)
            elif args.load_balance_scheme == "federated":
                # Only train on locally available training indices.
                print('Using FederatedDistributedShuffler', flush=True)
                self.train_shuffler = FederatedDistributedShuffler(
                    dataset.split_idx_parts[dist.get_rank()]['train'])
            else:
                raise ValueError("Supported load balance schemes are 'fully_random' & 'federated'.")
        else:
            self.train_shuffler = DistributedShuffler(
                self.dataset.split_idx['train'], ddp_cfg.world_size)
            self.test_shuffler = DistributedShuffler(
                self.dataset.split_idx['test'], ddp_cfg.world_size)
            self.valid_shuffler = DistributedShuffler(
                self.dataset.split_idx['valid'], ddp_cfg.world_size)

        self.reset()

    def __del__(self):
        dist.destroy_process_group()

    def _reset_model(self):
        if self.orig_model is None:
            self.orig_model = self.model
        self.orig_model.reset_parameters()
        print("before init ddp", flush=True)
        self.model = DistributedDataParallel(
            self.orig_model, device_ids=[self.main_device], broadcast_buffers=True)#, process_group=self.gradient_group)#, process_group=self.gradient_group)  # , find_unused_parameters=True)
        print("after init ddp", flush=True)

    def get_idx_test(self, name):
        if self.args.distribute_data:
            if name == 'test':
                #return self.test_shuffler.get_idx(self.global_rank)
                # not shuffling
                return self.dataset.split_idx_parts[dist.get_rank()][name]
            elif name == 'valid':
                #return self.valid_shuffler.get_idx(self.global_rank)
                # not shuffling
                return self.dataset.split_idx_parts[dist.get_rank()][name]
            else:
                raise ValueError('invalid test dataset name')
        else:
            if name == 'test':
                return self.test_shuffler.get_idx(self.global_rank)
            elif name == 'valid':
                return self.valid_shuffler.get_idx(self.global_rank)

    def get_idx(self, epoch: int):
        self.train_shuffler.set_epoch(10000*self.TRIAL_NUM + epoch)
        if self.args.distribute_data:
            if self.args.load_balance_scheme == "fully_random":
                idx = self.train_shuffler.get_idx(self.global_rank)
            elif self.args.load_balance_scheme == "federated":
                idx = self.train_shuffler.get_idx()
            else:
                raise ValueError('Error getting idx.')
        else:
            return self.train_shuffler.get_idx(self.global_rank)

        return idx.reshape(-1)

    # CACHING COMPONENT
    # HACK, since life cycle of the FastSampler is only one epoch, and the stats are collected in FastSampler,
    #   the way we run for multiple epochs is by running a single epoch of size = normal epoch size * num_simulated_epochs
    # Optimizing for training, so use the same shuffler as used for training.
    def get_create_cache_idx(self, num_simulated_epochs: int):
        idxs = []
        for epoch in range(num_simulated_epochs):
            # Picking arbitrary prime numbers.
            self.train_shuffler.set_epoch(17*self.TRIAL_NUM + 19*epoch + 311)
            if self.args.load_balance_scheme == "fully_random":
                idx = self.train_shuffler.get_idx(self.global_rank)
            elif self.args.load_balance_scheme == "federated":
                idx = self.train_shuffler.get_idx()
            else:
                raise ValueError('Error getting idx.')
            idxs.append(idx)
        return torch.cat(idxs, dim=0)

    def get_sampler(self, _seed):
        return torch.utils.data.distributed.DistributedSampler(
            self.dataset.split_idx['train'],
            num_replicas=self.ddp_cfg.world_size,
            rank=self.global_rank, seed=_seed)

    @property
    def my_name(self):
        return f'{super().my_name}_{self.ddp_cfg.node_num}_{self.main_device.index}'

    @property
    def is_main_proc(self):
        return self.global_rank == 0

    def create_vip_cache(self, num_cache_creation_epochs, cache_size) -> None:

        # Misnomer, this is the replication factor.
        # Divide by 100 as passed in as a percentage.
        num_features_to_cache = int(self.dataset.num_nodes / self.dataset.num_parts * (cache_size/100))

        partition_book = self.dataset.get_RangePartitionBook()

        if self.args.cache_strategy == "vip":
            print("USING vip for cache strategy", flush=True)
            num_vertices = self.dataset.num_nodes
                
            # VIP ordering
            partition_ids = partition_book.nid2partid(torch.arange(0, num_vertices))

            frequencies = get_frequency_tensors_fast(self.dataset, self.args.train_fanouts, partition_ids, self.devices[0], dist.get_rank(), self.args.train_batch_size)
            external_frequencies = torch.masked_select(frequencies, partition_ids != dist.get_rank())
            external_node_ids = torch.masked_select(torch.arange(0,num_vertices), partition_ids != dist.get_rank())

            num_nonzero = torch.count_nonzero(external_frequencies).item()
            num_features_to_cache = min(num_nonzero, num_features_to_cache)
            #remote_vertices_to_cache = torch.tensor(external_node_ids[external_frequencies.argsort(descending=True, stable=True)])[:num_features_to_cache]
            remote_vertices_to_cache = torch.tensor(external_node_ids[external_frequencies.argsort(descending=True)])[:num_features_to_cache]
            del partition_ids, frequencies, external_frequencies, external_node_ids
            # Attempt to mitigate influence of CUDACachingAllocator on remainder of computation.
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache() 
            gc.collect()
            torch.cuda.synchronize()
            #gc.disable()
        elif self.args.cache_strategy == "simulation":
            epoch = 0 # Part of the HACK.

            self.create_cache_loader.idx = self.get_create_cache_idx(num_cache_creation_epochs)
            it = iter(self.create_cache_loader)

            if self.is_main_proc:
                pbar = tqdm(total=self.create_cache_loader.idx.numel())
                pbar.set_description(f'Cache Creation ({num_cache_creation_epochs} epochs)')

            def cb(inputs, results):
                if self.is_main_proc:
                    pbar.update(sum(batch.batch_size for batch in inputs))

            while True:
                try:
                    # Could optimize to not return anything..
                    #   But should just be references anyway.
                    _ = next(it)
                except StopIteration:
                    break

            if self.is_main_proc:
                self.log((epoch, it.get_stats()))
                pbar.close()
                del pbar

            ds = it.get_distributed_stats()
            print ("INFO BELOW", flush=True)
            print(str(ds.remote_vertices_ordered_by_freq.numel()), flush=True)
            print(str(num_features_to_cache), flush=True)
            num_features_to_cache = min(num_features_to_cache, ds.remote_vertices_ordered_by_freq.numel())
            remote_vertices_to_cache = ds.remote_vertices_ordered_by_freq[:num_features_to_cache]
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache() 
            gc.collect()
            torch.cuda.synchronize()
            #gc.disable()
        elif self.args.cache_strategy == "degree":
            # Degree ordering
            rowptr, col, value  = self.dataset.adj_t().csr()
            num_vertices = self.dataset.num_nodes
            node_degrees = rowptr[1:] - rowptr[:-1]
            partition_ids = partition_book.nid2partid(torch.arange(0, num_vertices))
            external_node_degrees = torch.masked_select(node_degrees, partition_ids != dist.get_rank())
            external_node_ids = torch.masked_select(torch.arange(0, num_vertices), partition_ids != dist.get_rank())
            remote_vertices_to_cache = external_node_ids[external_node_degrees.argsort(descending=False)[:num_features_to_cache]]
        else:
            print(f"Error invalid cache strategy provided: {self.args.cache_strategy}", flush=True)
            exit(1)

        # Parameters for communication.
        device = self.devices[0] 

        # Determine which partitions the remote vertices we want to cache are on.
        partition_ids = partition_book.nid2partid(remote_vertices_to_cache)
        partitioned_vertices_to_cache = [None] * dist.get_world_size()
        partitioned_vertices_to_cache_gpu = [None] * dist.get_world_size()
        for i in range(dist.get_world_size()):
            partitioned_vertices_to_cache[i] = remote_vertices_to_cache[(partition_ids == i).nonzero()]
            partitioned_vertices_to_cache_gpu[i] = partitioned_vertices_to_cache[i].to(device, non_blocking=False)

        # Make sure not trying to put vertices we already have locally in the cache.
        assert partitioned_vertices_to_cache[dist.get_rank()].numel() == 0

        # Parameters for communication.
        device = self.devices[0] 
        meta_dtype = torch.int32
        indices_dtype = torch.int64
        features_dtype = self.dataset.x.dtype
        feature_dim = self.dataset.x.size(dim=1)

        # Perform communication to get features from remote machine that we would like to cache.
        # META
        # Exchange information of how many vertices need from each other machine.
        meta_scatter = [torch.tensor([vertices.numel()], dtype=meta_dtype, device=device) for vertices in partitioned_vertices_to_cache]
        meta_gather = [torch.empty((1,), dtype=meta_dtype, device=device) for i in range(dist.get_world_size())]
        meta_handle = dist.all_to_all(meta_gather, meta_scatter, group=None, async_op=False)
        # INDICES / VERTICES
        # Exchange information for exactly which vertices need from other machines.
        indices_scatter = partitioned_vertices_to_cache_gpu
        indices_gather = [torch.empty(meta_gather[i].item(), dtype=indices_dtype, device=device) for i in range(dist.get_world_size())]
        indices_handle = dist.all_to_all(indices_gather, indices_scatter, group=None, async_op=False)
        # FEATURES 
        # Exchange feature information corresponding to desired vertices.
        features_gather = [torch.empty((meta_scatter[j], feature_dim), dtype=features_dtype, device=device) for j in range(dist.get_world_size())]
        features_scatter = [None for j in range(dist.get_world_size())]
        for i in range(dist.get_world_size()):
            all_idx = partition_book.nid2localnid(indices_gather[i], dist.get_rank()).to(torch.int64)
            gpu_idx = torch.masked_select(all_idx, all_idx < self.x_gpu.size()[0]).to(device=device)
            cpu_idx = torch.masked_select(all_idx, all_idx >= self.x_gpu.size()[0]) - self.x_gpu.size()[0]
            gpu_features = self.x_gpu[gpu_idx]
            cpu_features = self.x_cpu[cpu_idx.cpu()].to(device=device, non_blocking=True)

            gpu_positions = torch.masked_select(torch.arange(0,all_idx.numel(), device=device), all_idx < self.x_gpu.size()[0])
            cpu_positions = torch.masked_select(torch.arange(0,all_idx.numel(), device=device), all_idx >= self.x_gpu.size()[0])
            assert gpu_features.size()[0] + cpu_features.size()[0] == all_idx.numel()

            all_features = torch.zeros([gpu_features.size()[0] + cpu_features.size()[0], gpu_features.size()[1]], device=device, dtype=torch.float16)
            all_features[gpu_positions] = gpu_features
            all_features[cpu_positions] = cpu_features
            features_scatter[i] = all_features 
        features_handle = dist.all_to_all(features_gather, features_scatter, group=None, async_op=False)
        del features_scatter
        # Now cached after communication.
        # Now have features for remote vertices, they are cached.
        cached_vertices = torch.cat(partitioned_vertices_to_cache, dim=0)
        cached_features = torch.cat(features_gather, dim=0)

        # Create Cache Object which is accessible to the FastSampler.
        # This Cache will be passed to the FastSamplerThrough the FastSamplerConfig. 
        cache = Cache(dist.get_rank(), dist.get_world_size(), cached_vertices, cached_features)

        effective_cache_size = cached_vertices.numel()
        self.log(f"EFFECTIVE_CACHE_SIZE(numel={effective_cache_size})")
        effective_replication_factor = effective_cache_size / (self.dataset.num_nodes / self.dataset.num_parts)
        self.log(f"EFFECTIVE_REPLICATION_FACTOR(factor={effective_replication_factor})")

        # log the cache size, can then calculate the intended replication facotr
        # Updating something like self.cache is not sufficient, must make sure the config is updated.
        self.train_loader.cache = cache
        print("Made it past the cache stage", flush=True)
