#import nvtx
import torch
import torch.distributed as dist


class Shuffler:
    initial_idx: torch.Tensor
    world_size: int
    initial_seed: int
    generator: torch.Generator
    epoch: int

    DEFAULT_INITIAL_SEED = 2147483647

    def __init__(self, idx, initial_seed=DEFAULT_INITIAL_SEED):
        assert idx.dim() == 1
        self.initial_idx = idx
        self.initial_seed = initial_seed
        self.generator = torch.Generator(device='cpu')
        self.set_epoch(0)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def get_idx(self):
        self.generator.manual_seed(self.initial_seed + self.epoch)
        return self.initial_idx[torch.randperm(self.initial_idx.numel(),
                                               generator=self.generator,
                                               device=self.initial_idx.device)]

class SubgraphShuffler:
    initial_idx: torch.Tensor
    world_size: int
    initial_seed: int
    generator: torch.Generator
    epoch: int

    DEFAULT_INITIAL_SEED = 2147483647

    # The sizes of idx and idx_to_subgraph_id tensors should match.
    # The idx_to_subgraph_id tensor is an integer tensor that contains the subgraph id for each vertex id in the idx tensor.
    def __init__(self, idx : torch.Tensor, idx_to_subgraph_id : torch.Tensor, initial_seed=DEFAULT_INITIAL_SEED):
        assert idx.dim() == 1
        # First we reorder the idx tensor so that subgraphs with the same ID are contiguous.
        sorted_indices = torch.argsort(idx_to_subgraph_id)
        reordered_idx = idx[sorted_indices]
        reordered_idx_to_subgraph_id = idx_to_subgraph_id[sorted_indices]
        subgraph_ids,subgraph_sizes = torch.unique_consecutive(reordered_idx_to_subgraph_id, return_counts=True)
        end_idx = torch.cumsum(subgraph_sizes, dim=-1)
        start_idx = torch.cat((torch.tensor([0]),torch.cumsum(subgraph_sizes, dim=-1)), dim=0).resize_(subgraph_sizes.size())
        idx_ranges = torch.stack([start_idx, end_idx]).t().to(dtype=torch.int)

        self.initial_idx = idx
        self.idx_ranges = idx_ranges
        self.num_subgraphs = idx_ranges.size()[0]
        self.initial_seed = initial_seed
        
        self.generator = torch.Generator(device='cpu')
        self.set_epoch(0)

    def get_num_subgraphs(self):
        return self.num_subgraphs

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def get_idx(self):
        self.generator.manual_seed(self.initial_seed + self.epoch)
        return self.initial_idx, self.idx_ranges[torch.randperm(self.num_subgraphs,
                                                                generator=self.generator)]

class DistributedShuffler(Shuffler):
    world_size: int

    def __init__(self, idx, world_size,
                 initial_seed=Shuffler.DEFAULT_INITIAL_SEED):
        super().__init__(idx, initial_seed)
        self.world_size = world_size

    def get_idx(self, rank):
        shuffled_idx = super().get_idx()
        n = shuffled_idx.numel()
        start = (n * rank) // self.world_size
        stop = (n * (rank + 1)) // self.world_size
        return shuffled_idx[start: stop]

class DistributedSubgraphShuffler(SubgraphShuffler):
    world_size: int

    def __init__(self, idx, idx_to_subgraph_id : torch.Tensor, world_size,
                 initial_seed=Shuffler.DEFAULT_INITIAL_SEED):
        super().__init__(idx, idx_to_subgraph_id, initial_seed)
        self.world_size = world_size

    def get_idx(self, rank):
        global_idx, global_idx_ranges = super().get_idx()
        n = global_idx_ranges.size()[0]
        if n % self.world_size != 0:
            global_idx_ranges = global_idx_ranges[n%self.world_size:]
            n = global_idx_ranges.size()[0]
        assert n % self.world_size == 0

        start = (n * rank) // self.world_size
        stop = (n * (rank + 1)) // self.world_size
        ret_idx_list = []
        ret_range_list = []
        total = 0
        for i in range(start, stop):
            ret_idx_list.append(global_idx[global_idx_ranges[i][0] : global_idx_ranges[i][1]])
            ret_range_list.append(torch.tensor([[total, total + global_idx_ranges[i][1] - global_idx_ranges[i][0]]]))
            total += global_idx_ranges[i][1] - global_idx_ranges[i][0]
        return torch.cat(ret_idx_list, dim=-1), torch.cat(ret_range_list, dim=-1) #global_idx_ranges[start : stop]

#class DistributedShuffler(Shuffler):
#    """
#    Minibatches are fully random. At the beginning of each epoch the rank 0 machine shuffles all training nodes and scatters
#    1/kth of the shuffled training indices to each of the k machines. Each iteration each machine will then compute a microbatch
#    composed of the next minibatch_size/k vertices it gathered
#    """
#
#    initial_idx: torch.Tensor
#    epoch: int
#    rank: int
#    world_size: int
#    device: torch.device 
#    generator: torch.Generator
#    initial_seed: int
#    DEFAULT_INITIAL_SEED = 2147483647
#
#    def __init__(self, local_idx, initial_seed=Shuffler.DEFAULT_INITIAL_SEED):
#        self.device = torch.device('cuda:0')
#        self.initial_idx = local_idx.to(self.device) # all_to_all is on GPU.
#        self.initial_seed = initial_seed
#        self.generator = torch.Generator(self.device) # Why on GPU? Does it matter?
#        self.world_size = dist.get_world_size()
#        self.set_epoch(0)
#
#    #@nvtx.annotate('get_idx', color='grey')
#    def get_idx(self, rank):
#        """
#        The rank 0 machine is in charge of shuffling and scattering training nodes to all machines.
#        Initial_idx must be all training nodes in the graph.
#        # MINOR WARNING: Splitting may lead to poorer performance in all_to_all, but as this is relatively tiny operation compared to the rest of the epoch, not a cause for concern.
#        """
#        self.generator.manual_seed(self.initial_seed + self.epoch)
#        recv_size = list(torch.tensor_split(self.initial_idx, self.world_size))[rank].numel()
#        machine_training_ids = torch.empty((recv_size,), dtype=self.initial_idx.dtype, device=self.device)
#        # NOTE: dist.scatter likely does not support different tensor sizes, all_to_all here is a workaround.
#        idx_gather = [machine_training_ids] + [torch.empty(1, device=self.device) for i in range(self.world_size - 1)]
#        if rank == 0:
#            idx_scatter = list(torch.tensor_split(self.initial_idx[torch.randperm(self.initial_idx.numel(), generator=self.generator, device=self.device)], self.world_size))
#        else:
#            idx_scatter = [torch.empty(1, device=self.device) for i in range(self.world_size)]
#        dist.all_to_all(idx_gather, idx_scatter, group=None, async_op=False)
#        # Transfer to CPU for Fast Sampler.
#        return machine_training_ids.to(torch.device('cpu'))
#
class FederatedDistributedShuffler(Shuffler):
    """
    Minibatches are not fully random. At the beginning of each epoch, each of the k machines shuffles the training ids on its
    partition, then each iteration each machine will compute a microbatch composed of the next minibatch_size/k vertices that it
    shuffled locally. Note: If the training nodes are not partitioned equally this can also lead to an uneven number of iterations/microbatches, which can be remedied
    by enforcing equal number of iterations but for example allowing different-sized microbatches.
    Should make sure that the initial_idx passed in are only the local training ids.
    """
    pass # The code is exactly the same as Shuffler.

    
