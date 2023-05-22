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

    
