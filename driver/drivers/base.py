from abc import abstractmethod
from typing import Callable, List, Mapping, Type, Iterable
from ogb.nodeproppred import Evaluator
from pathlib import Path
from tqdm import tqdm
import time
import torch
import gc
import os
import sys
import importlib
if importlib.util.find_spec("torch_geometric.loader") is not None:
    import torch_geometric.loader
    if hasattr(torch_geometric.loader, "NeighborSampler"):
        from torch_geometric.loader import NeighborSampler
    else:
        from torch_geometric.data import NeighborSampler
else:
    from torch_geometric.data import NeighborSampler

import torch.distributed as dist

from ..dataset import FastDataset
from fast_trainer.utils import Timer, CUDAAggregateTimer, append_runtime_stats, start_runtime_stats_epoch, DataCollector
from fast_trainer.samplers import *
from fast_trainer.transferers import *
from fast_trainer.concepts import TrainImpl
from fast_trainer import train, test

# testing
from fast_trainer.partition_book import RangePartitionBookLoader
from fast_sampler import RangePartitionBook, Cache


class BaseDriver:
    devices: List[torch.device]
    dataset: FastDataset
    lr: float
    train_loader: ABCNeighborSampler
    train_transferer: Type[DeviceIterator]
    test_transferer: Type[DeviceIterator]
    train_impl: TrainImpl
    train_max_num_batches: int
    model: torch.nn.Module
    make_subgraph_loader: Callable[[torch.Tensor], Iterable[PreparedBatch]]
    #evaluator: Evaluator
    log_file: Path


    def get_num_trainers(self):
        # This is for purposes of calculating adjusted minibatch sizes
        # to get even # of minibatches among all trainers.
        return self.args.total_num_nodes * self.args.max_num_devices_per_node

    def __init__(self, args, devices: List[torch.device],
                 dataset: FastDataset, model_type: Type[torch.nn.Module]):
        assert torch.cuda.is_available()

        self.args = args
        self.devices = devices
        self.dataset = dataset
        self.model_type = model_type
        self.lr = args.lr
        self.log_file = Path(args.log_file)
        self.logs = []
        self.firstRun = True
        self.TRIAL_NUM = 0
        assert len(self.devices) > 0

        if args.train_type == 'serial' and len(self.devices) > 1:
            raise ValueError('Cannot serial train with more than one device.')

        minibatch_size = args.train_batch_size * self.get_num_trainers()
    
        self.create_cache = (args.distribute_data and args.cache_size > 0) and \
                            ((args.execution_mode == "computation" and "cache" in args.computation_mode) or \
                             (args.execution_mode == "simulation" and "cache" in args.communication_simulation_mode))

        # Create a cache.
        if self.create_cache and self.args.cache_strategy == "simulation":
            create_cache_cfg = FastSamplerConfig(
                # Do not need features and labels. 
                x_cpu=torch.zeros(0), x_gpu=torch.zeros(0), y=torch.zeros(0),
                rowptr=self.dataset.rowptr, col=self.dataset.col,
                # After the initial creation, this config object will be updated with a shuffled idx, can use placeholder data.
                idx=torch.zeros(0, dtype=torch.int64),
                # Creating cache running with the same parameters as training on.
                batch_size=args.train_batch_size,
                sizes=args.train_fanouts,
                skip_nonfull_batch=False,
                # Not transfering to GPU, no need to pin memory.
                # Unless later find that want to do some analysis on GPU for performance?
                pin_memory=False,
                distributed=args.distribute_data,
                partition_book = self.dataset.get_RangePartitionBook(),
                # Creating a cache, one does not exist. Pass in an empty placeholder.
                cache = Cache(),
                # Force the exact num batches if distributing data to have an equal number of iterations.
                force_exact_num_batches=True,#args.distribute_data,
                exact_num_batches=self.dataset.get_num_iterations(minibatch_size)['train'],
                # Count the remote request frequency to determine what to cache.
                count_remote_frequency=True,
                # Cannot use cache if not yet created.
                use_cache=False,
            )

        if self.args.distribute_data:
            #gpu_percent = min(max(0.001,self.args.gpu_percent), 0.999)
            gpu_percent = min(max(0,self.args.gpu_percent), 1.0)

            limit = int(self.dataset.x.size()[0]*gpu_percent)
            self.x_cpu = self.dataset.x[limit:]
            self.x_gpu = self.dataset.x[:limit].to(self.devices[0]) 
        else:
            self.x_cpu = self.dataset.x
            self.x_gpu = torch.empty(0)

        self.simulate_communication = args.execution_mode == "simulate_communication" and args.distribute_data
        self.compute = args.execution_mode == "computation" and not self.simulate_communication

        # Simulate communication instead of computation (training, test, etc.)
        if self.simulate_communication and self.args.distribute_data:
            simulate_cfg = FastSamplerConfig(
                # Do not need features and labels. 
                x_cpu=torch.zeros(0), x_gpu=torch.zeros(0), y=torch.zeros(0),
                rowptr=self.dataset.rowptr, col=self.dataset.col,
                # After the initial creation, this config object with a shuffled idx, can use placeholder data.
                idx=torch.zeros(0, dtype=torch.int64),
                # Simulating with same parameters as training on.
                batch_size=args.train_batch_size,
                sizes=args.train_fanouts,
                skip_nonfull_batch=False,
                # Not transfering to GPU, no need to pin memory.
                # Unless later find that want to do some analysis on GPU for performance?
                pin_memory=False,
                distributed = args.distribute_data,
                partition_book = self.dataset.get_RangePartitionBook(),
                # Value will be overwritten if a cache is created.
                cache = Cache(),
                # Force the exact num batches if distributing data to have an equal number of iterations.
                force_exact_num_batches=True,#args.distribute_data,
                exact_num_batches=self.dataset.get_num_iterations(minibatch_size)['train'],
                count_remote_frequency=False,
                # If a cache is created then use it.
                use_cache=self.create_cache,
            )
        elif self.compute and self.args.distribute_data:
            # TODO: Add 1D version of serial_idx kernel
            train_cfg = FastSamplerConfig(
                x_cpu=self.x_cpu, x_gpu=self.x_gpu, y=self.dataset.y.unsqueeze(-1),
                rowptr=self.dataset.rowptr, col=self.dataset.col,
                # After the initial creation, this config object with a shuffled idx, can use placeholder data.
                idx=torch.zeros(1, dtype=torch.int64),
                batch_size=args.train_batch_size,
                sizes=args.train_fanouts,
                skip_nonfull_batch=False,
                pin_memory=True,
                distributed = args.distribute_data,
                partition_book = self.dataset.get_RangePartitionBook(),
                # Value will be overwritten if a cache is created.
                cache = Cache(),
                # Force the exact num batches if distributing data to have an equal number of iterations.
                force_exact_num_batches=True,#args.distribute_data,
                exact_num_batches=self.dataset.get_num_iterations(minibatch_size)['train'],
                count_remote_frequency=False,
                # If a cache is created then use it.
                use_cache=self.create_cache,
            )
        elif self.compute:
            # TODO: Add 1D version of serial_idx kernel
            train_cfg = FastSamplerConfig(
                x_cpu=self.dataset.x, x_gpu=self.x_gpu, y=self.dataset.y.unsqueeze(-1),
                rowptr=self.dataset.rowptr, col=self.dataset.col,
                # After the initial creation, this config object with a shuffled idx, can use placeholder data.
                idx=torch.empty(self.dataset.split_idx['train'].numel(), dtype=torch.int64),
                batch_size=args.train_batch_size,
                sizes=args.train_fanouts,
                skip_nonfull_batch=False,
                pin_memory=True,
                distributed = args.distribute_data,
                partition_book = None,
                # Value will be overwritten if a cache is created.
                cache = Cache(),
                # Force the exact num batches if distributing data to have an equal number of iterations.
                force_exact_num_batches=True, #args.distribute_data,
                exact_num_batches=self.dataset.get_num_iterations(minibatch_size)['train'],
                count_remote_frequency=False,
                # If a cache is created then use it.
                use_cache=self.create_cache,
            )
        else:
            raise ValueError(f'Did not create a valid FastSamplerConfig for {args.execution_mode=}')

        self.train_max_num_batches = min(args.train_max_num_batches,
                                         train_cfg.get_num_batches())
        def make_loader(sampler, cfg: FastSamplerConfig):
            kwargs = dict()
            if sampler == 'NeighborSampler' and self.args.one_node_ddp:
                kwargs = dict(sampler=self.get_sampler(self.TRIAL_NUM*1000 +
                                                       self.global_rank),
                              persistent_workers=True)
            return {
                'FastPreSampler': lambda: FastPreSampler(cfg),
                'FastSampler': lambda: FastSampler(
                    args.num_workers, self.train_max_num_batches, cfg),
                'NeighborSampler': lambda: NeighborSampler(
                    self.dataset.adj_t(), node_idx=cfg.idx,
                    batch_size=cfg.batch_size, sizes=cfg.sizes,
                    num_workers=args.num_workers, pin_memory=True, **kwargs)
            }[sampler]()

        if self.create_cache and self.args.cache_strategy == "simulation":
            self.create_cache_loader = FastSampler(self.args.num_workers, self.train_max_num_batches, create_cache_cfg)
        if self.simulate_communication:
            self.simulate_loader = FastSampler(self.args.num_workers, self.train_max_num_batches, simulate_cfg)
            self.simulation_reset()
        if self.compute:
            self.train_loader = make_loader(args.train_sampler, train_cfg)
            if not args.distribute_data:
                self.train_transferer = DevicePrefetcher if args.train_prefetch \
                    else DeviceTransferer
                self.test_transferer = DevicePrefetcher if args.test_prefetch \
                    else DeviceTransferer
            else:
                self.train_transferer = DeviceDistributedPrefetcher
                self.test_transferer = DeviceDistributedPrefetcher
            self.train_impl = {'dp': train.data_parallel_train,
                               'serial': train.serial_train}[args.train_type]
            self.model = self.model_type(
                self.dataset.num_features, args.hidden_features,
                self.dataset.num_classes,
                num_layers=args.num_layers).to(self.main_device)
            self.model_noddp = self.model_type(
                self.dataset.num_features, args.hidden_features,
                self.dataset.num_classes,
                num_layers=args.num_layers).to(self.main_device)
            self.idx_arange = torch.arange(self.dataset.y.numel())
            #self.evaluator = Evaluator(name=args.dataset_name)
            self.reset()

    def __del__(self):
        if len(self.logs) > 0:
            raise RuntimeError('Had unflushed logs when deleting BaseDriver')
        # NOTE: Cannot always flush logs for the user.
        # It might be impossible if __del__ is called during
        # the shutdown phase of the interpreter...

        # self.flush_logs()

    def _reset_model(self):
        self.model.reset_parameters()
        #print("Reset model")

    def _reset_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        #print("Reset optimizer")

    def simulation_reset(self):
        self.TRIAL_NUM += 1

    def reset(self):
        self._reset_model()
        self._reset_optimizer()
        self.TRIAL_NUM += 1

    @property
    def my_name(self) -> str:
        return self.args.job_name

    @property
    @abstractmethod
    def is_main_proc(self) -> bool:
        ...

    @property
    def main_device(self) -> torch.device:
        return self.devices[0]

    def get_idx_test(self) -> None:
        return self.dataset.split_idx['test']

    def make_train_devit(self) -> DeviceIterator:
        return self.train_transferer(self.devices, iter(self.train_loader), pipeline_on = not self.args.pipeline_disabled)

    def log(self, t) -> None:
        self.logs.append(t)
        if self.is_main_proc and self.args.verbose:
            print(str(t))

    def flush_logs(self) -> None:
        if len(self.logs) == 0:
            return

        with self.log_file.open('a') as f:
            f.writelines(repr(item) + '\n' for item in self.logs)
        self.logs = []

    def train(self, epochs, data_collector: DataCollector = None) -> None:
        self.model.train()
        if self.args.model_name.lower() == "sageresinception" or \
           self.args.use_lrs:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer, factor=0.8,
                patience=self.args.patience, verbose=True)
        else:
            lr_scheduler = None

        def record_sampler_init_time(x):
            self.log(x)
            # append_runtime_stats("sampler init", x.nanos/1000000)

        for epoch in epochs:
            #print("", flush=True)
            sys.stdout.flush()
            if dist.is_initialized():
                dist.barrier()
            start_runtime_stats_epoch()
            ctimer_preamble = CUDAAggregateTimer("Preamble")
            ctimer_preamble.start()

            with Timer((epoch, 'Preamble'), record_sampler_init_time):
                #gc.collect()
                runtime_stats_cuda.start_region("total")
                runtime_stats_cuda.start_region(
                    "preamble", runtime_stats_cuda.get_last_event())
                if self.args.train_sampler == 'NeighborSampler':
                    self.train_loader.node_idx = self.get_idx(epoch)
                    devit = self.train_loader
                else:
                    self.train_loader.idx = self.get_idx(epoch)
                    devit = self.make_train_devit()
                runtime_stats_cuda.end_region("preamble")
                runtime_stats_cuda.end_region(
                    "total", runtime_stats_cuda.get_last_event())
            ctimer_preamble.end()
            # append_runtime_stats("Sampler init", ctimer_preamble.report())

            if self.args.train_sampler != 'NeighborSampler' and \
               isinstance(devit.it, FastSamplerIter):
                self.log((epoch, devit.it.get_stats()))
                append_runtime_stats("Sampler init", devit.it.get_stats(
                ).total_blocked_dur.total_seconds() * 1000)
            if self.is_main_proc:
                if self.args.train_sampler == 'NeighborSampler':
                    pbar = tqdm(total=self.train_loader.node_idx.numel())
                else:
                    pbar = tqdm(total=self.train_loader.idx.numel())
                pbar.set_description(f'Epoch {epoch}')

            def cb(inputs, results):
                if self.is_main_proc:
                    pbar.update(sum(batch.batch_size for batch in inputs))

            def cb_NS(inputs, results):
                if self.is_main_proc:
                    pbar.update(sum(bs[0] for bs in inputs))

            def log_total_compute_time(x):
                append_runtime_stats("total", x.nanos/1000000)
                self.log(x)

            with Timer((epoch, 'Compute'), log_total_compute_time) as timer:
                if self.args.train_sampler == 'NeighborSampler':
                    self.train_impl(self.model, train.barebones_train_core,
                                    devit, self.optimizer, lr_scheduler,
                                    cb_NS, dataset=self.dataset,
                                    devices=self.devices)
                else:
                    self.train_impl(self.model, train.barebones_train_core,
                                    devit, self.optimizer, lr_scheduler,
                                    cb, dataset=None, devices=None)
                # Barrier is not needed for correctness. I'm also not sure it is needed for accurate
                #   timing either because of synchronization in DDP model. In any case, including it
                #   here to make sure there is a synchronization point inside the compute region.
                if dist.is_initialized():
                    dist.barrier()
                timer.stop()

                runtime_stats_cuda.report_stats({'total': 'Total', 'data_transfer': 'Data Transfer', 'sampling': 'Sampling + Slicing', 'train': 'Train', 'sampling2': 'Sampling Blocking'})
                
                # Log amount of communication during training.
                if self.args.distribute_data:
                    # NOTE: These values are off by a factor of 4. Only useful for relative comparisons.
                    self.log(f"NUM_SENT_BYTES(name={epoch}, bytes={devit.NUMBER_OF_SENT_BYTES})")
                    #print(f"NUM_SENT_BYTES(name={epoch}, bytes={devit.NUMBER_OF_SENT_BYTES})", flush=True)
                if self.is_main_proc:
                    if self.args.train_sampler != 'NeighborSampler' and \
                            isinstance(devit.it, FastSamplerIter):

                        self.log((epoch, devit.it.get_stats()))
                        # append runtime stats. Convert units to milliseconds
                        append_runtime_stats("Sampling block time", devit.it.get_stats(
                        ).total_blocked_dur.total_seconds()*1000)
                    pbar.close()
                    del pbar
                if self.args.train_sampler == 'FastSampler':

                    """
                    if __debug__:
                        dc = data_collector
                        num_sent_feature_bytes = devit.collect_data(dc)
                        bytes_in_GiB = 1024 ** 3
                        bandwidth_GiB_per_second = num_sent_feature_bytes / bytes_in_GiB / timer.elapsed_time_seconds
                        print(f'Utilized Bandwidth -- epoch {epoch}: {bandwidth_GiB_per_second} (GiB/s)', flush=True)
                        #dc.set_current_epoch(epoch)
                        ## Save the epoch compute time.
                        #epoch_times_f = dc.get_epoch_data_filepath('epoch_times', use_rank=True)
                        #epoch_times = {'epoch compute time': timer.elapsed_time_seconds}
                        #dc.np_savez_dict(epoch_times_f, epoch_times)
                    """

                    pass
                    """
                    # THIS IS WRONG, timing compute only.
                    num_sent_feature_bytes = devit.NUMBER_OF_SENT_BYTES
                    bytes_in_GiB = 1024 ** 3
                    num_sent_GiB = num_sent_feature_bytes / bytes_in_GiB
                    bandwidth_GiB_per_second = num_sent_GiB / timer.elapsed_time_seconds
                    average_GiB_per_batch = num_sent_GiB / devit.it.session.num_total_batches
                    print(f'Num total batches in epoch: {devit.it.session.num_total_batches}')
                    print(f'Utilized Bandwidth -- epoch {epoch}: {bandwidth_GiB_per_second} (GiB/s)', flush=True)
                    print(f'Average amt of data sent per batch -- epoch {epoch}: {average_GiB_per_batch} (GiB)', flush=True)
                    """
                

    def test(self, sets=None) -> Mapping[str, float]:
        if self.is_main_proc:
            print()

        if self.args.test_type == 'layerwise':
            assert False
            #results = self.layerwise_test(sets=sets)
        elif self.args.test_type == 'batchwise':
            results = self.batchwise_test(sets=sets)
        else:
            raise ValueError('unknown test_type')

        return results

    @torch.no_grad()
    def batchwise_test(self, sets=None) -> Mapping[str, float]:
        self.model.eval()

        if sets is None:
            sets = self.dataset.split_idx

        results = {}

        for name in sets:
            with Timer((name, 'Preamble'), self.log):
                local_fanouts = self.args.batchwise_test_fanouts
                # This is actually for validation, just named test_batch_size.
                local_batchsize = self.args.test_batch_size
                id_set_name = 'valid'
                minibatch_size = self.args.test_batch_size * self.get_num_trainers()
        

                if name == 'test':
                    local_fanouts = self.args.final_test_fanouts
                    local_batchsize = self.args.final_test_batchsize
                    # SEE NOTES: in __init__
                    id_set_name = 'test'
                    minibatch_size = self.args.final_test_batchsize * self.get_num_trainers()


                cfg = FastSamplerConfig(
                    x_cpu=self.x_cpu, x_gpu=self.x_gpu, y=self.dataset.y.unsqueeze(-1),
                    rowptr=self.dataset.rowptr, col=self.dataset.col,
                    idx=self.get_idx_test(name),
                    batch_size=local_batchsize,
                    sizes=local_fanouts,
                    skip_nonfull_batch=False,
                    pin_memory=True,
                    distributed = self.args.distribute_data,
                    partition_book = self.dataset.get_RangePartitionBook() if self.args.distribute_data else None,
                    # Can overwrite later, but method for doing so not yet available.
                    # Cache for inference not yet supported.
                    cache = Cache(),
                    # Force the exact num batches if distributing data to have an equal number of iterations.
                    force_exact_num_batches=True,#self.args.distribute_data,
                    exact_num_batches=self.dataset.get_num_iterations(minibatch_size)[id_set_name],
                    count_remote_frequency=False,
                    # Cache for inference not yet supported.
                    use_cache=False, 
                )

                loader = FastSampler(self.args.num_workers,
                                     self.args.test_max_num_batches, cfg)
                devit = self.test_transferer([self.main_device], iter(loader), pipeline_on = not self.args.pipeline_disabled)

            if self.is_main_proc:
                pbar = tqdm(total=cfg.idx.numel())
                if not dist.is_initialized():
                    pbar.set_description(f'Batchwise eval (one proc)')
                else:
                    pbar.set_description(
                        'Batchwise eval (multi proc, showing main proc progress)')

            def cb(batch):
                if self.is_main_proc:
                    pbar.update(batch.batch_size)

            with Timer((name, 'Compute'), self.log) as timer:
                if hasattr(self.model, 'module'):
                    self.model_noddp.load_state_dict(
                        self.model.module.state_dict())
                else:
                    self.model_noddp.load_state_dict(self.model.state_dict())
                result = test.batchwise_test(
                    self.model_noddp, len(loader), devit, cb)

                timer.stop()
                if self.is_main_proc:
                    pbar.close()
                    del pbar

            if dist.is_initialized():
                output_0 = torch.tensor([result[0]]).to(self.main_device)
                output_1 = torch.tensor([result[1]]).to(self.main_device)
                _ = dist.all_reduce(output_0)
                _ = dist.all_reduce(output_1)
                result = (output_0.item(), output_1.item())
            results[name] = result[0] / result[1]

        return results
