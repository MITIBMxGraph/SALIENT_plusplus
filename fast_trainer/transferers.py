from collections import deque, namedtuple
from dataclasses import dataclass
import nvtx

from typing import List, Iterator, Any
import torch
import torch.distributed as dist

from .samplers import ProtoBatch, PreparedBatch, ProtoDistributedBatch, NumpyProtoDistributedBatch
from .utils import append_runtime_stats, Timer, runtime_stats_cuda, DataCollector
import time

aggregate_time_results = dict()
def aggregate_time(result):
    global aggregate_time_results
    if result.name not in aggregate_time_results:
        aggregate_time_results[result.name] = 0
    aggregate_time_results[result.name] += result.nanos

class DeviceIterator(Iterator[List[PreparedBatch]]):
    '''
    Abstract class that returns PreparedBatch on devices (GPUs)
    '''
    devices: List[torch.cuda.device]

    def __init__(self, devices):
        assert len(devices) > 0
        self.devices = devices
        self.device = self.devices[0]

StageOutput = namedtuple('StageOutput', 'handle args kwargs')

class DeviceDistributedPrefetcher(DeviceIterator):
    streams = {}
    def __init__(self, devices, it: Iterator[ProtoDistributedBatch], pipeline_on=True):
        super().__init__(devices)
        #print("Init the device prefetcher")
        self.pipeline_on = pipeline_on
        self.it = it
        self.save_event = None

        # Get the partition book.
        self.partition_book = self.it.session.config.partition_book
        # Get the cache.
        self.cache = self.it.session.config.cache
        self.use_cache = self.it.session.config.use_cache

        # Execution params.
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.other_ranks = [rank for rank in range(self.world_size) if rank != self.rank]
        self.fanout = self.it.session.config.sizes
        self.batch_size = self.it.session.config.batch_size
        self.ids = self.it.session.config.idx
        # QUESTION: do we want the features from the fast sampler config or the partition_book
        # Can get it from the config as long as when create the config pass in the reordered features when using a RangePartitionBook.
        self.features_gpu = self.it.session.config.x_gpu
        #print("features GPU size is " + str(self.features_gpu.size()), flush=True)
        self.feature_gpu_cutoff = torch.tensor([self.features_gpu.size()[0]], dtype=torch.long, device=self.device)
        self.feature_gpu_cutoff_real = torch.tensor([self.features_gpu.size()[0]], dtype=torch.long, device=self.device)
        self.feature_gpu_cutoff_real2 = torch.tensor([self.features_gpu.size()[0]+1], dtype=torch.long, device=self.device)
        self.features_cpu = self.it.session.config.x_cpu
        self.labels = self.it.session.config.y
    
        # Datatype parameters.
        self.meta_dtype = torch.int32
        
        #NOTE(TFK): Choosing features_gpu arbitrarily..
        self.features_dtype = self.features_gpu.dtype
        self.feature_dim = self.features_gpu.size(dim=1)
        if True and self.pipeline_on:
            ## Streams mostly combined.
            if not bool(self.streams):
                DeviceDistributedPrefetcher.streams = {}
                DeviceDistributedPrefetcher.streams['default']          = torch.cuda.default_stream(self.device)
                DeviceDistributedPrefetcher.streams['data_transfer']    = torch.cuda.Stream(self.device)#, priority=-1)
                DeviceDistributedPrefetcher.streams['network']           = torch.cuda.default_stream(self.device)#, priority=-1)
                DeviceDistributedPrefetcher.streams['finalizer']           = torch.cuda.Stream(self.device)#, priority=-1)
                DeviceDistributedPrefetcher.streams['h2d']              = torch.cuda.Stream(self.device) 
                DeviceDistributedPrefetcher.streams['meta_all2all']     = self.streams['network']
                DeviceDistributedPrefetcher.streams['meta_d2h']         = self.streams['data_transfer']
                DeviceDistributedPrefetcher.streams['indices_all2all']  = self.streams['network']
                DeviceDistributedPrefetcher.streams['features_slicing1'] = torch.cuda.Stream(self.device)
                DeviceDistributedPrefetcher.streams['features_slicing2'] = torch.cuda.Stream(self.device)
                DeviceDistributedPrefetcher.streams['features_slicing3'] = torch.cuda.Stream(self.device)
                DeviceDistributedPrefetcher.streams['features_slicing4'] = torch.cuda.Stream(self.device)
                DeviceDistributedPrefetcher.streams['features_all2all'] = self.streams['network']
                DeviceDistributedPrefetcher.streams['combine_features'] = torch.cuda.Stream(self.device, priority=-1)
        elif False and self.pipeline_on:
            # Streams All separate, except for nccl streams.
            self.streams = {}
            self.streams['default']          = torch.cuda.default_stream(self.device)

            self.streams['h2d']              = torch.cuda.Stream(self.device) 
            self.streams['meta_all2all']     = torch.cuda.Stream(self.device)
            self.streams['meta_d2h']         = torch.cuda.Stream(self.device)
            self.streams['indices_all2all']  = torch.cuda.Stream(self.device)
            self.streams['features_slicing1'] = torch.cuda.Stream(self.device)
            self.streams['features_slicing2'] = torch.cuda.Stream(self.device)
            self.streams['features_slicing3'] = torch.cuda.Stream(self.device)
            self.streams['features_all2all'] = torch.cuda.Stream(self.device)
            self.streams['combine_features'] = torch.cuda.Stream(self.device)
        else:
            # Put everything on the same background stream.
            self.streams = {}
            self.streams['default']          = torch.cuda.default_stream(self.device)

            self.streams['h2d']               = torch.cuda.Stream(self.device) 
            self.streams['meta_all2all']      = self.streams['h2d'] 
            self.streams['meta_d2h']          = self.streams['h2d']
            self.streams['indices_all2all']   = self.streams['h2d']
            self.streams['features_slicing1'] = self.streams['h2d']
            self.streams['features_slicing2'] = self.streams['h2d']
            self.streams['features_slicing3'] = self.streams['h2d']
            self.streams['features_slicing4'] = self.streams['h2d']
            self.streams['features_all2all']  = self.streams['h2d']
            self.streams['combine_features']  = self.streams['h2d']

        self.all_streams = self.streams.values()

        # May need to record the features on the stream we want them on.
        self.features_gpu.record_stream(self.streams['features_slicing1'])
        if self.use_cache: self.cache.cached_features.record_stream(self.streams['features_slicing1'])

        self.copy_partition_offsets = self.partition_book.partition_offsets.clone()
        self.copy_partition_offsets = self.copy_partition_offsets.to(device=self.device)
        self.copy_partition_offsets.record_stream(self.streams['features_slicing1'])

        # Queues.
        self.queues = {}
        self.queues['combine_features'] = deque()
        self.queues['features_all2all'] = deque()
        self.queues['features_slicing1'] = deque()
        self.queues['features_slicing2'] = deque()
        self.queues['features_slicing3'] = deque()
        self.queues['features_slicing4'] = deque()
        self.queues['indices_all2all'] = deque()
        self.queues['meta_d2h'] = deque()
        self.queues['meta_all2all'] = deque()
        # Final output.
        self.next = []

        """
        Infrastructure to collect additional statistics, particularly timing information, batch properties, and load balance among nodes.
        Run with PYTHONOPTIMIZE=1 to disable.
        """
        if __debug__:
            # Dict of lists to collect converted batch data.
            self.ALL_BATCH_DATA = {k:[] for k in NumpyProtoDistributedBatch._fields}
            # However, this conversion only happens at the end of an epoch. Need another list to hold batch references.
            self.ALL_BATCHES = []
            # Save a reference to the training ids for the epoch so when converting batches can slice out the relevant training nodes.
            self.IDS = self.it.session.config.idx
            # Save a reference to the cache so can examine later.
            # When running for multiple epochs with the same cache can just check the passed-in data collection object to see if the cache is already there.
            self.CACHED_VERTICES = self.it.session.config.cache.cached_vertices
            # Collect number of sent bytes to all other machines
            #self.NUMBER_OF_SENT_FEATURE_BYTES = 0

        # NOT PERFECT, only update this once features have been sent, with meta, indices, and features bytes.
        self.NUMBER_OF_SENT_BYTES = 0

        self.TFK_SENT_BYTES = 0
        self.TFK_REC_BYTES = 0

        self.FILL_PIPELINE_CALLS = 0
        self.PRELOAD_CALLS = 0
        self.ITERATION = 0

        """
        # If want to time relative to start, can use a barrier here.
        if __debug__:
            dist.barrier()
        """
        if self.pipeline_on:
            FILL = 10
            for i in range(1, FILL+1):
                self.streams['default'].wait_stream(self.streams['combine_features'])
                self.fill_pipeline(i)
            #torch.cuda.synchronize()
        else:
            torch.cuda.synchronize()
            self.preload_nopipeline(start_sampling=False)
            torch.cuda.synchronize()

    @nvtx.annotate('stage_h2d', color='green')
    def stage_h2d(self):
        # This batch is on cpu.
        runtime_stats_cuda.start_region("sampling2")
        #handle = dist.all_reduce(torch.zeros(1, device='cuda:0'), async_op=True)#barrier(async_op=True)
        batch = next(self.it, None)
        #this is purely so we can track sampling time which could be delayed due to another machine.
        # failure to do this synchronization results in time being attributed to meta-all2all, which is annoying in profiles.
        # note that, a barrier operation induces a device synchronization --- which is why we effectively do a barrier here with
        #  a meaningless reduction of a cuda tensor.
        #dist.barrier()
        #handle.wait()
        if batch is None:
            runtime_stats_cuda.end_region("sampling2")
            return
        # CLEANUP, indices_dtype should be fixed. Could just set to torch.int64
        self.indices_dtype = batch.partition_nids[0].dtype
        runtime_stats_cuda.end_region("sampling2")
        # Execution on desired stream.
        with torch.cuda.stream(self.streams['h2d']):
            runtime_stats_cuda.start_region("stage_h2d")

            _out = self.preload_h2d(batch)
            self.queues['meta_all2all'].append(_out)

            runtime_stats_cuda.end_region("stage_h2d")


    @nvtx.annotate('stage_meta_all2all', color='yellow')
    def stage_meta_all2all(self):
        if not self.queues['meta_all2all']:
            #print("returning from meta_all2all", flush=True)
            return
         
        _in = self.queues['meta_all2all'][0] #self.queues['meta_all2all'].popleft()
        # Execution on desired stream.
        with torch.cuda.stream(self.streams['meta_all2all']):
            runtime_stats_cuda.start_region("stage_metaall2all")
            
            _out = self.preload_meta_all2all(*_in.args, **_in.kwargs)
            self.queues['meta_d2h'].append(_out)
            self.queues['meta_all2all'].popleft()
            runtime_stats_cuda.end_region("stage_metaall2all")

    @nvtx.annotate('stage_meta_d2h', color='red')
    def stage_meta_d2h(self):
        if not self.queues['meta_d2h']:
            #print("returning from meta_d2h", flush=True)
            return
        _in = self.queues['meta_d2h'][0] 

        with torch.cuda.stream(self.streams['meta_d2h']):
            runtime_stats_cuda.start_region("stage_meta_d2h")

            # Wait conditions.
            _in.handle.wait()

            _out = self.preload_meta_d2h(*_in.args, **_in.kwargs) 
            self.queues['indices_all2all'].append(_out)
            runtime_stats_cuda.end_region("stage_meta_d2h")

            self.queues['meta_d2h'].popleft() 


    @nvtx.annotate('stage_indices_all2all', color='blue')
    def stage_indices_all2all(self):
        if not self.queues['indices_all2all']:
            #print("returning from indices_all2all", flush=True)
            return
        _in = self.queues['indices_all2all'][0] 
        # Execution on desired stream.
        with torch.cuda.stream(self.streams['indices_all2all']):

            runtime_stats_cuda.start_region("stage_indices_all2all")

            _out = self.preload_indices_all2all(*_in.args, **_in.kwargs) 
            self.queues['features_slicing1'].append(_out)
            self.queues['indices_all2all'].popleft() 
            runtime_stats_cuda.end_region("stage_indices_all2all")

    @nvtx.annotate('stage_features_slicing1', color='orange')
    def stage_features_slicing1(self):

        if not self.queues['features_slicing1']:
            #print("returning from features_slicing1", flush=True)
            return
        _in = self.queues['features_slicing1'][0] 
        # Execution on desired stream.
        with torch.cuda.stream(self.streams['features_slicing1']):

            runtime_stats_cuda.start_region("stage_features_slicing1")
            # Wait conditions.
            _in.handle.wait()

            _out = self.preload_features_slicing1(*_in.args, **_in.kwargs) 
            self.queues['features_slicing2'].append(_out)
            self.queues['features_slicing1'].popleft()
            runtime_stats_cuda.end_region("stage_features_slicing1")


    @nvtx.annotate('stage_features_slicing2', color='orange')
    def stage_features_slicing2(self):

        if not self.queues['features_slicing2']:
            #print("returning from features_slicing2", flush=True)
            self.it.session.wait_slice_tensors()
            return
        _in = self.queues['features_slicing2'][0] 
        # Execution on desired stream.
        with torch.cuda.stream(self.streams['features_slicing2']):

            runtime_stats_cuda.start_region("stage_features_slicing2")

            _out = self.preload_features_slicing2(*_in.args, **_in.kwargs) 
            self.queues['features_slicing3'].append(_out)
            self.queues['features_slicing2'].popleft()
            runtime_stats_cuda.end_region("stage_features_slicing2")

    @nvtx.annotate('stage_features_slicing3', color='orange')
    def stage_features_slicing3(self):

        if not self.queues['features_slicing3']:
            #print("returning from features_slicing3", flush=True)
            return
        _in = self.queues['features_slicing3'][0] 
        # Execution on desired stream.
        with torch.cuda.stream(self.streams['features_slicing3']):

            runtime_stats_cuda.start_region("stage_features_slicing3")

            _out = self.preload_features_slicing3(*_in.args, **_in.kwargs) 
            self.queues['features_slicing4'].append(_out)
            self.queues['features_slicing3'].popleft()
            runtime_stats_cuda.end_region("stage_features_slicing3")

    @nvtx.annotate('stage_features_slicing4', color='orange')
    def stage_features_slicing4(self):

        if not self.queues['features_slicing4']:
            #print("returning from features_slicing3", flush=True)
            return
        _in = self.queues['features_slicing4'][0] 
        # Execution on desired stream.
        with torch.cuda.stream(self.streams['features_slicing4']):

            runtime_stats_cuda.start_region("stage_features_slicing4")

            _out = self.preload_features_slicing4(*_in.args, **_in.kwargs) 
            self.queues['features_all2all'].append(_out)
            self.queues['features_slicing4'].popleft()
            runtime_stats_cuda.end_region("stage_features_slicing4")

    @nvtx.annotate('stage_features_all2all', color='pink')
    def stage_features_all2all(self):
        if not self.queues['features_all2all']:
            #print("returning from features_all2all", flush=True)
            return
        _in = self.queues['features_all2all'][0]#.popleft() 
        # Execution on desired stream.
        with torch.cuda.stream(self.streams['features_all2all']):

            runtime_stats_cuda.start_region("stage_features_all2all")

            _out = self.preload_features_all2all(*_in.args, **_in.kwargs) 
            self.queues['combine_features'].append(_out)
            runtime_stats_cuda.end_region("stage_features_all2all")

            self.queues['features_all2all'].popleft() 


    @nvtx.annotate('stage_combine_features', color='cyan')
    def stage_combine_features(self):
        if not self.queues['combine_features']:
            self.next = None
            #print("returning from combine_features", flush=True)
            return
        _in = self.queues['combine_features'][0] 
        with torch.cuda.stream(self.streams['combine_features']):
             
            runtime_stats_cuda.start_region("stage_combine_features")
            # Wait conditions.
            _in.handle.wait()

            _out = self.preload_combine_features(*_in.args, **_in.kwargs) 
            self.next = [_out.args]
            self.queues['combine_features'].popleft() 
            runtime_stats_cuda.end_region("stage_combine_features")

    def preload_nopipeline(self, timing=True, start_sampling=True):

        self.PRELOAD_CALLS += 1

        if start_sampling:
            runtime_stats_cuda.end_region("sampling")
        self.stage_h2d()
        self.stage_meta_all2all()       
        self.stage_meta_d2h()

        # Need to synchronize due to the device to host transfer.
        self.streams['meta_d2h'].synchronize()

        self.stage_indices_all2all()
        self.stage_features_slicing1()

        # Need to synchronize due to the device to host transfer.
        self.streams['features_slicing1'].synchronize()

        self.stage_features_slicing2()
        self.stage_features_slicing3()
        self.stage_features_slicing4()
        self.stage_features_all2all()
        self.stage_combine_features()

    def preload(self, timing=True):

        self.PRELOAD_CALLS += 1

        with nvtx.annotate('synchronize nonnetwork streams', color='brown'):
            for x in ['combine_features', 'features_slicing4', 'features_slicing3', 'features_slicing2', 'features_slicing1', 'meta_d2h', 'h2d']:
                self.streams[x].synchronize()
        with nvtx.annotate('synchronize previous default stream work', color='brown'):
            if self.save_event != None:
                self.save_event.synchronize()

        self.stage_combine_features()
        self.stage_features_all2all()
        self.stage_features_slicing4()
        self.stage_features_slicing3()
        self.stage_features_slicing2()
        self.stage_features_slicing1()
        self.stage_indices_all2all()
        self.stage_meta_d2h()
        self.stage_meta_all2all()
        self.stage_h2d()
        self.save_event = torch.cuda.Event()
        self.save_event.record() 
    
    @nvtx.annotate('fill_pipeline', color='black')
    def fill_pipeline(self, fill: int):
         
        with nvtx.annotate('synchronize nonnetwork streams', color='brown'):
            for x in ['combine_features', 'features_slicing4', 'features_slicing3', 'features_slicing2', 'features_slicing1', 'meta_d2h', 'h2d']:
                self.streams[x].synchronize()
        with nvtx.annotate('synchronize previous default stream work', color='brown'):
            if self.save_event != None:
                self.save_event.synchronize()


        self.FILL_PIPELINE_CALLS += 1

        # Reverse order.
        #print("before stage combine", flush=True)
        if (fill >= 10): self.stage_combine_features()
        #print("before stage all2all", flush=True)
        if (fill >= 9): self.stage_features_all2all()
        #print("before stage slicing4", flush=True)
        if (fill >= 8): self.stage_features_slicing4()
        #print("before stage slicing3", flush=True)
        if (fill >= 7): self.stage_features_slicing3()
        #print("before stage slicing2", flush=True)
        if (fill >= 6): self.stage_features_slicing2()
        #print("before stage slicing1", flush=True)
        if (fill >= 5): self.stage_features_slicing1()
        #print("before stage meta_d2h", flush=True)
        if (fill >= 4): self.stage_indices_all2all()
        #print("before stage meta_d2h", flush=True)
        if (fill >= 3): self.stage_meta_d2h()
        #print("before stage meta_all2all", flush=True)
        if (fill >= 2): self.stage_meta_all2all()        
        #print("before stage h2d", flush=True)
        if (fill >= 1): self.stage_h2d()
        #print("end of fill pipeline", flush=True)

        self.save_event = torch.cuda.Event()
        self.save_event.record() 

    @nvtx.annotate('preload_combine_features', color='cyan')
    def preload_combine_features(self, features_gather, _features_scatter, tmp_tensor, **kwargs):

        y = kwargs['_y_gpu']
        adjs = kwargs['_adjs_gpu']
        idx_range = kwargs['_batch_cpu'].idx_range
        partition_nids_cpu = kwargs['_batch_cpu'].partition_nids
        perm_partition_to_mfg_gpu = kwargs['_perm_partition_to_mfg_gpu']
        cached_features = kwargs['_cached_features']

        gpu_features, cpu_features, size_tuple, gpu_positions, cpu_positions = tmp_tensor

        cpu_features = cpu_features#.to(device=self.device, non_blocking=True)
        features_gather[self.rank] = torch.empty(size_tuple, device=self.device, dtype=torch.float16)
        features_gather[self.rank][gpu_positions] = gpu_features
        features_gather[self.rank][cpu_positions] = cpu_features

        # Append cache features reference to list of features to cat.
        all_features = features_gather+[cached_features] if self.use_cache else features_gather

        # Cat and reorder to be compliant with ordering of features in mfg (adjs).
        x = torch.cat(all_features, dim=0)
        x[:] = x[perm_partition_to_mfg_gpu]

        prepared_batch = PreparedBatch(x, y, adjs, idx_range)

        # NOTE: This calculation is incorrect. I believe the 4x multiplier is not needed.
        self.NUMBER_OF_SENT_BYTES += (4 * len(self.other_ranks) * sum([partition_nids_cpu[other_rank].numel() for other_rank in self.other_ranks]) * ( 8 + 2 * self.feature_dim))

        for stm in [adjs, y, idx_range, partition_nids_cpu, perm_partition_to_mfg_gpu, cached_features, gpu_features, cpu_features, size_tuple, gpu_positions, cpu_positions] + [z for z in features_gather] + [z for z in _features_scatter] + [tmp_tensor]:
            if type(stm) == list or type(stm) == tuple:
                #print("type(stm) is list ", flush=True)
                #print(stm, flush=True)
                for z in stm:
                    if hasattr(z, 'is_cuda') and z.is_cuda:
                        z.record_stream(self.streams['combine_features'])
            if hasattr(stm, 'is_cuda') and stm.is_cuda:
                stm.record_stream(self.streams['combine_features'])

        return StageOutput(
            handle = None,
            args = prepared_batch,
            kwargs = None,
        )

    @nvtx.annotate('preload_features_all2all', color='pink')
    def preload_features_all2all(self, features_scatter, **kwargs):

        partition_nids_cpu = kwargs['_batch_cpu'].partition_nids

        # Only allocate for remote. Hopefully reduces latency but likely making use of a caching allocator anyway.
        features_gather = [None] * self.world_size
        for other_rank in self.other_ranks:
            features_gather[other_rank] = torch.empty((partition_nids_cpu[other_rank].numel(), self.feature_dim), dtype=self.features_dtype, device=self.device)

        tmp_tensor = features_scatter[self.rank]
        features_gather[self.rank] = torch.empty((0, self.feature_dim), dtype=self.features_dtype, device=self.device)
        features_scatter[self.rank] = torch.empty((0, self.feature_dim), dtype=self.features_dtype, device=self.device)

        features_handle = dist.all_to_all(features_gather, features_scatter, group=None, async_op=True)

        for stm in [z for z in partition_nids_cpu] + [z for z in features_scatter] + [z for z in tmp_tensor]:
            if hasattr(stm, 'is_cuda') and stm.is_cuda:
                stm.record_stream(self.streams['features_all2all'])

        return StageOutput(
            handle = features_handle, 
            args = (features_gather, features_scatter, tmp_tensor),
            kwargs = kwargs,
        )

    @nvtx.annotate('preload_features_slicing1', color='orange')
    def preload_features_slicing1(self, indices_gather, _indices_scatter, **kwargs):
        cached_nids_cpu = kwargs['_batch_cpu'].cached_nids

        # we need to get the cpu_idx
        cpu_idx_list = [None] * self.world_size
        for i in range(self.world_size):
            if True:#i != self.rank:
                all_idx = indices_gather[i]#-self.copy_partition_offsets[self.rank] - self.feature_gpu_cutoff_real
                all_idx = all_idx.to(torch.int64)

                with nvtx.annotate('cpu_idx_list async copy', color='pink'):
                    copy_tensor = all_idx-self.copy_partition_offsets[self.rank]-self.feature_gpu_cutoff_real
                    copy_tensor = copy_tensor.to(torch.int64)
                    cpu_idx_list[i] = torch.empty(all_idx.numel(), dtype=torch.int64, pin_memory=True).copy_(copy_tensor, non_blocking=True)
            else:
                cpu_idx_list[i] = torch.empty(0, dtype=torch.int64)
                

        cached_nids_gpu = kwargs['_batch_cpu'].cached_nids.to(device=self.device, non_blocking=True) # was true
        kwargs['cached_nids_gpu'] = cached_nids_gpu


        for stm in [kwargs['cached_nids_gpu']] + [z for z in indices_gather] + [z for z in _indices_scatter]:
            if stm.is_cuda:
                stm.record_stream(self.streams['features_slicing1'])

        return StageOutput(
            handle = None,
            args = (indices_gather, cpu_idx_list),
            kwargs = kwargs,
        )

    def print_stats(self):
        global aggregate_time_results
        #for k in aggregate_time_results:
        #    print(str(k) + ": " + str(aggregate_time_results[k]), flush=True)
        aggregate_time_results = dict()

    @nvtx.annotate('preload_features_slicing2', color='orange')
    def preload_features_slicing2(self, indices_gather, cpu_idx_list, **kwargs):
        with Timer("async_slice_tensors", aggregate_time):
            self.it.session.async_slice_tensors(cpu_idx_list, self.rank)

        for stm in [] + [z for z in indices_gather]:
            if stm.is_cuda:
                stm.record_stream(self.streams['features_slicing2'])

        return StageOutput(
            handle = None,
            args = (indices_gather,),
            kwargs = kwargs,
        )

    @nvtx.annotate('preload_features_slicing3', color='orange')
    def preload_features_slicing3(self, indices_gather, **kwargs):
        if not self.pipeline_on:
            self.it.session.wait_slice_tensors()
        with Timer("wait_slice_tensors", aggregate_time):
            self.it.session.wait_slice_tensors()
        with Timer("get_slice_tensors", aggregate_time):
            cpu_features_list = self.it.session.get_slice_tensors()

        local_slicing = 0
        remote_slicing = 0
        cpu_idx_pos_list = [None] * self.world_size
        gpu_idx_pos_list = [None] * self.world_size
        cpu_features_new_list = [None] * self.world_size 
        
        with nvtx.annotate('loop1', color='pink'):
            for i in range(self.world_size):

                all_idx = indices_gather[i]-self.copy_partition_offsets[self.rank]
                all_idx = all_idx.to(torch.int64)
                cpu_feats = cpu_features_list[i][0]
                cpu_idx_pos = cpu_features_list[i][1]
                gpu_idx_pos = cpu_features_list[i][2]
                cpu_idx_pos_list[i] = cpu_idx_pos.to(device=self.devices[0], non_blocking=True)
                gpu_idx_pos_list[i] = gpu_idx_pos.to(device=self.devices[0], non_blocking=True)

                if i == self.rank:
                    cpu_features = kwargs['_batch_gpu'].sliced_cpu_features.to(device=self.device, non_blocking=True)

                    cpu_features_new_list[i] = cpu_features
                    continue
                else:
                    cpu_features = cpu_feats.to(device=self.device, non_blocking=True)
                cpu_features_new_list[i] = cpu_features

        for stm in [kwargs['_batch_gpu'].sliced_cpu_features, kwargs['cached_nids_gpu']] + [z for z in indices_gather]:
            if stm.is_cuda:
                stm.record_stream(self.streams['features_slicing3'])

        return StageOutput(
            handle = None,
            args = (indices_gather, cpu_idx_pos_list, gpu_idx_pos_list, cpu_features_new_list,),
            kwargs = kwargs,
        )


    @nvtx.annotate('preload_features_slicing4', color='orange')
    def preload_features_slicing4(self, indices_gather, cpu_idx_pos_list, gpu_idx_pos_list, cpu_features_new_list, **kwargs):

        features_scatter = [None] * self.world_size
        with nvtx.annotate('loop2', color='pink'):
            for i in range(self.world_size):
                all_idx = indices_gather[i]-self.copy_partition_offsets[self.rank]
                all_idx = all_idx.to(torch.int64)

                cpu_idx_pos = cpu_idx_pos_list[i] 
                gpu_idx_pos = gpu_idx_pos_list[i] 

                gpu_idx = all_idx[gpu_idx_pos] 
                cpu_idx = all_idx[cpu_idx_pos] 
                cpu_features = cpu_features_new_list[i]
                if i == self.rank:
                    gpu_features = self.features_gpu[gpu_idx]
                    features_scatter[i] = (gpu_features, cpu_features, (indices_gather[i].size()[0], self.features_gpu.size()[1]), gpu_idx_pos, cpu_idx_pos)
                    continue
                else:
                    gpu_features = self.features_gpu[gpu_idx]

                all_features = torch.zeros([indices_gather[i].size()[0], self.features_gpu.size()[1]], device=self.device, dtype=torch.float16)
                all_features[gpu_idx_pos] = gpu_features
                all_features[cpu_idx_pos] = cpu_features
                features_scatter[i] = all_features

        # Note, do not actually need to do the cache slicing at this particular point.
        #   E.g. doesn't have to come after slicing of other features in this stage of the pipeline.
        #   Since it is local can basically happen in any stage.
        # If using the cache, slice from the cache.
        cached_features = []

        if self.use_cache:
            #cached_nids_cpu = kwargs['_batch_cpu'].cached_nids
            # NOTE(TFK): Need to fix this. Right now added line below..
            #cached_nids_cpu.to(self.device, non_blocking=True)
            #cached_features = self.cache.cached_features[cached_nids_cpu]
            cached_features = self.cache.cached_features[kwargs['cached_nids_gpu']]

        # Update kwargs
        kwargs['_cached_features'] = cached_features

        for stm in [kwargs['_batch_gpu'].sliced_cpu_features, kwargs['cached_nids_gpu']] + [z for z in indices_gather] + [cached_features] + [z for z in cpu_idx_pos_list]+ [z for z in gpu_idx_pos_list] + [z for z in cpu_features_new_list]:
            if type(stm) == list:
                for stm2 in stm:
                   stm2.record_stream(self.streams['features_slicing4'])
            elif hasattr(stm, 'is_cuda') and stm.is_cuda:
                stm.record_stream(self.streams['features_slicing4'])

        return StageOutput(
            handle = None,
            args = (features_scatter,),
            kwargs = kwargs,
        )


    @nvtx.annotate('preload_indices_all2all', color='blue')
    def preload_indices_all2all(self, meta_gather_cpu, **kwargs):

        partition_nids_gpu = kwargs['_partition_nids_gpu']

        """ Enqueue the indices exchange. """
        # Indices we want to request the corresponding features for are a property of the batch returned from the fast sampler, and were already transfered to gpu.
        indices_scatter = partition_nids_gpu

        # Allocate space to gather indices.

        # NOTE(TFK): New logic to avoid sending data to myself.
        indices_gather = [None] * self.world_size
        indices_gather[self.rank] = torch.empty(0, dtype=self.indices_dtype, device=self.device)
        for other_rank in self.other_ranks:
            indices_gather[other_rank] = torch.empty(meta_gather_cpu[other_rank], dtype=self.indices_dtype, device=self.device)
        handle_to_tensor = indices_scatter[self.rank]
        indices_scatter[self.rank] = torch.empty(0, dtype=self.indices_dtype, device=self.device)

        indices_handle = dist.all_to_all(indices_gather, indices_scatter, group=None, async_op=True)

        # NOTE(TFK): Avoiding sending data to myself, need to put back in right value for indices_gather for my rank.
        indices_gather[self.rank] = handle_to_tensor

        for stm in [z for z in kwargs['_partition_nids_gpu']] + [z for z in meta_gather_cpu] + [z for z in indices_gather]:
            if stm.is_cuda:
                stm.record_stream(self.streams['indices_all2all'])

        return StageOutput(
            handle = indices_handle,
            args = (indices_gather, indices_scatter),
            kwargs = kwargs,
        )


    @nvtx.annotate('preload_meta_d2h', color='red')
    def preload_meta_d2h(self, meta_gather_gpu, _meta_scatter, **kwargs):
        """
        Transfer the metadata communicated between GPUs to the CPU. The data is need on CPU to allocate appropriately-sized tensors for the indices and features.
        """

        local_rank_value = kwargs['_batch_cpu'].partition_nids[self.rank].numel()
        meta_gather_cpu = [None] * self.world_size
        for other_rank in self.other_ranks:
            meta_gather_cpu[other_rank] = meta_gather_gpu[other_rank].to(torch.device('cpu'), non_blocking=True)
        meta_gather_cpu[self.rank] = torch.tensor([local_rank_value], dtype=self.meta_dtype)

        # Record streams.
        # Nothing to record.
        # Recording _meta_scatter might be optional...
        for stm in [] + [z for z in meta_gather_gpu] + [z for z in _meta_scatter]:
            if stm.is_cuda:
                stm.record_stream(self.streams['meta_d2h'])

        return StageOutput(
            handle = None,
            args = (meta_gather_cpu,),
            kwargs = kwargs,
        )


    @nvtx.annotate('preload_meta_all2all', color='yellow')
    def preload_meta_all2all(self, meta_scatter, **kwargs):
        """ Enqueue the metadata exchange. """
        # Allocate a tensor of 1 element for each machine on the gpu to hold the metadata to receive.
        meta_gather_gpu = [torch.zeros((1,), dtype=self.meta_dtype, device=self.device) for i in range(self.world_size)]
        # Enqueue the exchange of metadata and obtain a handle for the asynchronous operation.
        meta_handle = dist.all_to_all(meta_gather_gpu, meta_scatter, group=None, async_op=True)

        for stm in [] + [z for z in meta_scatter] + [z for z in meta_gather_gpu]:
            if stm.is_cuda:
                stm.record_stream(self.streams['meta_all2all'])
        return StageOutput(
            handle = meta_handle,
            args = (meta_gather_gpu, meta_scatter,),
            kwargs = kwargs,
        )


    @nvtx.annotate('preload_h2d', color='green')
    def preload_h2d(self, batch_cpu):
        """
        h2d, more commonly capitalized H2D, is short for host to device transfer.
        Transfer the batch output by the FastSampler to the GPU.
            Could transfer y later to save more memory, but keeping all the h2d in one place for simplicity.
        This includes the implict H2D when creating meta_scatter.
        """

        """ Enqueue H2D operations. """
        batch_gpu = batch_cpu.to(self.device, non_blocking=True, stream=torch.cuda.current_stream(self.device))  # h2d_stream)
        with nvtx.annotate('xfer labels to gpu', color='pink'):
            y_gpu = batch_cpu.sliced_cpu_labels.to(self.device, non_blocking=True) #self.labels[self.ids[batch_cpu.idx_range]].to(self.device, non_blocking=True)
        # Allocate a tensor of 1 element for each machine on the gpu to hold the metadata to send.
        # Creating meta_scatter implicity invokes an H2D as the .numel() method is calculated on the CPU.
        # Here rewritten instead of allocating with torch.tenosr(...device=self.device) to use the .to() method so can specify non-blocking.
        meta_scatter = [None] * self.world_size
        for rank in range(self.world_size):
            meta_scatter[rank] = torch.tensor([batch_cpu.partition_nids[rank].numel()], dtype=self.meta_dtype, pin_memory=True).to(self.device, non_blocking=True) #NOTE(TFK): Experiment by making this blocking.

        # The partition_nids_gpu will be next use in the in the indices_all2_all stage.
        # The permutation_to_mfg will only be touched in the combine features stage.
        # The mfgs will only be touched in the forward and backward passes.
        # The labels will only be touched in the loss and backward pass.
        # Meta scatter wil be needed in the meta all2all stage.
        with nvtx.annotate('kwargs', color='brown'):
            kwargs = {
                '_perm_partition_to_mfg_gpu': batch_gpu.perm_partition_to_mfg,
                '_adjs_gpu': batch_gpu.adjs,
                '_partition_nids_gpu': batch_gpu.partition_nids,
                '_batch_cpu': batch_cpu,
                '_batch_gpu': batch_gpu,
                '_y_gpu': y_gpu,
            }
  
        # NOTE(TFK) 
        batch_gpu.record_stream(self.streams['h2d'])
        for stm in [y_gpu]:
            if stm.is_cuda:
                stm.record_stream(self.streams['h2d'])

 
        return StageOutput(
            handle = None,
            args = (meta_scatter,),
            kwargs = kwargs
        )


    def __next__(self):
        ret = self.next
        self.next = []
        self.ITERATION += 1
        
        runtime_stats_cuda.start_region(
            "data_transfer", runtime_stats_cuda.get_last_event())
        self.streams['default'].wait_stream(self.streams['combine_features'])
        runtime_stats_cuda.end_region("data_transfer")

        runtime_stats_cuda.start_region(
            "sampling", runtime_stats_cuda.get_last_event())
        if not ret:
            torch.cuda.synchronize()
            raise StopIteration
        if self.pipeline_on:
            self.preload()
        else:
            self.preload_nopipeline()
        return ret

    
    #@nvtx.annotate('collect epoch data', color='red')
    def collect_data(self, data_collector: DataCollector):

        out = None

        if __debug__:
            
            save_all_batch_data_to_disk = False
            
            if save_all_batch_data_to_disk == True:

                # Convert batch data to numpy arrays.
                #with nvtx.annotate('convert batch data', color='red'):
                for batch in self.ALL_BATCHES:
                    npbatch = NumpyProtoDistributedBatch.from_proto_batch(batch, self.IDS)
                    for (k, v) in npbatch._asdict().items():
                        self.ALL_BATCH_DATA[k].append(v)


                # Nvtx annotated in case want to examine overhead of saving statistics.
                #with nvtx.annotate('save epoch data to .npz files', color='red'):

                # Alias for brevity.
                dc = data_collector

                # Only save general graph data if it has not been saved in a previous epoch.
                # Not using rank as it is general graph data.
                graph_data_f = dc.get_run_data_filepath('graph_data', use_rank=False)
                if not graph_data_f.exists():
                    #NOTE(TFK): features_gpu picked arbitrarily.
                    graph_data = {'feature_dtype_bytes': self.features_gpu.element_size(),
                                  'feature_dim': self.feature_dim}
                    dc.np_savez_dict(graph_data_f, graph_data)

                # Only save the cache if it has not been saved in a previous epoch.
                # Using rank because different machines will have different caches.
                cache_f = dc.get_run_data_filepath('cached_vertices', use_rank=True)
                if not cache_f.exists():
                    dc.np_savez_dict(cache_f, {'cached_vertices': self.CACHED_VERTICES})

                # Save batch data.
                for k, lst in self.ALL_BATCH_DATA.items():
                    f = dc.get_epoch_data_filepath(k, use_rank=True)
                    dc.np_savez_list(f, lst)

            #out = self.NUMBER_OF_SENT_FEATURE_BYTES

        return out


class DevicePrefetcher(DeviceIterator):
    def __init__(self, devices, it: Iterator[PreparedBatch], pipeline_on = True):
        super().__init__(devices)
        
        self.it = it
        self.streams = [torch.cuda.Stream(device) for device in devices]
        self.next = []
        self.sampling_times = []
        self.record_stream_times = []
        self.start_prefetch_times = []
        self.wait_stream_times = []
        self.preload(False)
    def print_stats(self):
        return
        #global aggregate_time_results
        #for k in aggregate_time_results:
        #    print(str(k) + ": " + str(aggregate_time_results[k]), flush=True)
        #aggregate_time_results = dict()

    def preload(self, timing=True):
        self.next = []
        for device, stream in zip(self.devices, self.streams):
            timer_start = time.perf_counter_ns()
            batch = next(self.it, None)
            timer_end = time.perf_counter_ns()
            if batch is None:
                append_runtime_stats("total:load_batch:sampling", sum(
                    self.sampling_times)/1000000)
                self.sampling_times = []
                append_runtime_stats("total:load_batch:data_transfer:start_nonblocking_prefetch", sum(
                    self.start_prefetch_times)/1000000)
                self.start_prefetch_times = []
                break

            timer_start = time.perf_counter_ns()
            with torch.cuda.stream(stream):
                self.next.append(batch.to(device, non_blocking=True))
            timer_end = time.perf_counter_ns()
            self.start_prefetch_times.append(timer_end-timer_start)

    def __next__(self):
        runtime_stats_cuda.start_region(
            "data_transfer", runtime_stats_cuda.get_last_event())

        timer_start = time.perf_counter_ns()
        cur_streams = [torch.cuda.current_stream(
            device) for device in self.devices]

        for cur_stream, stream in zip(cur_streams, self.streams):
            cur_stream.wait_stream(stream)
        runtime_stats_cuda.end_region("data_transfer")

        runtime_stats_cuda.start_region(
            "sampling", runtime_stats_cuda.get_last_event())

        ret = self.next
        timer_end = time.perf_counter_ns()
        self.wait_stream_times.append(timer_end-timer_start)
        if not ret:
            torch.cuda.synchronize()
            append_runtime_stats("total:load_batch:data_transfer:wait_stream", sum(
                self.wait_stream_times)/1000000)
            self.wait_stream_times = []
            append_runtime_stats("total:load_batch:data_transfer:record_stream", sum(
                self.record_stream_times)/1000000)
            self.record_stream_times = []
            raise StopIteration

        # TODO: this might be a bit incorrect
        #
        # in theory, we want to record this event after all the
        # training computation on the default stream

        timer_start = time.perf_counter_ns()
        for cur_stream, batch in zip(cur_streams, ret):
            batch.record_stream(cur_stream)
        timer_stop = time.perf_counter_ns()
        self.record_stream_times.append(timer_stop-timer_start)

        self.preload()
        return ret


class DeviceTransferer(DeviceIterator):
    def __init__(self, devices, it: Iterator[PreparedBatch], pipeline_on = True):
        super().__init__(devices)

        self.it = it

    def __next__(self):
        ret = [batch.to(device, non_blocking=True)
               for device, batch in zip(self.devices, self.it)]
        if len(ret) == 0:
            raise StopIteration

        return ret


class DeviceSlicerTransferer(DeviceIterator):
    # NOTE: This class only exists to provide functionality
    #       that we used to have and no longer need (DATA_ON_MAIN).
    #       You likely do not need to use this.
    # NOTE: x and y can be GPU tensors too!
    def __init__(self, devices, x: torch.Tensor, y: torch.Tensor,
                 it: Iterator[ProtoBatch]):
        super().__init__(devices)

        self.x = x
        self.y = y
        self.it = it

    def __next__(self):
        ret = [PreparedBatch.from_proto_batch(
            self.x, self.y, proto_batch).to(device, non_blocking=True)
            for device, proto_batch in zip(self.devices, self.it)]

        if len(ret) == 0:
            raise StopIteration

        return ret
