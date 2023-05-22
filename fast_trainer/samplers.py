from abc import abstractmethod
import datetime
import itertools
from dataclasses import dataclass, fields
#from collections.abc import Iterable, Iterator, Sized
from typing import Iterable, Iterator, Sized
from typing import List, Optional, NamedTuple
import torch
from torch_sparse import SparseTensor

import fast_sampler
from .monkeypatch import Adj
from fast_sampler import RangePartitionBook, Cache

import nvtx

# For NumpyProtoDistributedBatch
import numpy as np
import scipy
from scipy.sparse import csr_matrix

def Adj__from_fast_sampler(adj) -> Adj:
    rowptr, col, e_id, sparse_sizes = adj
    return Adj(
        SparseTensor(rowptr=rowptr, row=None, col=col, value=None,
                     sparse_sizes=sparse_sizes, is_sorted=True,
                     trust_data=True),
        e_id,
        sparse_sizes[::-1]
    )

class ProtoDistributedBatch(NamedTuple):
    """
    A partially complete batch return from the fast sampler after sampling but without slicing
    and with additional information useful for making distributed requests and merging locally
    available and remote features.
    """

    # A list of length P (number of partitions) where the ith element of the list is a tensor
    #    containing the vertex ids in the MFG whose features are stored on partition i.
    partition_nids: List[torch.Tensor]

    sliced_cpu_features: torch.Tensor
    sliced_cpu_labels: torch.Tensor
    # A tensor containing the vertex ids in the MFG whose features are cached, so requests to them
    #   from remote machines do not need to be made. 
    # EDIT: these are not the actual cached_nids, they have been translated to local nids into the cache.
    # RENAME as these are not global ids but cache-specific ids.
    cached_nids: torch.Tensor

    # **Note on structure of communication/data layout**
    # We will make P requests to obtain the vertex features from each partition.
    #  Call the ith list of features features_i. Given [features_1, features_2, ..., features_P]
    #  We will concatenate this list to get features_all, then we will permute features_all so that
    #    order of vertices in the permuted tensor matches the order of vertices in the MFG.

    # Contains the permutation mapping features_all = concat([features_1, ..., features_P]) to
    #   a permuted tensor of vertex features that matches the order of vertices in the MFG.
    # If vertices where found in the cache, now a permutation mapping features_all = concat([features_1, ..., features_P, cached_features]) to
    perm_partition_to_mfg: torch.Tensor;

    # This is the full MFG for one compute node's micro-batch.
    adjs: List[Adj]

    # NOTE: Let's probably rename this... after some discussion.
    # This is the slice of indices into the local machine's list of shuffled vertices.
    #   This tells you the global ids of the vertices you are training on.
    idx_range: slice

    @classmethod
    def from_fast_sampler(cls, batch):
        assert batch.sliced_cpu_features != None
        #partition_nids, cached_nids, perm_partition_to_mfg, adjs, (start,stop) = batch
        (start, stop) = batch.idx_range
        #print(str([str(x.numel()) for x in batch.partition_nids]), flush=True)
        #print(batch.sliced_cpu_features.numel(), flush=True)
        #print(batch.sliced_cpu_labels.numel(), flush=True)
        #print(batch.cached_nids.numel(), flush=True)
        #print("[Batch order info] Start stop is " + str(batch.idx_range), flush=True)
        return cls(
            partition_nids=batch.partition_nids,
            sliced_cpu_features=batch.sliced_cpu_features,
            sliced_cpu_labels=batch.sliced_cpu_labels,
            cached_nids=batch.cached_nids,
            perm_partition_to_mfg=batch.perm_partition_to_mfg,
            adjs=[Adj__from_fast_sampler(adj) for adj in batch.adjs],
            idx_range=slice(start,stop)
        )

    # NOTE(IMPORTANT): Check whether recording stream is needed for the fields of this class and/or on the
    #   constructed feature/label tensors.
    @nvtx.annotate('batch record_stream', color='brown')
    def record_stream(self, stream):
        for part in self.partition_nids:
            part.record_stream(stream)
        #self.sliced_cpu_features.record_stream(stream)
        #self.cached_nids.record_stream(stream)
        self.perm_partition_to_mfg.record_stream(stream)
        for adj in self.adjs:
            adj.record_stream(stream)

    def to(self, device, stream=None, non_blocking=False, streams_to_sync=None, delay_feature_transfer=True):

        with torch.cuda.stream(stream):
            with nvtx.annotate('sending MFG structure to device', color='green'):
                adjs_gpu = [adj.to(device, non_blocking=non_blocking) for adj in self.adjs]

            #if streams_to_sync is not None:
            #    for s in streams_to_sync:
            #        s.synchronize()


            with nvtx.annotate('Sending partition_nids to GPU', color='green'):
                partition_nids_gpu = [part.to(device, non_blocking=non_blocking) for part in self.partition_nids]
            # Should not need to send cached_nids to GPU.
            #with nvtx.annotate('Sending cached_nids to GPU', color='purple'):
            #    cached_nids_gpu = self.cached_nids.to(device, non_blocking=non_blocking)
            with nvtx.annotate('Sending perm_partition to GPU', color='green'):
                perm_partition_to_mfg_gpu = self.perm_partition_to_mfg.to(device, non_blocking=non_blocking)


            if not delay_feature_transfer:
                with nvtx.annotate('Sending sliced_cpu_features to GPU', color='green'):
                    sliced_cpu_features = self.sliced_cpu_features.to(device, non_blocking=non_blocking)
        if delay_feature_transfer:
            return ProtoDistributedBatch(
                adjs = adjs_gpu,
                partition_nids = partition_nids_gpu,
                sliced_cpu_features = self.sliced_cpu_features,
                sliced_cpu_labels = self.sliced_cpu_labels,
                #cached_nids = cached_nids_gpu,
                cached_nids = self.cached_nids,
                perm_partition_to_mfg = perm_partition_to_mfg_gpu,
                idx_range = self.idx_range
            )
        else:
            return ProtoDistributedBatch(
                adjs = adjs_gpu,
                partition_nids = partition_nids_gpu,
                sliced_cpu_features = sliced_cpu_features,
                sliced_cpu_labels = self.sliced_cpu_labels,
                #cached_nids = cached_nids_gpu,
                cached_nids = self.cached_nids,
                perm_partition_to_mfg = perm_partition_to_mfg_gpu,
                idx_range = self.idx_range
            )

    @property
    def num_total_nodes(self):
        return self.perm_partition_to_mfg.size(0)

    @property
    def num_cached_nodes(self):
        return self.cached_nids.size(0)

    def get_num_local_nodes(self, local_rank):
        return self.partition_nids[local_rank]

    def get_num_communicated_nodes(self, local_rank):
        return sum([self.partition_nids[rank] for rank in range(len(partition_nids))]) - self.num_cached_nodes - self.get_num_local_nodes(local_rank)

    @property
    def num_cached_nodes(self):
        return self.cached_nids.size(0)


class NumpyProtoDistributedBatch(NamedTuple):
    """
    When collecting statistics on execution, this class converts the fields in the ProtoDistributedBatch class into numpy arrays
    so that they can be easily saved and used for plotting.
    """
    partition_nids: List[np.ndarray];
    cache_specific_nids: np.ndarray;
    perm_partition_to_mfg: np.array;
    # MFGs converted to scipy sparse csr matrices
    #adjs: List[scipy.sparse._csr.csr_matrix];
    adjs: List[scipy.sparse.csr_matrix];
    # seed indices
    seed_indices: np.ndarray;

    @classmethod
    def from_proto_batch(cls, batch: ProtoDistributedBatch, ids: torch.Tensor):
        print("NOTE(TFK): Disable this for now")
        assert False

        # NOTE: pass in training_ids or other ids so can use idx_range to slice seed_indices.
        return cls(
            partition_nids=[nids.numpy() for nids in batch.partition_nids],    
            cache_specific_nids=batch.cached_nids.numpy(),    
            perm_partition_to_mfg=batch.perm_partition_to_mfg.numpy(),    
            # Can specify the dtype in to_scipy but that only affects the edge data which we do not care about.
            adjs=[adj.adj_t.to_scipy(layout='csr') for adj in batch.adjs],
            seed_indices=ids[batch.idx_range]
        )


class ProtoBatch(NamedTuple):
    n_id: torch.Tensor
    adjs: List[Adj]
    idx_range: slice

    @classmethod
    def from_fast_sampler(cls, proto_sample):  # , idx_range):
        n_id, adjs, (start, stop) = proto_sample
        adjs = [Adj__from_fast_sampler(adj) for adj in adjs]
        return cls(n_id=n_id, adjs=adjs, idx_range=slice(start, stop))

    @property
    def batch_size(self):
        return self.idx_range.stop - self.idx_range.start


class PreparedBatch(NamedTuple):
    x: torch.Tensor
    y: Optional[torch.Tensor]
    adjs: List[Adj]
    idx_range: slice

    @classmethod
    def from_proto_batch(cls, x: torch.Tensor,
                         y: Optional[torch.Tensor],
                         proto_batch: ProtoBatch):
        return cls(
            x=x[proto_batch.n_id],
            y=y[proto_batch.n_id[:proto_batch.batch_size]
                ] if y is not None else None,
            adjs=proto_batch.adjs,
            idx_range=proto_batch.idx_range
        )

    @classmethod
    def from_fast_sampler(cls, prepared_sample):
        x, y, adjs, (start, stop) = prepared_sample
        return cls(
            x=x,
            y=y.squeeze() if y is not None else None,
            adjs=[Adj__from_fast_sampler(adj) for adj in adjs],
            idx_range=slice(start, stop)
        )

    def record_stream(self, stream):
        if self.x is not None:
            self.x.record_stream(stream)
        if self.y is not None:
            self.y.record_stream(stream)
        for adj in self.adjs:
            adj.record_stream(stream)

    def to(self, device, non_blocking=False):
        return PreparedBatch(
            x=self.x.to(
                device=device,
                non_blocking=non_blocking) if self.x is not None else None,
            y=self.y.to(
                device=device,
                non_blocking=non_blocking) if self.y is not None else None,
            adjs=[adj.to(device=device, non_blocking=non_blocking)
                  for adj in self.adjs],
            idx_range=self.idx_range
        )

    @property
    def num_total_nodes(self):
        return self.x.size(0)

    @property
    def batch_size(self):
        return self.idx_range.stop - self.idx_range.start


@dataclass
class FastSamplerConfig:
    x_cpu: torch.Tensor
    x_gpu: torch.Tensor
    y: torch.Tensor
    rowptr: torch.Tensor
    col: torch.Tensor
    idx: torch.Tensor
    batch_size: int
    sizes: List[int]
    skip_nonfull_batch: bool
    pin_memory: bool
    distributed: bool
    partition_book: RangePartitionBook
    cache: Cache
    force_exact_num_batches: bool
    exact_num_batches: int
    count_remote_frequency: bool
    use_cache: bool

    def to_fast_sampler(self) -> fast_sampler.Config:
        c = fast_sampler.Config()
        for field in fields(self):
            if not self.distributed and field.name == 'partition_book':
                continue
            setattr(c, field.name, getattr(self, field.name))

        return c

    def get_num_batches(self) -> int:
        if self.force_exact_num_batches: return self.exact_num_batches
        num_batches, r = divmod(self.idx.numel(), self.batch_size)
        if not self.skip_nonfull_batch and r > 0:
            num_batches += 1
        return num_batches


class FastSamplerStats(NamedTuple):
    total_blocked_dur: datetime.timedelta
    total_blocked_occasions: int

    @classmethod
    def from_session(cls, session: fast_sampler.Session):
        return cls(total_blocked_dur=session.total_blocked_dur,
                   total_blocked_occasions=session.total_blocked_occasions)

class FastSamplerDistributedStats(NamedTuple):
    remote_frequency_tensor: torch.Tensor
    remote_vertices_ordered_by_freq: torch.Tensor

    @classmethod
    def from_session(cls, session: fast_sampler.Session):
        assert session.num_consumed_batches == session.num_total_batches
        # Reduction should only be called once all the batches complete.
        # This is a no-op if the the reduction has already been performed.
        session.reduce_multithreaded_frequency_counts()
        return cls(remote_frequency_tensor=session.remote_frequency_tensor,
                   remote_vertices_ordered_by_freq=session.remote_vertices_ordered_by_freq,)


class FastSamplerIter(Iterator[PreparedBatch]):
    session: fast_sampler.Session

    def __init__(self, num_threads: int, max_items_in_queue:
                 int, cfg: FastSamplerConfig):
        ncfg = cfg.to_fast_sampler()
        self.session = fast_sampler.Session(
            num_threads, max_items_in_queue, ncfg)
        assert self.session.num_total_batches == cfg.get_num_batches()

    def __next__(self):
        if not self.session.config.distributed:
            sample = self.session.blocking_get_batch()
            if sample is None:
                raise StopIteration
            return PreparedBatch.from_fast_sampler(sample)
        else:
            sample = self.session.blocking_get_batch_distributed()
            if sample is None:
                raise StopIteration
            return ProtoDistributedBatch.from_fast_sampler(sample)

    def get_stats(self) -> FastSamplerStats:
        return FastSamplerStats.from_session(self.session)

    def get_distributed_stats(self) -> FastSamplerDistributedStats:
        return FastSamplerDistributedStats.from_session(self.session)


class ABCNeighborSampler(Iterable[PreparedBatch], Sized):
    @property
    @abstractmethod
    def idx(self) -> torch.Tensor:
        ...

    @idx.setter
    @abstractmethod
    def idx(self, idx: torch.Tensor) -> None:
        ...


@dataclass
class FastSampler(ABCNeighborSampler):
    num_threads: int
    max_items_in_queue: int
    cfg: FastSamplerConfig

    @property
    def idx(self):
        return self.cfg.idx

    @idx.setter
    def idx(self, idx: torch.Tensor) -> None:
        self.cfg.idx = idx

    @property
    def cache(self):
        return self.cfg.cache

    @cache.setter
    def cache(self, cache: Cache) -> None:
        self.cfg.cache = cache

    def __iter__(self):
        return FastSamplerIter(self.num_threads, self.max_items_in_queue,
                               self.cfg)

    def __len__(self):
        return self.cfg.get_num_batches()


@dataclass
class FastPreSampler(ABCNeighborSampler):
    cfg: FastSamplerConfig

    @property
    def idx(self):
        return self.cfg.idx

    @idx.setter
    def idx(self, idx: torch.Tensor) -> None:
        self.cfg.idx = idx

    def __iter__(self) -> Iterator[PreparedBatch]:
        cfg = self.cfg
        p = fast_sampler.full_sample(cfg.x, cfg.y, cfg.rowptr, cfg.col,
                                     cfg.idx, cfg.batch_size, cfg.sizes,
                                     cfg.skip_nonfull_batch, cfg.pin_memory)
        return (PreparedBatch.from_fast_sampler(sample)
                for sample in itertools.chain(*p))

    def __len__(self):
        return self.cfg.get_num_batches()
