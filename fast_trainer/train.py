from io import UnsupportedOperation
from typing import Optional
import torch
import torch.nn.functional as F

import nvtx
from .samplers import PreparedBatch
from .transferers import DeviceIterator
from .concepts import TrainCore, TrainCallback
import torch.distributed
import time
from .utils import runtime_stats_cuda, is_performance_stats_enabled


def barebones_train_core(model: torch.nn.Module, batch: PreparedBatch, preload_hook = None, optimizer=None, sync=True):
    #return 0
    #with model.no_sync():

    if sync:
        with nvtx.annotate('forward', color='brown'):
            out = model(batch.x, batch.adjs)
            # QUESTION: do we need to squeeze here?
            # ANSWER: pretty sure we do, otherwise
            #   RuntimeError: 0D or 1D target tensor expected, multi-target not supported
            loss = F.nll_loss(out, batch.y.squeeze(-1))
            #loss = F.nll_loss(out, batch.y.squeeze())


        #if preload_hook != None:
        #    preload_hook.preload()
        #for stm in wait_streams:
        #    torch.cuda.current_stream().wait_stream(stm)
        #    stm.synchronize()
        #torch.cuda.synchronize()
        #with nvtx.annotate('afterforward', color='brown'):
        #    #if preload_hook != None:
        #    #    preload_hook.preload()
        #    for stm in wait_streams:
        #        torch.cuda.current_stream().wait_stream(stm)
        #        stm.synchronize()

        #    runtime_stats_cuda.end_region("train")
        #    runtime_stats_cuda.start_region("sampling", runtime_stats_cuda.get_last_event())
        #    runtime_stats_cuda.end_region("sampling")




        with nvtx.annotate('loss.backward', color='brown'):
            if True:
                loss.backward()
                #with nvtx.annotate('backward.optimizer', color='brown'):
                optimizer.step()
            else:
                with model.no_sync():
                    loss.backward()
    else:
        with model.no_sync():
            with nvtx.annotate('forward', color='brown'):
                out = model(batch.x, batch.adjs)
                loss = F.nll_loss(out, batch.y.squeeze(-1))
            with nvtx.annotate('loss.backward', color='brown'):
                loss.backward()
                #optimizer.step()


    #runtime_stats_cuda.start_region("train", runtime_stats_cuda.get_last_event())


    #time.sleep(0.01)
    return loss
    return


def make_eval_and_loss(module, train_core):
    def eval_and_loss(*args, **_):
        return train_core(module, PreparedBatch(*args))

    return eval_and_loss


def data_parallel_train(model: torch.nn.Module,
                        train_core: TrainCore,
                        devit: DeviceIterator,
                        optimizer: torch.optim.Optimizer, lr_scheduler,
                        cb: Optional[TrainCallback] = None,
                        dataset=None,
                        devices=None) -> None:
    model.train()
    while True:
        optimizer.zero_grad()

        # Replicate the model (send weights) to devices
        #
        # TODO: This might not be non-blocking. If so, this is a
        #       PyTorch issue!
        #
        # NOTE: This creates "replica modules" whose gradients are
        #       automatically reduced during the computation of
        #       the backward pass.
        replicas = torch.nn.parallel.replicate(model, devit.devices)
        inputs = next(devit, [])

        if len(inputs) == 0:
            break

        replicas = replicas[:len(inputs)]
        devices = devit.devices[:len(inputs)]

        funcs = [make_eval_and_loss(replica, train_core)
                 for replica in replicas]

        # NOTE: devices can be inferred from inputs, but providing
        # them is faster
        results = torch.nn.parallel.parallel_apply(
            funcs, inputs, devices=devices)

        optimizer.step()

        if lr_scheduler is not None:
            for batch_res in results:
                lr_scheduler.step(batch_res)

        if cb is not None:
            cb(inputs, results)

        # skip replicating next iter, if we have no more data
        if len(inputs) < len(devit.devices):
            break

        del inputs
        del results
        del funcs


def serial_train_ns(model: torch.nn.Module,
                    train_core: TrainCore,
                    devit: DeviceIterator,
                    optimizer: torch.optim.Optimizer,
                    lr_scheduler,
                    cb: Optional[TrainCallback] = None,
                    dataset=None,
                    devices=None) -> None:
    print("serial_train_ns  disabled because of dataset.x_cpu/x_gpu split." )
    assert False
    ''' Serial training code that uses PyG's NeighborSampler '''
    model.train()

    runtime_stats_cuda.start_epoch()

    if devices is not None:
        assert len(devices) == 1
        device = devices[0]

    runtime_stats_cuda.start_region("total")
    runtime_stats_cuda.start_region(
        "sampling", runtime_stats_cuda.get_last_event())
    iterator = iter(devit)
    runtime_stats_cuda.end_region("sampling")
    runtime_stats_cuda.end_region("total", runtime_stats_cuda.get_last_event())

    while True:
        runtime_stats_cuda.start_region(
            "total", runtime_stats_cuda.get_last_event())


        #runtime_stats_cuda.start_region("synchronization", runtime_stats_cuda.get_last_event())
        #torch.distributed.barrier() 
        #runtime_stats_cuda.end_region("synchronization")

        runtime_stats_cuda.start_region(
            "sampling", runtime_stats_cuda.get_last_event())
        inputs = next(iterator, [])
        if len(inputs) == 0:
            runtime_stats_cuda.end_region("sampling")
            runtime_stats_cuda.end_region(
                "total", runtime_stats_cuda.get_last_event())
            break

        batch_size, n_id, adjs = inputs
        xs = torch.empty(len(n_id), dataset.x.shape[1], dtype=dataset.x.dtype,
                         layout=dataset.x.layout, pin_memory=True)
        torch.index_select(dataset.x, 0, n_id, out=xs)
        ys = torch.empty(batch_size, dtype=dataset.y.dtype,
                         layout=dataset.y.layout, pin_memory=True)
        torch.index_select(dataset.y, 0, n_id[:batch_size], out=ys)

        runtime_stats_cuda.end_region("sampling")

        runtime_stats_cuda.start_region(
            "data_transfer", runtime_stats_cuda.get_last_event())
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        adjs = [adj.to(device, non_blocking=True) for adj in adjs]
        runtime_stats_cuda.end_region("data_transfer")

        runtime_stats_cuda.start_region(
            "train", runtime_stats_cuda.get_last_event())
        optimizer.zero_grad()
        out = model(xs, adjs)
        loss = F.nll_loss(out, ys)
        loss.backward()
        result = loss
        optimizer.step()

        if lr_scheduler is not None:
            world_size = 1.0
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(result)
                world_size = 1.0*torch.distributed.get_world_size()
            lr_scheduler.step(result / world_size)

        if cb is not None:
            cb([inputs], [result])
        runtime_stats_cuda.end_region("train")
        runtime_stats_cuda.end_region(
            "total", runtime_stats_cuda.get_last_event())
    runtime_stats_cuda.end_epoch()
    runtime_stats_cuda.report_stats(
        {'total': 'Total', 'data_transfer': 'Data Transfer', 'sampling': 'Sampling + Slicing', 'train': 'Train'})


def serial_train_with_timing(model: torch.nn.Module,
                             train_core: TrainCore,
                             devit: DeviceIterator,
                             optimizer: torch.optim.Optimizer,
                             lr_scheduler,
                             cb: Optional[TrainCallback] = None,
                             dataset=None,
                             devices=None) -> None:
    print("in serial train with timing", flush=True)
    '''
    Serial train code that uses SALIENT's FastSampler with basic instrumentation to time different operations.
    This function is only designed for single GPU non-distributed setting.
    '''
    if dataset is not None and devices is not None:
        serial_train_ns(model, train_core, devit, optimizer, lr_scheduler,
                        cb=cb, dataset=dataset, devices=devices)
        return

    model.train()

    runtime_stats_cuda.start_region("total")
    runtime_stats_cuda.end_region("total")

    runtime_stats_cuda.start_epoch()
    i = 0
    while True:
        runtime_stats_cuda.start_region(
            "total", runtime_stats_cuda.get_last_event())
        runtime_stats_cuda.start_region(
            "load_batch", runtime_stats_cuda.get_last_event())
        try:
            inp, = next(devit)
            i += 1
            # The sampling region is opened, but not closed by next method of devit.
            runtime_stats_cuda.end_region("sampling")
        except StopIteration:
            # The sampling region is opened, but not closed by next method of devit.
            runtime_stats_cuda.end_region("sampling")
            runtime_stats_cuda.end_region(
                "load_batch", runtime_stats_cuda.get_last_event())
            runtime_stats_cuda.end_region(
                "total", runtime_stats_cuda.get_last_event())
            break
        runtime_stats_cuda.end_region(
            "load_batch", runtime_stats_cuda.get_last_event())

        runtime_stats_cuda.start_region(
            "train", runtime_stats_cuda.get_last_event())
        optimizer.zero_grad()
        if i % 4 == 0:
            do_sync = True
        else:
            do_sync = True 
        #result = train_core(model, inp, wait_streams=[devit.streams['meta_all2all'], devit.streams['indices_all2all'], devit.streams['features_all2all']], preload_hook=devit, optimizer=optimizer, sync=do_sync)
        train_core(model, inp, preload_hook=devit, optimizer=optimizer, sync=do_sync)

        #if True: #preload_hook != None:
        #    runtime_stats_cuda.end_region("train")
        #    devit.preload()
        #    runtime_stats_cuda.end_region("sampling")

        #runtime_stats_cuda.start_region("train", runtime_stats_cuda.get_last_event())


        # Use of the LR in the loop here may cause a performance penalty.
        if lr_scheduler is not None:
            print("Here were are!!!!")
            world_size = 1.0
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(result)
                world_size = 1.0*torch.distributed.get_world_size()
            lr_scheduler.step(result / world_size)

        if cb is not None:
            pass
            #cb([inp], [result])
        runtime_stats_cuda.end_region("train")
        #runtime_stats_cuda.start_region("synchronization", runtime_stats_cuda.get_last_event())
        #torch.distributed.barrier() 
        #runtime_stats_cuda.end_region("synchronization")

        runtime_stats_cuda.end_region("total")

    runtime_stats_cuda.start_region("total")
    torch.cuda.synchronize()
    runtime_stats_cuda.end_region("total")
    #print("Number of iterations " + str(i), flush=True)

    runtime_stats_cuda.end_epoch()
    devit.print_stats()
    #runtime_stats_cuda.report_stats({'block_1': 'block_1', 'block_2': 'block_2', 'block_3':'block_3', 'block_4':'block_4', 'block_5':'block_5', 'block_6':'block_6', 'block_7':'block_7', 'block_8':'block_8', 'block_9':'block_9', 'total':'Total'})
    #runtime_stats_cuda.report_stats({'total': 'Total', 'data_transfer': 'Data Transfer', 'sampling': 'Sampling + Slicing', 'train': 'Train', 'preamble': 'Preamble'})
    #runtime_stats_cuda.report_stats({'block_1': 'block_1', 'block_2': 'block_2', 'block_3':'block_3', 'block_4':'block_4', 'block_5':'block_5', 'block_6':'block_6', 'block_7':'block_7', 'block_8':'block_8', 'block_9':'block_9', 'total':'Total'})
    #print(model._get_ddp_logging_data())


def serial_train(model: torch.nn.Module,
                 train_core: TrainCore,
                 devit: DeviceIterator,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler,
                 cb: Optional[TrainCallback] = None,
                 dataset=None,
                 devices=None) -> None:
    ''' Serial train code that uses SALIENT's FastSampler '''
    if dataset is not None and devices is not None:
        serial_train_ns(model, train_core, devit, optimizer, lr_scheduler,
                        cb=cb, dataset=dataset, devices=devices)
        return

    if is_performance_stats_enabled():
        serial_train_with_timing(model, train_core, devit, optimizer, lr_scheduler,
                                 cb=cb, dataset=dataset, devices=devices)
        return

    model.train()

    iterator = iter(devit)

    while True:
        try:
            inp, = next(iterator)
        except StopIteration:
            break
        optimizer.zero_grad()
        #result = train_core(model, inp)
        result = train_core(model, inp, preload_hook=devit, optimizer=optimizer, sync=True)
        optimizer.step()
        if lr_scheduler is not None:
            world_size = 1.0
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(result)
                world_size = 1.0*torch.distributed.get_world_size()
            lr_scheduler.step(result.cpu()/world_size)
        if cb is not None:
            cb([inp], [result])
