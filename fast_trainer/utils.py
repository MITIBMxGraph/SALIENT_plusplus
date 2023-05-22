from contextlib import ContextDecorator
import statistics
import time
import torch
from typing import NamedTuple

import importlib.util

# imports for DataCollector
from argparse import Namespace
import datetime
import numpy as np

#import nvtx

from pathlib import Path
import torch.distributed as dist

# Convenience utility for recording runtime statistics during execution
#   and reporting the results after the run.
runtime_statistics = None
runtime_statistics_enabled = False

# prevents runtime statistics from being enabled.
performance_stats_enabled = False


class DataCollector:
    """
    A class designed to make collection of batch data simple.
    Supports multiple epochs.
    Warning:
        Only supports just running on training or communication simulation. Running evaluation may not be supported.
        Currently only supports single trials.
    Before every epoch that you want to save data for, call set_current_epoch to create a dir to save data in.
    """

    #@nvtx.annotate('DataCollector.__init__', color='red')
    def __init__(self, root: str, args: Namespace):
        """
        root:   The parent dir in which to create a dir to save data.
        args:   The original args used to launch the training process on each machine.
        """

        if args.datacollector_save:
            self.root = Path(root)
            # Within the root dir will create a dir for the whole run / execution.
            self.run_dir = None
            # Within the run dir will create a dir for each epoch.
            self.epoch_dirs = []

            # Start creating directories.
            self.run_path = self.distributed_create_dir('collected_data', self.root, suffix_time=True) 

            # Save metadata. Namely, the args running execution with.
            args_file = self.run_path.joinpath('args')
            self.np_savez_dict(args_file, vars(args))

        # Internal reference for the current epoch.
        self._current_epoch = None
        
    def distributed_create_dir(self, dirname: str, parent_dir: Path, suffix_time=False):
        """
        Create a dir when running in a distributed setting.
        Have the rank-0 machine create the directories.

        dirname         Name of the directory to create.
        parent_dir      Parent dir in which to create a directory.
        suffix_time     Whether to use datetimestr to create a unique directory. (E.g. for each run).
        """

        datetimestr = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        dirname = '_'.join((dirname, datetimestr)) if suffix_time else dirname
        new_dir = parent_dir.joinpath(dirname)
        if dist.get_rank() == 0:
            new_dir.mkdir(parents=True)
            print(f'Created data collector directory: {new_dir}..', flush=True)
        # Ensure the directory has been created by the rank 0 node before returning.
        dist.barrier()
        return new_dir 

    def new_epoch(self):
        # Sometimes do not have access to what epoch it is, just keep an internal count.
        # E.g. can just call this in the __init__ of the DistributedDevicePrefetcher.
        epoch = self._current_epoch + 1 if self._current_epoch is not None else 0
        self.set_current_epoch(epoch)
        
    def set_current_epoch(self, epoch: int):
        # Update the internal reference to the current epoch create an epoch dir for that epoch, and record it in the list of epoch dirs.
        self._current_epoch = epoch
        epoch_str = 'epoch'+str(self._current_epoch)
        self.epoch_dirs.append(self.distributed_create_dir(epoch_str, self.run_path, suffix_time=False))

    def get_epoch_data_filepath(self, data_name, use_rank=True):
        # Handy function for creating filepath from data_name when saving data specific for the last epoch.
        # E.g. the batch data within an epoch.
        name = '_'.join((f'rank{dist.get_rank()}', data_name)) if use_rank else data_name
        return self.epoch_dirs[-1].joinpath(name)

    def get_run_data_filepath(self, data_name, use_rank=True):
        # Handy function for creating filepath from data_name when saving data applicable to all epochs in a run.
        # E.g. the cache used across multiple epochs.
        name = '_'.join((f'rank{dist.get_rank()}', data_name)) if use_rank else data_name
        return self.run_path.joinpath(name)

    # Conveniece wrappers for flush printing saved files when using multiple processes.
    @staticmethod
    def np_savez_dict(f, data_dict):
        np.savez(f, **data_dict)
        print(f'Saved {f}..', flush=True)

    @staticmethod
    def np_savez_list(f, data_list):
        np.savez(f, *data_list)
        print(f'Saved {f}..', flush=True)


def is_performance_stats_enabled():
    global performance_stats_enabled
    return performance_stats_enabled


class RuntimeStatisticsCUDA:
    def __init__(self, name: str):
        self.stat_lists = dict()
        self.stat_lists_cuda = dict()
        self.name = name
        self.epoch_counter = 0
        self.cuda_timer_lists = dict()
        self.last_event = None
        self.cuda_times = dict()
        self.cuda_timer_start = dict()
        self.cuda_timer_end = dict()

    def start_epoch(self):
        self.epoch_counter += 1

    def get_last_event(self):
        return self.last_event

    def start_region(self, region_name, use_event=None):
        if not runtime_statistics_enabled:
            return
        if use_event is not None:
            self.cuda_timer_start[region_name] = use_event
            self.last_event = use_event
        else:
            self.cuda_timer_start[region_name] = torch.cuda.Event(
                enable_timing=True)
            self.last_event = self.cuda_timer_start[region_name]
            #print(torch.cuda.current_stream(), flush=True)
            self.cuda_timer_start[region_name].record()

    def end_region(self, region_name, use_event=None):
        if not runtime_statistics_enabled:
            return
        if use_event is not None:
            self.cuda_timer_end[region_name] = use_event
            self.last_event = use_event
        else:
            self.cuda_timer_end[region_name] = torch.cuda.Event(
                enable_timing=True)
            self.last_event = self.cuda_timer_end[region_name]
            self.cuda_timer_end[region_name].record(stream=torch.cuda.default_stream(device=torch.cuda.current_device()))
        if region_name not in self.cuda_timer_lists:
            self.cuda_timer_lists[region_name] = []
        self.cuda_timer_lists[region_name].append(
            (self.cuda_timer_start[region_name], self.cuda_timer_end[region_name]))

    def end_epoch(self):
        torch.cuda.synchronize()
        for x in self.cuda_timer_lists.keys():
            total = self.cuda_timer_lists[x][0][0].elapsed_time(
                self.cuda_timer_lists[x][0][1])
            for y in self.cuda_timer_lists[x][1:]:
                total += y[0].elapsed_time(y[1])
            if x not in self.cuda_times:
                self.cuda_times[x] = []
            if self.epoch_counter > 1:
                self.cuda_times[x].append(total)
        self.cuda_timer_lists = dict()
        self.cuda_timer_start = dict()
        self.cuda_timer_end = dict()

    def report_stats(self, display_keys=None):
        #print("PARSE_STATS_DICT:" + str(self.cuda_times), flush=True)
        rows = []
        for x in sorted(self.cuda_times.keys()):
            print_name = x
            if display_keys is not None and x not in display_keys:
                continue
            elif display_keys is not None:
                print_name = display_keys[x]
            row = []
            if len(self.cuda_times[x]) < 2:
                row = [print_name, "N/A", "N/A"]
                if len(self.cuda_times[x]) == 1:
                    row = [print_name, statistics.mean(
                        self.cuda_times[x]), "N/A"]
            else:
                row = [print_name, statistics.mean(
                    self.cuda_times[x]), statistics.stdev(self.cuda_times[x])]
            rows.append(row)
        exists = importlib.util.find_spec("prettytable") is not None
        if not exists:
            print("activity, mean time (ms), stdev", flush=True)
            for row in rows:
                print(", ".join([str(a) for a in row]))
        else:
            import prettytable
            tab = prettytable.PrettyTable()
            num_samples = -1
            for x in sorted(self.cuda_times.keys()):
                assert num_samples < 0 or num_samples == len(
                    self.cuda_times[x])
                num_samples = len(self.cuda_times[x])
            tab.field_names = [
                "Activity ("+self.name+")", "Mean time (ms) (over " + str(num_samples) + " epochs)", "Stdev"]
            for x in rows:
                tab.add_row(x)
            # for x in sorted(self.cuda_times.keys()):
            #    print_name = x
            #    if display_keys is not None and x not in display_keys:
            #        continue
            #    elif display_keys is not None:
            #        print_name = display_keys[x]
            #    if len(self.cuda_times[x]) < 2:
            #        tab.add_row([print_name, "N/A", "N/A"])
            #    else:
            #        tab.add_row([print_name, statistics.mean(self.cuda_times[x]), statistics.stdev(self.cuda_times[x])])
            print(tab.get_string(sortby=tab.field_names[1]), flush=True)

        #print("===Showing runtime stats for: " + self.name + " ===")

        # for x in sorted(self.cuda_times.keys()):
        #    if len(self.cuda_times[x]) < 2:
        #        print (x + ": N/A")
        #    else:
        #        print (x + " Mean: " + str(statistics.mean(self.cuda_times[x])) + " Stdev: " + str(statistics.stdev(self.cuda_times[x])))
        return str(rows)

    def clear_stats(self):
        self.cuda_times = dict()
        self.cuda_timer_lists = dict()
        self.cuda_timer_start = dict()
        self.cuda_timer_end = dict()


runtime_stats_cuda = RuntimeStatisticsCUDA("SALIENT ogbn-arxiv")


def setup_runtime_stats(args):
    global runtime_statistics
    global performance_stats_enabled
    runtime_statistics = RuntimeStatistics("")
    if args.train_sampler == 'NeighborSampler':
        sampler = "PyG"
    else:
        sampler = "SALIENT"
    model = args.model_name
    dataset = args.dataset_name
    if args.performance_stats:
        performance_stats_enabled = True
    else:
        performance_stats_enabled = False
    runtime_stats_cuda.name = " ".join([sampler, dataset, model])


def enable_runtime_stats():
    if not performance_stats_enabled:
        return
    global runtime_statistics_enabled
    runtime_statistics_enabled = True


def disable_runtime_stats():
    if not performance_stats_enabled:
        return
    global runtime_statistics_enabled
    runtime_statistics_enabled = False


def start_runtime_stats_epoch():
    runtime_statistics.start_epoch()


def report_runtime_stats(logger=None):
    # if runtime_statistics is not None:
    #    runtime_statistics.report_stats()
    if is_performance_stats_enabled() and runtime_stats_cuda is not None:
        string_output = runtime_stats_cuda.report_stats(
            {'total': 'Total', 'data_transfer': 'Data Transfer', 'sampling': 'Sampling + Slicing', 'train': 'Train', 'preamble': 'Preamble'})
        if logger is not None:
            logger(('performance_breakdown_stats', string_output))


def append_runtime_stats(name, value):
    if runtime_statistics is not None and runtime_statistics_enabled:
        runtime_statistics.append_stat(name, value)


class RuntimeStatistics:
    def __init__(self, name: str):
        self.stat_lists = dict()
        self.name = name
        self.epoch_counter = 0

    def start_epoch(self):
        self.epoch_counter += 1

    def append_stat(self, name, value):
        # skip the first epoch.
        if self.epoch_counter == 1:
            return
        if name not in self.stat_lists:
            self.stat_lists[name] = []
        self.stat_lists[name].append(value)

    def report_stats(self):
        print("===Showing runtime stats for: " + self.name + " ===")
        for x in sorted(self.stat_lists.keys()):
            if len(self.stat_lists[x]) == 0:
                print(x + ": N/A")
            else:
                print(x + " Mean: " + str(statistics.mean(self.stat_lists[x])) + " Stdev: " + str(
                    statistics.stdev(self.stat_lists[x])))

    def clear_stats(self):
        self.stat_lists = dict()


class TimerResult(NamedTuple):
    name: str
    nanos: int

    def __str__(self):
        return f'{self.name} took {self.nanos / 1e9} sec'


class CUDAAggregateTimer:
    def __init__(self, name: str):
        self.name = name
        self.timer_list = []
        self._start = None
        self._end = None

    def get_start(self):
        return self._start

    def get_end(self):
        return self._end

    def start(self, timer=None):
        if timer is None:
            self._start = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._start = timer

    def end(self, timer=None):
        # print(torch.cuda.current_stream())
        # print(stream)
        if timer is None:
            self._end = torch.cuda.Event(enable_timing=True)
            self._end.record()
        else:
            self._end = timer
            # self._end.record(stream)
        self.timer_list.append((self._start, self._end))

    def report(self, do_print=False):
        torch.cuda.synchronize()
        total_time = self.timer_list[0][0].elapsed_time(self.timer_list[0][1])
        for x in self.timer_list[1:]:
            total_time += x[0].elapsed_time(x[1])
        if do_print:
            print("CUDA Aggregate (" + self.name + "): "+str(total_time)+" msec")
        return total_time


class Timer(ContextDecorator):
    def __init__(self, name: str, fn=print):
        self.name = name
        self._fn = fn

    def __enter__(self):
        self.start_ns = time.perf_counter_ns()
        self.stop_ns = None
        return self

    def stop(self):
        self.stop_ns = time.perf_counter_ns()

    @property
    def elapsed_time_seconds(self):
        nanos = self.stop_ns - self.start_ns
        return nanos / 1e9

    def __exit__(self, *_):
        if self.stop_ns is None:
            self.stop()
        nanos = self.stop_ns - self.start_ns
        self._fn(TimerResult(self.name, nanos))
        
