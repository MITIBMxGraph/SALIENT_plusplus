import argparse


class LoadFromFile (argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            contents = f.read()
            # parse arguments in the file and store them in a blank namespace
            data = parser.parse_args(contents.split(), namespace=None)
            for k, v in vars(data).items():
                if k not in ["dataset_name", "job_name"]:
                    setattr(namespace, k, v)


def make_parser():
    parser = argparse.ArgumentParser(description="start an experiment")
    parser.add_argument("dataset_name",
                        help="name of the OGB dataset",
                        type=str)
    parser.add_argument("job_name",
                        help="name of the job",
                        type=str)
    parser.add_argument("--experimental_explicit_batches",
                        help="Experimental feature. Support training with explicitly selected batches of vertices.",
                        action="store_true")
    parser.add_argument("--dataset_root",
                        help="dataset root path",
                        type=str, default=f"fast_dataset/")
    parser.add_argument("--total_num_nodes",
                        help="total number of nodes to use",
                        type=int, default=1)
    parser.add_argument("--max_num_devices_per_node",
                        help="maximum number of devices per node",
                        type=int, default=1)
    parser.add_argument("--gpu_percent",
                        help="percent of data on gpu. min 1/1000 max 999/1000",
                        type=float)
    parser.add_argument("--cache_creation_epochs",
                        help="number of epochs to run before deciding cache",
                        type=int, default=0)
    # Cache size is a percentage to replicate.  E.g. passing in 30
    # here will give an intended replication factor of 30%.  Note:
    # this is different from the effective replication factor.
    parser.add_argument("--cache_size",
                        help="replication factor. between 0 and 100",
                        type=int, default=0)
    parser.add_argument("--train_batch_size",
                        help="size of training batches",
                        type=int, default=1024)
    parser.add_argument("--test_batch_size",
                        help="size of validation/testing batches",
                        type=int, default=4096)
    parser.add_argument("--final_test_batchsize",
                        help="size of testing batches",
                        type=int, default=1024)
    parser.add_argument("--train_fanouts",
                        help="training fanouts",
                        type=int, default=[15, 10, 5], nargs="*")
    parser.add_argument("--batchwise_test_fanouts",
                        help="testing fanouts",
                        type=int, default=[20, 20, 20], nargs="*")
    parser.add_argument("--final_test_fanouts",
                        help="testing fanouts",
                        type=int, default=[20, 20, 20], nargs="*")
    parser.add_argument("--test_epoch_frequency",
                        help="number of epochs to train before testing occurs",
                        type=int, default=20)
    parser.add_argument("--epochs",
                        help="total number of epochs to train",
                        type=int, default=21)
    parser.add_argument("--hidden_features",
                        help="number of hidden features",
                        type=int, default=256)
    parser.add_argument("--lr",
                        help="learning rate",
                        type=float, default=0.003)
    parser.add_argument("--patience",
                        help="patience to use in the LR scheduler",
                        type=int, default=1000)
    parser.add_argument("--use_lrs",
                        help="use learning rate scheduler",
                        action="store_true")
    # See driver/main.py/get_model_type() for available choices
    parser.add_argument("--model_name",
                        help="name of the model to use",
                        type=str, default="SAGE")
    parser.add_argument("--trials",
                        help="number of trials to run",
                        type=int, default=10)
    # A rule of thumb for num_workers is between the number of
    # physical CPUs and 3/2 times of hardware threads.
    parser.add_argument("--num_workers",
                        help="number of workers",
                        type=int, default=70)
    parser.add_argument("--train_max_num_batches",
                        help="max number of training batches waiting in queue",
                        type=int, default=100)
    parser.add_argument("--test_max_num_batches",
                        help="max number of testing batches waiting in queue",
                        type=int, default=50)
    parser.add_argument("--output_root",
                        help="the root of output storage",
                        type=str, default=f"job_output/")
    parser.add_argument("--ddp_dir",
                        help="coordination directory for ddp multinode jobs",
                        type=str, default=f"NONE")
    parser.add_argument("--pipeline_disabled",
                        help="whether to disable pipelining",
                        action="store_true")
    parser.add_argument("--distribute_data",
                        help="distribute the node features",
                        action="store_true")
    # Setting determinstic comes with a performance penalty --- due,
    # in part, to device-to-host data transfers to pageable memory by
    # deterministic versions of algorithms.
    parser.add_argument("--make_deterministic",
                        help="make the training/inference deterministic",
                        action="store_true")
    parser.add_argument("--one_node_ddp",
                        help="do DDP when total_num_nodes=1",
                        action="store_true")
    parser.add_argument("--do_test_run",
                        help="only run inference on the test set",
                        action="store_true")
    parser.add_argument("--do_test_run_filename",
                        help="the filename of model to load for the test run",
                        type=str, default=f"NONE", nargs='*')
    parser.add_argument("--num_layers",
                        help="number of layers",
                        type=int, default=3)
    parser.add_argument("--overwrite_job_dir",
                        help="if a job directory exists, delete it",
                        action="store_true")
    parser.add_argument("--performance_stats",
                        help="collect detailed performance statistics",
                        action="store_true")
    parser.add_argument("--train_type",
                        help="training type",
                        type=str, default="serial",
                        choices=("serial", "dp"))
    parser.add_argument("--test_type",
                        help="testing type",
                        type=str, default="batchwise",
                        choices=("layerwise", "batchwise"))
    parser.add_argument("--train_prefetch",
                        help="prefetch for training",
                        type=int, default=1),
    parser.add_argument("--test_prefetch",
                        help="prefetch for testing",
                        type=int, default=1)
    parser.add_argument("--train_sampler",
                        help="training sampler",
                        type=str, default="FastSampler")
    parser.add_argument("--verbose",
                        help="print log entries to stdout",
                        action="store_true")

    # The following arguments involve various caching strategies. A
    # user should just use the default.
    
    parser.add_argument("--cache_strategy",
                        help="",
                        type=str, default="vip")
    # Example of data: which vertices get cache hits
    parser.add_argument("--datacollector_root",
                        help="the root of collected data output",
                        type=str, default=f"data_collector/")
    parser.add_argument("--datacollector_save",
                        help="whether to write collected data to file",
                        action="store_true")
    # Whether to run computation with the model (train, test, etc.) or
    # simulate communication (no forward/backward passes).
    parser.add_argument("--execution_mode",
                        help="",
                        choices=["computation", "communication_simulation"],
                        default="computation")
    # If execution_mode == 'computation', whether to run normally or
    # create a cache before computation and then use it to allievate
    # communicaiton.
    parser.add_argument("--computation_mode",
                        help="",
                        choices=["normal", "frequency_cache"],
                        default="normal")
    # If execution_mode == 'communication_simulation', whether to
    # simulate normal execution or to simulate after creating a cache
    parser.add_argument("--communication_simulation_mode",
                        help="",
                        choices=["normal", "frequency_cache"],
                        default="normal")
    # Two schemes.
    #
    # Fully random (default): Minibatches are fully random. At the
    # beginning of each epoch the rank 0 machine shuffles all training
    # nodes and scatters 1/kth of the shuffled training indices to
    # each of the k machines. Each iteration each machine will then
    # compute a microbatch composed of the next minibatch_size/k
    # vertices it gathered.
    #
    # Federated: Minibatches are not fully random. At the beginning of
    # each epoch, each of the k machines shuffles the training ids on
    # its partition, then each iteration each machine will compute a
    # microbatch composed of the next minibatch_size/k vertices that
    # it shuffled locally. Note: If the training nodes are not
    # partitioned equally this can also lead to an uneven number of
    # iterations/microbatches for each machine. Currently we force
    # even number of iterations which can lead to uneven minibatch
    # sizes for different machines.
    parser.add_argument("--load_balance_scheme",
                        help="",
                        choices=["fully_random", "federated"],
                        default="fully_random")

    return parser
