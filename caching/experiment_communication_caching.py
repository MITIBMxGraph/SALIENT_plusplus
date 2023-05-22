"""
This module contains functions for evaluating the communication volume among
partitions in distributed GNN training with remote-vertex caching.

This file can be executed as a script with various command-line arguments to
run simulation experiments and measure communication volume with different
caching schemes.
"""


from driver.main import get_dataset

import argparse
import importlib
import torch

from pathlib import Path

import caching.vip as vip
import caching.util as util
import caching.parse_communication_volume_results as comm_parser

if importlib.util.find_spec("torch_geometric.loader") is not None:
    import torch_geometric.loader
    if hasattr(torch_geometric.loader, "NeighborSampler"):
        from torch_geometric.loader import NeighborSampler
    else:
        from torch_geometric.data import NeighborSampler
else:
    from torch_geometric.data import NeighborSampler


# ==================================================
# COMMAND-LINE ARGUMENTS
# ==================================================


ALL_CACHE_SCHEMES = ['degree-reachable', 'num-paths-reachable', 'halo-1hop',
                     'vip-simulation', 'vip-analytical']
ALL_REPLICATION_FACTORS = [0, 0.01, 0.05, 0.10, 0.20, 0.50, 1.00]


def parse_args_cache_simulation():
    """Parse command-line arguments for VIP caching simulation experiments."""
    parser = argparse.ArgumentParser(
        description="Run VIP caching communication simulation experiments.",
        add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_name", required=True, type=str,
                        help="Dataset name")
    parser.add_argument("--dataset_dir", required=True, type=str,
                        help="Path to the partitioned dataset directory")
    parser.add_argument("--partition_labels_dir", required=True, type=str,
                        help="Path to the partition-label tensors directory")
    parser.add_argument("--num_partitions", required=True, type=int,
                        help="Number of partitions")
    parser.add_argument("--fanouts", default=[15, 10, 5], type=int, nargs="+",
                        help="Layer-wise sampling fanouts")
    parser.add_argument("--minibatch_size", default=1024, type=int,
                        help="Minibatch size")
    parser.add_argument("--num_epochs_eval", default=10, type=int,
                        help="Number of simulated epochs for communication evaluation")
    parser.add_argument("--cache_schemes", default=ALL_CACHE_SCHEMES, type=str, nargs="+",
                        help="VIP caching scheme names")
    parser.add_argument("--replication_factors", default=ALL_REPLICATION_FACTORS, type=float, nargs="+",
                        help="Cache replication factors (alpha)")
    parser.add_argument("--num_epochs_vip_sim", default=2, type=int,
                        help="Number of epochs for simulation-based VIP weight estimation")
    parser.add_argument("--num_workers_sampler", default=20, type=int,
                        help="Number of CPU workers used by the SALIENT fast sampler")
    parser.add_argument("--output_prefix", default="results-sim-comm", type=str,
                        help="Prefix name for communication simulation results output .pobj file.")
    parser.add_argument("--store_sim_accesses", default=False, type=bool,
                        help="set to 1 to store vertex-wise access statistics after simulation.")
    parser.add_argument("--use_sim_accesses_file", default=None,
                        help="If specified, skip evaluation simulation and use vertex accesses from file")
    return parser.parse_args()


# ==================================================
# RUN VIP CACHING SIMULATION EXPERIMENTS
# ==================================================


def run_vip_cache_experiments(
        dataset_name,
        dir_dataset,
        dir_partitions,
        num_partitions,
        fanouts,
        minibatch_size,
        num_epochs_eval,
        schemes,
        replication_factors,
        num_epochs_vip_sim,
        num_workers_sampler,
        prefix_output,
        file_vertex_accesses_per_partition,
        verbose = True
):
    """Run VIP cache simulation experiments."""
    util.print_if(verbose, f"Loading '{dataset_name}', partitions={num_partitions}", 0)
    dataset = get_dataset(dataset_name, dir_dataset, skip_features=True)
    path_partitions = f"{dir_partitions}/{dataset_name}-{num_partitions}.pt"
    partition_ids = torch.load(path_partitions)
    num_vertices = partition_ids.size()[0]

    if file_vertex_accesses_per_partition is None:
        vertex_accesses_per_partition = vip.simulate_vertex_accesses(
            dataset, partition_ids, fanouts, minibatch_size,
            num_epochs_eval, num_workers_sampler, verbose
        )
    else:
        util.print_if(verbose, "Skipping simulation, using access statistics from file: " +
                      file_vertex_accesses_per_partition, 0)
        vertex_accesses_per_partition = torch.load(file_vertex_accesses_per_partition)

    comm_result_list = []

    for s in schemes:
        fun_cache_idx = vip.get_lambda_vip_cache(
            dataset, partition_ids, fanouts, minibatch_size, s,
            vertex_accesses_per_partition, num_epochs_vip_sim,
            num_workers_sampler, verbose
        )

        util.print_if(verbose, "Evaluating communication volumes", 0)
        for alpha in replication_factors:
            util.print_if(verbose, f"Replication factor = {alpha}", 1)

            cache_idx_per_partition = fun_cache_idx(alpha)
            comm_results = vip.evaluate_communication_volume(
                vertex_accesses_per_partition, partition_ids, cache_idx_per_partition
            )
            comm_results_avg = {k: v / num_epochs_eval for k, v in comm_results.items()}

            comm_results_avg['rfactor'] = alpha
            comm_results_avg['rfactor_achieved'] = \
                sum([x.size()[0] for x in cache_idx_per_partition]) / num_vertices
            comm_results_avg['strategy'] = s
            comm_results_avg['nparts'] = num_partitions

            comm_result_list.append(comm_results_avg)

        # END replication factor (alpha) loop
    # END vip scheme loop

    return comm_result_list, vertex_accesses_per_partition


# ==================================================
# MAIN
# ==================================================


if __name__ == "__main__":
    args = parse_args_cache_simulation()
    comm_result_list, vertex_accesses_per_partition = run_vip_cache_experiments(
        args.dataset_name, args.dataset_dir, args.partition_labels_dir,
        args.num_partitions, args.fanouts, args.minibatch_size,
        args.num_epochs_eval, args.cache_schemes, args.replication_factors,
        args.num_epochs_vip_sim, args.num_workers_sampler, args.output_prefix,
        args.use_sim_accesses_file, verbose=True
    )

    table_results, _ = comm_parser.tabulate_comm_results(comm_result_list)
    print("Average per-epoch communication in # of vertices")
    print(f"- {args.dataset_name}, " +
          f"{args.num_partitions} partitions, " +
          f"{args.num_epochs_eval} epochs, " +
          f"fanout ({','.join(str(x) for x in args.fanouts)}), " +
          f"minibatch size {args.minibatch_size}")
    print(table_results)

    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    file_out_comm_result = (f"{args.output_prefix}-{args.dataset_name}" +
                            f"-partitions-{args.num_partitions}" +
                            f"-minibatch-{args.minibatch_size}" +
                            f"-fanout-{'-'.join(str(f) for f in args.fanouts)}" +
                            f"-epochs-{args.num_epochs_eval}" +
                            f".pobj")
    print(f"Saving communication results ({file_out_comm_result})")
    with open(file_out_comm_result, "w+") as fp:
        fp.write(str(comm_result_list))

    if args.store_sim_accesses:
        file_out_vertex_accesses = (f"{args.output_prefix}-{args.dataset_name}" +
                                    f"-partitions-{args.num_partitions}" +
                                    f"-minibatch-{args.minibatch_size}" +
                                    f"-fanout-{'-'.join(str(f) for f in args.fanouts)}" +
                                    f"-epochs-{args.num_epochs_eval}" +
                                    f"_vertex-accesses.pt")
        print(f"Saving vertex-wise access stats per partition ({file_out_vertex_accesses})")
        torch.save(vertex_accesses_per_partition, file_out_vertex_accesses)
