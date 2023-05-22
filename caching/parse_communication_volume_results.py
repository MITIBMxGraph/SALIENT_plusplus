"""
This module provides a simple interface for parsing cross-partition
communication volumes from the .pobj output file of
`run_cache_simulation_experiments.py` and printing them as a CSV table.
"""

import argparse
from prettytable import PrettyTable


def parse_args():
    """Parse command-line arguments for communication-volume parser script."""
    parser = argparse.ArgumentParser(
        description="Parse VIP caching communication experiment results.",
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path",
        required=True,
        type=str,
        help="Path to input .pobj file",
    )
    parser.add_argument(
        "--path-csv-out",
        required=False,
        type=str,
        default=None,
        help="Path to output .csv file",
    )
    args = parser.parse_args()
    return args


def tabulate_comm_results(results):
    """VIP caching communication experiment results file -> list of CSV rows."""
    schemes = list(set([x['strategy'] for x in results]))
    alphas = list(set([x['rfactor'] for x in results]))

    comm_cross_dict = {(x['rfactor'], x['strategy']): x['cross'] for x in results}

    table_rows = [['alpha'] + schemes]
    csv_rows = [",".join(table_rows[0])]
    for a in sorted(alphas):
        row = [a]
        for s in schemes:
            row.append(comm_cross_dict[(a, s)])
        table_rows.append([str(a)] + ['%.2E' % x for x in row[1:]])
        csv_rows.append(",".join([str(x) for x in row]))

    table = PrettyTable()
    table.field_names = table_rows[0]
    table.add_rows(table_rows[1:])
    table.align = "r"

    return table, csv_rows


if __name__ == "__main__":
    args = parse_args()
    with open(args.path, "r") as fp:
        results = eval(fp.read())

    table, csv_rows = tabulate_comm_results(results)

    print(table)

    if args.path_csv_out is not None:
        with open(args.path_csv_out, "w"):
            for row in csv_rows:
                fp.write(row + "\n")
