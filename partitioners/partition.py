from driver.dataset import FastDataset
from driver.main import get_dataset
import torch
import torch_sparse
import os
import sys

import pathlib
from pathlib import Path
import subprocess

import os, re


def get_libmetis_path():
    script_path = Path(os.path.realpath(os.path.dirname(__file__)))
    lib_path = script_path / ".." / "pkgs" / "lib" / "libmetis.so"
    return str(lib_path)
    #path = os.environ['PATH'].split(':')
    #bin_path = None
    #for x in path:
    #    if (Path(x) / 'ndmetis').is_file():
    #        bin_path = Path(x) / 'ndmetis'
    #        break
    #if bin_path == None:
    #    print("Error: Could not locate metis binaries, e.g. ndmetis. This script locates libmetis.so by inspecting the dynamically linked libraries for metis.")
    #    print("You may need to manually set the libmetis DLL path.")
    #    quit()
    #ldd_out = subprocess.check_output(['ldd', bin_path]).decode('utf-8')
    ##print(ldd_out)
    #libraries = {}
    #for line in ldd_out.splitlines():
    #  match = re.match(r'\t(.*) => (.*) \(0x', line)
    #  if match:
    #    libraries[match.group(1)] = match.group(2)
    #if 'libmetis.so' not in libraries:
    #    print ("Error: Could not locate libmetis.so, but some metis binaries (e.g., ndmetis) appear to be on PATH. You might need to manually set the libmetis DLL path.")
    #    quit()
    #return "/efs/home/tfk/parsnip-develop/pkgs/lib/libmetis.so"


#os.environ["METIS_DLL"] = os.environ["HOME"] + "/local/lib/libmetis.so"
#os.environ["METIS_DLL"] = "/efs/home/tfk/miniconda3/envs/salientplus/bin/../lib/libmetis.so" 
os.environ["METIS_DLL"] = get_libmetis_path() 
#/efs/home/tfk/miniconda3/envs/salientplus/lib/python3.9/site-packages/met
os.environ["METIS_IDXTYPEWIDTH"] = "64"
os.environ["METIS_REALTYPEWIDTH"] = "64"

import torch_metis as metis

def metis_partition_lax(rowptr, col, node_weights, edge_weights, nodew_dim=1, num_parts=2):
    G = metis.csr_to_metis(rowptr.contiguous(), col.contiguous(), node_weights, edge_weights, nodew_dim=nodew_dim)
    print("After")
    objval, parts = metis.part_graph(G, nparts=num_parts, ubvec=[1.001, 1.01])
    parts = torch.tensor(parts)
    print("Cost is " + str(objval))
    
    print("Partitions:")
    print(parts)
    
    print("Partition bin counts:")
    bincounts = torch.bincount(parts, minlength=4)
    print(bincounts)
    return parts


def metis_partition(rowptr, col, node_weights, edge_weights, nodew_dim=1, num_parts=2):
    G = metis.csr_to_metis(rowptr.contiguous(), col.contiguous(), node_weights, edge_weights, nodew_dim=nodew_dim)
    print("After")
    print(str([1.001]*nodew_dim))
    objval, parts = metis.part_graph(G, nparts=num_parts, ubvec=[1.001]*nodew_dim)
    parts = torch.tensor(parts)
    print("Cost is " + str(objval))
    
    print("Partitions:")
    print(parts)
    
    print("Partition bin counts:")
    bincounts = torch.bincount(parts, minlength=num_parts)
    print(bincounts)
    return parts

