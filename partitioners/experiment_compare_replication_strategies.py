from driver.dataset import FastDataset
from driver.main import get_dataset
import torch
import torch_sparse
import os
import sys
import torch_geometric
import time

from partitioners.partition import metis_partition
from partitioners.eval_quality import evaluate_quality, generate_cache, refine_partition, get_frequency_tensors_fast

print (sys.argv)
print(sys.argv[1])
if len(sys.argv) < 2:
    print("Not enough arguments, provide NUM_PARTS arg")
    quit()
else:
    print("Running for NUM_PARTS = " + str(int(sys.argv[1])))
    #quit()

NUM_NODE_SAMPLES = 5
NUM_PARTS = int(sys.argv[1])
FANOUTS = [15,10,5]

dataset = get_dataset('ogbn-papers100M', './dataset', skip_features=True)

rowptr,col,_ = dataset.adj_t().csr()

FILENAME='ogbn-papers100M-'+str(NUM_PARTS)
CACHE_FILENAME = 'saved_caches/' + FILENAME + '_cache_'+str(NUM_PARTS)+'_parts'
parts = torch.load('good_partitions/'+str(FILENAME)+'.pt')


num_parts = int(NUM_PARTS)
result_list = []
normalized_result_list = []


# STRAT_LIST = ['frequency', 'frequency_analytic', 'frequency_analytic_notaylor',\
# 'frequency_analytic_cascade_notaylor', 'degree', 'degree-reachable',\
# 'numpaths-reachable', 'random-walk']

STRAT_LIST = ['frequency_analytic']

for strategy in STRAT_LIST:
    new_parts = torch.tensor(parts)
    time_start = time.perf_counter()
    get_frequency_tensors_fast(dataset, FANOUTS, new_parts, 'cuda:0')
    time_end = time.perf_counter()
    print("Total time = " + str(time_end - time_start))
    quit()


    cache_list_func = generate_cache(dataset, FANOUTS, torch.tensor(parts), 0.05, cache_strategy=strategy, cache_filename=None)
    for rfactor in [0,0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28]:
        cache_list = cache_list_func(rfactor)
        result = evaluate_quality(dataset, FANOUTS, torch.tensor(parts), cache_list=cache_list)
        result['rfactor_target'] = rfactor
 
        total_size = 0
        for x in cache_list:
            total_size += x.size()[0]
        
        num_vertices = parts.size()[0]
        result['rfactor_achieved'] = total_size/num_vertices
        result['normalized_rfactor_target'] = rfactor/num_parts
        result['normalized_rfactor_achieved'] = result['rfactor_achieved']/num_parts
        result['strategy'] = strategy
        result['nparts'] = NUM_PARTS
        result_list.append(result)
        print(result) 

    for normalized_rfactor in [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]:
        cache_list = cache_list_func(normalized_rfactor*(num_parts-1))
        result = evaluate_quality(dataset, FANOUTS, torch.tensor(parts), cache_list=cache_list)
        result['rfactor_target'] = normalized_rfactor*(num_parts-1)
 
        total_size = 0
        for x in cache_list:
            total_size += x.size()[0]
        
        num_vertices = parts.size()[0]
        result['rfactor_achieved'] = total_size/num_vertices
        result['normalized_rfactor_target'] = normalized_rfactor
        result['normalized_rfactor_achieved'] = result['rfactor_achieved']/(num_parts-1)
        result['strategy'] = strategy
        result['nparts'] = NUM_PARTS
        normalized_result_list.append(result)
        print(result) 

    
open('updated-results-'+FILENAME+'.pobj', 'w+').write(str(result_list))
open('updated-normalized-results-'+FILENAME+'.pobj', 'w+').write(str(normalized_result_list))
print("Done")
