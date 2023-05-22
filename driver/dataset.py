from ogb.nodeproppred import PygNodePropPredDataset
from typing import Mapping, NamedTuple, Any, Optional
from pathlib import Path
import torch
from torch_sparse import SparseTensor
from fast_sampler import to_row_major

# Temporarily using the old partition_book as an intermediate format.
from fast_trainer.partition_book import *


def get_sparse_tensor(edge_index, num_nodes=None, return_e_id=False):
    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if return_e_id:
            value = torch.arange(adj_t.nnz())
            adj_t = adj_t.set_value(value, layout='coo')
        return adj_t

    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    value = torch.arange(edge_index.size(1)) if return_e_id else None
    return SparseTensor(row=edge_index[0], col=edge_index[1],
                        value=value,
                        sparse_sizes=(num_nodes, num_nodes)).t()


class FastDataset(NamedTuple):
    name: str
    x: torch.Tensor
    y: torch.Tensor
    rowptr: torch.Tensor
    col: torch.Tensor
    split_idx: Mapping[str, torch.Tensor]
    meta_info: Mapping[str, Any]

    @classmethod
    def from_ogb(cls, name: str, root='dataset'):
        return cls.from_pyg(PygNodePropPredDataset(name=name, root=root))

    @classmethod
    def import_mag240(cls, adj_t, _x, _y, _split_idx, meta_info_dict):
        #data = dataset.data
        x = to_row_major(_x).to(torch.float16)
        y = _y.squeeze()

        if y.is_floating_point():
            y = y.nan_to_num_(-1)
            y = y.long()
        
        #adj_t = get_sparse_tensor(data.edge_index, num_nodes=x.size(0))
        rowptr, col, _ = adj_t.to_symmetric().csr()
        return cls(name='MAG240', x=x, y=y,
                   rowptr=rowptr, col=col,
                   split_idx=_split_idx,
                   meta_info=meta_info_dict)


    def get_num_iterations(self, minibatch_size: int):
        id_sizes = dict()
        names = ['train', 'valid', 'test']
        for n in names:
            id_sizes[n] = max(1,int(self.split_idx[n].numel()/minibatch_size))
        return id_sizes

    @classmethod
    def from_pyg(cls, dataset):
        data = dataset.data
        x = to_row_major(data.x).to(torch.float16)
        y = data.y.squeeze()

        if y.is_floating_point():
            y = y.nan_to_num_(-1)
            y = y.long()

        adj_t = get_sparse_tensor(data.edge_index, num_nodes=x.size(0))
        rowptr, col, _ = adj_t.to_symmetric().csr()
        return cls(name=dataset.name, x=x, y=y,
                   rowptr=rowptr, col=col,
                   split_idx=dataset.get_idx_split(),
                   meta_info=dataset.meta_info.to_dict())

    @classmethod
    def from_path(cls, _path, name, skip_features=False):
        path = Path(_path).joinpath(name)
        print("Path to dataset is " + str(path), flush=True)
        if not (path.exists() and path.is_dir()):
            dataset = cls.from_ogb(name, root=_path)
            dataset.save(_path)
            return dataset
        else:
            return cls.from_path_if_exists(_path, name, skip_features=skip_features)

    @classmethod
    def from_path_if_exists(cls, path, name, skip_features=False):
        path = Path(path).joinpath(name)
        assert path.exists() and path.is_dir()
        data = {
            field: torch.load(path.joinpath(field + '.pt'))
            for field in cls._fields if not skip_features or (field != 'y' and field != 'x')
        }
        if not skip_features:
          data['y'] = data['y'].long()
          data['x'] = data['x'].to(torch.float16)
        else:
          data['y'] = torch.tensor([])
          data['x'] = torch.tensor([])
        assert data['name'] == name
        return cls(**data)

    def save(self, path):
        path = Path(path).joinpath(self.name)
        path.mkdir()
        for i, field in enumerate(self._fields):
            torch.save(self[i], path.joinpath(field + '.pt'))

    def adj_t(self):
        num_nodes = self.rowptr.size(0)-1
        return SparseTensor(rowptr=self.rowptr, col=self.col,
                            sparse_sizes=(num_nodes, num_nodes),
                            is_sorted=True, trust_data=True)

    def share_memory_(self):
        self.x.share_memory_()
        self.y.share_memory_()
        self.rowptr.share_memory_()
        self.col.share_memory_()

        for v in self.split_idx.values():
            v.share_memory_()

    @property
    def num_features(self):
        return self.x.size(1)

    @property
    def num_classes(self):
        if 'num classes' in self.meta_info:
            return int(self.meta_info['num classes'])
        else:
            return int(self.meta_info['num_classes'])


class DisjointPartFeatReorderedDataset(NamedTuple):
    """
    An extension of the FastDataset class for the setting in which node features are disjointly partitioned.
    It is intended for the distributed setting where each machine/GPU loads in a subset of the features.
    Key differences from FastDataset:
        1. The features (x) loaded in will only be a disjoint subset of all features in the graph.
           (FastDataset loaded all features in the graph into x). 
        2. The vertices have different vertex ids from the original graph.
           The vertices have been reordered so that vertices in a partition fall in a contiguous range.
           This would allow faster lookup of a vertex's partition and local index of the corresponding feature tensor in a partition.
        3. Training, validation, testing may follow a different execution scheme in the distributed setting.
           split_idx_parts is provided as full information of each machine's train, val, test vertices.
           It is essentially a nested mapping, with a split_idx for each partition.
    Member variables:
        name:               Same as FastDataset.
        rank:               The rank of the partition this dataset corresponds to.
        num_parts:          The number of partitions the full dataset has been partitoned into. 
        x:                  The disjoint subset of all features in the graph. 
        y:                  Same as FastDataset.
        rowptr:             Same as FastDataset.
        col:                Same as FastDataset.
        split_idx:          Same as FastDataset.
        split_idx_parts:    A split idx for each partition.
        part_offsets:       The vertex offsets at which partitions start (after the vertex reordering).
        meta_info:          May contain additional info compared to FastDataset.
    """
    name: str
    rank: int
    num_parts: int
    x: torch.Tensor
    y: torch.Tensor
    rowptr: torch.Tensor
    col: torch.Tensor
    split_idx: Mapping[str, torch.Tensor]
    split_idx_parts: Mapping[int, Mapping[str, torch.Tensor]]
    part_offsets: torch.Tensor
    meta_info: Mapping[str, Any]

    @classmethod
    def from_path(cls, _path, name, rank):
        path = Path(_path).joinpath(name)
        if not (path.exists() and path.is_dir()):
            print("Path is " + str(path))
            raise ValueError('ERROR dataset does not exist at specified path.')
        return cls.from_path_if_exists(_path, name, rank)

    @classmethod
    def from_path_if_exists(cls, path, name, rank):
        """
        E.g.:
            path = '~/metis-reordered-k2/'
            name = 'ogbn-products'
            rank = 0 (or 1 for 2 partitions)
        """
        #print("Split idx info")
        path = Path(path).joinpath(name)
        assert path.exists() and path.is_dir()
        some_fields = [field for field in cls._fields]
        some_fields.remove('x')
        some_fields.remove('rank')
        data = {
            #field: torch.load(path.joinpath(field.replace('rowptr', 'mod_rowptr').replace('col', 'mod_col') + '.pt'))
            field: torch.load(path.joinpath(field + '.pt'))
            for field in some_fields
        }
        data['y'] = data['y'].long()
        data['x'] = torch.load(path.joinpath('x' + str(rank) + '.pt')).to(torch.float16)
        print("Feature data sizes " + str(data['x'].size()))
        data['rank'] = rank
        assert data['name'] == name
        return cls(**data)

    """
    This method will be removed.
    Data related to the partitioning: features, train vertices, etc. is currently saved in a different format.
    This method reformat and saves the data from the older format to a format suitable to this class.
    Hack: passing in num_classes manually.
    """
    @classmethod
    def reformat_and_save(cls, name, num_parts, old_path, new_path, num_classes):
        """
        name:       The name of the dataset (e.g. ogbn-products).
        num_parts:  The expected number of partitions in the old-format dataset.
        old_path:   The path to the old-format dataset.
                    The dataset will be loaded from exactly this path.
                    E.g. rowptr is stored in old_path/rowptr.pt
        new_path:   The path to where the newly-formatted dataset will be saved.
                    The dataset will be saved to a subdir name in this path.
                    E.g. rowptr will be stored in new_path/name/rowptr.pt
        """
        print('Running..')
        if not old_path.endswith('/'): old_path += '/'
        new_path = Path(new_path).joinpath(name)
        print('Loading with RangePartitionBookLoader..')
        old = RangePartitionBookLoader(num_parts, old_path, rank=None, load_features=True, load_all_train_nid=False)
        data = {'name': name}
        print('Loading feature data and converting to half datatype..')
        for i, feature_partition in enumerate(['x'+str(j) for j in range(num_parts)]):
            data[feature_partition] = old.features[i].to(torch.float16)
        print('Loading rest of data..')
        data['y'] = old.labels
        data['rowptr'] = old.rowptr
        data['col'] = old.col
        data['split_idx_parts'] = {
            i: {
                'train': old.train_nid[i],
                'valid': old.val_nid[i],
                'test':  old.test_nid[i],
            }
            for i in range(num_parts)
        }
        data['split_idx'] = {
           'train': torch.cat([data['split_idx_parts'][i]['train'] for i in range(num_parts)]), 
           'valid': torch.cat([data['split_idx_parts'][i]['valid'] for i in range(num_parts)]), 
           'test': torch.cat([data['split_idx_parts'][i]['test'] for i in range(num_parts)]), 
        }
        data['part_offsets'] = old._partition_offsets
        data['meta_info'] = {'num classes': num_classes}
        data['num_parts'] = num_parts
        print('Saving data..')
        new_path.mkdir(parents=True)
        for field, datum in data.items():
            torch.save(datum, new_path.joinpath(field + '.pt'))
        print('[DONE]')

    @classmethod
    def reorder_and_save(cls,
                         dataset: FastDataset,
                         partition_labels: torch.Tensor,
                         probability_of_access: Optional[torch.Tensor],
                         dir: Path):
        """
        Save partitioned dataset object members based on partitioning labels.

        Relabel vertices based on a partitioning s.t. vertices that belong
        to the same partition have contiguous IDs and reorder all tensor
        accordingly.

        If vertex-wise probabilities of access are also provided, order
        vertices within each partition in descending order of probability of
        access.  The probabilty_of_access tensor may be 1D for global access
        probabilities or 2D (#parts x #vertices) for partition-wise access
        probabilities.

        """
        def csr_permute_symmetric(rowptr, col, value, invperm):
            A_csr = SparseTensor(rowptr=rowptr, col=col, value=value,
                                 is_sorted=True)
            rows, cols, vals = A_csr.coo()
            rows = invperm[rows]        # relabel rows
            cols = invperm[cols]        # relabel columns
            A_coo = SparseTensor(row=rows, col=cols, value=vals).coalesce()
            return A_coo.csr()

        num_parts = partition_labels.max() + 1
        sizes_partition = torch.bincount(partition_labels, minlength=num_parts)

        # Generate an `ordering_vals` tensor which when sorted gives the
        # desired ordering of vertices.  Vertices should be in ascending order
        # of partition ID globally and in descending order of probability of
        # access locally within each partition.  Since the probabilities are in
        # [0,1], we can simply add them to the (reversed) partition labels and
        # sort the resulting tensor.  Just to be safe, make the step between
        # partition IDs 2 instead of 1.
        ordering_vals = 2 * (partition_labels.max() - partition_labels.float())
        if probability_of_access is not None:
            if len(probability_of_access.size()) == 1: # global probs
                ordering_vals += probability_of_access
            elif len(probability_of_access.size()) == 2: # partition-wise probs
                for part in range(probability_of_access.size()[0]):
                    mask_part = partition_labels == part
                    ordering_vals[mask_part] += probability_of_access[part][mask_part]
            else:
                print("*WARNING* Unexpected dimensionality of probability_of_access"
                      + f" ({len(probability_of_access.size())})")


        perm = ordering_vals.argsort(descending=True)
        invperm = perm.argsort()

        rowptr_p, col_p, _ = csr_permute_symmetric(dataset.rowptr, dataset.col, None,
                                                   invperm)

        split_idx_p = dict()
        split_idx_parts = dict()
        for r in range(0, num_parts):
            split_idx_parts[r] = dict()

        for k, v in dataset.split_idx.items():
            indices = v
            partition_ids = partition_labels[indices]
            local_part_size = torch.bincount(partition_ids, minlength=num_parts)
            local_part_offset = torch.cat((torch.tensor([0]),
                                           torch.cumsum(local_part_size, 0)))
            relabeled_indices = invperm[v]
            isort = partition_ids.argsort()
            sorted_relabeled_indices = relabeled_indices[isort]
            for r in range(0, num_parts):
                split_idx_parts[r][k] = \
                    sorted_relabeled_indices[local_part_offset[r] : local_part_offset[r+1]]

        x_p = dataset.x[perm]
        y_p = dataset.y[perm]

        part_offsets_p = torch.cat((torch.tensor([0]),
                                    torch.cumsum(sizes_partition, 0)))
        x_split_parts = dict()
        for r in range(0, num_parts):
            x_split_parts[r] = x_p[part_offsets_p[r] : part_offsets_p[r+1]]

        # save

        prefix = dir / f"metis-reordered-k{num_parts}" / dataset.name
        prefix.mkdir(parents=True, exist_ok=False)
        torch.save(num_parts, prefix / "num_parts.pt")
        torch.save(rowptr_p, prefix / "rowptr.pt")
        torch.save(col_p, prefix / "col.pt")
        torch.save(split_idx_p, prefix / "split_idx.pt")
        torch.save(split_idx_parts, prefix / "split_idx_parts.pt")
        torch.save(part_offsets_p, prefix / "part_offsets.pt")
        torch.save(y_p, prefix / "y.pt")
        torch.save(dataset.meta_info, prefix / "meta_info.pt")
        torch.save(dataset.name, prefix / "name.pt")
        for r in range(0, num_parts):
            torch.save(x_split_parts[r].to(torch.float16).clone(), prefix / f"x{r}.pt")

    def get_RangePartitionBook(self):
        return RangePartitionBook(self.rank, self.num_parts, self.part_offsets)

    def get_num_iterations(self, minibatch_size: int):
        """
        Given a minibatch size, using the known train/valid/test splits across partitions, compute the the number of iterations
        such that each machine runs for an equal number of iterations during each of training, val, and test.
        Returns a dictionary out:
            {'valid': int, 'test': int, 'train': int}
            Where for example  out['train'] gives the number of training iterations for each machine to execute.
        """
        id_sizes = dict()
        names = ['train', 'valid', 'test']
        for name in names:
            id_sizes[name] = torch.Tensor([self.split_idx_parts[i][name].numel() for i in range(self.num_parts)])
        out = {k: None for k in id_sizes.keys()}
        for id_set_name, id_set_sizes in id_sizes.items():
            total = torch.sum(id_set_sizes)
            # Flooring will set fewer iterations, so we will err on creating slightly larger minibatches.
            num_iterations = max(1, torch.floor(total / minibatch_size).item())
            out[id_set_name] = int(num_iterations)
        return out

    @property
    def num_nodes(self):
        return self.rowptr.numel() - 1

    def adj_t(self):
        num_nodes = self.num_nodes
        return SparseTensor(rowptr=self.rowptr, col=self.col,
                            sparse_sizes=(num_nodes, num_nodes),
                            is_sorted=True, trust_data=True)

    def share_memory_(self):
        self.x.share_memory_()
        self.y.share_memory_()
        self.rowptr.share_memory_()
        self.col.share_memory_()
        # NOTE: Do not need all of this information for every machine for every execution scheme.
        # However, this is general and any other approach would be a premature optimization.
        for v in self.split_idx.values():
            v.share_memory_()
        for v in self.split_idx_parts.values():
            for v2 in v.values:
                v2.share_memory_()

    @property
    def num_features(self):
        return self.x.size(1)

    @property
    def num_classes(self):
        print (self.meta_info) 
        if 'num classes' in self.meta_info:
            return int(self.meta_info['num classes'])
        else:
            return int(self.meta_info['num_classes'])





