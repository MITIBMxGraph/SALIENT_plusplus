from fast_sampler import RangePartitionBook
import os
import torch

class PartitionBookLoader():

    def __init__(self, k, path, rank=None, load_features=True, load_all_train_nid=True, load_id_sizes=False):
        # if partid is None load all partitions
        self.partid = rank
        self.k = k
        self.root = path
        self.load_csr()
        self.load_labels()
        if load_id_sizes:
            self.load_id_sizes()
        if not load_features:
            self.features = None
        else:
            self.load_features()

        # NOT ALLOWING THIS
        # Load both the locally available training ids and all training ids.
        # Which one is used dependends on load balancing scheme.
        """
        self.all_train_nid = self.load_all_train_nid()
        self.local_train_nid = self.load_nid('train')
        self.val_nid = self.load_nid('val')
        self.test_nid = self.load_nid('test')
        """

        # DEFAULT: load in all train, test, val ids on all machines
        # only change this after working
        # NOTE: atm val and test are local, update this
        if load_all_train_nid:
            self.train_nid = self.load_all_train_nid()
        else:
            self.train_nid = self.load_nid('train')
        self.val_nid = self.load_nid('val')
        self.test_nid = self.load_nid('test')

    def load_csr(self):
        rowptr = torch.load(''.join((self.root, 'rowptr.pt')))
        col = torch.load(''.join((self.root, 'col.pt')))
        edge_ids = torch.load(''.join((self.root, 'edge_ids.pt')))
        #self.csr = Csr(rowptr, col, edge_ids)
        self.rowptr, self.col, self.edge_ids = rowptr, col, edge_ids

    def load_labels(self):
        """ CURRENTLY assuming each machine has access to all labels, not partitioned. """
        self.labels = torch.load(''.join((self.root, 'labels.pt')))

    def load_features(self):
        if self.partid is not None:
            self.features = torch.load(''.join((self.root, 'part', str(self.partid), '/features.pt')))
        else:
            self.features = [None] * self.k
            for i in range(self.k):
                self.features[i] = torch.load(''.join((self.root, 'part', str(i), '/features.pt')))

    def load_nid(self, name):
        assert name == 'train' or name == 'val' or name == 'test'
        if self.partid is not None:
            result = torch.load(''.join((self.root, 'part', str(self.partid), '/', name, '_nid.pt')))
        else:
            result = [None] * self.k
            for i in range(self.k):
                result[i] = torch.load(''.join((self.root, 'part', str(i), '/', name, '_nid.pt')))
        return result

    def load_all_train_nid(self):
        # NOTE: made a separeate function as older partition books do not have an all_train_nid.pt file.
        file = ''.join((self.root, 'all_train_nid.pt'))
        if os.path.exists(file):
            result = torch.load(file)
        else:
            print('Detected that PartitionBook does not have all_train_nids yet, calculating and saving..')
            tensors = [None] * self.k
            for part in range(self.k):
                tensors[part] = torch.load(''.join((self.root, 'part', str(part), '/train_nid.pt')))
            result = torch.cat(tensors)
            # Since result didn't exist, save it.
            print('all_train_ids calculated, saving..')
            num_nodes = self.rowptr.numel() - 1
            ps = PartitionSaver(self.k, num_nodes, self.root)
            ps.save_all_train_nids(result)
        return result

    def load_id_sizes(self):
        """
        PartitionBook.id_sizes returns a dict of tensors where each tensor is of world_size, essentially a bincount by parititon.
        {'val': torch.Tensor, 'test': torch.Tensor, 'train': torch.Tensor}
        NOTE: if id_sizes.pt does not exist, to create self.csr must be already loaded to get num_nodes.
        NOTE: higher mem consumption, because have to load all tes, train, and val to calculate id_sizes as well.
        NOTE: would be nice if could tell size of tensor in .pt file without loading entirely,
                didn't see anything yet in https://pytorch.org/docs/stable/_modules/torch/serialization.html#load
              For now, a small optimization is to del the tensors after calculating the size.
        """
        file = ''.join((self.root, 'id_sizes.pt'))
        if os.path.exists(file):
            self.id_sizes = torch.load(file)
        # Some older saved partition books don't have the file, so just calculate and save now.
        else:
            print('Detected that PartitionBook does not have id_sizes yet, calculating and saving..')
            # A little hacky for num_nodes, requir
            num_nodes = self.rowptr.numel() - 1
            ps = PartitionSaver(self.k, num_nodes, self.root)

            print('May have higher memory consumption, need to load train, val, test ids for all partitions to calc sizes..')
            names = ['train', 'val', 'test']
            id_sizes = dict()
            for name in names:
                tensors = [None] * self.k
                for part in range(self.k):
                    tensors[part] = torch.load(''.join((self.root, 'part', str(part), '/', name, '_nid.pt')))
                id_sizes[name] = torch.Tensor([tensors[i].numel() for i in range(self.k)])
                # Try to free up memory immediately.
                for i in range(len(tensors)): del tensors[-1]

            print('id_sizes calculated, saving..')
            ps.save_id_sizes(id_sizes)
            self.id_sizes = id_sizes

    """
    # DEPRECATED / WRONG
    # NOTE: a little weird to do this computation in the loader
    def get_microbatch_sizes(self, minibatch_size: int):
        #Given a minibatch size, using the known train/val/test splits across partitions, compute the microbatches sizes
        #such that each machine runs for an equal number of iterations.
        #Returns a dictionary:
        #    {'val': torch.Tensor, 'test': torch.Tensor, 'train': torch.Tensor}
        #    Where the tensors are of length k and the appropriate micobatch size for each rank.
        ## NOTE: there is a small edge case here when have extreme imbalance, if want to divide 100 vertices into 101 minibatches.
        out = {k: None for k in self.id_sizes.keys()}
        for id_set_name, id_set_sizes in self.id_sizes.items():
            total = torch.sum(id_set_sizes)
            # Using a ceil will err on the side of making the batches too small.
            num_iterations = torch.ceil(total / minibatch_size)
            out[id_set_name] = id_set_sizes / num_iterations
        return out
    """

    def get_num_iterations(self, minibatch_size: int):
        """
        Given a minibatch size, using the known train/val/test splits across partitions, compute the the number of iterations
        such that each machine runs for an equal number of iterations during each of training, val, and test.
        Returns a dictionary out:
            {'val': int, 'test': int, 'train': int}
            Where for example  out['train'] gives the number of training iterations for each machine to execute.
        """
        out = {k: None for k in self.id_sizes.keys()}
        for id_set_name, id_set_sizes in self.id_sizes.items():
            total = torch.sum(id_set_sizes)
            # Flooring will set fewer iterations, so we will err on creating slightly larger minibatches.
            num_iterations = max(1, torch.floor(total / minibatch_size).item())
            out[id_set_name] = int(num_iterations)
        return out


class RangePartitionBookLoader(PartitionBookLoader):

    def __init__(self, k, path, rank=None, load_features=True, load_all_train_nid=True):
        super().__init__(k, path, rank, load_features, load_all_train_nid)
        self.load_partition_offsets() 

    def load_partition_offsets(self):
        """ Offsets include 0 idx followed by idx indicating the start of each subsequent partition. """
        self._partition_offsets = torch.load(''.join((self.root, 'partition_offsets.pt')))

    def get_RangePartitionBook(self):
        return RangePartitionBook(self.partid,
                                  self.k,
                                  self.rowptr,
                                  self.col,
                                  self.edge_ids,
                                  self.labels,
                                  self.train_nid,
                                  self.val_nid,
                                  self.test_nid,
                                  self._partition_offsets)

