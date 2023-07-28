from . import BaseDriver
import torch
from fast_trainer.shufflers import Shuffler, SubgraphShuffler


class SingleProcDriver(BaseDriver):
    def __init__(self, args, devices, dataset, model_type):
        super().__init__(args, devices, dataset, model_type)

        if self.args.experimental_explicit_batches:
            fake_subgraph_ids_train = torch.randint(low=0, high=max(self.dataset.split_idx['train'].numel() // args.train_batch_size, 10), size=self.dataset.split_idx['train'].size())
            self.train_shuffler = SubgraphShuffler(self.dataset.split_idx['train'], fake_subgraph_ids_train)

            fake_subgraph_ids_valid = torch.randint(low=0, high=max(self.dataset.split_idx['valid'].numel() // args.train_batch_size, 10), size=self.dataset.split_idx['valid'].size())
            self.valid_shuffler = SubgraphShuffler(self.dataset.split_idx['valid'], fake_subgraph_ids_valid)

            fake_subgraph_ids_test = torch.randint(low=0, high=max(self.dataset.split_idx['test'].numel() // args.train_batch_size, 10), size=self.dataset.split_idx['test'].size())
            self.test_shuffler = SubgraphShuffler(self.dataset.split_idx['test'], fake_subgraph_ids_test)
        else:
            self.train_shuffler = Shuffler(self.dataset.split_idx['train'])

    def get_idx(self, epoch: int):
        self.train_shuffler.set_epoch(epoch)
        return self.train_shuffler.get_idx()

    def get_idx_test(self, name):
        return self.dataset.split_idx[name]

    # Called druing the training loop
    def get_explicit_batches(self, epoch : int):
        self.train_shuffler.set_epoch(epoch)
        return self.train_shuffler.get_idx()

    # Called during evaluation. The "name" argument refers to the name of the split: train, valid, test
    def get_explicit_batches_test(self, name : str):
        if name == "valid":
            return self.valid_shuffler.get_idx()
        elif name == "test":
            return self.test_shuffler.get_idx()
        else:
            raise Exception("get_explicit_batches_test(name) must be called with name : 'test' | 'valid' ")

    @property
    def is_main_proc(self):
        return True
