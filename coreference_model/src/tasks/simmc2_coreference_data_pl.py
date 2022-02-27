
import collections
from typing import *

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from tasks.simmc2_coreference_data import SIMMC2CoreferenceDataset, SIMMC2CoreferenceTorchDataset, \
    SIMMC2CoreferenceEvaluator


DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


class SIMMC2DataModule(pl.LightningDataModule):

    def __init__(self, splits: list, batch_size: int, num_workers: int, max_sequence_length: int):
        super().__init__()
        self.splits = splits
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_sequence_length = max_sequence_length
        self.train_tuple, self.dev_tuple, self.devtest_tuple = None, None, None
        # self.setup()

    @classmethod
    def train_data_module(cls, batch_size: int, num_workers: int, max_sequence_length: int):
        return cls(['train', 'dev'], batch_size, num_workers, max_sequence_length)

    @classmethod
    def train_test_data_module(cls, batch_size: int, num_workers: int, max_sequence_length: int):
        return cls(['train', 'dev', 'devtest'], batch_size, num_workers, max_sequence_length)

    @classmethod
    def test_data_module(cls, batch_size: int, num_workers: int, max_sequence_length: int):
        return cls(['dev', 'devtest'], batch_size, num_workers, max_sequence_length)

    @classmethod
    def empty_data_module(cls, batch_size: int, num_workers: int, max_sequence_length: int):
        return cls([], batch_size, num_workers, max_sequence_length)

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        for split in self.splits:
            if getattr(self, f"{split}_tuple") is not None:
                # already done
                continue
            else:
                if split == 'train':
                    data_tuple = self._get_data_tuple(split, self.max_sequence_length, True, True)
                else:
                    data_tuple = self._get_data_tuple(split, self.max_sequence_length, False)

                setattr(self, f"{split}_tuple", data_tuple)
                print(f"{split} data tuple ready\n\n")

    def train_dataloader(self):
        if 'train' not in self.splits:
            raise ValueError('train split not loaded!')

        return self.train_tuple.loader

    def val_dataloader(self):
        if 'dev' not in self.splits:
            raise ValueError('dev split not loaded!')

        return self.dev_tuple.loader

    def test_dataloader(self):
        if 'devtest' not in self.splits and 'test-std' not in self.splits:
            raise ValueError('devtest split not loaded!')
        return self.devtest_tuple.loader

    def custom_dataloader(self, split: str, shuffle=False):
        # load a dataset on the fly, used to e.g., predict
        return self._get_data_tuple(split, self.max_sequence_length, shuffle).loader

    def _get_data_tuple(self, split, max_seq_length, shuffle, drop_last=False) -> DataTuple:
        dset = SIMMC2CoreferenceDataset(split)
        tset = SIMMC2CoreferenceTorchDataset(dset, max_seq_length)
        evaluator = SIMMC2CoreferenceEvaluator(dset, tset)
        data_loader = DataLoader(
            tset, batch_size=self.batch_size,
            shuffle=shuffle, num_workers=self.num_workers,
            drop_last=drop_last, pin_memory=False
        )

        return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)
