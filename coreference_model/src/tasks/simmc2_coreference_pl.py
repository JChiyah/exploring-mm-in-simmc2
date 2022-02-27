# coding=utf-8
# Copyleft 2019 project LXRT.

import sys
import traceback
import os
import json
import random
import collections
import shutil

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from tqdm import tqdm
import numpy as np
import cv2
from detectron2.structures import BoxMode
import wandb

import utils
from param import args
from tasks.simmc2_coreference_data_pl import SIMMC2DataModule
from tasks.simmc2_coreference_model_pl import SIMMC2CoreferenceModelWithDescriptions


USE_MODEL_WITH_DESCRIPTIONS = True

SEEDS = np.arange(122, 300, 5)

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


wandb.login()

args.qa = False


class CustomModelCheckpoint(pl.callbacks.ModelCheckpoint):

    def __init__(self, dirpath, filename):
        self._token_to_replace = '$run_id$'
        self.filename = filename
        super().__init__(
            monitor='val/object_f1',
            dirpath=dirpath,
            filename=self.filename,
            save_last=True,
            save_top_k=2,
            mode='max',
            every_n_epochs=1,
            auto_insert_metric_name=False
        )

    def on_validation_end(self, trainer: pl.Trainer, pl_module):
        """
        ModelCheckpoint hardcodes self.filename = '{epoch}' in its on_train_start().
        But custom callbacks are called _before_ ModelCheckpoint, meaning setting it
        in our on_train_start() would just get overwritten. Therefore, we set it here in
        on_validation_end(), as checkpointing in Lightning is currently tied to
        Validation performance.

        https://github.com/PyTorchLightning/pytorch-lightning/issues/2042
        """
        if self._token_to_replace in self.filename and trainer.global_rank == 0:
            self.filename = self.filename.replace(
                self._token_to_replace, str(trainer.logger.version))

        super().on_validation_end(trainer, pl_module)


def get_model(batches_per_epoch=-1):
    # create model class
    if args.load:
        print(f"Loading model from '{checkpoint_dir}/{args.load}'")
        coreference_model = SIMMC2CoreferenceModelWithDescriptions.load_from_checkpoint(
            f"{checkpoint_dir}/{args.load}", name=model_name,
            max_seq_length=model_config['max_seq_length'], f=model_config['input_features'],
            batches_per_epoch=batches_per_epoch, lr=args.lr, final_layer=args.final_layer,
            ablation=args.ablation)
    else:
        coreference_model = SIMMC2CoreferenceModelWithDescriptions(
            model_name, model_config['max_seq_length'], f=model_config['input_features'],
            batches_per_epoch=batches_per_epoch, lr=args.lr, final_layer=args.final_layer,
            ablation=args.ablation)

    return coreference_model


def main_train():
    # basic datamodule
    data_module_train_test = SIMMC2DataModule.train_test_data_module(
        args.batch_size, args.num_workers, model_config['max_seq_length'])
    # data_module_train_test.setup()
    # avoid duplicating data, so comment out this line
    # batches_per_epoch = len(data_module_train_test.train_dataloader())
    coreference_model = get_model(batches_per_epoch=9110)

    trainer = pl.Trainer(**trainer_config)
    trainer.fit(coreference_model, datamodule=data_module_train_test)

    if coreference_model.global_rank == 0:
        # only print once in main process
        print(f"\nBest model saved at '{checkpoint_callback.best_model_path}'")

        if log_with_wandb and checkpoint_callback.best_model_path:
            # log final devtest results to wandb
            wandb_run_id = coreference_model.logger.version

            args.load = checkpoint_callback.best_model_path.split('/simmc2_533/')[-1]
            # trainer_config['logger']
            print()

            coreference_model.logger.experiment.notes = f"Model saved as: '{args.load}'"

            # a limitation of DDP is that we cannot run .fit and .test in the same script
            # main_test(wandb_run_id)

            print(f"To test this model in devtest, run: "
                  f"test \"--wandb_id {wandb_run_id} --load {args.load}\"\n\n")


def main_test(wandb_run_id: str = None):
    trainer_config['gpus'] = 1      # do not test in more than 1 GPU to avoid some DDP issues
    trainer_config['logger'] = WandbLogger(
            name=model_name, project='exloring-mm-in-simmc2',
            settings=wandb.Settings(_disable_stats=True), version=wandb_run_id) \
        if log_with_wandb else True

    # basic datamodule
    data_module_train_test = SIMMC2DataModule.train_test_data_module(
        args.batch_size, args.num_workers, model_config['max_seq_length'])
    coreference_model = get_model()

    trainer = pl.Trainer(**trainer_config)
    test_results = trainer.test(
        coreference_model,
        datamodule=data_module_train_test)

    print(test_results)

    if log_with_wandb is not None:
        if wandb_run_id is None:
            wandb_run_id = coreference_model.logger.version

        # log test results
        print(f"Logging test results to {wandb_run_id}")

        if len(test_results) > 1:
            raise ValueError(
                f"test_results is too long: len={len(test_results)}\ntest_results={test_results}")

        # need to log into wandb: object f1, recall, precision and object f1 std, object similarity
        for key, value in test_results[0].items():
            # we only want to have 1 value for test at most
            coreference_model.logger.experiment.summary[key] = value

        # coreference_model.logger.experiment.summary.update()


def main_predict():
    trainer_config['gpus'] = 1      # do not predict in more than 1 GPU to avoid some DDP issues
    data_module = SIMMC2DataModule.empty_data_module(
        args.batch_size, args.num_workers, model_config['max_seq_length'])
    data_module.setup()
    coreference_model = get_model()

    trainer = pl.Trainer(**trainer_config)

    # create prediction files for each dataset split
    splits_to_predict = ['devtest']     # , 'teststd_public']
    for split in splits_to_predict:
        predictions = trainer.predict(
            coreference_model,
            dataloaders=data_module.custom_dataloader(split))

        coreference_model.post_process_predictions(
            predictions, f"snap/dstc10-simmc-{split}-pred-subtask-2.json",
            extra={'load': args.load})


if __name__ == "__main__":
    keys_to_print = ['load', 'output', 'num_runs', 'lr', 'dropout', 'llayers', 'xlayers', 'rlayers']
    # 'train_data_ratio', 'simmc2_input_features', 'simmc2_max_turns']
    _info = {k: getattr(args, k) for k in keys_to_print}

    model_config = {
        'batch_size': args.batch_size,
        'dropout': args.dropout,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        # max_seq_length 50 is fine for lxr322 to lxr533
        # max_seq_length 30 is fine for lxr744
        # max_seq_length 25 is fine for lxr955
        'max_seq_length': 50,
        'random_seed': 123,
        'image_feature_file': args.image_feature_file,
        'image_categories': args.image_categories,
        'pred_threshold': 0.35,
        'input_features': args.simmc2_input_features
    }

    feature_str = '-'.join([f"{k}={v}" for k, v in args.simmc2_input_features.items()])
    feature_str = feature_str.replace('True', 'T').replace('False', 'F')

    if args.tiny:
        feature_str += '-tiny'
    model_name = f"{feature_str}-epochs={args.epochs}"

    log_with_wandb = not ('WANDB_MODE' in os.environ and os.environ['WANDB_MODE'] == 'disabled')

    if log_with_wandb and args.mode in ['train', 'test'] and not args.tiny:
        logger = WandbLogger(
            name=model_name, project='exloring-mm-in-simmc2',
            settings=wandb.Settings(_disable_stats=True), version=args.wandb_id)
    else:
        logger = True

    checkpoint_dir = f"{args.output}_{args.llayers}{args.xlayers}{args.rlayers}"
    # instead of saving as val/object_f1, we do val@object_f1 to avoid creating folders due to the /
    checkpoint_filename = 'coref-$run_id$-val@f1={val/object_f1:.3f}-' \
                          'step={step}-train@loss{train/loss:.3f}-' + feature_str

    checkpoint_callback = CustomModelCheckpoint(checkpoint_dir, checkpoint_filename)

    trainer_config = {
        'max_epochs': args.epochs,
        'gpus': 4 if not args.tiny else 1,
        'accelerator': 'ddp',
        'precision': 16,
        'accumulate_grad_batches': 8,
        'profiler': None,
        # for debugging: runs 1 train, val, test batch and program ends
        'fast_dev_run': False,
        'log_every_n_steps': 100,
        'deterministic': True,
        'default_root_dir': checkpoint_dir,
        'logger': logger,
        # turn off the warning
        'plugins': [pl.plugins.DDPPlugin(find_unused_parameters=False)],
        'callbacks': [checkpoint_callback],
        'resume_from_checkpoint': args.load
    }

    if args.tiny:
        trainer_config['limit_train_batches'] = 10
        trainer_config['limit_val_batches'] = 10
        trainer_config['limit_test_batches'] = 10

    _info = {
        **_info,
        'output_checkpoint_filename': checkpoint_filename,
        'model_config': model_config,
        'trainer_config': trainer_config
    }

    print(f"Info: {json.dumps(_info, indent=4, default=str)}")

    sweep_config = {
        'method': 'random',  # grid, random, bayesian
        'metric': {
            'name': 'val_object_f1',
            'goal': 'maximize'
        },
        'parameters': {
            'random_seed': {
                'values': [model_config['random_seed']]
            },
            'learning_rate': {
                'values': [model_config['learning_rate']]
            },
            'batch_size': {
                'values': [model_config['batch_size']]
            },
            'epochs': {'value': model_config['epochs']},
            'dropout': {
                'values': [model_config['dropout']]
            },
            'max_seq_length': {'value': model_config['max_seq_length']},
        }
    }

    model_config['random_seed'] = int(SEEDS[0])
    pl.seed_everything(model_config['random_seed'], workers=True)

    if args.mode == 'train':
        main_train()

    else:
        if args.load is None:
            print(f"WARNING! No model loaded, so testing with untrained model")
        elif feature_str.replace('-tiny', '') not in args.load:
            raise ValueError(
                f"Input features do not match with loaded model: \n"
                f"\t'{feature_str.replace('-tiny', '').replace('-object_counts=True', '')}' vs \n"
                f"\t'{args.load}'")

        if args.mode == 'test':
            main_test(args.wandb_id)

        elif args.mode == 'predict':
            main_predict()

        else:
            print(f"mode not recognised: {args.mode}")
