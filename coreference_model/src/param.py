# coding=utf-8
# Copyleft 2019 project LXRT.

import ast
import argparse
import random

import numpy as np
import torch


def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        optimizer = 'bert'      # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args():
    parser = argparse.ArgumentParser()

    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='valid')
    parser.add_argument("--test", default=None)

    # Training Hyper-parameters
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=256)
    parser.add_argument('--optim', default='bert')
    parser.add_argument('--lr', type=float, default=3e-4)   # 1e-4
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    # changed from 9595 seed to None so we always have to set it to value from 0 to 100
    parser.add_argument('--seed', type=int, default=None, help='random seed')

    # Debugging
    parser.add_argument('--output', type=str, default='snap/test')
    parser.add_argument("--fast", action='store_const', default=False, const=True)
    parser.add_argument("--tiny", action='store_const', default=False, const=True)
    parser.add_argument("--tqdm", action='store_const', default=False, const=True)

    # Model Loading
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--loadLXMERT', dest='load_lxmert', type=str, default=None,
                        help='Load the pre-trained LXMERT model.')
    parser.add_argument('--loadLXMERTQA', dest='load_lxmert_qa', type=str, default=None,
                        help='Load the pre-trained LXMERT model with QA answer head.')
    parser.add_argument("--fromScratch", dest='from_scratch', action='store_const', default=False, const=True,
                        help='If none of the --load, --loadLXMERT, --loadLXMERTQA is set, '
                             'the model would be trained from scratch. If --fromScratch is'
                             ' not specified, the model would load BERT-pre-trained weights by'
                             ' default. ')

    # Optimization
    parser.add_argument("--mceLoss", dest='mce_loss', action='store_const', default=False, const=True)

    # LXRT Model Config
    # Note: LXRT = L, X, R (three encoders), Transformer
    parser.add_argument("--llayers", default=9, type=int, help='Number of Language layers')
    parser.add_argument("--xlayers", default=5, type=int, help='Number of CROSS-modality layers.')
    parser.add_argument("--rlayers", default=5, type=int, help='Number of object Relationship layers.')

    # LXMERT Pre-training Config
    parser.add_argument("--taskMatched", dest='task_matched', action='store_const', default=False, const=True)
    parser.add_argument("--taskMaskLM", dest='task_mask_lm', action='store_const', default=False, const=True)
    parser.add_argument("--taskObjPredict", dest='task_obj_predict', action='store_const', default=False, const=True)
    parser.add_argument("--taskQA", dest='task_qa', action='store_const', default=False, const=True)
    parser.add_argument("--visualLosses", dest='visual_losses', default='obj,attr,feat', type=str)
    parser.add_argument("--qaSets", dest='qa_sets', default=None, type=str)
    parser.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.15, type=float)
    parser.add_argument("--objMaskRate", dest='obj_mask_rate', default=0.15, type=float)

    # Training configuration
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument("--numWorkers", dest='num_workers', default=0, type=int)

    # JChiyah for multiple runs and averages
    parser.add_argument("--num_runs", default=1, type=int)
    parser.add_argument('--mode', required=True, choices=['train', 'test', 'predict'])
    parser.add_argument(
        '--simmc2_input_features',
        help='The input features to use in inference, default=["user_utterance", "image"]',
        default='')
    parser.add_argument(
        '--image_feature_file', type=str, required=True,
        help='Suffix of the feature .tsv files to use',
        default='all_gt_boxes')
    parser.add_argument(
        '--image_categories', type=str,
        help='Suffix of the category feature .tsv files to use',
        default='colour_types_compound')
    parser.add_argument(
        '--visualise', action='store_true',
        help='Visualise some examples, load must be true',
        default=False)
    parser.add_argument(
        '--simmc2_sample', type=int,
        help='Load only a sample of the datasets of this size',
        default=5)
    parser.add_argument(
        '--wandb_id', type=str,
        help='id to use for wandb if loading an older model',
        default=None)
    parser.add_argument(
        '--final_layer', type=str,
        help='final layer of the model, linear usually works best but sequential was the submitted',
        default='linear')
    parser.add_argument(
        '--ablation', type=str,
        help='what to ignore from the model architecture during more in-depth ablations',
        choices=['visual_encoder', 'lxrt_encoder', 'multiple_similar_objects', None], default=None)

    # Parse the arguments.
    args = parser.parse_args()

    # Bind optimizer class.
    args.optimizer = get_optimizer(args.optim)

    default_input_features = {
        'user_utterance': True,
        'img_features': True,
        'img_bounding_boxes': GOLD_DATA,
        'dialogue_history_turns': 1,
        'previous_object_turns': 1,
        'descriptions': True,
        'object_indexes_in_descriptions': 'number',
        'object_counts': True,
    }
    input_feature_choices = {
        'user_utterance': [True, False],
        'img_features': [True, False],
        'img_bounding_boxes': [False, GOLD_DATA],
        'dialogue_history_turns': [False, 'all'] + list(range(0, 10)),
        'previous_object_turns': [False, 'all'] + list(range(0, 10)),
        'descriptions': [True, False, GOLD_DATA, 'colours_only', 'asset_only'],
        'object_indexes_in_descriptions': [False, 'number', 'token_local', 'token_global'],
        'object_counts': [True, False],
    }

    tmp_input_feats = args.simmc2_input_features.split(',')
    args.simmc2_input_features = default_input_features.copy()
    # fill args.simmc2_input_features with each feature value
    for feat_str in tmp_input_feats:
        if feat_str == '':
            continue

        if '=' not in feat_str:
            raise ValueError(
                f"Features should have the format: feat_name=feat_value, ... but found {tmp_input_feats}")
        key, value = feat_str.strip().split('=')

        if key not in args.simmc2_input_features.keys():
            raise ValueError(
                f"Key {key} not recognised in possible inputs: {args.simmc2_input_features.keys()}")
        value = eval(value)
        if value not in input_feature_choices[key]:
            raise ValueError(
                f"Feature {key} cannot have value {value}. Choices: {input_feature_choices[key]}")

        if 'turns' in key and not value:
            value = 0                       # set minimum value to be 0, not False or None

        args.simmc2_input_features[key] = value

    print(args.simmc2_input_features)

    if args.visualise and not args.load:
        raise ValueError(f"You need to give the --load argument for visualise!")

    # # Set seeds
    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)

    return args


GOLD_DATA = 'gold'
args = parse_args()
