#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""

    Author : JChiyah
    Date   : July 2021
    Python : 3.8.10

Code for the experiments from sections 4 and 5 of the paper "Exploring Multi-Modal Representations
for Ambiguity Detection & Coreference Resolution in the SIMMC 2.0 Challenge".

Before running this file, download the SpaCy model:

    python -m spacy download en_core_web_lg

Example run:

    python experiments.py

"""

import collections
import os
import copy
import argparse
import glob
import random
import re
import shutil
import json
import typing
import pathlib
import builtins as __builtin__

import numpy as np
import cv2
from tqdm import tqdm
from scipy.spatial import distance
import pingouin as pg
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import spacy
# pip install neuralcoref --no-binary neuralcoref
# downgrade spacy to <3.0.0 if using neuralcoref
# import neuralcoref
import scipy.stats
from scipy.stats import pearsonr, ttest_ind

from simmc2.model.mm_dst.utils.evaluate_dst import evaluate_from_flat_list, reformat_turn

from generate_simmc2_data import analyse_data, generate_object_classes, DOMAINS, SCENE_JSONS, METADATA


TEMPLATE_TRUE = 'simmc2_dials_dstc10_{split}.json'
TEMPLATE_PREDICTED = 'dstc10-simmc-{split}-pred-subtask-{subtask_index}.json'
SUBTASK_DISAMBIGUATION = 1
SUBTASK_COREFERENCE = 2


def _read_json_data(file_path):
    with open(file_path, 'r') as in_file:
        content = json.load(in_file)
    return content


_print_indentation = 0
# Overwrite print to add indentation
def print(*args, **kwargs):
    __builtin__.print(' ' * _print_indentation, end='')
    return __builtin__.print(*args, **kwargs)


def get_predicted_data(subtask_index, split):
    if subtask_index not in [1, 2] or split not in ['train', 'dev', 'devtest']:
        raise ValueError

    if subtask_index == 1:
        parent_folder = 'disambiguation'
    elif subtask_index == 2:
        parent_folder = 'coreference'
    else:
        raise ValueError

    parent_folder += '_model'
    parent_folder += '/snap' if subtask_index == 2 else ''
    # parent_folder = 'submission'

    return _read_json_data(os.path.join(
        parent_folder, TEMPLATE_PREDICTED).format(
        subtask_index=subtask_index, split=split))


def get_true_data(split):
    all_data = _read_json_data(os.path.join(
        'simmc2/data',
        TEMPLATE_TRUE.format(split=split)))

    scene_jsons = get_scene_data()

    for dialogue_index, dialogue_datum, turn_index, turn_datum in get_turn_iterator(all_data['dialogue_data']):
        if 'scene_objects' not in turn_datum:
            turn_datum['scene_idx'], turn_datum['previous_scene_idx'], turn_datum['image_name'] = \
                _get_scene_idx(dialogue_datum["scene_ids"], turn_datum["turn_idx"])

            turn_datum['scene_objects'] = get_scene(scene_jsons, turn_datum['scene_idx'])['objects']

    return all_data['dialogue_data']


def get_metadata():
    simmc2_metadata = {}
    for domain in DOMAINS:
        simmc2_metadata = {
            **simmc2_metadata,
            **_read_json_data(os.path.join('simmc2/data', METADATA.format(domain)))
        }

    return simmc2_metadata
    # with open(), "r") as f_in:
    #     simmc2_metadata = {**simmc2_metadata, **json.load(f_in)}


def get_scene_data():
    simmc2_scenes_jsons = {}
    _files = glob.glob(f"simmc2/data/{SCENE_JSONS}/*.json")
    for file in _files:
        simmc2_scenes_jsons[os.path.splitext(os.path.basename(file))[0]] = _read_json_data(file)
        # with open(file, "r") as f_in:
        #     simmc2_scenes_jsons[os.path.splitext(os.path.basename(file))[0]] = json.load(f_in)

    return simmc2_scenes_jsons


def get_joint_data(split):
    print(f"getting joint data for {split}")
    json_disambiguation = get_predicted_data(SUBTASK_DISAMBIGUATION, split)
    json_coreference = get_predicted_data(SUBTASK_COREFERENCE, split)

    json_true_data = get_true_data(split)

    if len(json_coreference['dialogue_data'][0]['dialogue']) == 1:
        print('fixing data')
        # need to fix it
        tmp_dialogue_data = []
        index = 0
        while index < len(json_coreference['dialogue_data']):
        # for index in range(len(json_coreference['dialogue_data']))[:2]:
            dialogue_turns = []
            new_dialogue = json_coreference['dialogue_data'][index]

            dialogue_idx = new_dialogue['dialogue_idx']
            # while the dialogue entry has the same idx
            _internal_turn_idx = 0
            while json_coreference['dialogue_data'][index]['dialogue_idx'] == dialogue_idx:
                # append turns to initial dialogue
                dialogue_turns.append(json_coreference['dialogue_data'][index]['dialogue'][0])
                assert dialogue_turns[-1]['turn_idx'] == _internal_turn_idx, \
                    f"{dialogue_turns[-1]['turn_idx']} == {_internal_turn_idx} at {dialogue_idx}"
                index += 1
                _internal_turn_idx += 1
                if index >= len(json_coreference['dialogue_data']):
                    break

            # end of loop, dialogue gathered to one

            new_dialogue['dialogue'] = dialogue_turns
            tmp_dialogue_data.append(new_dialogue)
            # print(len(dialogue_turns))
            # print(dialogue_idx)
            # exit()
            # index += 1

        json_coreference['dialogue_data'] = tmp_dialogue_data
        # print(len(json_coreference['dialogue_data']))
        # print(len(json_true_data['dialogue_data']))

    return join_true_and_predicted(json_true_data, json_disambiguation, json_coreference)


def get_turn_iterator(all_data: list, limit=None):
    i = 0
    for dialogue_index, dialogue_datum in enumerate(all_data):
        for turn_index, turn_datum in enumerate(dialogue_datum['dialogue']):
            yield dialogue_index, dialogue_datum, turn_index, turn_datum
            if limit is not None and i > limit:
                return
            i += 1


def is_disambiguation_turn(turn_datum):
    return 'disambiguation_label' in turn_datum


def count_turns(all_data: list) -> int:
    return len(list(get_turn_iterator(all_data)))


def join_true_and_predicted(true_data, pred_data_disambiguation, pred_data_coreference):
    true_data = true_data.get('dialogue_data', true_data) if isinstance(true_data, dict) else true_data
    pred_data_coreference = pred_data_coreference['dialogue_data']
    # disambiguation data has a different format/order, so make it a dict to search for dialogue_idx
    pred_data_disambiguation = {x['dialog_id']: x for x in pred_data_disambiguation}

    # output has the same shape as true_data
    for i in range(len(true_data)):

        # assume they are in order, except for disambiguation data
        assert true_data[i]['dialogue_idx'] == pred_data_coreference[i]['dialogue_idx'], \
            f"{true_data[i]['dialogue_idx']} == {pred_data_coreference[i]['dialogue_idx']}"

        # Iterate through each dialog
        dialogue_true = true_data[i]['dialogue']
        dialogue_pred_dis = pred_data_disambiguation.get(true_data[i]['dialogue_idx'], None)
        # convert to dict by turn_idx, similar to above
        if dialogue_pred_dis:
            # print(dialogue_pred_dis)
            dialogue_pred_dis = {x['turn_id']: x for x in dialogue_pred_dis['predictions']}
        dialogue_pred_cor = pred_data_coreference[i]['dialogue']

        for t, turn_true in enumerate(dialogue_true):
            # print(turn_true)
            # add disambiguation pred
            if 'disambiguation_label' in turn_true:
                turn_true['predicted_disambiguation_label'] = \
                    dialogue_pred_dis[t]['disambiguation_label']

            # add coreference pred
            assert dialogue_pred_cor[t]['turn_idx'] == t, f"{dialogue_pred_cor[t]['turn_idx']} == {t}"
            turn_true['predicted_objects'] = \
                dialogue_pred_cor[t]['transcript_annotated']['act_attributes']['objects']
            turn_true['prediction_outputs'] = dialogue_pred_cor[t]['prediction_outputs']

    return true_data


def evaluate_joint_data_object_f1(joint_data) -> dict:
    d_true_flattened = []
    d_pred_flattened = []

    # flatten turns
    for dialogue_index, dialogue_datum, turn_index, turn_datum in get_turn_iterator(joint_data):
        turn_true = reformat_turn(turn_datum["transcript_annotated"])
        turn_pred = reformat_turn(turn_datum["transcript_annotated"])
        # print(turn_pred)
        turn_pred[0]['objects'] = turn_datum['predicted_objects']

        d_true_flattened.append(turn_true)
        d_pred_flattened.append(turn_pred)

    eval_result = evaluate_from_flat_list(d_true_flattened, d_pred_flattened)
    # remove everything that doesn't have to do with object f1
    for key in list(eval_result.keys()):        # list avoids error when deleting
        if 'object' not in key:
            del eval_result[key]

    return eval_result


def evaluate_single_turn_object_f1(turn_datum):
    turn_true = reformat_turn(turn_datum["transcript_annotated"])
    turn_pred = reformat_turn(turn_datum["transcript_annotated"])
    # print(turn_pred)
    turn_pred[0]['objects'] = turn_datum['predicted_objects']

    eval_result = evaluate_from_flat_list([turn_true], [turn_pred])
    # remove everything that doesn't have to do with object f1
    for key in list(eval_result.keys()):        # list avoids error when deleting
        if 'object' not in key:
            del eval_result[key]

    return eval_result


def get_data_where(all_data, filter_level, key, value, _func=None):
    if filter_level not in ['dialogue', 'turn']:
        raise ValueError

    final_data = copy.deepcopy(all_data)
    indexes_to_remove = []

    if filter_level == 'dialogue':
        for dialogue_index, dialogue_datum, turn_index, turn_datum in get_turn_iterator(all_data):
            if dialogue_datum[key] != value and dialogue_index not in indexes_to_remove:
                indexes_to_remove.append(dialogue_index)
            if _func:
                raise NotImplementedError

        # remove starting from the last ones, so it doesn't mess up the order
        for dialogue_index in reversed(indexes_to_remove):
            del final_data[dialogue_index]

    elif filter_level == 'turn':
        for dialogue_index, dialogue_datum, turn_index, turn_datum in get_turn_iterator(all_data):
            if _func is not None:
                if not _func(turn_datum[key]):
                    indexes_to_remove.append((dialogue_index, turn_index))
            elif turn_datum[key] != eval(value):
                indexes_to_remove.append((dialogue_index, turn_index))

            # print(f"{turn_datum[key]} != {value} ({eval(value)}): {turn_datum[key] != value}")

        # remove starting from the last ones, so it doesn't mess up the order
        for dialogue_index, turn_index in reversed(indexes_to_remove):
            del final_data[dialogue_index]['dialogue'][turn_index]
    else:
        raise ValueError

    # for i, dialogue_datum in enumerate(all_data):
    #     if len(dialogue_datum['dialogue']) == 0:
    #         del all_data[i]
    final_data = [x for x in final_data if len(x['dialogue']) > 0]

    return final_data


def calculate_key_proportions(data, key, key_type=None, extra_condition=None) -> dict:
    proportions = {}

    def _get_nice_label(label):
        if key_type == bool:
            # lowercase is better
            return f"{'true' if bool(label) else 'false'}"
        elif key_type == 'with_without':
            return f"{'with' if bool(label) else 'without'}"
        else:
            return f"{label}"

    for dialogue_index, dialogue_datum, turn_index, turn_datum in get_turn_iterator(data):
        if extra_condition:
            value = _get_nice_label(extra_condition(turn_datum[key]))
        else:
            value = _get_nice_label(turn_datum[key])

        if value not in proportions:
            proportions[value] = 0

        proportions[value] += 1

    # order dict for better printing
    ord_list = ['true', 'false', 'with', 'without']
    ord_list += [key for key in proportions.keys() if key not in ord_list]

    res = dict()
    for key in ord_list:
        if key in proportions:
            res[key] = proportions[key]

    proportions = res   # {key: proportions[key] for key in ord_list}

    all_labels = list(proportions.keys())
    proportions['total'] = sum([proportions[label] for label in all_labels])

    for label in all_labels:
        proportions[f"{label}_%"] = proportions[label] / proportions['total']

    return proportions


def _print_examples(data, limit=None):
    i = 0
    print('Printing examples:')
    for _, dialogue_datum, _, turn_datum in get_turn_iterator(data):
        status = 'CORRECT' if turn_datum['disambiguation_label'] == \
                             turn_datum['predicted_disambiguation_label'] else 'FAILED'
        print(f"\t[{status}, {dialogue_datum['dialogue_idx']}-{turn_datum['turn_idx']}] "
              f"\"{turn_datum['transcript']}\" "
              f"(gold: {turn_datum['disambiguation_label']} / "
              f"predicted: {turn_datum['predicted_disambiguation_label']})")
        if limit is not None and i > limit:
            return
        i += 1


def _get_scene_idx(dialogue_scenes: dict, turn_idx: int) -> tuple:
    # resolve scene id from the dialogue list
    scene_idx_list = []
    for scene_id in reversed(list(dialogue_scenes.keys())):
        if turn_idx >= int(scene_id):
            scene_idx_list.append(dialogue_scenes[scene_id])

    if len(scene_idx_list) == 0 or len(scene_idx_list) > 2:
        raise ValueError

    # remove the initial 'm_' from the image or otherwise it won't be recognised
    image_name = scene_idx_list[0].lstrip('m_')

    # current scene, previous scene, image_name
    return scene_idx_list[0], scene_idx_list[1] if len(scene_idx_list) > 1 else None, image_name


def get_scene(scene_jsons, scene_idx):
    # if 'm_' in scene_idx:
    #     scene_idx = scene_idx.lstrip('m_')
    scene_idx = scene_idx + '_scene'

    try:
        if scene_idx in scene_jsons:
            scene = scene_jsons[scene_idx]
        else:
            scene = scene_jsons[f"m_{scene_idx}"]
        return scene['scenes'][0]

    except KeyError:
        raise KeyError(
            f"KeyError: scene_idx '{scene_idx}' not found in data\n\n"
            f"Scenes available: {scene_jsons.keys()}")


def get_object_from_datum_by_index(entry_datum, object_index_id) -> typing.Union[dict, None]:
    for scene_object in entry_datum['scene_objects']:
        if scene_object['index'] == object_index_id:
            # if scene_object['index'] == 0:
            #     print(scene_object)
            #     raise NotImplementedError
            return scene_object
    else:
        raise ValueError(
            f"Object index {object_index_id} not found in entry_datum {entry_datum}: "
            f"{[scene_object['index'] for scene_object in entry_datum['scene_objects']]}")


def get_object_type(metadata, _object) -> str:
    item = metadata[_object['prefab_path']]
    return item['assetType' if 'assetType' in item else 'type']


def counter_to_distribution(counter: dict) -> dict:
    # result = {}
    total = sum(counter.values())

    result = {k: v / total for k, v in counter.items()}
    result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

    return result


def analyse_coreference_target_objects(all_data, include_past_objects=False):
    metadata = get_metadata()

    coreference_target_types = collections.Counter()
    coreference_target_prefabs = collections.Counter()
    coreference_target_indexes = collections.Counter()
    coreference_target_length = collections.Counter()
    coreference_target_in_the_past = collections.Counter()
    coreference_target_previously_system_referenced = collections.Counter()       # turn-n, never
    coreference_target_previously_user_referenced = collections.Counter()       # turn-n, never

    # What are the most common co-ref output types?
    for dialogue_index, dialogue_datum, turn_index, turn_datum in get_turn_iterator(all_data):
        target_object_ids = turn_datum['transcript_annotated']['act_attributes']['objects']

        coreference_target_length[str(len(target_object_ids))] += 1

        if len(target_object_ids) == 0:
            coreference_target_types[''] += 1
        else:
            for target_object_idx in target_object_ids:
                try:
                    target_object = get_object_from_datum_by_index(turn_datum, target_object_idx)
                    coreference_target_types[get_object_type(metadata, target_object)] += 1
                    coreference_target_prefabs[target_object['prefab_path']] += 1
                    coreference_target_indexes[target_object['index']] += 1
                    coreference_target_in_the_past['false'] += 1

                    # find if target was mentioned previously by system or user
                    for prev_index, previous_turn in enumerate(dialogue_datum['dialogue'][:turn_index:-1]):
                        if target_object_idx in previous_turn['system_transcript_annotated']['act_attributes']['objects']:
                            coreference_target_previously_system_referenced[f"turn-{prev_index+1}"] += 1
                            break
                    else:
                        coreference_target_previously_system_referenced['never'] += 1

                    for prev_index, previous_turn in enumerate(dialogue_datum['dialogue'][:turn_index:-1]):
                        if target_object_idx in previous_turn['transcript_annotated']['act_attributes']['objects']:
                            coreference_target_previously_user_referenced[f"turn-{prev_index+1}"] += 1
                            break
                    else:
                        coreference_target_previously_user_referenced['never'] += 1

                except ValueError:
                    # skip if object is in the past for now
                    coreference_target_in_the_past['true'] += 1

    labels = [
        'coreference_target_types',
        # 'coreference_target_prefabs',
        'coreference_target_indexes',
        'coreference_target_length',
        'coreference_target_in_the_past',
        'coreference_target_previously_system_referenced',
        'coreference_target_previously_user_referenced',
    ]
    analysis = {}
    for variable in labels:
        analysis[variable] = locals().get(variable)
        analysis[variable] = collections.OrderedDict(analysis[variable].most_common())
        # print(json.dumps(analysis[variable], indent=4))
        analysis[variable] = counter_to_distribution(analysis[variable])

    # print(json.dumps(analysis, indent=4))
    for x in [100, 50, 20, 10, 5]:
        analysis[f"percentage_times_target_index_below_{x}"] = sum(
            [v for k, v in analysis['coreference_target_indexes'].items() if int(k) < x])

    # print(sum([v for k, v in analysis['coreference_target_indexes'].items() if int(k) < 10]))
    # print(sum([v for k, v in analysis['coreference_target_indexes'].items() if int(k) < 5]))

    # print(json.dumps(counter_to_distribution(coreference_target_types), indent=4))
    # print(json.dumps(counter_to_distribution(coreference_target_prefabs), indent=4))

    return analysis


def extract_object_info(all_data, scene_jsons, metadata):
    objects_per_index = {}
    objects_per_unique_id = {}
    all_train_data = get_true_data('train')
    all_devtest_data = get_true_data('devtest')

    def add_objects(scene_objects):
        for elem in scene_objects:
            if elem['index'] not in objects_per_index:
                objects_per_index[elem['index']] = collections.Counter()

            object_class = get_object_type(metadata, elem)
            objects_per_index[elem['index']][object_class] += 1

            if elem['unique_id'] not in objects_per_unique_id:
                objects_per_unique_id[elem['unique_id']] = collections.Counter()

            objects_per_unique_id[elem['unique_id']][object_class] += 1

    # absolute, across all scenes
    all_scenes = [
        scene['scenes'][0]['objects'] for key, scene in scene_jsons.items()
        if key.endswith('scene')]
    for scene_objects in all_scenes:
        add_objects(scene_objects)

    objects_per_index = collections.OrderedDict(sorted(objects_per_index.items()))
    # print(objects_per_index)
    # print('----')
    objects_per_index = {}
    objects_per_unique_id = {}

    # relative, across objects mentioned at training time
    # iterate per dialogue and find all object IDs
    for dialogue_index, dialogue_datum in enumerate(all_train_data):
        for scene_idx in dialogue_datum['scene_ids'].values():
            scene = get_scene(scene_jsons, scene_idx)
            add_objects(scene['objects'])

    objects_per_index = collections.OrderedDict(sorted(objects_per_index.items()))
    # print(objects_per_index)
    # print(json.dumps({k: counter_to_distribution(v) for k, v in objects_per_index.items()}, indent=4))
    # print(json.dumps({k: counter_to_distribution(v) for k, v in objects_per_unique_id.items()}, indent=4))
    # exit()

    all_objects = {}
    for counter in objects_per_index.values():
        for object_type, value in counter.items():
            if object_type not in all_objects:
                all_objects[object_type] = 0
            all_objects[object_type] += value

    # analyse_coreference_target_objects(all_train_data)
    # analyse_coreference_target_objects(all_devtest_data)

    all_joint_data = get_joint_data('devtest')
    performance = {}

    performance['distribution_of_target_object_types'] = counter_to_distribution(all_objects)
    performance['original_model'] = evaluate_joint_data_object_f1(all_joint_data)

    # baseline where output is always []
    for dialogue_index, dialogue_datum, turn_index, turn_datum in get_turn_iterator(all_joint_data):
        turn_datum['predicted_objects'] = []

    max_turns = 1
    for dialogue_index, dialogue_datum, turn_index, turn_datum in get_turn_iterator(all_joint_data):
        # prev_turn = dialogue_datum['dialogue'][turn_index-1] if turn_index > 0 else None
        prev_objects = []
        # prev_objects += prev_turn['transcript_annotated']['act_attributes']['objects']
        # prev_objects += prev_turn['system_transcript_annotated']['act_attributes']['objects']
        for prev_turn in dialogue_datum['dialogue'][turn_index-max_turns if turn_index > 0 else 0:turn_index]:  # turn_index+1 is current
            # prev_objects += prev_turn['transcript_annotated']['act_attributes']['objects']
            prev_objects += prev_turn['system_transcript_annotated']['act_attributes']['objects']
            # print(f"{turn_index}: {prev_objects}")

        turn_datum['predicted_objects'] = sorted(list(set(prev_objects)))

    performance[f"baseline_previous_system_mentioned_objects_t-{max_turns}"] = evaluate_joint_data_object_f1(all_joint_data)

    max_turns = 2
    for dialogue_index, dialogue_datum, turn_index, turn_datum in get_turn_iterator(all_joint_data):
        # prev_turn = dialogue_datum['dialogue'][turn_index-1] if turn_index > 0 else None
        prev_objects = []
        # prev_objects += prev_turn['transcript_annotated']['act_attributes']['objects']
        # prev_objects += prev_turn['system_transcript_annotated']['act_attributes']['objects']
        for prev_turn in dialogue_datum['dialogue'][turn_index-max_turns if turn_index > 0 else 0:turn_index]:  # turn_index+1 is current
            # prev_objects += prev_turn['transcript_annotated']['act_attributes']['objects']
            prev_objects += prev_turn['system_transcript_annotated']['act_attributes']['objects']
            # print(f"{turn_index}: {prev_objects}")

        turn_datum['predicted_objects'] = sorted(list(set(prev_objects)))

    performance[f"baseline_previous_system_mentioned_objects_t-{max_turns}"] = evaluate_joint_data_object_f1(all_joint_data)

    max_turns = 99
    for dialogue_index, dialogue_datum, turn_index, turn_datum in get_turn_iterator(all_joint_data):
        # prev_turn = dialogue_datum['dialogue'][turn_index-1] if turn_index > 0 else None
        prev_objects = []
        # prev_objects += prev_turn['transcript_annotated']['act_attributes']['objects']
        # prev_objects += prev_turn['system_transcript_annotated']['act_attributes']['objects']
        for prev_turn in dialogue_datum['dialogue'][turn_index-max_turns if turn_index > 0 else 0:turn_index]:  # turn_index+1 is current
            # prev_objects += prev_turn['transcript_annotated']['act_attributes']['objects']
            prev_objects += prev_turn['system_transcript_annotated']['act_attributes']['objects']
            # print(f"{turn_index}: {prev_objects}")

        turn_datum['predicted_objects'] = sorted(list(set(prev_objects)))

    performance[f"baseline_previous_system_mentioned_objects_t-{max_turns}"] = evaluate_joint_data_object_f1(all_joint_data)

    # baseline where output is a random int below 5, or empty (6 possibilities)
    for dialogue_index, dialogue_datum, turn_index, turn_datum in get_turn_iterator(all_joint_data):
        turn_datum['predicted_objects'] = []
        pred_obj = random.randint(0, 5)
        if pred_obj > 0:
            turn_datum['predicted_objects'].append(pred_obj)

    performance['baseline_random_index_below_5'] = evaluate_joint_data_object_f1(all_joint_data)

    # baseline where output is a random int below 10, or empty (11 possibilities)
    for dialogue_index, dialogue_datum, turn_index, turn_datum in get_turn_iterator(all_joint_data):
        turn_datum['predicted_objects'] = []
        pred_obj = random.randint(0, 10)
        if pred_obj > 0:
            turn_datum['predicted_objects'].append(pred_obj)

    performance['baseline_random_index_below_10'] = evaluate_joint_data_object_f1(all_joint_data)

    # baseline where output is a random int below 141, or empty (142 possibilities)
    for dialogue_index, dialogue_datum, turn_index, turn_datum in get_turn_iterator(all_joint_data):
        turn_datum['predicted_objects'] = []
        pred_obj = random.randint(0, 141)
        if pred_obj > 0:
            turn_datum['predicted_objects'].append(pred_obj)

    performance['baseline_random_index_below_141'] = evaluate_joint_data_object_f1(all_joint_data)

    data_without_empty_coreferences = get_data_where(
        all_joint_data, 'turn', 'transcript_annotated', None, _func=lambda x: len(x['act_attributes']['objects']) > 0)
    # print(count_turns(all_joint_data))
    # print(count_turns(data_without_empty_coreferences))
    # baseline where output is a random int below 141, or empty (142 possibilities)
    for dialogue_index, dialogue_datum, turn_index, turn_datum in get_turn_iterator(data_without_empty_coreferences):
        turn_datum['predicted_objects'] = []
        pred_obj = random.randint(0, 5)
        if pred_obj > 0:
            turn_datum['predicted_objects'].append(pred_obj)

    performance['baseline_random_index_below_5_no_empty'] = evaluate_joint_data_object_f1(data_without_empty_coreferences)

    # print(json.dumps(performance, indent=4))

    return performance


# Subtask#1

def get_experimental_analysis_subtask1(split):
    analysis = {
        'data': analyse_disambiguation_pos(split),
        **disambiguation_analysis(split),
    }

    return analysis


def get_all_ambiguous_coreferences(split):
    all_data = get_joint_data(split)
    indexes_to_remove = []
    # print(len(list(get_turn_iterator(all_data))))

    # iterate whilst keeping track of what to remove
    for dialogue_index, dialogue_datum, turn_index, turn_datum in get_turn_iterator(all_data):
        if 'disambiguation_label' not in turn_datum:
            indexes_to_remove.append((dialogue_index, turn_index))

    # remove starting from the last ones, so it doesn't mess up the order
    for dialogue_index, turn_index in reversed(indexes_to_remove):
        # print(dialogue_index, turn_index)
        del all_data[dialogue_index]['dialogue'][turn_index]

    # print(len(list(get_turn_iterator(all_data))))
    all_data = [x for x in all_data if len(x['dialogue']) > 0]

    return all_data


def get_data_after_disambiguation(split, disambiguation_label=1):
    all_data = get_joint_data(split)
    indexes_to_remove = []
    previous_dialogue = None
    previous_turn = -1
    after_disambiguation = False
    # print(len(list(get_turn_iterator(all_data))))

    # iterate whilst keeping track of what to remove
    for dialogue_index, dialogue_datum, turn_index, turn_datum in get_turn_iterator(all_data):
        # if 'disambiguation_label' not in turn_datum:
        if not after_disambiguation or previous_dialogue != dialogue_index:
            # changed dialogue, so not the turn after anymore
            indexes_to_remove.append((dialogue_index, turn_index))

        after_disambiguation = False

        if 'disambiguation_label' in turn_datum \
            and turn_datum['disambiguation_label'] == disambiguation_label:
            after_disambiguation = True
            previous_dialogue = dialogue_index
            # previous_turn = turn_index

    # remove starting from the last ones, so it doesn't mess up the order
    for dialogue_index, turn_index in reversed(indexes_to_remove):
        # print(dialogue_index, turn_index)
        del all_data[dialogue_index]['dialogue'][turn_index]

    # print(len(list(get_turn_iterator(all_data))))
    all_data = [x for x in all_data if len(x['dialogue']) > 0]

    return all_data


def disambiguation_analysis(split) -> dict:
    analysis = {}

    ambiguous_data = get_all_ambiguous_coreferences(split)
    analysis['total_dialogues'] = len(ambiguous_data)
    analysis['total_turns'] = count_turns(ambiguous_data)
    analysis['turns_disambiguate'] = {}

    analysis['turns_disambiguate']['gold'] = calculate_key_proportions(
        ambiguous_data, 'disambiguation_label', bool)

    assert analysis['turns_disambiguate']['gold']['true'] + \
           analysis['turns_disambiguate']['gold']['false'] \
           == analysis['total_turns']

    analysis['turns_disambiguate']['predicted'] = calculate_key_proportions(
        ambiguous_data, 'predicted_disambiguation_label', bool)

    disambiguation_true_pred_data = get_data_where(
        ambiguous_data, 'turn', 'predicted_disambiguation_label', '1')
    disambiguation_false_pred_data = get_data_where(
        ambiguous_data, 'turn', 'predicted_disambiguation_label', '0')

    disambiguation_true_pred_data_after = get_data_after_disambiguation(split)

    disambiguation_correct_pred_data = get_data_where(
        ambiguous_data, 'turn', 'predicted_disambiguation_label',
        "turn_datum['disambiguation_label']")
    # note we invert 0s to 1s to get the ones failed
    disambiguation_failed_pred_data = get_data_where(
        ambiguous_data, 'turn', 'predicted_disambiguation_label',
        "abs(turn_datum['disambiguation_label'] - 1)")

    analysis['turns_disambiguate']['predicted']['accuracy_%'] = \
        count_turns(disambiguation_correct_pred_data) / analysis['total_turns']

    analysis['turns_disambiguate']['predicted']['object_f1'] = evaluate_joint_data_object_f1(
        ambiguous_data)

    analysis['turns_disambiguate']['predicted']['true_object_f1'] = \
        evaluate_joint_data_object_f1(disambiguation_true_pred_data)
    analysis['turns_disambiguate']['predicted']['false_object_f1'] = \
        evaluate_joint_data_object_f1(disambiguation_false_pred_data)
    analysis['turns_disambiguate']['predicted']['after_true_object_f1'] = \
        evaluate_joint_data_object_f1(disambiguation_true_pred_data_after)

    analysis['turns_disambiguate']['predicted']['correct'] = calculate_key_proportions(
        disambiguation_correct_pred_data, 'predicted_disambiguation_label', bool)
    analysis['turns_disambiguate']['predicted']['correct']['object_f1'] = \
        evaluate_joint_data_object_f1(disambiguation_correct_pred_data)

    analysis['turns_disambiguate']['predicted']['failed'] = calculate_key_proportions(
        disambiguation_failed_pred_data, 'predicted_disambiguation_label', bool)
    analysis['turns_disambiguate']['predicted']['failed']['object_f1'] = \
        evaluate_joint_data_object_f1(disambiguation_failed_pred_data)
    # analysis['turns_disambiguate']['predicted']['failed_gold'] = calculate_key_proportions(
    #     disambiguation_failed_pred_data, 'disambiguation_label', bool)

    # how many dialogues are there in the failed ones? does it fail twice in the same dialogues?
    analysis['turns_disambiguate']['predicted_failed_dialogues'] = \
        len(disambiguation_failed_pred_data)
    # what is the proportion of true/false predicted?
    analysis['turns_disambiguate']['predicted_failed_turns'] = \
        count_turns(disambiguation_failed_pred_data)

    # then, can you tell, one
    for label, pattern in [
        # ('then', r"then"),
        ('can_you_tell', r"can you tell"),
        ('ones?', r"one")
    ]:
        def matches_condition(x):
            return re.search(rf"([^a-z]|^){pattern}([^a-z]|$)", x, re.IGNORECASE)

        for key, data in [
            (f"containing_{label}", ambiguous_data),
            (f"failed_containing_{label}", disambiguation_failed_pred_data)
        ]:
            analysis['turns_disambiguate']['predicted'][key] = \
                calculate_key_proportions(
                    data, 'transcript', 'with_without',
                    extra_condition=matches_condition)

            with_predicted_data = get_data_where(
                data, 'turn', 'transcript', None, _func=matches_condition)

            # save value since it will be overwritten
            total = analysis['turns_disambiguate']['predicted'][key]['total']
            analysis['turns_disambiguate']['predicted'][key] = {
                **analysis['turns_disambiguate']['predicted'][key],
                **calculate_key_proportions(
                    with_predicted_data, 'predicted_disambiguation_label', bool)
            }
            analysis['turns_disambiguate']['predicted'][key]['total'] = total
        # _print_examples(with_predicted_data)

    # _print_examples(disambiguation_correct_pred_data, 150)
    # _print_examples(disambiguation_failed_pred_data)

    return analysis


def analyse_disambiguation_pos(split):
    # python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_lg")
    # neuralcoref.add_to_pipe(nlp)

    all_data = get_true_data(split)
    analysis = {}
    # ambiguous_data = get_all_ambiguous_coreferences(split)
    # disambiguation_failed_pred_data = get_data_where(
    #     ambiguous_data, 'turn', 'predicted_disambiguation_label',
    #     "abs(turn_datum['disambiguation_label'] - 1)")

    counter_disambiguation_true = collections.Counter()
    counter_disambiguation_false = collections.Counter()
    general_counter = collections.Counter()
    sentence_lengths = {0: [], 1: []}
    disambiguation_labels = []
    # correlation = {'the': []}
    tag_count = {k: [] for k in nlp.pipe_labels['tagger'] + ['_SP']}
    # # then, can you tell, one
    # for label, pattern in [
    #     # ('then', r"then"),
    #     ('can_you_tell', r"can you tell"),
    #     ('ones?', r"one")
    # ]:
    #     def matches_condition(x):
    #         return re.search(rf"([^a-z]|^){pattern}([^a-z]|$)", x, re.IGNORECASE)

    correlation_special = {k: [] for k in ['one', 'then', 'an you tell', 'in ']}
    only_count_once_per_doc = True
    for dialogue_index, dialogue_datum, turn_index, turn_datum in get_turn_iterator(all_data):
        if not is_disambiguation_turn(turn_datum):
            continue

        system_utt = dialogue_datum['dialogue'][turn_index - 1]['system_transcript'] if turn_index > 0 else ''
        sentence = f"{turn_datum['transcript']}"
        doc = nlp(sentence)

        sentence_lengths[turn_datum['disambiguation_label']].append(len(doc))

        if turn_datum['disambiguation_label'] == 1:
            _counter = counter_disambiguation_true
        elif turn_datum['disambiguation_label'] == 0:
            _counter = counter_disambiguation_false
        else:
            raise ValueError

        disambiguation_labels.append(turn_datum['disambiguation_label'])
        # add a placeholder 0
        for tag in tag_count.keys():
            tag_count[tag].append(0)

        tags_already_counted = []
        for token in doc:
            if token.tag_ not in tags_already_counted:
                _counter[token.tag_] += 1
                tag_count[token.tag_][-1] += 1
                if only_count_once_per_doc:
                    tags_already_counted.append(token.tag_)
            # if token.tag_ == 'IN':
            #     print(sentence)
            #     print(token)
            #     print('-----')

        # if sentence == 'Can you tell me the sizes and prices of the two grey jeans in the back two cubbies on the right?':
        #     print(doc)
        #     print([(token.tag_, token) for token in doc])
        #     exit()

        for word in correlation_special.keys():
            correlation_special[word].append(word in sentence.lower())

        # if 'WP' not in tags_already_counted and not turn_datum['disambiguation_label']:
        #     print(f"** {sentence} **")

        # disambiguation_labels.append(turn_datum['disambiguation_label'])
        # for tag in correlation.keys():
        #     correlation[tag].append(tag in tags_already_counted)

        # if doc._.has_coref:
        #     general_counter[f"has_coref_{turn_datum['disambiguation_label']}"] += 1
        #     # print(f"HAS coref: {sentence}")
        #     #
        #     # print(doc._.coref_clusters)
        #     # exit()
        # else:
        #     # print(f"no coref: {sentence}")
        #     general_counter[f"no_coref_{turn_datum['disambiguation_label']}"] += 1

        # times ['the', 'that', 'this', 'it'] appears in sentence
        # _counter['total_sentences'] += 1
        # the_amount = 0
        # for span in ['the', 'that', 'this', 'it']:
        #
        #     # _counter[token.tag_] += 1
        #     for token in doc:
        #         if span == token.text.lower():
        #             _counter[span] += 1
        #             if span == 'the':
        #                 the_amount += 1
        #             break
        # correlation['disambiguation_label'].append(turn_datum['disambiguation_label'])
        # correlation['the'].append(the_amount)

    # sentence length mean/SD
    # print(f"Sentence Length Analysis for {split}")
    for k, v in sentence_lengths.items():
        analysis[f"sentence_length_mean_disambiguate_{bool(k)}"] = np.mean(v)
        analysis[f"sentence_length_std_disambiguate_{bool(k)}"] = np.std(v)

        # print(f"\tDisambiguate={bool(k)}: {np.mean(v)} (SD: {np.std(v)})")
    # print(np.mean(sentence_lengths[0]))
    # print(np.std(sentence_lengths[0]))
    # print(np.mean(sentence_lengths[1]))
    # print(np.std(sentence_lengths[1]))

    print(f"Are the variances in sentence length similar? \n\tNull Hypothesis: variances are equal"
          f"\n\t{scipy.stats.levene(sentence_lengths[0], sentence_lengths[1])}"
          f"\n\t-> Reject NH if p-value < 0.05")

    # res = ttest_ind(sentence_lengths[0], sentence_lengths[1], equal_var=True)

    res = pg.ttest(sentence_lengths[0], sentence_lengths[1], correction=False)

    print(f"Two-tailed Student T-test:\n{res}\n")
    analysis['sentence_length_two_tailed_t-test'] = res

    analysis['sentence_length_correlation_to_disambiguate'] = pearsonr(
        [0] * len(sentence_lengths[0]) + [1] * len(sentence_lengths[1]),
        [x for x in sentence_lengths[0] + sentence_lengths[1]])

    # make both the same length for plot
    min_len = min([len(v) for v in sentence_lengths.values()])
    sentence_lengths = {k: v[:min_len] for k, v in sentence_lengths.items()}

    joint_lengths = {
        'sentence_length': [x for x in sentence_lengths[0] + sentence_lengths[1]],
        'disambiguate': [0] * len(sentence_lengths[0]) + [1] * len(sentence_lengths[1])
    }

    con = pd.DataFrame.from_dict(joint_lengths)
    # con = pd.DataFrame.from_dict({str(bool(k)): v for k, v in sentence_lengths.items()})

    sns.lmplot(x="disambiguate", y="sentence_length", data=con)
    # plt.savefig('correlation_sentence_length_disambiguate.png')

    # for counter in [counter_disambiguation_false, counter_disambiguation_true]:
    #     for key in counter_disambiguation_true:
    #         print(f"{'T' if counter is counter_disambiguation_true else 'F'} - {key}: {counter[key]} / {counter['total_sentences']}, {counter[key] / counter['total_sentences'] * 100}%")
    #
    #     print('---')

    # corr, _ = pearsonr(correlation['disambiguation_label'], correlation['the'])
    # print(f"Correlation: {corr}")

    # group positive cor
    tag_count['WP_VBZ'] = [x + y for x, y in zip(tag_count['WP'], tag_count['VBZ'])]
    # group negative
    tag_count['IN_JJ_PRP$'] = [x + y + z for x, y, z in zip(tag_count['IN'], tag_count['JJ'], tag_count['PRP$'])]

    correlation = {**tag_count, **correlation_special}

    # find combinations of correlations to compare to
    final_correlations = {}
    for key, values in correlation.items():
        if sum(values) == 0:
            # skip
            continue
        final_correlations[key] = pearsonr(disambiguation_labels, values)
        if final_correlations[key] is np.nan:
            del final_correlations[key]

    # grouped correlations
    _positive, _negative = '', ''
    for key in final_correlations.keys():
        if final_correlations[key][0] < 0:
            _negative += key + '+'
        else:
            _positive += key + '+'

    _positive = _positive[:-1]
    _negative = _negative[:-1]

    final_correlations[_positive] = pearsonr(disambiguation_labels, [sum(x) for x in zip(*[correlation[y] for y in _positive.split('+')])])
    final_correlations[_negative] = pearsonr(disambiguation_labels, [sum(x) for x in zip(*[correlation[y] for y in _negative.split('+')])])

    # order by strength
    final_correlations = dict(sorted(final_correlations.items(), key=lambda item: abs(item[1][0]), reverse=True))

    analysis['pos_correlations'] = final_correlations
    # print(f"Correlations: {json.dumps(final_correlations, indent=4)}")

    distribution_true = counter_to_distribution(counter_disambiguation_true)
    distribution_false = counter_to_distribution(counter_disambiguation_false)

    # print(json.dumps(distribution_true, indent=4))
    # print(json.dumps(distribution_false, indent=4))

    # not used at this moment
    difference = {}
    for key in counter_disambiguation_true.keys():
        if key not in counter_disambiguation_true:
            counter_disambiguation_true[key] = 0
        if key not in counter_disambiguation_false:
            counter_disambiguation_false[key] = 0

        difference[key] = counter_disambiguation_false[key] - counter_disambiguation_true[key]

    difference = dict(sorted(difference.items(), key=lambda item: abs(item[1]), reverse=True))

    # analysis['POS_distribution_difference_false-minus-true'] = difference
    # print(json.dumps(difference, indent=4))

    def substitute(tag):
        swap_dict = {
            'JJ': 'adjective',
            'VBP': 'verb non-3rd person',
            'VBZ': 'verb 3rd person',
            'VB': 'verb base',
            '.': 'full stop',
            ',': 'comma',
            'WP': 'wh-pronoun',
            'PRP$': 'pronoun, possessive'
        }
        return swap_dict[tag] if tag in swap_dict else spacy.explain(tag)   #   + ' ' + tag

    # tag distribution
    # order them both with the same order
    # distribution_false = {k: distribution_false[k] if k in distribution_false else 0 for k in distribution_true.keys()}
    all_tags = list(distribution_true.keys()) + list(distribution_false.keys())
    for tag in all_tags:
        distribution_true.setdefault(tag, 0)
        distribution_false.setdefault(tag, 0)

    # print(len(all_tags))
    distribution_final = {k: distribution_true[k] - distribution_false[k] for k in all_tags}
    # print(distribution_final)
    # filter those with a difference smaller than 0.01
    distribution_final = {k: v for k, v in distribution_final.items() if abs(v) > 0.007}
    all_tags = distribution_final.keys()
    # print(len(all_tags))

    distribution_true = {i: distribution_true[k] if k in distribution_true else 0 for i, k in enumerate(all_tags)}
    distribution_false = {i: distribution_false[k] if k in distribution_false else 0 for i, k in enumerate(all_tags)}

    plt.figure(figsize=(100,100),dpi=200)
    fig, ax = plt.subplots()
    # fig, axes = plt.subplots(figsize=(7, 5), dpi=100)
    tag_range = np.array(range(0, len(all_tags)))
    # print(tag_range - 0.25)

    plt.barh(tag_range - 0.2, distribution_true.values(), label='Disambiguate True', height=0.37)
    plt.barh(tag_range + 0.2, distribution_false.values(), label='Disambiguate False', height=0.37)
    ax.set_yticks(range(0, len(all_tags)))
    ax.set_yticklabels([substitute(x) for x in all_tags])
    ax.set_xlabel('Percentage')
    # ax.tick_params(axis='y', which='major', height=2)
    plt.title('Distribution of POS Tags')
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(left=0.25)
    plt.savefig('pos_distribution.png')
    print(f"Part of Speech distribution plot saved at 'pos_distribution.png'")

    return analysis


# Subtask#2

def get_experimental_analysis_subtask2(split):
    coreference_data = get_joint_data(split)

    analysis = {
        'total_turns': count_turns(coreference_data),
        'data': analyse_coreference_target_objects(coreference_data),
        'gpt2_baseline': analyse_gpt2_coreference_baseline(split),
        'ambiguous_coreferences': relation_between_ambiguous_coreferences(coreference_data)
    }

    return analysis


def relation_between_ambiguous_coreferences(all_data):
    counter = collections.Counter()
    scores = {}
    object_f1_thresholds = [0.5, 0.66, 0.99]

    def _eval_turn(counter_name, _turn_datum):
        object_f1 = evaluate_single_turn_object_f1(_turn_datum)['object_f1']

        for threshold in object_f1_thresholds:
            if object_f1 >= threshold:
                key = counter_name.format(threshold=threshold)
                counter[key] += 1

                if key not in scores:
                    scores[key] = []
                scores[key].append(np.mean([x[2] for x in turn_datum['prediction_outputs'].values()]))

    # general: object f1 when disambiguation label exists

    # What percentage of the failed co-reference predictions are actually ambiguous?
    for dialogue_index, dialogue_datum, turn_index, turn_datum in get_turn_iterator(all_data):
        if is_disambiguation_turn(turn_datum):

            if turn_datum['disambiguation_label'] == 1:
                # ambiguous coreferences
                counter['total_disambiguations=1'] += 1

                for threshold in object_f1_thresholds:
                    current_object_f1 = evaluate_single_turn_object_f1(turn_datum)['object_f1']

                    if current_object_f1 >= threshold:
                        counter[f"object_f1_above_{threshold:.2f}"] += 1

                    # also evaluate in the next turn
                    if len(dialogue_datum['dialogue']) > turn_index + 1:
                        next_turn_datum = dialogue_datum['dialogue'][turn_index + 1]
                        assert next_turn_datum['turn_idx'] == turn_index + 1

                        next_object_f1 = evaluate_single_turn_object_f1(next_turn_datum)['object_f1']

                        if next_object_f1 >= threshold:
                            counter[f"object_f1_above_{threshold:.2f}_after"] += 1

                            if current_object_f1 >= threshold:
                                counter[f"object_f1_above_{threshold:.2f}_both_correct"] += 1
                            elif current_object_f1 < threshold:
                                counter[f"object_f1_above_{threshold:.2f}_improved_after"] += 1
                            else:
                                raise ValueError
                        else:
                            if current_object_f1 >= threshold:
                                counter[f"object_f1_above_{threshold:.2f}_worse_after"] += 1
                            elif current_object_f1 < threshold:
                                counter[f"object_f1_above_{threshold:.2f}_both_wrong"] += 1
                            else:
                                raise ValueError

                # it seems worse, can I see what is the object f1 is I remove the turns after the disambiguation?

                # calculate Object F1 score here
                _eval_turn('object_f1_above_{threshold:.2f}', turn_datum)

                # also evaluate in the next turn
                if len(dialogue_datum['dialogue']) > turn_index + 1:
                    next_turn_datum = dialogue_datum['dialogue'][turn_index + 1]
                    assert next_turn_datum['turn_idx'] == turn_index + 1

                    _eval_turn('object_f1_above_{threshold:.2f}_after', next_turn_datum)

            else:
                counter['total_disambiguations=0'] += 1

        counter['total'] += 1

    # print(json.dumps(counter, indent=4))
    # final_scores = {}
    # for key, value in scores.items():
    #     final_scores[key] = np.mean(value), np.std(value)
    # print(json.dumps(final_scores, indent=4))

    return {
        'explanation': 'if _worse_after is higher than _improved_after, then in general, prediction '
                       'accuracy reduced after a disambiguation question, and vice-versa',
        **counter
    }


def analyse_gpt2_coreference_baseline(split) -> dict:
    """
    Analyse the output from the GPT-2 baseline offered in Kottur et al. 2021 for Subtask#2

    To check this, we applied several "replace with" actions (the action from any code editor)
    to eliminate everything else but input and predicted object IDs from the output .txt files.

    Belief State :  .*\) < to Belief State : <
    \>  \<EOB\>  .*\. to >
    \<EOM\> .* \<SOM\> to <EOM> <SOM>
    <EOM> User : [^\<]* => Belief State : to <EOM> =>
    [^\<\n]* => Belief State : to =>
    [^\n\<\>]* <SOM> to <SOM>
    <EOB> [^\n]* to <EOB>

    The file has a row for each prediction, e.g., "<SOM> 37, 40, 12 <EOM> =>  < 12, 40 >"
    """

    with open('simmc2/model/mm_dst/gpt2_dst/results/simmc2_dials_dstc10_dev_predicted_f1=0.3637_modified.txt', 'r') as in_file:
        pred_content = in_file.readlines()

    assert len(pred_content) == 3494 - 2

    analysis = {}

    def parse_object_ids(string) -> list:
        result = []
        all_strings = string.split()
        for obj in all_strings:
            # .replace('<SOM>', '').replace('<EOM>', '').replace('<EOM>', '')
            tmp_object_id = obj.strip(' <SOM><EOM>,<EOB>')
            if tmp_object_id != '':
                result.append(int(tmp_object_id))

        return result

    gpt2_pred_data = []         # input, prediction
    for line in pred_content:
        # print(line)
        input_part, predicted_part = line.split('=>')

        input_part = ' '.join(input_part.split('<')[0:])
        predicted_part = predicted_part.split('>')[0]
        input_part = parse_object_ids(input_part)
        predicted_part = parse_object_ids(predicted_part)

        # print(f"{line} -> {input_part} | {predicted_part}")
        gpt2_pred_data.append((input_part, predicted_part))

    # tmp = [x[1] for x in gpt2_pred_data]
    all_predicted_ids = []
    [all_predicted_ids.extend(x[1]) for x in gpt2_pred_data]
    # all_predicted_ids = [x[1] for x in gpt2_pred_data]
    analysis['max_predicted_id'] = max(all_predicted_ids)
    analysis['unique_predicted_ids'] = len(set(all_predicted_ids))

    counter = collections.Counter()
    counter_pred_input = collections.Counter()

    for row in gpt2_pred_data:
        for pred_object_id in row[1]:
            counter[pred_object_id] += 1

            # print(f"check {pred_object_id} in {row[0]}")
            if pred_object_id in row[0]:
                # predicted was part of input
                counter_pred_input['pred_object_in_input'] += 1
            else:
                counter_pred_input['pred_object_NOT_in_input'] += 1

    analysis['pred_objects_in_input'] = counter_pred_input['pred_object_in_input'] / \
        (counter_pred_input['pred_object_in_input'] + counter_pred_input['pred_object_NOT_in_input'])

    analysis['distribution'] = counter_to_distribution(counter)

    analysis['%_pred_ids_below_5'] = sum([v for k, v in analysis['distribution'].items() if k < 5 ])
    analysis['%_pred_ids_below_10'] = sum([v for k, v in analysis['distribution'].items() if k < 10 ])
    analysis['%_pred_ids_below_20'] = sum([v for k, v in analysis['distribution'].items() if k < 20 ])

    return analysis


def analyse_roi_features(split) -> dict:
    analysis = {}
    # get roi feats first
    from coreference_model.src.utils import load_obj_tsv

    img_data = load_obj_tsv(
        os.path.join('simmc2_data_generated', 'image_features/', '%s_%s_detectron_feats_%s.tsv' % (
            split, 'colour_types_compound', 'all_gt_boxes')))

    cos_similarity_per_img = []
    for img_datum in img_data:
        # find two objects with the same category
        found_so_far = {}
        cosine_similarities = []
        for index_i, obj_name in enumerate(img_datum['t_category']):
            if obj_name == '':
                # skip these placeholders
                continue

            if obj_name not in found_so_far:
                found_so_far[obj_name] = [index_i]
            else:
                for index_j in found_so_far[obj_name]:
                    cosine_similarities.append(
                        1 - distance.cosine(img_datum['features'][index_i], img_datum['features'][index_j])
                    )

                found_so_far[obj_name].append(index_i)

        if len(cosine_similarities) > 0:
            cos_similarity_per_img.append(np.mean(cosine_similarities))

        # print(f"mean cos similarity: {np.mean(cosine_similarities)}, SD: {np.std(cosine_similarities)}")

    # print(f"mean cos similarity: {np.mean(cos_similarity_per_img)}, SD: {np.std(cos_similarity_per_img)}")

    # size of bounding boxes
    all_box_areas = []
    for img_datum in img_data:
        img_h, img_w = img_datum['img_h'], img_datum['img_w']

        boxes = img_datum['boxes'].copy()
        # boxes[:, (0, 2)] /= img_w
        # boxes[:, (1, 3)] /= img_h

        for bounding_box in boxes:
            width = bounding_box[2] - bounding_box[0]
            height = bounding_box[3] - bounding_box[1]

            area = width * height
            assert area >= 0
            # if area == 0:
            #     print('0!')

            all_box_areas.append(area)
            # print(bounding_box)
            # print(area)

    analysis['total_images'] = len(img_data)
    analysis['bounding_box_count'] = len(all_box_areas)
    analysis['bounding_box_mean'] = len(all_box_areas) / len(img_data)
    analysis['bounding_box_area_mean'] = np.mean(all_box_areas)
    analysis['bounding_box_area_sd'] = np.std(all_box_areas)
    analysis['mean_cosine_similarity'] = np.mean(cos_similarity_per_img)
    analysis['mean_cosine_similarity_std'] = np.std(cos_similarity_per_img)

    return analysis


def data_analysis(all_data) -> dict:
    simmc2_metadata = get_metadata()
    simmc2_scenes_jsons = get_scene_data()

    object_classes = generate_object_classes(simmc2_metadata)

    analysis = {}

    analysis = {**analysis, **extract_object_info(all_data, simmc2_scenes_jsons, simmc2_metadata)}

    objects_per_entry = []

    for dialogue_index, dialogue_datum, turn_index, turn_datum in get_turn_iterator(all_data):
        objects_per_entry.append(len(turn_datum['scene_objects']))

    analysis['total_turns'] = len(objects_per_entry)
    analysis['total_objects'] = sum(objects_per_entry)
    analysis['objects_per_entry_mean'] = np.mean(objects_per_entry)
    analysis['objects_per_entry_std'] = np.std(objects_per_entry)

    # print(json.dumps(analysis, indent=4))

    return {**analysis, **analyse_data({}, simmc2_scenes_jsons, simmc2_metadata, object_classes)}


def main(args):
    print(f"Dataset split: {args.dataset_split}\n")

    all_data = get_true_data(args.dataset_split)
    # random_baselines(args.dataset_split)
    analysis = {
        'dataset_split': args.dataset_split,
        'total_dialogues': len(all_data),
        'total_turns': count_turns(all_data),
        'images': analyse_roi_features(args.dataset_split),
        'data': data_analysis(all_data),
        'subtask#1_disambiguation': get_experimental_analysis_subtask1(args.dataset_split),
        'subtask#2_coreference': get_experimental_analysis_subtask2(args.dataset_split),
    }

    # print(f"{args.dataset_split} results:\n{json.dumps(analysis, indent=4, default=str)}")
    with open(args.output_file, 'w') as out_file:
        json.dump(analysis, out_file, indent=4, default=str)

    print(f"\nSaved to '{args.output_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_data_folder_path", default="simmc2/data",
        help="Path to folder with SIMMC2 data"
    )
    # parser.add_argument(
    #     "--input_disambiguation_folder_path", default="disambiguation_model/",
    #     help="Path to folder with the disambiguation model"
    # )
    # parser.add_argument(
    #     "--input_coreference_folder_path", default="coreference_model/",
    #     help="Path to folder with the coreference model"
    # )
    parser.add_argument(
        "--dataset_split", default="devtest",
        help="dataset split"
    )
    # parser.add_argument(
    #     '--subtask', required=True, default=None,
    #     choices=[
    #         SUBTASK_ALL])

    parser.add_argument(
        "--output_file", default="experiment_results.json",
        help="Name of file to save experimental analysis results"
    )

    print(f"Performing Experimental Analysis\n{'='*32}")
    _print_indentation = 2

    main(parser.parse_args())

    _print_indentation = 0
    print('\nFinished\n')
