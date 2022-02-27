#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
    Author : JChiyah
    Date   : July 2021
    Python : 3.8.10

"""
import collections
import os
import copy
import argparse
import glob
import shutil
import json
import pathlib

import numpy as np
import cv2
from tqdm import tqdm


SPLITS = ['train', 'dev', 'devtest', 'teststd_public']
# SPLITS = ['teststd_public']
DOMAINS = ['fashion', 'furniture']

DIALOGUES = 'simmc2_dials_dstc10_{}.json'       # train|dev|devtest|test
SCENES = 'simmc2_scene_images_dstc10_public'
SCENES_TESTSTD = 'simmc2_scene_images_dstc10_teststd'
SCENE_JSONS = 'simmc2_scene_jsons_dstc10_*'
# SCENE_JSONS = 'simmc2_scene_jsons_dstc10_teststd'
METADATA = '{}_prefab_metadata_all.json'        # fashion|furniture

SUBTASK_DISAMBIGUATION = 'disambiguation'
SUBTASK_COREFERENCE = 'coreference'
SUBTASK_ANALYSIS = 'analysis'
SUBTASK_METADATA = 'metadata'
SUBTASK_IMAGE_EXTRACTION = 'image_extraction'
SUBTASK_ALL = 'all'

OUTPUT_IMAGE_FOLDER = 'images'

TOKEN_USER = 'USER'
TOKEN_SYSTEM = 'SYSTEM'
TOKEN_ENTITY_SEP = ':'
# example of input data to predict belief state in simmc2_dials_dstc10_devtest_predict.txt:3
# System : What do you think of the grey pair on the left? <SOM> 29 <EOM>
# User : Sorry, I misspoke. Can you show me dresses instead?
# System : There's a maroon one on the wall on the right, and a brown one and a grey one on the
# rack. <SOM> 42, 14, 36 <EOM> User : Does the grey have good reviews? => Belief State :

# from https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.BoxMode
XYXY_ABS = 0
XYWH_ABS = 1


def _copy_image(input_folder, output_folder, image_name):
    final_path = os.path.join(output_folder, image_name + '.png')
    # copy image to correct folder if it doesn't exist
    if not os.path.exists(final_path):
        try:
            shutil.copy(os.path.join(input_folder, SCENES, image_name + '.png'), final_path)
        except FileNotFoundError:
            # check the teststd image folder
            shutil.copy(os.path.join(input_folder, SCENES_TESTSTD, image_name + '.png'), final_path)

    return final_path


def _output_json_data(data: dict, output_data_folder, split):
    if not os.path.exists(output_data_folder):
        pathlib.Path(output_data_folder).mkdir(parents=True)

    if 'entries' in data:
        print(f"\tTotal entries [{split}]: {len(data['entries'])}")
    save_file_path = os.path.join(output_data_folder, f"{split}.json")
    # print(f"Saving: {save_file_path}")
    with open(save_file_path, "w") as file_out:
        json.dump(data, file_out, indent=4)


def _check_missing_data(missing_entries):
    if len(missing_entries) > 0:
        print(f"Found {len(missing_entries)} entries with missing data! Examples:")
        for missing_entry in missing_entries[:10]:
            print(f"\t{missing_entry[0]}: {missing_entry[1]} for {missing_entry[2]}")

        raise ValueError


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


def _build_dialogue_history_utterance(speaker, turn_datum):
    if speaker == 'user':
        speaker = TOKEN_USER
        utterance = turn_datum['transcript']
        # todo: this value cannot be used at inference time, even from a previous turn
        if 'transcript_annotated' in turn_datum:
            objects_referenced = turn_datum['transcript_annotated']['act_attributes']['objects']
        else:
            objects_referenced = []
    elif speaker == 'system':
        speaker = TOKEN_SYSTEM
        utterance = turn_datum['system_transcript'] if 'system_transcript' in turn_datum else ''
        objects_referenced = turn_datum['system_transcript_annotated']['act_attributes']['objects']
    else:
        raise ValueError

    return {
        'utterance': f"{speaker} {TOKEN_ENTITY_SEP} {utterance}",
        'objects': objects_referenced
    }


def process_disambiguation_data(
        input_folder: str, output_folder: str, dialogues: dict, scene_jsons: dict, metadata: dict):
    subtask = SUBTASK_DISAMBIGUATION
    output_data_folder = os.path.join(output_folder, subtask)
    # output_image_folder = os.path.join(output_folder, FOLDER_IMAGES)

    # if not os.path.exists(output_image_folder):
    #     pathlib.Path(output_image_folder).mkdir(parents=True)

    missing_entries = []        # list of (entry_idx, attribute, attribute_idx)
    domains = []
    for split in SPLITS:
        print(f"Processing {subtask} data [{split}]...")
        next_idx = 1
        disambiguation_data = {
            'entries': [],
            'scenes': {},
            'scene_bboxes': {},
            # 'object_metadata': metadata
        }
        # print(len(dialogues[split]["dialogue_data"]))
        for dialog_id, dialog_datum in enumerate(dialogues[split]["dialogue_data"]):
            history = []
            domains.append(dialog_datum['domain'])
            for turn_datum in dialog_datum["dialogue"]:
                history.append(_build_dialogue_history_utterance('user', turn_datum))

                label = turn_datum.get("disambiguation_label", None)
                if "disambiguation_label" in turn_datum:
                    scene_idx, _, image_name = _get_scene_idx(
                        dialog_datum["scene_ids"], turn_datum["turn_idx"])

                    # _copy_image(input_folder, output_image_folder, image_name)

                    entry = {
                        'entry_idx': f"simmc2_{subtask}_{split}_d{dialog_datum['dialogue_idx']}_{next_idx}",
                        'dialogue_idx': dialog_datum['dialogue_idx'],
                        'turn_idx': turn_datum['turn_idx'],
                        'dialogue_history': copy.deepcopy(history[:-1]),
                        'user_utterance': copy.deepcopy(history[-1]['utterance']),
                        'disambiguation_label': label,
                        'image_name': image_name,
                        'scene_idx': scene_idx
                    }
                    disambiguation_data['entries'].append(entry)

                    # sometimes, the scene _bbox file and/or the _scene file will be with/without m_
                    # see https://github.com/facebookresearch/simmc2/issues/6
                    # So, need to check for 2 filenames
                    try:
                        scene = scene_jsons[
                            f"{scene_idx if scene_idx + '_scene' in scene_jsons else image_name}_scene"]
                    except KeyError:
                        missing_entries.append((entry['entry_idx'], 'scene_info', scene_idx))
                        scene = None

                    try:
                        scene_bbox = scene_jsons[
                            f"{scene_idx if scene_idx + '_bbox' in scene_jsons else image_name}_bbox"]
                    except KeyError:
                        missing_entries.append((entry['entry_idx'], 'scene_bbox', scene_idx))
                        scene_bbox = None

                    disambiguation_data['scenes'][scene_idx] = json.dumps(scene)
                    disambiguation_data['scene_bboxes'][scene_idx] = json.dumps(scene_bbox)

                    next_idx += 1

                history.append(_build_dialogue_history_utterance('system', turn_datum))

        _output_json_data(disambiguation_data, output_data_folder, split)

    _check_missing_data(missing_entries)


def process_coreference_data(
        input_folder: str, output_folder: str, dialogues: dict, scene_jsons: dict, metadata: dict):
    subtask = SUBTASK_COREFERENCE
    output_data_folder = os.path.join(output_folder, subtask)
    output_image_folder = os.path.join(output_folder, OUTPUT_IMAGE_FOLDER)

    if not os.path.exists(output_image_folder):
        pathlib.Path(output_image_folder).mkdir(parents=True)

    missing_entries = []        # list of (entry_idx, attribute, attribute_idx)
    for split in SPLITS:
        print(f"Processing {subtask} data [{split}]...")
        next_idx = 1
        coreference_data = {
            'entries': [],
            'scenes': {},
            'scene_bboxes': {},
            'object_metadata': metadata if split != 'teststd_public' else None
        }

        # We need to predict the object indexes in 'target_object_ids', which is a list of obj_index
        # for teststd data, this field is null to predict accordingly

        for dialog_id, dialog_datum in enumerate(dialogues[split]["dialogue_data"]):
            history = []
            for turn_datum in dialog_datum["dialogue"]:
                history.append(_build_dialogue_history_utterance('user', turn_datum))

                scene_idx, previous_scene_idx, image_name = _get_scene_idx(
                    dialog_datum["scene_ids"], turn_datum["turn_idx"])

                _copy_image(input_folder, output_image_folder, image_name)

                entry = {
                    'entry_idx': f"simmc2_{SUBTASK_COREFERENCE}_{split}_d{dialog_datum['dialogue_idx']}_{next_idx}",
                    'dialogue_idx': dialog_datum['dialogue_idx'],
                    'turn_idx': turn_datum['turn_idx'],
                    'dialogue_history': copy.deepcopy(history[:-1]),
                    'user_utterance': copy.deepcopy(history[-1]['utterance']),
                    'target_object_ids':
                        turn_datum['transcript_annotated']['act_attributes']['objects'] \
                        if 'transcript_annotated' in turn_datum else None,
                    'image_name': image_name,
                    'scene_idx': scene_idx,
                    'previous_scene_idx': previous_scene_idx
                }
                coreference_data['entries'].append(entry)

                # sometimes, the scene _bbox file and/or the _scene file will be with/without m_
                # see https://github.com/facebookresearch/simmc2/issues/6
                # So, need to check for 2 filenames
                try:
                    scene = scene_jsons[
                        f"{scene_idx if scene_idx + '_scene' in scene_jsons else image_name}_scene"]
                except KeyError:
                    missing_entries.append((entry['entry_idx'], 'scene_info', scene_idx))
                    scene = None

                try:
                    scene_bbox = scene_jsons[
                        f"{scene_idx if scene_idx + '_bbox' in scene_jsons else image_name}_bbox"]
                except KeyError:
                    missing_entries.append((entry['entry_idx'], 'scene_bbox', scene_idx))
                    scene_bbox = None

                # note that the bboxes coming out of here have the format XYHW, not XYWH (standard)
                # fixme: convert them as with the image processing func
                coreference_data['scenes'][scene_idx] = json.dumps(scene)
                coreference_data['scene_bboxes'][scene_idx] = json.dumps(scene_bbox)

                next_idx += 1

                history.append(_build_dialogue_history_utterance('system', turn_datum))

        _output_json_data(coreference_data, output_data_folder, split)

    _check_missing_data(missing_entries)


def process_image_extraction_data(
        input_folder: str, output_folder: str, dialogues: dict, scene_jsons: dict, metadata: dict,
        object_classes: dict):
    subtask = SUBTASK_IMAGE_EXTRACTION
    output_data_folder = os.path.join(output_folder, subtask)
    output_image_folder = os.path.join(output_folder, OUTPUT_IMAGE_FOLDER)

    if not os.path.exists(output_image_folder):
        pathlib.Path(output_image_folder).mkdir(parents=True)

    def _get_entry_template():
        # Detectron2/COCO format
        return {
            'image_id': '',         # image_id (str or int): a unique id that identifies this image
            'file_name': '',        # full path to the image file
            'height': -1,           # height, width: integer. The shape of the image.
            'width': -1,            # height, width: integer. The shape of the image.
            'annotations': []       # annotations (list[dict]): Required
        }

    def _get_annotation_entry_template():
        return {
            'bbox': [],             # bbox (list[float]): list of 4 numbers for the bounding box
            'bbox_mode': XYWH_ABS,  # bbox_mode (int): the format of bbox: XYXY_ABS or XYWH_ABS
            'category_id': -1,      # category_id (int): in the range [0, num_categories-1]
            'segmentation': []      # segmentation (list[list[float]]): segmentation mask NOT USED
        }

    print(f"Processing {subtask} data...")
    missing_entries = []        # list of (entry_idx, attribute, attribute_idx)
    for split in SPLITS:
        image_extraction_data = {
            'entries': {}
        }

        for dialog_id, dialog_datum in tqdm(enumerate(dialogues[split]["dialogue_data"]), total=len(dialogues[split]["dialogue_data"]), desc=split):
            # not sure whether images are split in datasets (e.g., no train images appear in dev),
            # so iterating over images from dialogues instead
            # Also, [img_name] and [m_ + img_name] have very different metadata for some reason
            # so need to have separate entries for these, and not repeat previous ones

            for turn_idx in dialog_datum["scene_ids"].keys():
                scene_idx, _, image_name = _get_scene_idx(
                    dialog_datum["scene_ids"], int(turn_idx))
                if scene_idx in image_extraction_data['entries']:
                    # skip to avoid duplications
                    continue

                # copy image first
                image_path = _copy_image(input_folder, output_image_folder, image_name)

                # this could probably be faster but it is the way that lxmert does it at preprocess
                im = cv2.imread(image_path)
                try:
                    np.size(im, 0)
                except IndexError:
                    # see corrupted images from https://github.com/facebookresearch/simmc2/issues/2
                    if image_name not in [
                        'cloth_store_1416238_woman_4_8', 'cloth_store_1416238_woman_19_0',
                            'cloth_store_1416238_woman_20_6']:
                        # corrupted image, ignore
                        print(f"Ignoring corrupted image: {image_name}")
                    continue

                entry  =_get_entry_template()
                entry['file_name'] = os.path.join(OUTPUT_IMAGE_FOLDER, image_name)
                entry['height'] = np.size(im, 0)
                entry['width'] = np.size(im, 1)
                entry['image_id'] = scene_idx

                scene = scene_jsons[
                    f"{scene_idx if scene_idx + '_scene' in scene_jsons else image_name}_scene"]

                # print(scene['scenes'][0].keys())
                for item in scene['scenes'][0]['objects']:
                    annotation = _get_annotation_entry_template()
                    annotation['simmc2_obj_index'] = item['index']
                    # bbox in simmc2 are `x`, `y`, `height`, `width`, so need to swap h by w
                    annotation['bbox'] = item['bbox'][:2] + [item['bbox'][3]] + [item['bbox'][2]]

                    annotation['types_id'] = object_classes[
                        'types'][object_classes['prefab_path_to_types'][item['prefab_path']]]
                    annotation['types_name'] = object_classes['prefab_path_to_types'][item['prefab_path']]

                    annotation['colour_types_single_id'] = object_classes[
                        'colour_types_single'][object_classes['prefab_path_to_colour_types_single'][item['prefab_path']]]
                    annotation['colour_types_single_name'] = \
                        object_classes['prefab_path_to_colour_types_single'][item['prefab_path']]
                    annotation['colour_types_compound_id'] = object_classes[
                        'colour_types_compound'][object_classes['prefab_path_to_colour_types_compound'][item['prefab_path']]]
                    annotation['colour_types_compound_name'] = object_classes[
                        'prefab_path_to_colour_types_compound'][item['prefab_path']]

                    entry['annotations'].append(annotation)

                if entry['image_id'] in image_extraction_data['entries'] \
                    and entry != image_extraction_data['entries'][entry['image_id']]:
                    raise KeyError(f"{entry['image_id']} already in entries: \nnew: {entry}\n\nold: {image_extraction_data['entries'][entry['image_id']]}")
                image_extraction_data['entries'][entry['image_id']] = entry
                # image_extraction_data['entries'].append(entry)

        _output_json_data(image_extraction_data, output_data_folder, split)

    _check_missing_data(missing_entries)


def generate_object_classes(metadata: dict, output_folder: str = None):
    data = {
        'total_types': 0,
        'total_colours': 0,
        'total_colour_types_single': 0,
        'total_colour_types_compound': 0,
        'total_prefab_path_to_types': 0,
        'types': {},
        'colours': {},
        'colour_types_single': {},
        'colour_types_compound': {},
        'prefab_path_to_types': {},
        'prefab_path_to_colour_types_single': {},
        'prefab_path_to_colour_types_compound': {},
        'prefab_path_to_global_token': {}
    }
    for prefab_path, item in metadata.items():
        if prefab_path not in data['prefab_path_to_types'] \
                or prefab_path not in data['prefab_path_to_colour_types_compound']:
            object_class = item['assetType' if 'assetType' in item else 'type']

            # colours = item['color'].replace(',', '').split()
            object_colour_class = None
            colours = item['color'].split(', ')
            # reduce the amount of colours, so light grey becomes grey
            colours = [colour.split()[-1] for colour in colours]
            for colour in colours:
                if colour not in data['colours']:
                    data['colours'][colour] = len(data['colours'].keys())
                colour_class = f"{colour}_{object_class}"
                if colour_class not in data['colour_types_single']:
                    data['colour_types_single'][colour_class] = len(data['colour_types_single'].keys())
                if object_colour_class is None:
                    object_colour_class = colour_class

            compound_colour = f"{'_'.join(colours)}_{object_class}".replace(' ', '')
            if compound_colour not in data['colour_types_compound']:
                data['colour_types_compound'][compound_colour] = len(data['colour_types_compound'].keys())

            if prefab_path not in data['prefab_path_to_types']:
                data['prefab_path_to_types'][prefab_path] = object_class
            if prefab_path not in data['prefab_path_to_colour_types_single']:
                data['prefab_path_to_colour_types_single'][prefab_path] = object_colour_class
            if prefab_path not in data['prefab_path_to_colour_types_compound']:
                data['prefab_path_to_colour_types_compound'][prefab_path] = compound_colour

            if object_class not in data['types']:
                data['types'][object_class] = len(data['types'].keys())  # = {
                # 'color': item['color'],
                # 'pattern': item['pattern'],
                # 'sleeveLength': item['sleeveLength'],
                # 'type': item['type']
                # }
            # global token representation
            data['prefab_path_to_global_token'][prefab_path] = '[OBJ_{}]'.format(
                len(data['prefab_path_to_global_token']))

        # else ignore, already handled

    data['total_types'] = len(data['types'])
    data['total_colours'] = len(data['colours'])
    data['total_colour_types_single'] = len(data['colour_types_single'])
    data['total_colour_types_compound'] = len(data['colour_types_compound'])
    data['total_prefab_path_to_types'] = len(data['prefab_path_to_types'])

    if output_folder is not None:
        _output_json_data(data, output_folder, 'object_classes')

    return data


def analyse_data(dialogues: dict, scene_jsons: dict, metadata: dict, object_classes: dict):
    analysis = {}
    # scene_bboxes = [len(scene['Items'])
    # for key, scene in scene_jsons.items() if key.endswith('bbox')]
    # it seems that the _scene.json files don't have the "camera" objects as in _bbox.json files
    # using those bboxes instead, as all the field but "name" seem the same
    scene_bboxes = [
        len(scene['scenes'][0]['objects']) for key, scene in scene_jsons.items()
        if key.endswith('scene')]

    analysis['bboxes_mean'] = np.mean(scene_bboxes)
    analysis['bboxes_max'] = max(scene_bboxes)
    analysis['bboxes_median'] = np.median(scene_bboxes)

    # find MAX_OBJECTS_PER_SCENE for the positional embeddings per type of item
    max_objects_per_scene = (None, 0)
    max_asset_type_per_scenes = {}
    max_asset_type_per_scene_counter = collections.Counter()
    scenes_above_100_objects = {}
    total_scenes = 0
    objects_per_scene = []
    for scene_key, scene in scene_jsons.items():
        if not scene_key.endswith('scene'):
            # ignore those about bboxes
            continue

        total_scenes += 1
        if len(scene['scenes'][0]['objects']) > max_objects_per_scene[1]:
            max_objects_per_scene = (scene_key, len(scene['scenes'][0]['objects']))

        counter = collections.Counter()
        for item in scene['scenes'][0]['objects']:
            asset_type = object_classes['prefab_path_to_types'][item['prefab_path']]
            if asset_type not in counter:
                counter[asset_type] = 0
            counter[asset_type] += 1

        most_common = counter.most_common()[0]
        if most_common[1] > max_asset_type_per_scene_counter[most_common[0]]:
            max_asset_type_per_scene_counter[most_common[0]] = most_common[1]
            max_asset_type_per_scenes[most_common[0]] = scene_key

        if len(scene['scenes'][0]['objects']) > 100:
            scenes_above_100_objects[scene_key] = len(scene['scenes'][0]['objects'])

        objects_per_scene.append(len(scene['scenes'][0]['objects']))

    max_asset_type_per_scene_counter = {
        f"{asset_type}-{max_asset_type_per_scenes[asset_type]}": value
        for asset_type, value in max_asset_type_per_scene_counter.most_common()}

    analysis['total_scenes'] = total_scenes
    analysis['objects_per_scene_mean'] = np.mean(objects_per_scene)
    analysis['objects_per_scene_std'] = np.std(objects_per_scene)
    analysis['max_objects_per_scene'] = max_objects_per_scene
    analysis['max_asset_type_per_scene'] = max_asset_type_per_scene_counter
    analysis['scenes_above_100_objects'] = len(scenes_above_100_objects)
    analysis['scenes_above_100_objects_mean'] = len(scenes_above_100_objects) / total_scenes

    # scenes_above_100_objects = collections.OrderedDict(
    # sorted(scenes_above_100_objects.items(), key=lambda k: k[1]))

    print(json.dumps(analysis, indent=4))

    return analysis
    # print(json.dumps(scenes_above_100_objects, indent=4))


def main(args):
    assert args.subtask == SUBTASK_ANALYSIS or args.output_data_folder_path is not None, \
        f"--output_data_folder_path cannot be None"

    print('Reading all data...')
    # read all data
    simmc2_dialogues = {}
    for split in SPLITS:
        with open(os.path.join(args.input_data_folder_path, DIALOGUES.format(split)), "r") as f_in:
            simmc2_dialogues[split] = json.load(f_in)

    simmc2_scenes_jsons = {}
    _files = glob.glob(f"{args.input_data_folder_path}/{SCENE_JSONS}/*.json")
    for file in _files:
        with open(file, "r") as f_in:
            simmc2_scenes_jsons[os.path.splitext(os.path.basename(file))[0]] = json.load(f_in)

    simmc2_metadata = {}
    for domain in DOMAINS:
        with open(os.path.join(args.input_data_folder_path, METADATA.format(domain)), "r") as f_in:
            simmc2_metadata = {**simmc2_metadata, **json.load(f_in)}

    object_classes = generate_object_classes(simmc2_metadata, args.output_data_folder_path)

    if args.subtask in [SUBTASK_DISAMBIGUATION, SUBTASK_ALL]:
        process_disambiguation_data(
            args.input_data_folder_path, args.output_data_folder_path,
            simmc2_dialogues, simmc2_scenes_jsons, simmc2_metadata)

    if args.subtask in [SUBTASK_COREFERENCE, SUBTASK_ALL]:
        process_coreference_data(
            args.input_data_folder_path, args.output_data_folder_path,
            simmc2_dialogues, simmc2_scenes_jsons, simmc2_metadata)

    if args.subtask in [SUBTASK_ANALYSIS, SUBTASK_ALL]:
        analyse_data(simmc2_dialogues, simmc2_scenes_jsons, simmc2_metadata, object_classes)

    if args.subtask in [SUBTASK_IMAGE_EXTRACTION, SUBTASK_ALL]:
        process_image_extraction_data(
            args.input_data_folder_path, args.output_data_folder_path,
            simmc2_dialogues, simmc2_scenes_jsons, simmc2_metadata, object_classes)

    # if args.clear_output:
    # 	print(f"Deleting all contents of {args.output_data_folder_path}")
    # 	shutil.rmtree(args.output_data_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_data_folder_path", default="simmc2/data",
        help="Path to folder with SIMMC2 data"
    )
    parser.add_argument(
        '--subtask', required=True, default=None,
        choices=[
            SUBTASK_DISAMBIGUATION, SUBTASK_COREFERENCE, SUBTASK_ANALYSIS, SUBTASK_IMAGE_EXTRACTION,
            SUBTASK_METADATA, SUBTASK_ALL])
    parser.add_argument(
        "--output_data_folder_path", default=None,
        help="Path to save SIMMC2 JSONs",
    )
    parser.add_argument(
        "--clear_output", default=False, action="store_true",
        help="Delete all contents of output_data_folder_path",
    )

    main(parser.parse_args())
