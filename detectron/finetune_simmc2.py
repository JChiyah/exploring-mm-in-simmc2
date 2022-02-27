"""

Author: JChiyah
Date: 2021

Fine-tune Detectron2 on SIMMC2.0 data and generate feature files.

"""

import os
import copy
import json
import pickle
import sys
import base64
import shutil
import logging
import csv
import random
import itertools

import torch

import numpy as np
import cv2

import detectron2
import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch, default_setup, default_argument_parser
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.checkpoint import DetectionCheckpointer
# import detectron2.data.transforms as T
from detectron2.structures import BoxMode

from tqdm import tqdm

import tweaks


FIELDITEMS = ["img_id", "img_h", "img_w", "num_boxes","t_num_boxes", "boxes", "features","names","t_boxes","t_names","box_order"]\
             + ['t_simmc2_obj_indexes', 'category', 't_category', 'category_scores']


DEBUG = False
if DEBUG:
    print('DEBUG is True')


# The following func come from simmc2/model/mm_dst/utils/evaluate_dst.py
# Used to evaluate object coreference
def rec_prec_f1(n_correct, n_true, n_pred):
    rec = n_correct / n_true if n_true != 0 else 0.
    # print(f"correct: {n_correct} / {n_pred} / ntrue: {n_true}")
    prec = n_correct / n_pred if n_pred != 0 else 0.
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0.

    return rec, prec, f1


def get_category_name(category_mode, category_id: int):
    for category_name, cat_id in object_classes[category_mode].items():
        if cat_id == category_id:
            return category_name

    raise ValueError(
        f"Cannot find category id {category_id} in {object_classes[category_mode]} for '{category_mode}'")


def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box
    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
    Returns:
        float: value of the IoU for the two boxes.
    Raises:
        AssertionError: if the box is obviously malformed
    """
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou


def get_simmc2_dicts(input_image_annotations_folder, category_mode):
    splits = {
        'train': [],
        'dev': [],
        'devtest': [],
        'teststd_public': []
    }
    print('Pre-processing datasets')

    for split in splits.keys():
        with open(os.path.join(input_image_annotations_folder, f"{split}.json"), 'r') as input_file:
            annotations = json.load(input_file)

        if DEBUG:
            splits[split] = annotations['entries'][:20]
        else:
            splits[split] = list(annotations['entries'].values())

        split_proposals = {
            'ids': [],
            'boxes': [],
            'objectness_logits': [],
            'bbox_mode': BoxMode.XYXY_ABS
        }

        for index, img_datum in tqdm(enumerate(splits[split]), total=len(splits[split]), desc=split):
            # print(img_datum)
            # exit(1)
            # make paths absolute
            img_datum['file_name'] = os.path.abspath(img_datum['file_name']) + '.png'
            img_datum['file_name'] = img_datum['file_name'].replace('detectron', 'simmc2_data_generated')
            # fix for detectron2@v0.3
            for bbox_datum in img_datum['annotations']:
                # change bbox class from integer (for compatibility with detectron2@v0.1)
                bbox_datum['bbox_mode'] = BoxMode.XYWH_ABS
                # transform bboxes to BoxMode.XYXY_ABS so code is compatible with marios_tweaks
                # ignored because otherwise visualisation is wrong
                # bbox_datum['bbox'] = BoxMode.convert(
                #     bbox_datum['bbox'], from_mode=BoxMode.XYWH_ABS, to_mode=BoxMode.XYXY_ABS)

                bbox_datum['category_id'] = bbox_datum[f"{category_mode}_id"]
                bbox_datum['name'] = bbox_datum[f"{category_mode}_name"] # get_category_name(bbox_datum['category_id'])

            # add segmentation information
            for bbox_datum in img_datum['annotations']:
                xmin, ymin, width, height = bbox_datum['bbox']
                # need the segmentation for it to work, even if it is approx
                poly = [
                    (xmin, ymin), (xmin + width, ymin),
                    (xmin + width, ymin + height), (xmin, ymin + height)
                ]
                poly = list(itertools.chain.from_iterable(poly))

                bbox_datum['segmentation'] = [poly]

            # prepare proposal files (gold bounding boxes)
            raw_boxes = np.asarray([
                BoxMode.convert(b['bbox'], from_mode=b['bbox_mode'], to_mode=BoxMode.XYXY_ABS)
                for b in img_datum['annotations']])
            # raw_boxes = detectron2.structures.Boxes(torch.from_numpy(raw_boxes))

            split_proposals['ids'].append(img_datum['image_id'])
            split_proposals['boxes'].append(raw_boxes)
            split_proposals['objectness_logits'].append(np.ones(len(img_datum['annotations'])))

        with open(f"simmc2_proposals_{split}.json", 'wb') as out_file:
            pickle.dump(split_proposals, out_file)

    # splits['dev'] = splits['dev'][:10]

    print('Finished pre-processing datasets')
    return splits


with open('../simmc2_data_generated/object_classes.json', 'r') as in_file:
    object_classes = json.load(in_file)


def build_feat_model(cfg):
    return tweaks.rcnn.GeneralizedRCNNFeat(cfg)


class CustomTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("_evaluation", exist_ok=True)
            output_folder = "_evaluation"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    # function that generates the tsv file gtsv -> generate tsv file
    # modeling is mappert

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:
        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        # model = build_model(cfg)
        model = build_feat_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model


class CustomPredictor(DefaultPredictor):

    def __init__(self, cfg, model):
        # super(CustomPredictor, self).__init__(cfg)
        # skip parent's constructor to avoid calling the wrong build_model() because
        # ROIHeads with Feat is not available yet at the registry level
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = model  # build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.transform_gen = detectron2.data.transforms.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def predict_with_bboxes(self, original_image, gt_boxes):
        """
        JChiyah: overwrite __call__ so it accepts GT bboxes
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        raw_boxes = np.asarray([
            BoxMode.convert(b['bbox'], from_mode=b['bbox_mode'], to_mode=BoxMode.XYXY_ABS)
            for b in gt_boxes])
        raw_boxes = detectron2.structures.Boxes(torch.from_numpy(raw_boxes).cuda())

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # from https://github.com/airsplay/py-bottom-up-attention/blob/master/demo/demo_feature_extraction_given_box.ipynb
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            raw_height, raw_width = original_image.shape[:2]
            image = self.transform_gen.get_transform(original_image).apply_image(original_image)
            new_height, new_width = image.shape[:2]
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            # Scale the box
            scale_x = 1. * new_width / raw_width
            scale_y = 1. * new_height / raw_height
            # print(scale_x, scale_y)
            boxes = raw_boxes.clone()
            boxes.scale(scale_x=scale_x, scale_y=scale_y)

            proposals = detectron2.structures.Instances(
                (new_height, new_width), proposal_boxes=boxes, objectness_logits=torch.ones(len(gt_boxes)))

            # print(proposals)
            # print(boxes)
            inputs = {"image": image, "height": raw_height, "width": raw_width, "proposals": proposals}
            predictions = self.model([inputs])[0]
            # changed model so that pred_boxes have the same order as raw_boxes (and feats, classes, etc)

            predictions['instances'].pred_boxes = raw_boxes.to('cpu')

            return predictions['instances']

    def __call__(self, original_image):
        # Overwrite so we use the new method to predict above
        raise NotImplementedError


def get_config(args):
    cfg = get_cfg()
    # Config: https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/defaults.py

    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.merge_from_file("./faster_rcnn_R_101_C4_caffe.yaml")
    cfg.DATASETS.TRAIN = (f"simmc2_train_{args.category}",)
    cfg.DATASETS.TEST = (f"simmc2_dev_{args.category}",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
    cfg.OUTPUT_DIR = f"output_{args.category}"
    if not args.train and args.resume:
        print('Restoring model')
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # cfg.SOLVER.IMS_PER_BATCH = 16
    # cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    # cfg.SOLVER.MAX_ITER = 500    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    # cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, f"model_final_{cfg.SOLVER.MAX_ITER}.pth")
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NAME = cfg.MODEL.ROI_HEADS.NAME + ('Feat' if args.return_feat else '')
    cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True
    # changed for compatibility with https://github.com/airsplay/py-bottom-up-attention/blob/master/demo/demo_feature_extraction_given_box.ipynb
    # cfg.MODEL.ROI_HEADS.NAME = 'Res5ROIHeads' + ('Feat' if args.return_feat else '')
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(object_classes[args.category])
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    if not args.train:
        cfg.MODEL.LOAD_PROPOSALS = True
        cfg.MODEL.PROPOSAL_GENERATOR.NAME = 'PrecomputedProposals'
        cfg.DATASETS.PROPOSAL_FILES_TRAIN = ('simmc2_proposals_train.json',)
        cfg.DATASETS.PROPOSAL_FILES_TEST = ('simmc2_proposals_dev.json',)

    # cfg.TEST.DETECTIONS_PER_IMAGE = 100
    cfg.TEST.EVAL_PERIOD = 500
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.MAX_ITER = 2000
    cfg.SOLVER.STEPS = (1000, 1500, 2000, 2500)  # , 1550, 1600, 1650, 1700, 1750, 1800, 2000, 2500, 3750, 3000)
    cfg.SOLVER.GAMMA = 0.05
    # cfg.SOLVER.WEIGHT_DECAY = 0.0001
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0  # set a custom testing threshold

    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.8
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 141
    cfg.TEST.DETECTIONS_PER_IMAGE = 141

    # from marios... do they improve?
    cfg.MODEL.ROI_BOX_HEAD.FC_DIM = 2048
    cfg.SOLVER.BASE_LR = 0.0045
    cfg.SOLVER.MAX_ITER = 3000

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


# config: TEST.EVAL_PERIOD = 500, SOLVER.IMS_PER_BATCH = 4, SOLVER.BASE_LR = 0.001,
# SOLVER.WARMUP_ITERS = 500, SOLVER.MAX_ITER = 2000, SOLVER.STEPS = (1000, 1500)
# SOLVER.GAMMA = 0.05, MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5


def test_model(args, cfg, model):
    # todo: maybe test with all other datasets?
    # model = CustomTrainer.build_model(cfg)
    res = CustomTrainer.test(cfg, model)
    # if cfg.TEST.AUG.ENABLED:
    #     res.update(NewTrainer.test_with_TTA(cfg, model))
    if comm.is_main_process():
        verify_results(cfg, res)
    return res


def train_model(args, cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer.train()


def generate_tsv_data(cfg, model, category_mode, data_percentage=0):
    if data_percentage == 0:
        data_percentage = 100

    predictor = CustomPredictor(cfg, model)
    eval_results = {}

    output_folder_path = os.path.join('..', 'simmc2_data_generated', 'image_features')
    os.makedirs(output_folder_path, exist_ok=True)

    for split in dataset_dicts.keys():

        bb_acc = {}
        eval_metrics = {
            # on one side, we have class accuracy (e.g., jacket -> jacket) (accuracy %)
            'category_accuracy': {
                'n_total_objects': 0,
                'n_correct_objects': 0
            },
            # on the other hand, we have no of boxes predicted,
            # and whether those match ground truth or not (object f1, recall, precision)
            'bbox_accuracy': {
                "n_true_objects": 0.0,
                "n_pred_objects": 0.0,
                "n_correct_objects": 0.0,
            }
        }
        data_type = dataset_dicts[split][:int(data_percentage * len(dataset_dicts[split]) / 100)]

        tsv_path = os.path.join(
            output_folder_path,
            f"{split}_{category_mode}_detectron_feats_"
            f"{'all' if data_percentage == 100 else data_percentage}_gt_boxes.tsv")

        with open(tsv_path, 'w', encoding='utf-8') as tsv_file:

            writer = csv.DictWriter(tsv_file, delimiter='\t', fieldnames=FIELDITEMS)
            for d in tqdm(data_type, desc=f"Generating TSV [{split}]"):

                # print(d['file_name'])
                im = cv2.imread(d["file_name"])
                _instances = predictor.predict_with_bboxes(im, d['annotations'])

                # note that boxes should have the same order as d['annotations']
                # object_f1_mean would go down from 1 if the order is changed
                boxes = np.array(_instances.pred_boxes.tensor.tolist()).tolist()
                classes = np.array(_instances.pred_classes.tolist()).tolist()
                features = np.array(_instances.features.tolist()).tolist()
                class_scores = np.array(_instances.pred_class_scores.tolist()).tolist()

                # print(_instances)
                assert len(boxes) == len(d['annotations']), f"{len(boxes)} != {len(d['annotations'])}"
                gt_raw_boxes = np.asarray([
                    BoxMode.convert(b['bbox'], from_mode=b['bbox_mode'], to_mode=BoxMode.XYXY_ABS)
                    for b in d['annotations']])

                assert np.all(boxes == gt_raw_boxes), f"{boxes} != {gt_raw_boxes}"

                # max_items = 141
                f_features = np.copy(features)  # [-max_items:]
                # print(f_features.shape)

                # f_features = f_features[-50:]
                # print(f_features.shape)
                f_boxes = np.copy(boxes)    # [-max_items:]
                if len(boxes) == 0:
                    # ignored until we can fix it
                    print(f"Error: no bboxes in prediction!! {d['file_name']}")
                    continue

                gt_idx = {}
                # print(len(d['annotations']))

                t_boxes = np.copy(f_boxes)[:len(d['annotations'])]
                while len(t_boxes) < len(d['annotations']):
                    t_boxes = np.concatenate((t_boxes, [t_boxes[0]]), axis=0)
                # print(type(t_boxes))

                new_f_boxes = np.copy(t_boxes)[:len(d['annotations'])]
                new_f_features = np.copy(f_features)[:len(d['annotations'])]
                while len(new_f_features) < len(d['annotations']):
                    new_f_features = np.concatenate((new_f_features, [new_f_features[0]]), axis=0)
                names = ['unk'] * np.size(d['annotations'], 0)
                t_names = ['unk'] * np.size(d['annotations'], 0)
                pred_class_scores = np.zeros((np.size(d['annotations']), cfg.MODEL.ROI_HEADS.NUM_CLASSES), dtype=float)
                t_simmc2_obj_indexes = np.zeros(np.size(d['annotations'], 0), dtype=int)
                # print(t_boxes)
                # print(t_names)
                # print(len(new_f_features))
                # print(len(names))

                eval_metrics['category_accuracy']['n_total_objects'] += len(d['annotations'])
                eval_metrics['bbox_accuracy']['n_true_objects'] += len(d['annotations'])
                eval_metrics['bbox_accuracy']['n_pred_objects'] += len(_instances)

                # list of indexes that have already been used
                # this fixes Marios' issue that allows precision, recall and object f1 to be above 0
                # (by not selecting unique bboxes)
                pred_index_used = []
                # I should probably use the matrix method that I used in LXMERT, as currently,
                # this way is very optimistic and probably has better results than it should
                # get the names, new_f_features and new_f_boxes
                for index_gt_box, gt_box in enumerate(d['annotations']):
                    # convert to XYXY format to make compatible with this code
                    gt_box = copy.deepcopy(gt_box)
                    gt_box['bbox'] = BoxMode.convert(
                        gt_box['bbox'], from_mode=gt_box['bbox_mode'], to_mode=BoxMode.XYXY_ABS)
                    # print(f"{len(d['annotations'])}, {len(f_boxes)}, {len(t_boxes)}")
                    # moved down
                    # t_boxes[index_gt_box] = gt_box['bbox']
                    # JChiyah: name is the category_name, as done in pre-processing
                    # t_names[index_gt_box] = gt_box['name']
                    # this is the ground truth (in case we want to use gold data)
                    t_boxes[index_gt_box] = gt_box['bbox']
                    t_names[index_gt_box] = gt_box['name']
                    # remember the +1 so 0 is always empty object
                    t_simmc2_obj_indexes[index_gt_box] = gt_box['simmc2_obj_index'] + 1

                    # changed to do max over the whole thing due to using GT bboxes and output being in diff order
                    max_iou = (0, None)
                    for index_pred_box, pred_box in enumerate(f_boxes):
                        if index_pred_box in pred_index_used:
                            # already used!
                            continue
                        iou = calc_iou_individual(pred_box, gt_box['bbox'])
                        if iou > max_iou[0]:
                            max_iou = iou, index_pred_box

                    # print(max_iou)
                    index_pred_box = max_iou[1]
                    # if iou > iou_index and names[index_gt_box] == 'unk':
                    # print(f_boxes.shape)
                    # print(f_boxes[index_pred_box].shape)
                    # print(new_f_boxes.shape)
                    # print(new_f_boxes[index_pred_box].shape)
                    new_f_boxes[index_gt_box] = f_boxes[index_pred_box]     # pred_box  # gt_box['bbox']
                    new_f_features[index_gt_box] = f_features[index_pred_box]
                    names[index_gt_box] = get_category_name(category_mode, classes[index_pred_box]) # gt_box['name']
                    pred_class_scores[index_gt_box] = class_scores[index_pred_box]
                    # print(f"Pred: {names[igb]} vs GT: {gt_box['name']}")
                    # count for evaluation
                    if names[index_gt_box] == gt_box['name']:
                        eval_metrics['category_accuracy']['n_correct_objects'] += 1
                    eval_metrics['bbox_accuracy']['n_correct_objects'] += 1
                    pred_index_used.append(index_pred_box)

                    # max_iou has the max iou and index

                object_rec, object_prec, object_f1 = rec_prec_f1(
                    n_correct=eval_metrics['bbox_accuracy']['n_correct_objects'],
                    n_true=eval_metrics['bbox_accuracy']['n_true_objects'],
                    n_pred=eval_metrics['bbox_accuracy']['n_pred_objects'],
                )

                bb_acc[d["image_id"]] = len(gt_idx.keys()) / len(d['annotations'])

                try:
                    names = np.array(names, dtype='<U100')
                    t_names = np.array(t_names, dtype='<U100')

                    tmp_h, tmp_w = im.shape[:2]

                    writer.writerow({
                        "img_id": d['image_id'],
                        "img_h": int(tmp_h),
                        "img_w": int(tmp_w),
                        "num_boxes": len(new_f_boxes),
                        "t_num_boxes": len(t_boxes),
                        "boxes": base64.b64encode(new_f_boxes),  # float64
                        "t_boxes": base64.b64encode(t_boxes),  # float64
                        "features": base64.b64encode(new_f_features),  # float64
                        "category": base64.b64encode(names),  # dtype='<U100'
                        "t_category": base64.b64encode(t_names),  # dtype='<U100'
                        "category_scores": base64.b64encode(pred_class_scores),
                        "t_simmc2_obj_indexes": base64.b64encode(t_simmc2_obj_indexes)  # int
                        # "box_order": base64.b64encode(ids_order)  # float64
                    })

                except Exception as e:
                    type, value, traceback = sys.exc_info()
                    print(value)
                    print(type)
                    print(traceback)
                    print(e)
                    break

        eval_results[split] = {
            'total_entries': eval_metrics['category_accuracy']['n_total_objects'],
            'category_accuracy_mean':
                eval_metrics['category_accuracy']['n_correct_objects'] /
                eval_metrics['category_accuracy']['n_total_objects'],
            'object_recall_mean': object_rec,
            'object_precision_mean': object_prec,
            'object_f1_mean': object_f1,
        }

        print(f"[{split}] Results: {json.dumps(eval_results[split], indent=4, default=str)}")
        print(f"Saved at '{tsv_path}'")

    print(f"Results: {json.dumps(eval_results, indent=4, default=str)}")
    print(f"Feature files saved in folder '{output_folder_path}'")


def visualise_model_outputs(cfg, model, category_mode, img_separator_width=30):
    simmc2_metadata = MetadataCatalog.get(f"simmc2_train_{category_mode}")
    predictor = CustomPredictor(cfg, model)

    FOLDER_IMAGE_OUTPUT = f"{cfg.OUTPUT_DIR}/images"
    shutil.rmtree(FOLDER_IMAGE_OUTPUT, ignore_errors=True)
    os.makedirs(FOLDER_IMAGE_OUTPUT, exist_ok=True)

    dataset = dataset_dicts['train']
    # filter here
    dataset = [x for x in dataset if 'cloth_store_paul_5_2' in x['file_name']]

    for d in random.sample(dataset, 10 if len(dataset) > 10 else len(dataset)):
        im = cv2.imread(d["file_name"])
        _instances = predictor.predict_with_bboxes(im, d['annotations'])
        # format at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        # print(outputs)

        pred_v = Visualizer(
            im[:, :, ::-1],
            metadata=simmc2_metadata,  # MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            scale=1,
           # instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        ground_truth_v = Visualizer(im[:, :, ::-1], metadata=simmc2_metadata, scale=1)
        predicted = pred_v.draw_instance_predictions(_instances)
        ground_truth = ground_truth_v.draw_dataset_dict(d)

        concat_img = np.concatenate((
            ground_truth.get_image()[:, :, ::-1],
            # add a black stripe to separate images
            ground_truth.get_image()[:, :img_separator_width, ::-1],
            predicted.get_image()[:, :, ::-1]), axis=1)

        # make a black line to separate images
        concat_img[
        :, ground_truth.get_image().shape[1]:ground_truth.get_image().shape[1] + img_separator_width,
        :] = 0

        # out = v.overlay_instances(boxes=outputs["instances"].pred_boxes.to("cpu"))
        # cv2.imshow('<- ground truth   |     predicted    ->', concat_img)
        image_path = os.path.join(FOLDER_IMAGE_OUTPUT, f"output_{os.path.basename(d['file_name'])}")
        cv2.imwrite(image_path, concat_img)
        print(f"Saved image at {image_path}")
        # cv2.waitKey(0) # waits until a key is pressed

    # cv2.destroyAllWindows()  # destroys the window showing image


dataset_dicts = None


def get_split(split):
    return dataset_dicts[split]


def main(args):
    global dataset_dicts

    assert args.category in object_classes, \
        f"Category {args.category} not in object_classes.json: {object_classes.keys()}"

    cfg = get_config(args)

    dataset_dicts = get_simmc2_dicts('../simmc2_data_generated/image_extraction', args.category)

    for split in dataset_dicts.keys():
        DatasetCatalog.register(f"simmc2_{split}_{args.category}", lambda d=split: get_split(d))
        MetadataCatalog.get(f"simmc2_{split}_{args.category}").set(
            thing_classes=list(object_classes[args.category].keys()))
        print(f"Dataset [{split}_{args.category}] loaded, # instances: {len(dataset_dicts[split])}")

    if args.train:
        # I cannot seem to get performance to be the same when trained in multiple GPUs
        # E.g., 1 GPU =~ 32 AP; 2 GPUs =~ 21 AP
        res = launch(
            train_model,
            num_gpus_per_machine=args.num_gpus,
            num_machines=1,
            machine_rank=0,
            dist_url="auto",
            args=(args, cfg),
        )
        # todo: get model
        print(type(res))
        model = None

    args.resume = True
    model = build_feat_model(cfg)
    # this line of code seems to load the model correctly, but not sure why since it is loading
    # the weights in get_config(). Not doing this makes the model output garbage bboxes
    DetectionCheckpointer(
        model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)

    model.is_training = False

    if args.test:
        assert args.resume, 'Missing model to load'
        test_model(args, cfg, model)

    if args.gtsv:
        print('Generating TSV files')
        assert args.resume, 'Missing model to load'
        args.return_feat = True
        generate_tsv_data(cfg, model, args.category, args.data_percentage)

    if args.vis:
        print('Visualising model')
        assert args.resume, 'Missing model to load'
        visualise_model_outputs(cfg, model, args.category)


if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument(
        '--return_feat', action='store_true',
        help='for prediction only to return RoI features')
    parser.add_argument(
        '--train', action='store_true', help='to start training')
    parser.add_argument(
        '--test', action='store_true', help='to start testing')
    parser.add_argument(
        '--gtsv', action='store_true', help='to generate rcnn features in tsv')
    parser.add_argument(
        '--data_percentage', type=int, default=100,
        help='percentage of images to output, 0 (default) is all files, from 1% to 100%')
    parser.add_argument(
        '--category', type=str, help='category to use for training, could be colour, types, etc.', default='types')
    parser.add_argument(
        '--vis', action='store_true', help='visualise 10 random examples')
    parsed_args = parser.parse_args()

    # note that all of these modes are not mutually exclusive (so can train, test and generate)
    assert parsed_args.train or parsed_args.test or parsed_args.gtsv or parsed_args.vis

    main(parsed_args)
