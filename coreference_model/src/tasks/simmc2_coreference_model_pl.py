# coding=utf-8

import sys
import json
import itertools
from typing import List

import numpy as np
import transformers
import torch.nn as nn
import torch
import pytorch_lightning as pl
import allennlp.nn.util
import Levenshtein
from tqdm import tqdm

sys.path.append('..')
from simmc2.model.mm_dst.utils.evaluate_dst import evaluate_from_flat_list

from param import args
from lxrt.entry_custom import LXRTEncoder, convert_sents_to_features
from lxrt.modeling_custom import BertLayerNorm, GeLU
from simmc2_coreference_data import decode_batch_strings, get_values_above_threshold_mask, get_object_indexes_from_array

# transformers             4.5.1 -> 4.10 if error comes up


class SIMMC2CoreferenceModelWithDescriptions(pl.LightningModule):

    def __init__(
            self, name, max_seq_length, *, f=None, final_layer='linear', ablation=None,
            batches_per_epoch=-1, lr=0.0001):
        super().__init__()
        self.name = name
        self.lr = lr
        self.batches_per_epoch = batches_per_epoch

        if self.batches_per_epoch > 0:
            # this means we are training, to avoid saving parameters at test time
            self.save_hyperparameters(ignore='batches_per_epoch')

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=max_seq_length,
            ablation=ablation
        )
        self.hid_dim = self.lxrt_encoder.dim

        if final_layer == 'linear':
            self.logit_fc = nn.Linear(self.hid_dim, 1)
        elif final_layer == 'sequential':
            self.logit_fc = nn.Sequential(
                nn.Linear(self.hid_dim, self.hid_dim * 2),
                GeLU(),
                BertLayerNorm(self.hid_dim * 2, eps=1e-12),
                nn.Linear(self.hid_dim * 2, 1)
            )
        else:
            raise ValueError(f"final_layer cannot be '{final_layer}'")

        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

        # self.logger
        # self.log('logger_version', self.logger.version, logger=False)

    def forward(self, *, visual_features, visual_feature_mask, boxes, object_counts, sentences):
        """
L
        :param visual_features: (batch_size, num_obj, 2048) - float
        :param visual_feature_mask: (batch_size, num_obj) - int
        :param boxes:  (batch_size, num_obj, 4) - float
        :param object_counts:  (batch_size, num_obj) - int
        :param sentences: (batch_size, num_obj, max_seq_length) - List[str]
        :return: (batch_size, num_obj)
        """
        x_output = self.lxrt_encoder(
            sentences, (visual_features, boxes, None, object_counts),
            visual_attention_mask=visual_feature_mask)

        logit = self.logit_fc(x_output)
        # print(logit.shape)
        logit = torch.flatten(logit, 1, 2)
        # print(logit.shape)

        return logit

    @staticmethod
    def _compose_output(logits, loss, batch):
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2608
        # logits = torch.sigmoid(logits.detach())
        # print(f"logits={logits}")

        # logits is (batch_size, num_objects)
        output = {
            'loss': loss,  # required
            'logits': logits.detach(),
            **batch,
        }

        # move from GPU to save space
        for k in output.keys():
            if k == 'loss':
                continue
            try:
                if output[k].device:
                    output[k] = output[k].cpu()
                else:
                    print(k)
            except AttributeError:
                pass

        return output

    def _prepare_forward_input(self, batch) -> dict:
        return {
            'visual_features': batch['features'].cuda(),
            'visual_feature_mask': batch['feat_mask'].cuda(),
            'boxes': batch['boxes'].cuda(),
            'object_counts': batch['object_counts'].cuda(),
            'sentences': self._prepare_sentence_data(batch),
        }

    @staticmethod
    def _prepare_sentence_data(batch):
        # do sentence tokenization only once to speed up training
        if 'processed_sentences' not in batch:
            # this is now done in __getitem__
            raise ValueError('Sentences are not processed, key missing in batch!')

        return tuple(batch['processed_sentences'])

    def _forward_step(self, batch):
        logits = self(**self._prepare_forward_input(batch))

        if 'target' in batch:
            # only calculate loss if target is in batch
            loss = self.bce_loss(logits, batch['target'])
            loss = allennlp.nn.util.masked_mean(loss, batch['feat_mask'].bool(), -1).mean()
        else:
            loss = torch.zeros(logits.size())

        return torch.sigmoid(logits), loss

    def training_step(self, batch, batch_idx):
        logits, loss = self._forward_step(batch)

        self.log('loss', loss, prog_bar=False, logger=False)

        # returning a dict at training causes too much memory use
        # calculate scores and average it at the end of the epoch instead
        # return self._compose_output(logits, loss, batch)
        training_outputs = self._compose_output(logits, loss, batch)
        training_outputs = self._aggregate_step_outputs([training_outputs])
        results = evaluate_model_outputs(training_outputs)

        best = results['best']
        self.log('train/loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train/object_f1', best['object_f1'], on_step=False, on_epoch=True, prog_bar=True)
        # self.log('train/best_cap', best['method_cap'], prog_bar=False, logger=False)
        self.log('train/object_similarity', best['object_similarity'], on_step=False, on_epoch=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        logits, loss = self._forward_step(batch)

        self.log('val/loss', loss, prog_bar=True, logger=True)

        return self._compose_output(logits, loss, batch)

    def test_step(self, batch, batch_idx):
        logits, loss = self._forward_step(batch)

        return self._compose_output(logits, loss, batch)

    @staticmethod
    def _aggregate_step_outputs(step_outputs) -> dict:
        # this function is called on a per-GPU basis and divided into batches, so aggregate results
        outputs = {}
        for key in step_outputs[0].keys():
            # print(key)
            tmp_elem = step_outputs[0][key]
            # print(tmp_elem)
            # print(validation_step_outputs[0][key].shape)
            if isinstance(tmp_elem, torch.Tensor):
                if tmp_elem.shape != torch.Size([]):
                    outputs[key] = torch.cat([x[key].cpu() for x in step_outputs])
                else:
                    # this is loss, ignore since it's only of size num_batches and not batch_size
                    outputs[key] = torch.cat([x[key].unsqueeze(0) for x in step_outputs])

                # print(validation_outputs[key].shape)
            else:
                outputs[key] = list(itertools.chain.from_iterable([x[key] for x in step_outputs]))
            # print(len(validation_outputs[key]))
            # if key == 'target':
            #     exit()

        # outputs is now a dict of aggregated results
        return outputs

    # def training_epoch_end(self, training_step_outputs):
    #     training_outputs = self._aggregate_step_outputs(training_step_outputs)
    #     results = evaluate_model_outputs(training_outputs)
    #
    #     best = results['best']
    #     self.log('train/object_f1', best['object_f1'])
    #     self.log('train/best_cap', best['method_cap'], prog_bar=False, logger=False)
    #     self.log('train/object_similarity', best['object_similarity'])

    def validation_epoch_end(self, validation_step_outputs):
        validation_outputs = self._aggregate_step_outputs(validation_step_outputs)
        results = evaluate_model_outputs(validation_outputs)

        best = results['best']
        self.log('val/object_f1', best['object_f1'], prog_bar=True, logger=True)
        self.log('val/best_cap', best['method_cap'], prog_bar=True, logger=False)
        self.log('val/object_similarity', best['object_similarity'], prog_bar=True, logger=True)

    def test_epoch_end(self, test_step_outputs):
        test_outputs = self._aggregate_step_outputs(test_step_outputs)
        results = evaluate_model_outputs(test_outputs)

        print(json.dumps(results['best'], indent=4, default=str))
        self.log('test/loss', results['mean_loss'], logger=False)
        self.log('test/object_f1', results['best']['object_f1'], logger=False)
        self.log('test/object_f1_std', results['best']['object_f1_stderr'], logger=False)
        self.log('test/object_rec', results['best']['object_rec'], logger=False)
        self.log('test/object_prec', results['best']['object_prec'], logger=False)
        self.log('test/best_threshold', results['best']['method_cap'], logger=False)
        self.log('test/object_similarity', results['best']['object_similarity'], logger=False)

        # the return of trainer.test() is whatever you log above, not this result below
        return results

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # this is the same as test so we don't care much
        return self.test_step(batch, batch_idx)

    def post_process_predictions(
            self, prediction_step_outputs, output_path, pred_threshold=0.35, extra=None):
        prediction_outputs = self._aggregate_step_outputs(prediction_step_outputs)
        results = evaluate_model_outputs(
            prediction_outputs, include_all_info=True, pred_threshold=pred_threshold)

        generate_prediction_file(output_path, results, self.name, extra)

        print(json.dumps(results['best'], indent=4, default=str))

        return results

    def configure_optimizers(self):
        from lxrt.optimization import BertAdam

        optimizer = BertAdam(
            list(self.parameters()), lr=args.lr, warmup=0.1,
            t_total=self.batches_per_epoch * args.epochs if self.batches_per_epoch > 0 else -1)
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def evaluate_model_outputs(
        output: dict, return_examples=False, include_all_info=False, pred_threshold=None):
    error_examples = []
    final_answers = {}

    thresholds = np.arange(0, 1, step=0.05)
    if pred_threshold is not None:
        # this means we are predicting without a target, so go only for one threshold
        thresholds = [pred_threshold]

    output['descriptions'] = np.array(decode_batch_strings(output['descriptions']))
    output['t_descriptions'] = np.array(decode_batch_strings(output['t_descriptions']))
    output['sentence'] = np.array(decode_batch_strings(output['sentence']))

    for cap in tqdm(thresholds, 'Checking thresholds', disable=True):
        final_answers[f"threshold_{cap}"] = []

        for i, logits in enumerate(output['logits']):

            true_mask = get_values_above_threshold_mask(output['target'][i]) if 'target' in output else None
            pred_mask = get_values_above_threshold_mask(logits, cap)

            true_objects = output['t_simmc2_obj_indexes'][i][true_mask].tolist() if true_mask else None
            pred_objects = output['t_simmc2_obj_indexes'][i][pred_mask].tolist()

            # assert true_objects == true_objects2, f"{true_objects} == {true_objects2}"

            # note these true descriptions are doubly true: from detectron and from true objects
            # change true_descriptions to use 'descriptions' otherwise
            true_descriptions = output['t_descriptions'][i][true_mask].tolist() if true_mask else None
            pred_descriptions = output['descriptions'][i][pred_mask].tolist()

            frame = {
                'dialogue_idx': output['dialogue_idx'][i],
                'turn_idx': output['turn_idx'][i],
                'true_objects': true_objects,
                'pred_objects': pred_objects,
                'true_descriptions': true_descriptions,
                'pred_descriptions': pred_descriptions,
            }
            if include_all_info:
                # include information like object indexes, descriptions, etc
                frame['object_indexes'] = output['t_simmc2_obj_indexes'][i]
                frame['descriptions'] = output['descriptions'][i]
                frame['logits'] = output['logits'][i]
                frame['sentence'] = output['sentence'][i]

            final_answers[f"threshold_{cap}"].append(frame)

    result = {
        'mean_loss': output['loss'].mean().item(),
    }
    best_object_f1 = {
        'object_f1': -1,
    }
    if pred_threshold is not None:
        method = f"threshold_{pred_threshold}"
        best_object_f1 = evaluate_method_frames(final_answers[method])
        best_object_f1['method'] = method
        best_object_f1['method_cap'] = pred_threshold
    else:
        # target is available, so evaluate
        for method in tqdm(final_answers.keys(), 'Evaluating frames', disable=True):
            result[method] = evaluate_method_frames(final_answers[method])

            if result[method]['object_f1'] > best_object_f1['object_f1']:
                best_object_f1 = result[method]
                best_object_f1['method'] = method
                best_object_f1['method_cap'] = float(method.split('_')[1])

    result = {**result, 'best': best_object_f1}

    # print(json.dumps(best_object_f1, indent=4))
    # print(json.dumps(result, indent=4))
    if include_all_info:
        result['answers'] = final_answers

    # else:
    return result


def _reformat_frame_turn(frame_objects: list):
    frame = {
        'act': [],
        'slots': [],
        'request_slots': [],
        'objects': frame_objects,
    }
    return [frame]


def evaluate_method_frames(frames: List[dict]) -> dict:
    if frames[0]['true_objects'] is None:
        # no target, we are doing prediction
        return {}

    d_true_flattened = []
    d_pred_flattened = []
    object_similarity = 0.0

    # flatten turns
    for frame in frames:
        turn_true = _reformat_frame_turn(frame['true_objects'])
        turn_pred = _reformat_frame_turn(frame['pred_objects'])

        object_similarity += letter_wise_levenshtein_object_similarity(
            frame['true_descriptions'], frame['pred_descriptions'])

        d_true_flattened.append(turn_true)
        d_pred_flattened.append(turn_pred)

    eval_result = evaluate_from_flat_list(d_true_flattened, d_pred_flattened)
    # remove everything that doesn't have to do with object f1
    for key in list(eval_result.keys()):        # list avoids error when deleting
        if 'object' not in key:
            del eval_result[key]

    # mean of Levenshtein naive object similarity
    eval_result['object_similarity'] = object_similarity / len(frames)

    return eval_result


def letter_wise_levenshtein_object_similarity(true_descriptions, pred_descriptions) -> float:
    # it is difficult to evaluate this when doing multi-label due to differences in output
    # so, split string into letters, order them alphabetically and then join to get a final str
    # print(true_descriptions)
    # print(pred_descriptions)
    true_letters = list(itertools.chain.from_iterable([list(x) for x in true_descriptions]))
    pred_letters = list(itertools.chain.from_iterable([list(x) for x in pred_descriptions]))
    # print('ratio:')
    # print(true_letters)
    # print('--')
    # print(pred_letters)

    true_letters = sorted(true_letters)
    pred_letters = sorted(pred_letters)

    # print(pred_letters)

    true_final = ''.join(true_letters)
    pred_final = ''.join(pred_letters)
    # print(true_final)
    # print(pred_final)

    similarity = Levenshtein.ratio(true_final, pred_final)
    # print(similarity)
    # exit()
    return similarity


def generate_prediction_file(output_path, predictions, model_name, extra=None):
    best_threshold = predictions['best']['method']
    dialogue_results = {}

    # format data as the evaluation script uses
    for frame in predictions['answers'][best_threshold]:
        if frame['dialogue_idx'] not in dialogue_results:
            dialogue_results[frame['dialogue_idx']] = {
                'dialog_id': int(frame['dialogue_idx']),
                'dialogue_idx': int(frame['dialogue_idx']),
                'dialogue': []
            }

        dialogue_results[frame['dialogue_idx']]['dialogue'].append({
            'turn_idx': int(frame['turn_idx']),
            'transcript_annotated': {
                'act': None,
                'act_attributes': {
                    'slot_values': {},
                    'request_slots': {},
                    # remember that objects have +1 in index to avoid clashes with 0
                    'objects': [x - 1 for x in frame['pred_objects']]
                }
            },
            # duplicated, but easier to read
            'pred_objects': [x - 1 for x in frame['pred_objects']],
            'true_objects': [x - 1 for x in frame['true_objects']] \
            if frame['true_objects'] is not None else None,
            'input_sequence': '[SEP]'.join(frame['sentence'][0].split('[SEP]')[:-1]),
            'prediction_outputs': {
                index - 1: [index - 1, description, score]
                for index, description, score in zip(
                    frame['object_indexes'].tolist(), frame['descriptions'],
                    frame['logits'].tolist()) if index - 1 >= 0
            }
        })
        # print(dialogue_results[frame['dialogue_idx']]['dialogue'][-1])
    # end of formatting data loop

    export_data = {
        'model_name': model_name,
        **predictions['best'],
        **extra,
        'dialogue_data': list(dialogue_results.values()),
    }
    with open(output_path, 'w') as out_file:
        json.dump(export_data, out_file, indent=4, default=str)
    print(f"Predictions file saved at '{output_path}'")
