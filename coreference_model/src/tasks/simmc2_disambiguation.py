# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import json
import random
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.simmc2_disambiguation_model import SIMMC2DisambiguationModel
from tasks.simmc2_disambiguation_data import SIMMC2DisambiguationDataset, SIMMC2DisambiguationTorchDataset, SIMMC2DisambiguationEvaluator

SEEDS = np.arange(0, 100, 5)

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_data_tuple(splits: str, bs: int, shuffle=False, drop_last=False) -> DataTuple:
    dset = SIMMC2DisambiguationDataset(splits)
    tset = SIMMC2DisambiguationTorchDataset(dset)
    evaluator = SIMMC2DisambiguationEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class SIMMC2Disambiguation:
    # copy of VQA, with stuff from NLVR2

    def __init__(self, random_seed):
        self.seed = random_seed
        # Set seeds
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "" and args.test:
            self.valid_tuple = get_data_tuple(
                args.valid, bs=1024,
                shuffle=False, drop_last=False
            )
            # self.test_tuple = get_data_tuple(
            #     args.test, bs=128,
            #     shuffle=False, drop_last=False
            # )
        else:
            raise NotImplementedError
            # self.valid_tuple = None

        # Model
        self.model = SIMMC2DisambiguationModel()

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            raise NotImplementedError
            # load_lxmert_qa(args.load_lxmert_qa, self.model,
            #                label2ans=self.train_tuple.dataset.label2ans)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        # self.bce_loss = nn.BCEWithLogitsLoss()
        # use cross entropy loss, from NLVR2
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam

            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)

        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)
        self.best_epoch = 0.

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent)
                # print("batch_size: {}, logits: {}, target: {}".format(args.batch_size, len(logit), len(target)))
                # exit(1)
                # assert logit.dim() == target.dim() == 2
                # loss = self.bce_loss(logit, target)
                # loss = loss * logit.size(1)
                # use cross entropy loss, from NLVR2
                loss = self.mce_loss(logit, target)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, predict = logit.max(1)
                for qid, l in zip(ques_id, predict.cpu().numpy()):
                    quesid2ans[qid] = l

                # score, label = logit.max(1)
                # for qid, l in zip(ques_id, label.cpu().numpy()):
                #     ans = dset.label2ans[l]
                #     quesid2ans[qid.item()] = ans

                # moved so it evaluates twice per epoch
                if i % (len(loader) / 2) == 0:
                    _exact_epoch = epoch + i / len(loader)
                    log_str = "\nEpoch %0.1f: Train %0.5f\n" % (_exact_epoch, evaluator.evaluate(quesid2ans))

                    # if self.valid_tuple is not None:  # Do Validation
                    valid_score = self.evaluate(eval_tuple)
                    if valid_score > best_valid:
                        best_valid = valid_score
                        self.save("BEST")
                        self.best_epoch = _exact_epoch

                    log_str += "Epoch %0.1f: Valid %0.5f\n" % (_exact_epoch, valid_score) + \
                               "Epoch %0.1f: Best %0.5f at epoch %0.1f\n" % (_exact_epoch, best_valid, self.best_epoch)

                    print(log_str, end='')

                    with open(self.output + "/log.log", 'a') as f:
                        f.write(log_str)
                        f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]  # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                # score, label = logit.max(1)
                # for qid, l in zip(ques_id, label.cpu().numpy()):
                #     ans = dset.label2ans[l]
                #     quesid2ans[qid.item()] = ans
                score, predict = logit.max(1)
                for qid, l in zip(ques_id, predict.cpu().numpy()):
                    quesid2ans[qid] = l
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    # @staticmethod
    # def oracle_score(data_tuple):
    #     dset, loader, evaluator = data_tuple
    #     quesid2ans = {}
    #     for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
    #         _, label = target.max(1)
    #         for qid, l in zip(ques_id, label.cpu().numpy()):
    #             ans = dset.label2ans[l]
    #             quesid2ans[qid.item()] = ans
    #     return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


def evaluate_best_model(_trainer):

    evaluation_results = {
        '_best_epoch': _trainer.best_epoch,
        '_output_dir_origin': _trainer.output,
        '_seed': _trainer.seed,
        '_total_epochs': args.epochs
    }
    # unload model so not to mix anything
    _trainer = None

    trainer = SIMMC2Disambiguation(evaluation_results['_seed'])
    trainer.load(evaluation_results['_output_dir_origin'] + '/BEST')

    for split in ['train', 'dev', 'devtest']:

        # todo: should prob get loss as in the evaluation of ToD-BERT/main.py
        split_eval_accuracy = trainer.evaluate(
            get_data_tuple(split, bs=64, shuffle=False, drop_last=False))

        evaluation_results["{}_accuracy".format(split)] = split_eval_accuracy

    keys_to_print = ['load', 'output', 'batch_size', 'epochs', 'num_runs', 'lr']
    # 'train_data_ratio', 'simmc2_input_features', 'simmc2_max_turns']
    info = {k: getattr(args, k) for k in keys_to_print}

    print(f"Training Info: {json.dumps(info, indent=4, default=str)}\nResults: {json.dumps(evaluation_results, indent=4, default=str)}")

    return evaluation_results


if __name__ == "__main__":

    # Load model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        # Build Class
        simmc2_disambiguation = SIMMC2Disambiguation(SEEDS[0])
        # changed the way this works whenever a model is loaded then it is test only
        raise NotImplementedError
        simmc2_disambiguation.load(args.load)

        # Test
        if args.test is not None:
            args.fast = args.tiny = False  # Always loading all data in test
            if 'devtest' in args.test:
                simmc2_disambiguation.predict(
                    get_data_tuple(args.test, bs=64,
                                   shuffle=False, drop_last=False),
                    dump=os.path.join(args.output, 'devtest_predict.json')
                )
                eval_results = {}
                for split in ['train', 'dev', 'devtest']:
                    split_eval_result = simmc2_disambiguation.evaluate(
                        get_data_tuple(split, bs=64,
                                       shuffle=False, drop_last=False),
                        dump=None  # os.path.join(args.output, 'devtest_predict.json')
                    )
                    eval_results["{}_accuracy".format(split)] = split_eval_result

                print("\nEvaluation Results")
                for key, value in eval_results.items():
                    print("\t{}: {}".format(key, value))

            elif 'dev' in args.test:
                # Since part of valididation data are used in pre-training/fine-tuning,
                # only validate on the minival set.
                result = simmc2_disambiguation.evaluate(
                    get_data_tuple(args.test, bs=64,
                                   shuffle=False, drop_last=False),
                    dump=os.path.join(args.output, 'dev_predict.json')
                )
                print(result)
            else:
                assert False, "No such test option for %s" % args.test

    else:
        output_dir_origin = args.output
        result_runs = []
        for run in range(args.num_runs):
            args.output = output_dir_origin + "/run" + str(run)
            # Build Class
            simmc2_disambiguation = SIMMC2Disambiguation(SEEDS[run])

            print("Training and Evaluating Run {}/{}".format(run, args.num_runs))
            # all train, dev and devtest splits must be true from here

            # print('Splits in Train data:', simmc2_disambiguation.train_tuple.dataset.splits)
            # if simmc2_disambiguation.valid_tuple is not None:
            #     print('Splits in Valid data:', simmc2_disambiguation.valid_tuple.dataset.splits)
            #     # print("Valid Oracle: %0.2f" % (simmc2_disambiguation.oracle_score(vqa.valid_tuple) * 100))
            # else:
            #     print("DO NOT USE VALIDATION")
            #     raise NotImplementedError   # always use validation

            # train
            simmc2_disambiguation.train(simmc2_disambiguation.train_tuple, simmc2_disambiguation.valid_tuple)
            # evaluate
            result_runs.append(evaluate_best_model(simmc2_disambiguation))

        if len(result_runs) > 1:
            average_results = {
                'description': "Average over {} runs and {} evals ({} epochs each)".format(
                    len(result_runs), 1, args.epochs),
                'results_per_run': result_runs,
                'args': vars(args)
            }
            print(f"\nEvaluation Results: {average_results['description']}\n")
            for key in result_runs[0].keys():
                if key.startswith('_'):
                    continue    # ignore _epoch
                mean = np.mean([r[key] for r in result_runs])
                std = np.std([r[key] for r in result_runs])
                average_results[f"{key}_mean"] = mean
                average_results[f"{key}_std"]  = std
                # f_out.write("{}: mean {} std {} \n".format(key, mean, std))
                print(f"\t{key} mean: {mean} (SD: {std})")

            save_file = output_dir_origin + "/eval_results_multi-runs.json"
            with open(save_file, "w") as f_out:
                json.dump(average_results, f_out, indent=4, default=str)
                # print(json.dumps(average_results, indent=4))
                # f_out.close()
            print(f"\nResults saved in {save_file}\n")
