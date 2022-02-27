import json

from tqdm import tqdm
import torch.nn as nn
import logging
import ast
import glob
import numpy as np
import copy

# utils
from utils.config import *
from utils.utils_general import *
from utils.utils_multiwoz import *
from utils.utils_oos_intent import *
from utils.utils_universal_act import *
from utils.utils_simmc2_mm_disambiguation import *

# models
from models.multi_label_classifier import *
from models.multi_class_classifier import *
from models.BERT_DST_Picklist import *
from models.dual_encoder_ranking import *

# hugging face models
from transformers import *

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

## model selection
MODELS = {"bert": (BertModel,       BertTokenizer,       BertConfig),
          "todbert": (BertModel,       BertTokenizer,       BertConfig),
          "gpt2": (GPT2Model,       GPT2Tokenizer,       GPT2Config),
          "todgpt2": (GPT2Model,       GPT2Tokenizer,       GPT2Config),
          "dialogpt": (AutoModelWithLMHead, AutoTokenizer, GPT2Config),
          "albert": (AlbertModel, AlbertTokenizer, AlbertConfig),
          "roberta": (RobertaModel, RobertaTokenizer, RobertaConfig),
          "distilbert": (DistilBertModel, DistilBertTokenizer, DistilBertConfig),
          "electra": (ElectraModel, ElectraTokenizer, ElectraConfig)}

## Fix torch random seed
if args["fix_rand_seed"]:
    torch.manual_seed(args["rand_seed"])


## Reading data and create data loaders
datasets = {}
for ds_name in ast.literal_eval(args["dataset"]):
    data_trn, data_dev, data_tst, data_meta = globals()["prepare_data_{}".format(ds_name)](args)
    datasets[ds_name] = {"train": data_trn, "dev":data_dev, "test": data_tst, "meta":data_meta}
unified_meta = get_unified_meta(datasets)
if "resp_cand_trn" not in unified_meta.keys(): unified_meta["resp_cand_trn"] = {}
args["unified_meta"] = unified_meta


## Create vocab and model class
args["model_type"] = args["model_type"].lower()
model_class, tokenizer_class, config_class = MODELS[args["model_type"]]
tokenizer = tokenizer_class.from_pretrained(args["model_name_or_path"], cache_dir=args["cache_dir"])
args["model_class"] = model_class
args["tokenizer"] = tokenizer
if args["model_name_or_path"]:
    config = config_class.from_pretrained(args["model_name_or_path"], cache_dir=args["cache_dir"])
else:
    config = config_class()
args["config"] = config
args["num_labels"] = unified_meta["num_labels"]

best_epoch = -1

# JChiyah: better function to evaluate results
def evaluate_simmc2_model(_model, _output_dir_origin) -> dict:
    print(f"[Info] Start Evaluation on dev and test set (best epoch: {best_epoch})...")
    trn_loader = get_loader(args, "train", tokenizer, datasets, unified_meta)
    dev_loader = get_loader(args, "dev"  , tokenizer, datasets, unified_meta)
    tst_loader = get_loader(args, "test" , tokenizer, datasets, unified_meta, shuffle=args["task_name"]=="rs")
    _model.eval()

    evaluation_results = {
        '_best_epoch': best_epoch,
        '_output_dir_origin': _output_dir_origin
    }
    # for d_eval in ["tst"]: #["dev", "tst"]: commented out by JChiyah
    for d_eval in ["trn", "dev", "tst"]:

        ## Start evaluating on the test set
        test_loss = 0
        preds, labels = [], []
        pbar = tqdm(locals()["{}_loader".format(d_eval)])
        for d in pbar:
            with torch.no_grad():
                outputs = _model(d)
            test_loss += outputs["loss"]
            preds += [item for item in outputs["pred"]]
            labels += [item for item in outputs["label"]]

        test_loss = test_loss / len(tst_loader)
        tmp_results = _model.evaluation(preds, labels)
        # JChiyah, also show loss
        tmp_results["loss"] = test_loss
        for key, value in tmp_results.items():
            evaluation_results["{}_{}".format(d_eval, key)] = value

    with open(os.path.join(_output_dir_origin, "{}_results.txt".format(d_eval)), "w") as f_w:
        f_w.write(json.dumps(evaluation_results, indent=4))

    keys_to_print = ['do_train', 'model_name_or_path', 'batch_size', 'epoch', 'nb_runs',
                     'train_data_ratio', 'simmc2_input_features', 'simmc2_max_turns']
    info = {k: args[k] for k in keys_to_print}

    print(f"Training Info: {json.dumps(info, indent=4)}\nResults: {json.dumps(evaluation_results, indent=4)}")

    return evaluation_results


## Training and Testing Loop
if args["do_train"]:
    result_runs = []
    output_dir_origin = str(args["output_dir"])

    tmp_save_file = output_dir_origin + "_tmp_eval_results_multi-runs.json"

    if args["simmc2_load_tmp_results"]:
        if os.path.exists(tmp_save_file):
            print(f"Loading tmp results from {tmp_save_file}")
            with open(tmp_save_file, "r") as f_in:
                result_runs = json.load(f_in)
            for i_index in range(len(result_runs)):
                seed_to_remove = result_runs[i_index]["_seed"]
                for j_index in range(SEEDS.shape[0]):
                    if int(SEEDS[j_index]) == seed_to_remove:
                        SEEDS = np.delete(SEEDS, j_index)
                        args["nb_runs"] -= 1
                        break
                else:
                    print(f"Seed not found in {tmp_save_file}: {seed_to_remove} ({SEEDS=})")
                    exit(1)

            # SEEDS = SEEDS[len(result_runs):]
            # args["nb_runs"] -= len(result_runs)
            print(f"Loaded {len(result_runs)} runs from tmp file at {tmp_save_file}")

            print(f"Next seed is {int(SEEDS[0])} (last loaded was {result_runs[-1]['_seed']})")

            if args["nb_runs"] > SEEDS.shape[0]:
                raise ValueError(f"Not enough seeds for {len(args['nb_runs'])} runs ({SEEDS.shape=})")
        else:
            print(f"Nothing to load from {tmp_save_file}")

    ## Setup logger
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(args["output_dir"], "train.log"),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    ## training loop
    for run in range(args["nb_runs"]):

        ## Setup random seed and output dir
        rand_seed = SEEDS[run]
        if args["fix_rand_seed"]:
            torch.manual_seed(rand_seed)
            args["rand_seed"] = rand_seed
        args["output_dir"] = os.path.join(output_dir_origin, "run{}".format(run))
        os.makedirs(args["output_dir"], exist_ok=False)
        logging.info("Running Random Seed: {}".format(rand_seed))

        ## Loading model
        model = globals()[args['my_model']](args)
        if torch.cuda.is_available(): model = model.cuda()

        ## Create Dataloader
        trn_loader = get_loader(args, "train", tokenizer, datasets, unified_meta)
        dev_loader = get_loader(args, "dev"  , tokenizer, datasets, unified_meta, shuffle=args["task_name"]=="rs")
        tst_loader = get_loader(args, "test" , tokenizer, datasets, unified_meta, shuffle=args["task_name"]=="rs")

        ## Create TF Writer
        tb_writer = SummaryWriter(comment=args["output_dir"].replace("/", "-"))

        # Start training process with early stopping
        loss_best, acc_best, cnt, train_step = 1e10, -1, 0, 0

        try:
            for epoch in range(args["epoch"]):
                logging.info("Epoch:{}".format(epoch+1))
                train_loss = 0
                pbar = tqdm(trn_loader)
                for i, d in enumerate(pbar):
                    model.train()
                    outputs = model(d)
                    train_loss += outputs["loss"]
                    train_step += 1
                    pbar.set_description("Training Loss: {:.4f}".format(train_loss/(i+1)))

                    ## Dev Evaluation
                    if (train_step % args["eval_by_step"] == 0 and args["eval_by_step"] != -1) or \
                                                  (i == len(pbar)-1 and args["eval_by_step"] == -1):
                        model.eval()
                        dev_loss = 0
                        preds, labels = [], []
                        ppbar = tqdm(dev_loader)
                        for d in ppbar:
                            with torch.no_grad():
                                outputs = model(d)
                            #print(outputs)
                            dev_loss += outputs["loss"]
                            preds += [item for item in outputs["pred"]]
                            labels += [item for item in outputs["label"]]

                        dev_loss = dev_loss / len(dev_loader)
                        results = model.evaluation(preds, labels)
                        dev_acc = results[args["earlystop"]] if args["earlystop"] != "loss" else dev_loss

                        ## write to tensorboard
                        tb_writer.add_scalar("train_loss", train_loss/(i+1), train_step)
                        tb_writer.add_scalar("eval_loss", dev_loss, train_step)
                        tb_writer.add_scalar("eval_{}".format(args["earlystop"]), dev_acc, train_step)

                        if (dev_loss < loss_best and args["earlystop"] == "loss") or \
                            (dev_acc > acc_best and args["earlystop"] != "loss"):
                            loss_best = dev_loss
                            acc_best = dev_acc
                            best_epoch = epoch + (train_step / (args['eval_by_step'] * 4))
                            cnt = 0 # reset

                            if args["not_save_model"]:
                                model_clone = globals()[args['my_model']](args)
                                model_clone.load_state_dict(copy.deepcopy(model.state_dict()))
                            else:
                                output_model_file = os.path.join(args["output_dir"], "pytorch_model.bin")
                                if args["n_gpu"] == 1:
                                    torch.save(model.state_dict(), output_model_file)
                                else:
                                    torch.save(model.module.state_dict(), output_model_file)
                                logging.info("[Info] Model saved at epoch {} step {}".format(epoch, train_step))
                        else:
                            cnt += 1
                            logging.info("[Info] Early stop count: {}/{}...".format(cnt, args["patience"]))

                        if cnt > args["patience"]:
                            logging.info("Ran out of patient, early stop...")
                            break

                        logging.info("Trn loss {:.4f}, Dev loss {:.4f}, Dev {} {:.4f}".format(train_loss/(i+1),
                                                                                              dev_loss,
                                                                                              args["earlystop"],
                                                                                              dev_acc))

                if cnt > args["patience"]:
                    tb_writer.close()
                    break

        except KeyboardInterrupt:
            logging.info("[Warning] Earlystop by KeyboardInterrupt")

        ## Load the best model
        if args["not_save_model"]:
            model.load_state_dict(copy.deepcopy(model_clone.state_dict()))
        else:
            # Start evaluating on the test set
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(output_model_file))
            else:
                model.load_state_dict(torch.load(output_model_file, lambda storage, loc: storage))

        ## Run test set evaluation
        pbar = tqdm(tst_loader)
        for nb_eval in range(args["nb_evals"]):
            test_loss = 0
            preds, labels = [], []
            for d in pbar:
                with torch.no_grad():
                    try:
                        outputs = model(d)
                    except Exception as ex:
                        # special code to run everything again
                        print(f"There was a problem testing the model: {ex}")
                        exit(2)
                test_loss += outputs["loss"]
                preds += [item for item in outputs["pred"]]
                labels += [item for item in outputs["label"]]

            test_loss = test_loss / len(tst_loader)
            # results = model.evaluation(preds, labels)
            results = evaluate_simmc2_model(model, output_dir_origin)
            results["_seed"] = int(rand_seed)
            results["_total_epochs"] = args["epoch"]
            result_runs.append(results)
            logging.info("[{}] Test Results: ".format(nb_eval) + str(results))

        # save results temporarily
        with open(tmp_save_file, "w") as f_out:
            json.dump(result_runs, f_out, indent=4)
        print(f"Temporal results saved in {tmp_save_file}\n")

    ## Average results over runs
    if len(result_runs) > 1:
        average_results = {
            'description': "Average over {} runs and {} evals ({} epochs each)".format(
                len(result_runs), args["nb_evals"], args["epoch"]),
            'results_per_run': result_runs,
            'args': args
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
            print(f"   {key[:3]} {key[4:]} mean: {mean} (SD: {std})")

        save_file = output_dir_origin + "-eval_results_multi-runs.json"
        with open(save_file, "w") as f_out:
            json.dump(average_results, f_out, indent=4, default=str)
            # print(json.dumps(average_results, indent=4))
            # f_out.close()
        print(f"Results saved in {save_file}\n")
        if os.path.exists(tmp_save_file):
            os.remove(tmp_save_file)

else:
    ## Load Model
    print("[Info] Loading model from {}".format(args['my_model']))
    model = globals()[args['my_model']](args)
    if args["load_path"]:
        print("MODEL {} LOADED".format(args["load_path"]))
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(args["load_path"]))
        else:
            model.load_state_dict(torch.load(args["load_path"], lambda storage, loc: storage))
    else:
        print("[WARNING] No trained model is loaded...")

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    output_dir_origin = args["output_dir"]

    # add test to available datasets
    data_trn, data_dev, data_devtst, data_meta = prepare_data_simmc2_mm_disambiguation(args)
    data_tst, _ = prepare_data_simmc2_mm_disambiguation_test(args)
    eval_datasets = {}
    eval_datasets[args["dataset"]] = {
        "train": data_trn, "dev": data_dev, "devtest": data_devtst, "teststd_public": data_tst, "meta": data_meta
    }

    for d_eval in ["train", "dev", "devtest", "teststd_public"]:
        print(f"[Info] Start Evaluation on {d_eval}")
        dialogue_results = {}
        loader = get_loader(args, d_eval, tokenizer, eval_datasets, unified_meta, shuffle=False)
        f_w = open(os.path.join(args["output_dir"], "{}_results.txt".format(d_eval)), "w")

        test_loss = 0
        preds, labels = [], []
        pbar = tqdm(loader)
        for d in pbar:
            with torch.no_grad():
                outputs = model(d)
            test_loss += outputs["loss"]
            preds += [item for item in outputs["pred"]]
            labels += [item for item in outputs["label"]]

            # for idx, pred_label, gt_label, plain in zip(d['ID'], outputs["pred"], outputs["label"], d["intent_plain"]):
            for idx, pred_label in zip(d['ID'], outputs["pred"]):
                tmp = idx.split('-')
                dialogue_idx, turn_idx = tmp[-1].split('+')

                if dialogue_idx not in dialogue_results:
                    dialogue_results[dialogue_idx] = {
                        "dialog_id": int(dialogue_idx),
                        # "dialogue_idx": int(dialogue_idx),
                        "predictions": []
                    }

                # WARNING labels are switched, so 0 == disambiguation and 1 == no_disambiguation
                # print(f"{plain} == {gt_label}")
                dialogue_results[dialogue_idx]["predictions"].append({
                    "turn_id": int(turn_idx),
                    # "turn_idx": int(turn_idx),
                    "disambiguation_label": 1 if int(pred_label) == 0 else 0,
                    # "gt_disambiguation_label": 1 if not int(gt_label) else 0,
                })

        test_loss = test_loss / len(loader)
        results = model.evaluation(preds, labels)

        # JChiyah, also show loss
        results["loss"] = test_loss
        print("{} Results: {}".format(d_eval, str(results)))
        f_w.write(str(results))
        f_w.close()

        json_file = "dstc10-simmc-{}-pred-subtask-1.json".format(d_eval)
        with open(json_file, "w") as out_file:
            json.dump(list(dialogue_results.values()), out_file, indent=4)

        print(f"[Info] Evaluation on {d_eval} finished, file saved at {json_file}")

    print()

    logging.info(f"{args['output_dir']=}\n{args['simmc2_input_features']=}")


if args["nb_runs"] == 1 and len(result_runs) == 1:
    evaluate_simmc2_model(model, output_dir_origin)
