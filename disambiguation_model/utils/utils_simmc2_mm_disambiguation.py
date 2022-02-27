import json
import ast
import os
import random
from .utils_function import get_input_example


def read_langs(args, dtype, _data):
    print(("Reading [SIMMC2 Multimodal Disambiguation] for read_langs {}".format(dtype)))
    features = ast.literal_eval(args["simmc2_input_features"])
    if args["simmc2_max_turns"] <= 0 and "dialogue_history" in features:
        raise ValueError(f"args.simmc_max_turns must be above 0, remove the feature instead")
    # if args["simmc2_max_turns"] == -1:
    #     args["simmc2_max_turns"] = None

    data = []
    intent_counter = {}

    for d in _data['entries']:
        sentence = d['user_utterance']
        dialogue_history = [past_utt['utterance'] for past_utt in d['dialogue_history']]
        label_int = d.get("disambiguation_label", None)
        label = "disambiguation" if label_int == 1 else "no_disambiguation"
        # dialogue_history, sentence,  = d[0], d[1], d[2]

        data_detail = get_input_example("turn")
        # put dialogue and turn id for retrieval later
        data_detail["ID"] = "SIMMC2-MM_DISAMBIGUATION-{}-{}-{}+{}".format(
            dtype, len(data), d['dialogue_idx'], d['turn_idx'])

        if "dialogue_history" in features:
            data_detail["dialog_history"] = dialogue_history[:args["simmc2_max_turns"]]

            if args["simmc2_max_turns"] is not None \
                and len(data_detail["dialog_history"]) > args["simmc2_max_turns"]:
                raise ValueError(
                    f"Length of dialogue_history ({len(data_detail['dialog_history'])}) "
                    f"does not match args.simmc2_max_turns ({args['simmc2_max_turns']})")

        if "user_utterance" in features:
            data_detail["turn_usr"] = sentence

        if label_int is not None:
            data_detail["intent"] = label

            # count number of each label
            if label not in intent_counter.keys():
                intent_counter[label] = 0
            intent_counter[label] += 1

        data.append(data_detail)


    # print("len of OOS Intent counter: ", len(intent_counter))

    return data, intent_counter


def prepare_data_simmc2_mm_disambiguation(args):
    example_type = args["example_type"]
    max_line = args["max_line"]

    data_path = args["data_path"] + 'disambiguation/{}.json'

    pair_trn, intent_counter_trn = read_langs(
        args, "trn", json.load(open(data_path.format("train"), "r"))) # '[:len(data["train"]) // 2])
    pair_dev, intent_counter_dev = read_langs(
        args, "dev", json.load(open(data_path.format("dev"), "r")))
    pair_tst, intent_counter_tst = read_langs(
        args, "tst", json.load(open(data_path.format("devtest"), "r")))

    print(
        f"Read {len(pair_trn)} pairs train from SIMMC2 Multimodal Disambiguation, "
        f"{intent_counter_trn['disambiguation'] / len(pair_trn) * 100:.1f}% disambiguation")
    print(
        f"Read {len(pair_dev)} pairs valid from SIMMC2 Multimodal Disambiguation, "
        f"{intent_counter_dev['disambiguation'] / len(pair_dev) * 100:.1f}% disambiguation")
    print(
        f"Read {len(pair_tst)} pairs test from SIMMC2 Multimodal Disambiguation, "
        f"{intent_counter_tst['disambiguation'] / len(pair_tst) * 100:.1f}% disambiguation")
    intent_class = list(intent_counter_trn.keys())

    meta_data = {"intent": intent_class, "num_labels": len(intent_class)}
    print("len(intent_class)", len(intent_class))

    return pair_trn, pair_dev, pair_tst, meta_data


def prepare_data_simmc2_mm_disambiguation_test(args):
    data_path = args["data_path"] + 'disambiguation/{}.json'
    pair_test, intent_counter_tst = read_langs(args, "tst", json.load(
        open(data_path.format("teststd_public"), "r")))

    print(
        f"Read {len(pair_test)} pairs teststd_public from SIMMC2 Multimodal Disambiguation")
        # f"{intent_counter_tst['disambiguation'] / len(pair_test) * 100:.1f}% disambiguation")

    intent_class = list(intent_counter_tst.keys())
    meta_data = {"intent": intent_class, "num_labels": len(intent_class)}
    print("len(intent_class)", len(intent_class))

    return pair_test, meta_data
