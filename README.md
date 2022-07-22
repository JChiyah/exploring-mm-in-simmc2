# Exploring Multi-Modal Representations for Ambiguity Detection & Coreference Resolution in the SIMMC 2.0 Challenge


Repository for the paper "*Exploring Multi-Modal Representations for Ambiguity Detection & Coreference Resolution in the SIMMC 2.0 Challenge*" due to appear in the [AAAI 2022 DSTC10 Workshop](https://sites.google.com/dstc.community/dstc10/aaai-22-workshop). The code and data here are given as-is for reproducibility without any warranties. It has only been tested in Ubuntu 18 and 20, with Python 3. If there are any issues, or you cannot run certain parts, please let us know and we will fix it as soon as possible. Cite us like [this](#citation).

Note that there is a mix of licenses involved, see below for more information.

This repository is divided into 4 main folders:

- `simmc2_data_generated` contains pre-processed data extracted from the original SIMMC2.0 data that is easier to use for models and experiments.
- `detectron` has the code used to extract information from the images. It uses a modified version of [Detectron2](https://github.com/facebookresearch/detectron2/) and it has an Apache-2.0 License.
- `disambiguation_model` has the code for a modified version of [ToD-BERT](https://github.com/jasonwu0731/ToD-BERT) for disambiguation prediction and is released under their BSD 2-Clause License. It is the base for the disambiguation model for [Subtask#1](#).
- `coreference_model` contains the code for the multimodal coreference model of [Subtask#2](). It uses a pre-trained [LXMERT](https://github.com/airsplay/lxmert/) as the encoder and it is released under the MIT License.


This repository is released under an MIT License, unless a more restrictive license applies to some files (Apache-2.0 for the Detectron2 files and BSD 2-Clause License for ToD-BERT).

Four different environments are used throughout the code due to using different versions of the same packages (e.g., Detectron2, PyTorch). Conda is heavily recommended if running anything below. All the environments files and requirements.txt are provided but please get in touch if there is anything broken!


## Subtask #1 (Multimodal Disambiguation): Task-Oriented Disambiguation Model

The model uses a base of [ToD-BERT](https://github.com/jasonwu0731/ToD-BERT) with some small modifications to accept the SIMMC2.0 data. Conversation history is optional and the model often learns slightly better without it. In the best model, the batch size is 64 and it does not use any previous system or user utterances in the dialogue, just the current sentence.

Its high accuracy is due to a bias in the data about what the user says before a disambiguation turn, with some occurrences appearing more often when the system needs to disambiguate (e.g., more anaphoric references than normal turns, few or no entities mentioned, etc.).

Submission JSONs are inside the `disambiguation-model/` folder.

Evaluation results from fine-tuning the model for 5 epochs and evaluating the best dev model on devtest. These results are averaged over 10 runs with different seeds.


| Dataset | Accuracy |
|---------|----------|
| train   | 0.951726 |
| dev     | 0.932957 |
| devtest | 0.918539 |


### Training and Testing

```shell
# 1. Create conda environment & activate
conda env create -f environment-disambiguation.yml && conda activate simmc2-disam

# 2. Install packages
pip install -r requirements.txt
pip install -r disambiguation_model/requirements.txt

# 3. Generate data for this model
python generate_simmc2_data.py --subtask disambiguation --output_data_folder_path simmc2_data_generated

cd disambiguation-model

# Train model
./evaluation_simmc2_pipeline.sh 0 train "user_utterance" 0

# Evaluate and generate submission jsons
./evaluation_simmc2_pipeline.sh 0 eval "user_utterance" 0

python disambiguator_evaluation.py --model_result_path="../../../disambiguation-model/dstc10-simmc-dev-pred-subtask-1.json" --data_json_path="../../../disambiguation-model/dstc10-simmc-dev-true-subtask-1.json"
```

If you want to load our model downloaded from GitHub, download the checkpoint from the releases and check the script `evaluation_simmc2_pipeline.sh` to change line 49 to point to this downloaded file.


## Subtask #2 (Multimodal Coreference Resolution): Object-Sentence Relational Model

This model uses a pre-trained [LXMERT](https://github.com/airsplay/lxmert/) model and makes several modifications.

### Summary

For each object in the image/scene, we assemble a natural language sentence with the user utterance and the object description extracted from the Detectron2 model.
We then append to this sentence the dialogue history and/or previously mentioned objects (from the previous `system_transcript_annotated`). An example final sentence could be _"SYSTEM : We have this red, white and yellow blouse on the left, and the white blouse on the right [SEP] 60 56 [SEP] USER : I'd like to get that blouse, please [SEP] 56 white blouse hanging"_.

Each sentence is then fed to the model along with the visual RoI features, bounding boxes and object counters.

The model first extracts BERT embeddings from the input sentence, as well as object positional embeddings from the object counters. The object counters represent the amount of objects per asset type in the scene, derived from the object descriptions from Detectron. For example, the first jacket will be 1, the second jacket will be 2, etc.

An encoder combines the bounding boxes, RoI features and object positional embeddings through several linear and normalisation layers to obtain a visual feature state.

The sentence embeddings and visual features are passed through the typical set of LXMERT encoders (language, relational and cross-modality layers) to obtain two feature sequences: language and vision. We then use these to extract a hidden representation, that we pass through a sequence of simple GeLU, normalisation and linear layers to obtain a vector with as many values as initial sentences were given to the model.

The output is a score between 0 and 1 for the likelihood of a sentence referencing that particular object.
The final object IDs are resolved using a threshold on these scores, usually between 0.3 and 0.4. For example, an object is selected if its score is above this threshold.


We train with a batch size of 4 in a RTX 2080 for 10 epochs and evaluate the results with the script provided for the following scores:

| Dataset | Object F1 |  Recall   | Precision |
|---------|-----------|-----------|-----------|
| train   | 0.7261765 | 0.6247054 | 0.6362249 |
| dev     | 0.5357559 | 0.5408953 | 0.5307133 |
| devtest | 0.5658407 | 0.5955493 | 0.5389553 |

Submission JSONs are inside the model folder.

We use considerably fewer layers, batch size and sequence length than in the original LXMERT architecture. However, ablations show that increasing the sequence length and the number of LXMERT layers used benefit the Object F1, so further improving the results obtained with this model could be possible with a larger GPU or more optimisations.

One of the automatic metrics that we also track is object similarity between the predicted and true object descriptions. It quickly reaches 0.9 after a single epoch, signalling that the model is able to learn the type of object we are talking about.
Qualitative examples show that this model picks the correct asset type and colour most of the time, even if it doesn't pick the correct object ID in the end. E.g., top scores when the correct object is of type jeans_hanging are all very likely to be jeans_hanging too, and probably the same colour. However, the model doesn't know exactly which one the user may be talking about when there are many similar objects, thus leading to confusion. Improving the input visual features could help, and this is part of future work.



### Training and Testing

Requisites: run the `generate_simmc2_data.py` script for the data and extract the images features using the Detectron2 model. Instructions for both of these are below.

```shell
# 1. Create conda environment & activate
conda env create -f environment-coreference.yml && conda activate simmc2-coref && cd coreference_model

# 2. Install packages
pip install torch==1.9.1 && pip install -r requirements.txt

# 3. Check that it can train with a small data size
WANDB_MODE=disabled bash run/simmc2_coreference_finetune.bash 0 lxr533 4 "user_utterance image named_entities dialogue_history object_index previous_objects" 1 5 all_gt_boxes "--llayers 5 --xlayers 3 --rlayers 3 --tiny"
# updated version
WANDB_MODE=disabled bash run/simmc2_coreference_finetune.bash 0,1,2,3 4 "previous_object_turns='all'" 5 train "--tiny"

# 4. Train model
WANDB_MODE=disabled bash run/simmc2_coreference_finetune.bash 0 lxr533 4 "user_utterance image named_entities dialogue_history object_index previous_objects" 1 5 all_gt_boxes "--llayers 5 --xlayers 3 --rlayers 3"

# 5. Test and generate JSONs for submission
WANDB_MODE=disabled bash run/simmc2_coreference_finetune.bash 0 lxr533 4 "user_utterance image named_entities dialogue_history object_index previous_objects" 1 5 all_gt_boxes "--llayers 5 --xlayers 3 --rlayers 3 --load subtask2_coreference_model"

# 6. Evaluate with scripts provided
python ../simmc2/model/mm_dst/utils/evaluate_dst.py --input_path_target=../simmc2/data/simmc2_dials_dstc10_devtest.json --input_path_predicted=dstc10-simmc-devtest-pred-subtask-2.json --output_path_report=eval_result.json
```

Note that the script provided to evaluate Subtask#2 `simmc2/model/mm_dst/utils/evaluate_dst.py` does not work if only attempting Subtask#2 (as we are). We commented out the lines that were causing the issues (slot prediction) and uploaded a modified file in the root folder.


Other useful commands below:

```shell
# Test
bash run/simmc2_coreference_finetune.bash 0 lxr533 4 "user_utterance image named_entities dialogue_history object_index previous_objects" 1 1 all_gt_boxes "--load snap/simmc2/coreference_lxr533_4_user_utterance_image/run0/BEST"

# Visualise output in images (compare predicted vs ground truth)
bash run/simmc2_coreference_finetune.bash 0 lxr533 4 "user_utterance image named_entities dialogue_history object_index previous_objects" 1 1 all_gt_boxes "--load snap/simmc2/coreference_lxr533_4_user_utterance_image/run0/BEST --visualise --simmc2_sample 10"
```

Check [coreference_model/run/simmc2_coreference_finetune.bash](coreference_model/run/simmc2_coreference_finetune.bash) for several examples and experiments (e.g., train/test without X feature).

We tracked our experiments with [Weight and Biases](https://wandb.ai), you can check the open access table of results here: [https://wandb.ai/jchiyah/exloring-mm-in-simmc2](https://wandb.ai/jchiyah/exloring-mm-in-simmc2)


## Generate Pre-processed Data

Generate the SIMMC2.0 data for use in different models/places of this repository. Check `generate_simmc2_data.py` for more information.


```shell
# 1. Create conda environment & activate
conda env create -f environment-base.yml && conda activate simmc2-base

# 2. Install packages
pip install -r requirements.txt

# 3. Generate data for everything (might take a few minutes)
python generate_simmc2_data.py --subtask all --output_data_folder_path simmc2_data_generated

# Run some data analysis
python generate_simmc2_data.py --subtask analysis --output_data_folder_path tmp
```


## Image Feature Extraction: Detectron2

Extract object features and descriptions using Detectron2. The model used for downstream tasks is provided as a binary in a release [here](https://github.com/JChiyah/simmc2-jchiyah/releases/tag/detectron2-trained-model)

The object descriptions are a combination of the assetType and colour of the objects in the SIMMC2.0 data. For example, "blue jeans display" or "red white dress hanging". It is fine-tuned for 3000 iterations (see config in `finetune_simmc2.py`) and the accuracy results for predicting the correct object description are 0.71/0.66/0.65 for train/dev/devtest respectively.


### Commands

```shell
# 1. Create conda environment & activate
conda create -n simmc2-detectron python=3.7.11 && conda activate simmc2-detectron && cd detectron

# 2. Install packages
pip install -r requirements.txt

# 3. Fix cuda if needed (we are using Detectron2v0.1, so it is a bit old)
conda install cudatoolkit=9.2

# 4. Train the model to classify descriptions of colours and asset types
CUDA_VISIBLE_DEVICES=1 python finetune_simmc2.py --train --test --return_feat --category colour_types_compound

# 5. Generate TSV files with image features after fine-tuning
CUDA_VISIBLE_DEVICES=1 python finetune_simmc2.py --resume --return_feat --category colour_types_compound --gtsv --data_percentage 100

# > Example output after generating the feature files:
#{
#    "train": {
#        "total_entries": 30768,
#        "category_accuracy_mean": 0.7065782631305252,
#        "object_recall_mean": 1.0,
#        "object_precision_mean": 1.0,
#        "object_f1_mean": 1.0
#    },
#    "dev": {
#        "total_entries": 3979,
#        "category_accuracy_mean": 0.6559437044483538,
#        "object_recall_mean": 1.0,
#        "object_precision_mean": 1.0,
#        "object_f1_mean": 1.0
#    },
#    "devtest": {
#        "total_entries": 7492,
#        "category_accuracy_mean": 0.6542979177789642,
#        "object_recall_mean": 1.0,
#        "object_precision_mean": 1.0,
#        "object_f1_mean": 1.0
#    }
#    "teststd_public": {
#	     "total_entries": 5665,
#        "category_accuracy_mean": 0.659135039717564,
#        "object_recall_mean": 1.0,
#        "object_precision_mean": 1.0,
#        "object_f1_mean": 1.0
#    }
#}
```
We are using gold bounding boxes at prediction time, so object F1 is always 1. Regarding teststd_public, gold data for categories (metadata about asset type and colours) is only used to calculate an approximate accuracy at evaluation.

The generated tsv files are saved to `simmc2_data_generated/image_features/`.


# Acknowledgements

Chiyah-Garcia’s PhD is funded under the EPSRC iCase with Siemens (EP/T517471/1). This work was also supported by the EPSRC CDT in Robotics and Autonomous Systems (EP/L016834/1).

# Citation

Check the paper in [Arxiv.org]()

Chiyah-Garcia, F., Suglia, A., Lopes, J., Eshghi, A., and Hastie, H. 2022. Exploring Multi-Modal Representations for Ambiguity Detection & Coreference Resolution in the SIMMC 2.0 Challenge. In AAAI 2022 DSTC10 Workshop.

```bibtex
@inproceedings{chiyah-garcia2022dstc10,
  title={Exploring Multi-Modal Representations for Ambiguity Detection \& Coreference Resolution in the SIMMC 2.0 Challenge},
  author={Chiyah-Garcia, Francisco J. and Suglia, Alessandro and Lopes, Jos{'e} David and Eshghi, Arash and Hastie, Helen},
  booktitle={AAAI 2022 DSTC10 Workshop},
  year={2022}
}
```
