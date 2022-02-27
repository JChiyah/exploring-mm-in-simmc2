### SIMMC2.0

#### Disambiguation Baseline

1. **Preprocess** the datasets to reformat the data for GPT-2 input.

```shell
python format_disambiguation_data.py \
	--simmc_train_json="../../data/simmc2_dials_dstc10_train.json" \
	--simmc_dev_json="../../data/simmc2_dials_dstc10_dev.json" \
	--simmc_devtest_json="../../data/simmc2_dials_dstc10_devtest.json" \
	--disambiguate_save_path="../../data/"
```

2. **Train** and simultaneously test the baseline model.

```shell
./evaluate_mm_disambiguation_model.sh 0 5 4
```

#### Coreference Baseline

1. **Preprocess** the datasets to reformat the data for GPT-2 input.

```shell
cd model/mm_dst
./run_preprocess_gpt2.sh
```

2. **Train** the baseline model

```shell
./run_train_gpt2.sh
```

3. **Generate** prediction for `devtest` data

```shell
./run_generate_gpt2.sh
```

The generation results are saved in the `/mm_dst/results` folder. Change the `path_output` to a desired path accordingly.


4. **Evaluate** predictions for `devtest` data

```shell
./run_evaluate_gpt2.sh
```

# best so far: object f1=0.308, epochs=1, batch=1, gradient acc=4, eval batch=4