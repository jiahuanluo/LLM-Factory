#!/usr/bin/env python
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "transformers>=4.57.0",
#     "accelerate >= 0.12.0",
#     "datasets >= 2.14.0",
#     "sentencepiece != 0.1.92",
#     "scipy",
#     "scikit-learn",
#     "protobuf",
#     "torch >= 1.3",
# ]
# ///

"""Finetuning the library models for text classification."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field

import datasets
import numpy as np
import scipy.special
from datasets import Value, load_dataset, Sequence
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, mean_squared_error, f1_score

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.57.0.dev0")

# require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: str | None = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: str | None = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    do_regression: bool = field(
        default=None,
        metadata={
            "help": "Whether to do regression instead of classification. If None, will be inferred from the dataset."
        },
    )
    text_column_names: str | None = field(
        default=None,
        metadata={
            "help": (
                "The name of the text column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "sentence" column for single/multi-label classification task.'
            )
        },
    )
    text_column_delimiter: str | None = field(
        default=" ", metadata={"help": "The delimiter to use to join text columns into a single sentence."}
    )
    train_split_name: str | None = field(
        default=None,
        metadata={
            "help": 'The name of the train split in the input dataset. If not specified, will use the "train" split when do_train is enabled'
        },
    )
    validation_split_name: str | None = field(
        default=None,
        metadata={
            "help": 'The name of the validation split in the input dataset. If not specified, will use the "validation" split when do_eval is enabled'
        },
    )
    test_split_name: str | None = field(
        default=None,
        metadata={
            "help": 'The name of the test split in the input dataset. If not specified, will use the "test" split when do_predict is enabled'
        },
    )
    remove_splits: str | None = field(
        default=None,
        metadata={"help": "The splits to remove from the dataset. Multiple splits should be separated by commas."},
    )
    remove_columns: str | None = field(
        default=None,
        metadata={"help": "The columns to remove from the dataset. Multiple columns should be separated by commas."},
    )
    label_column_name: str | None = field(
        default=None,
        metadata={
            "help": (
                "The name of the label column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "label" column for single/multi-label classification task'
            )
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    preprocessing_num_workers: int | None = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    shuffle_train_dataset: bool = field(
        default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."}
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    metric_name: str | None = field(default=None, metadata={"help": "The metric to use for evaluation."})
    train_file: str | None = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_files: str | None = field(
        default=None, metadata={"help": "Comma-separated paths to csv or json validation files. Supports single or multiple files."}
    )
    test_file: str | None = field(default=None, metadata={"help": "Comma-separated paths to csv or json test files for prediction. Supports single or multiple files."})

    def __post_init__(self):
        if self.dataset_name is None:
            if self.train_file is None or self.validation_files is None:
                raise ValueError("Need either a training/validation file or a dataset name.")

            train_extension = self.train_file.split(".")[-1]
            if train_extension not in ["csv", "json"]:
                raise ValueError("`train_file` should be a csv or a json file.")
            for val_file in self.validation_files.split(","):
                val_file = val_file.strip()
                if not val_file:
                    raise ValueError("Empty path found in --validation_files.")
                validation_extension = val_file.split(".")[-1]
                if validation_extension != train_extension:
                    raise ValueError(
                        f"`validation_files` entry '{val_file}' should have the same extension (csv or json) as `train_file`."
                    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: str | None = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: str | None = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: str | None = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `hf auth login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def get_label_list(raw_dataset, split="train") -> list[str]:
    """Get the list of labels from a multi-label dataset"""

    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list


def _load_dataset(filepath, is_csv, cache_dir, token):
    fmt = "csv" if is_csv else "json"
    return load_dataset(fmt, data_files={"validation": filepath}, cache_dir=cache_dir, token=token)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    from args_parser import read_args

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = read_args(parser)

    # Setup logging
    os.makedirs(training_args.output_dir, exist_ok=True)
    log_handlers = [logging.StreamHandler(sys.stdout)]
    # Only rank 0 writes to log file to avoid multi-process file conflicts
    if training_args.process_index == 0:
        log_file = os.path.join(training_args.output_dir, "train.log")
        log_handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=log_handlers,
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(logging.WARNING)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_process_index}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files, or specify a dataset name
    # to load from huggingface/datasets. In ether case, you can specify a the key of the column(s) containing the text and
    # the key of the column containing the label. If multiple columns are specified for the text, they will be joined together
    # for the actual text value.
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        # Try print some info about the dataset
        logger.info(f"Dataset loaded: {raw_datasets}")
        logger.info(raw_datasets)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file}

        # Get the test dataset: you can provide your own CSV/JSON test file(s)
        if training_args.do_predict:
            if data_args.test_file is not None:
                test_files_list = [f.strip() for f in data_args.test_file.split(",") if f.strip()]
                train_extension = data_args.train_file.split(".")[-1]
                for tf in test_files_list:
                    test_extension = tf.split(".")[-1]
                    if test_extension != train_extension:
                        raise ValueError(
                            f"`test_file` entry '{tf}' should have the same extension (csv or json) as `train_file`."
                        )
                if len(test_files_list) == 1:
                    data_files["test"] = test_files_list[0]
                # Multiple test files are loaded separately after the main dataset
            else:
                raise ValueError("Need either a dataset name or a test file for `do_predict`.")

        for key in data_files:
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )

        # Load multiple validation files into separate splits
        validation_files_list = [f.strip() for f in data_args.validation_files.split(",") if f.strip()]
        if not validation_files_list:
            raise ValueError("`validation_files` must contain at least one non-empty file path.")
        is_csv = data_args.train_file.endswith(".csv")
        if len(validation_files_list) == 1:
            # Single validation file: load as "validation" split (backward compatible)
            val_datasets = _load_dataset(validation_files_list[0], is_csv, model_args.cache_dir, model_args.token)
            raw_datasets["validation"] = val_datasets["validation"]
            logger.info(f"Loaded validation file '{validation_files_list[0]}' as split 'validation'")
        else:
            # Multiple validation files: load each as "validation_N"
            for i, val_file in enumerate(validation_files_list):
                val_datasets = _load_dataset(val_file, is_csv, model_args.cache_dir, model_args.token)
                raw_datasets[f"validation_{i}"] = val_datasets["validation"]
                logger.info(f"Loaded validation file '{val_file}' as split 'validation_{i}'")

        # Load multiple test files into separate splits (if more than one)
        if training_args.do_predict and data_args.test_file is not None:
            test_files_list = [f.strip() for f in data_args.test_file.split(",") if f.strip()]
            if len(test_files_list) > 1:
                is_csv = data_args.train_file.endswith(".csv")
                for i, tf in enumerate(test_files_list):
                    test_ds = _load_dataset(tf, is_csv, model_args.cache_dir, model_args.token)
                    raw_datasets[f"test_{i}"] = test_ds["validation"]
                    logger.info(f"Loaded test file '{tf}' as split 'test_{i}'")

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    if data_args.remove_splits is not None:
        for split in data_args.remove_splits.split(","):
            logger.info(f"removing split {split}")
            raw_datasets.pop(split)

    if data_args.train_split_name is not None:
        logger.info(f"using {data_args.train_split_name} as train set")
        raw_datasets["train"] = raw_datasets[data_args.train_split_name]
        raw_datasets.pop(data_args.train_split_name)

    if data_args.validation_split_name is not None:
        logger.info(f"using {data_args.validation_split_name} as validation set")
        raw_datasets["validation"] = raw_datasets[data_args.validation_split_name]
        raw_datasets.pop(data_args.validation_split_name)

    if data_args.test_split_name is not None:
        logger.info(f"using {data_args.test_split_name} as test set")
        raw_datasets["test"] = raw_datasets[data_args.test_split_name]
        raw_datasets.pop(data_args.test_split_name)

    if data_args.remove_columns is not None:
        for split in raw_datasets:
            for column in data_args.remove_columns.split(","):
                logger.info(f"removing column {column} from split {split}")
                raw_datasets[split] = raw_datasets[split].remove_columns(column)

    if data_args.label_column_name is not None and data_args.label_column_name != "label":
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].rename_column(data_args.label_column_name, "label")

    # Trying to have good defaults here, don't hesitate to tweak to your needs.

    is_regression = data_args.do_regression is True

    is_multi_label = False
    if is_regression:
        label_list = None
        num_labels = 1
        # regression requires float as label type, let's cast it if needed
        for split in raw_datasets:
            if raw_datasets[split].features["label"].dtype not in ["float32", "float64"]:
                logger.warning(
                    f"Label type for {split} set to float32, was {raw_datasets[split].features['label'].dtype}"
                )
                features = raw_datasets[split].features
                features.update({"label": Value("float32")})
                try:
                    raw_datasets[split] = raw_datasets[split].cast(features)
                except TypeError as error:
                    logger.error(
                        f"Unable to cast {split} set to float32, please check the labels are correct, or maybe try with --do_regression=False"
                    )
                    raise error

    else:  # classification
        _label_feature = raw_datasets["train"].features["label"]
        _is_list_label = isinstance(_label_feature, Sequence)
        # Cast float labels to int for classification (e.g., 0.0/1.0 -> 0/1)
        if not _is_list_label and getattr(_label_feature, "dtype", None) in ["float32", "float64"]:
            logger.info("Label dtype is float, casting to int32 for classification.")
            features = raw_datasets["train"].features.copy()
            features.update({"label": Value("int32")})
            for split in raw_datasets:
                if "label" in raw_datasets[split].features:
                    raw_datasets[split] = raw_datasets[split].cast(features)

        if _is_list_label:  # multi-label classification
            is_multi_label = True
            logger.info("Label type is list, doing multi-label classification")
        # Trying to find the number of labels in a multi-label classification task
        # We have to deal with common cases that labels appear in the training set but not in the validation/test set.
        # So we build the label list from the union of labels in train/val/test.
        label_list = get_label_list(raw_datasets, split="train")
        for split in raw_datasets:
            if split.startswith("validation") or split == "test":
                val_or_test_labels = get_label_list(raw_datasets, split=split)
                diff = set(val_or_test_labels).difference(set(label_list))
                if len(diff) > 0:
                    # add the labels that appear in val/test but not in train, throw a warning
                    logger.warning(
                        f"Labels {diff} in {split} set but not in training set, adding them to the label list"
                    )
                    label_list += list(diff)
        # if label is -1, we throw a warning and remove it from the label list
        for label in label_list:
            if label == "-1":
                logger.warning("Label -1 found in label list, removing it.")
                label_list.remove(label)

        label_list.sort()
        num_labels = len(label_list)
        if num_labels <= 1:
            raise ValueError("You need more than one label to do classification.")

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="text-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    if is_regression:
        config.problem_type = "regression"
        logger.info("setting problem type to regression")
    elif is_multi_label:
        config.problem_type = "multi_label_classification"
        logger.info("setting problem type to multi label classification")
    else:
        config.problem_type = "single_label_classification"
        logger.info("setting problem type to single label classification")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Check UNK token ratio on a random sample before training
    def check_unk_ratio(tokenizer, dataset, text_column, num_samples=100, threshold=0.1, max_length=512):
        import random
        unk_id = tokenizer.unk_token_id
        if unk_id is None:
            return
        n = min(num_samples, len(dataset))
        indices = random.sample(range(len(dataset)), n)
        samples = [dataset[i][text_column] for i in indices]
        samples = [s for s in samples if s and not s.isspace()]
        if not samples:
            return
        encodings = tokenizer(samples, truncation=True, max_length=max_length, return_attention_mask=False)
        total_tokens = 0
        unk_tokens = 0
        for ids in encodings["input_ids"]:
            total_tokens += len(ids)
            unk_tokens += ids.count(unk_id)
        ratio = unk_tokens / total_tokens if total_tokens > 0 else 0
        logger.info(f"UNK token 检查: 采样 {len(samples)} 条, 总 token {total_tokens}, UNK {unk_tokens}, 占比 {ratio:.2%}")
        if ratio > threshold:
            raise ValueError(
                f"UNK token 占比 {ratio:.2%} 超过阈值 {threshold:.0%}，"
                f"请检查 tokenizer 是否与训练数据匹配（语言、词表等）"
            )

    if training_args.do_train and training_args.local_rank in [-1, 0]:
        if data_args.text_column_names is not None:
            _check_text_col = data_args.text_column_names.split(",")[0]
        else:
            _features = list(raw_datasets["train"].features)
            _check_text_col = "sentence" if "sentence" in _features else _features[0]
        check_unk_ratio(tokenizer, raw_datasets["train"], _check_text_col,
                        max_length=data_args.max_seq_length or 512)

    if model_args.model_name_or_path:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
    else:
        logger.info("model_name_or_path is None, initializing model from config (random weights)")
        model = AutoModelForSequenceClassification.from_config(
            config, trust_remote_code=model_args.trust_remote_code
        )

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # for training ,we will update the config with label infos,
    # if do_train is not set, we will use the label infos in the config
    if training_args.do_train and not is_regression:  # classification, training
        label_to_id = {v: i for i, v in enumerate(label_list)}
        # update config with label infos
        if model.config.label2id != label_to_id:
            logger.warning(
                "The label2id key in the model config.json is not equal to the label2id key of this "
                "run. You can ignore this if you are doing finetuning."
            )
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in label_to_id.items()}
    elif not is_regression:  # classification, but not training
        logger.info("using label infos in the model config")
        logger.info(f"label2id: {model.config.label2id}")
        label_to_id = model.config.label2id
    else:  # regression
        label_to_id = None

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def multi_labels_to_ids(labels: list[str]) -> list[float]:
        ids = [0.0] * len(label_to_id)  # BCELoss requires float as target type
        for label in labels:
            ids[label_to_id[str(label)]] = 1.0
        return ids

    def preprocess_function(examples):
        if data_args.text_column_names is not None:
            text_column_names = data_args.text_column_names.split(",")
            # join together text columns into "sentence" column
            examples["sentence"] = examples[text_column_names[0]]
            for column in text_column_names[1:]:
                for i in range(len(examples[column])):
                    examples["sentence"][i] += data_args.text_column_delimiter + examples[column][i]
        # Tokenize the texts
        result = tokenizer(examples["sentence"], padding=padding, max_length=max_seq_length, truncation=True)
        if label_to_id is not None and "label" in examples:
            if is_multi_label:
                result["label"] = [multi_labels_to_ids(l) for l in examples["label"]]
            else:
                mapped_labels = []
                for l in examples["label"]:
                    key = str(l)
                    if key in label_to_id:
                        mapped_labels.append(label_to_id[key])
                    elif l == -1:
                        mapped_labels.append(-1)
                    else:
                        # For prediction with mismatched labels, use 0 as placeholder
                        # (labels will be removed before prediction anyway)
                        mapped_labels.append(0)
                result["label"] = mapped_labels
        return result

    # Running the preprocessing pipeline on all the datasets
    # Rank 0 tokenizes and saves; other ranks load from disk (avoids redundant tokenize)
    tokenized_cache_dir = os.path.join(training_args.output_dir, "_tokenized_cache")
    with training_args.main_process_first(desc="dataset map pre-processing"):
        if training_args.process_index == 0:
            raw_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            raw_datasets.save_to_disk(tokenized_cache_dir)
        else:
            from datasets import load_from_disk
            raw_datasets = load_from_disk(tokenized_cache_dir)

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset.")
        train_dataset = raw_datasets["train"]
        if data_args.shuffle_train_dataset:
            logger.info("Shuffling the training dataset")
            train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        # Build eval_dict: collect all validation splits
        eval_dict = {}
        for key in raw_datasets:
            if key.startswith("validation"):
                ds = raw_datasets[key]
                if data_args.max_eval_samples is not None:
                    max_eval_samples = min(len(ds), data_args.max_eval_samples)
                    ds = ds.select(range(max_eval_samples))
                eval_dict[key] = ds

        if len(eval_dict) == 0:
            # Fallback to test set
            if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                raise ValueError("--do_eval requires a validation or test dataset if validation is not defined.")
            else:
                logger.warning("Validation dataset not found. Falling back to test dataset for validation.")
                eval_dataset = raw_datasets["test"]
                if data_args.max_eval_samples is not None:
                    max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                    eval_dataset = eval_dataset.select(range(max_eval_samples))
        elif len(eval_dict) == 1:
            # Single validation: use the dataset directly for backward compatibility
            eval_dataset = list(eval_dict.values())[0]
        else:
            # Multiple validations: use dict
            eval_dataset = eval_dict

    if training_args.do_predict:
        # Collect all test splits (single "test" or multiple "test_0", "test_1", ...)
        predict_datasets = {}
        for key in raw_datasets:
            if key == "test" or key.startswith("test_"):
                ds = raw_datasets[key]
                if data_args.max_predict_samples is not None:
                    max_predict_samples = min(len(ds), data_args.max_predict_samples)
                    ds = ds.select(range(max_predict_samples))
                predict_datasets[key] = ds
        if not predict_datasets:
            raise ValueError("--do_predict requires a test dataset")

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if data_args.metric_name is not None:
        metric_name = data_args.metric_name
    elif is_regression:
        metric_name = "mse"
    elif is_multi_label:
        metric_name = "f1"
    else:
        metric_name = "accuracy"
    logger.info(f"Using metric '{metric_name}' for evaluation.")

    if training_args.metric_for_best_model is not None:
        expected_prefix = f"eval_{metric_name}"
        best_metric = training_args.metric_for_best_model
        # auc/ks 会额外输出 per-label 指标如 eval_auc_xxx，所以也允许前缀匹配
        if best_metric != expected_prefix and not best_metric.startswith(expected_prefix + "_"):
            raise ValueError(
                f"metric_for_best_model='{best_metric}' 与 metric_name='{metric_name}' 不匹配。"
                f"预期以 '{expected_prefix}' 开头，请检查配置。"
            )

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        if is_regression:
            preds = np.squeeze(preds)
            return {"mse": mean_squared_error(p.label_ids, preds)}
        elif is_multi_label:
            preds = np.array([np.where(p > 0, 1, 0) for p in preds])
            result = {"f1": f1_score(p.label_ids, preds, average="micro")}
            # Compute per-label AUC and KS
            if metric_name in ("auc", "ks"):
                n_labels = p.label_ids.shape[1]
                for i in range(n_labels):
                    label_name = label_list[i] if label_list else str(i)
                    try:
                        auc = roc_auc_score(p.label_ids[:, i], preds[:, i])
                        result[f"auc_{label_name}"] = auc if not np.isnan(auc) else 0.0
                    except ValueError as e:
                        raise RuntimeError(
                            f"无法计算标签 '{label_name}' 的 AUC 指标: {e}。"
                            f"请检查验证集是否包含足够多的正负样本。"
                        ) from e
                    try:
                        fpr, tpr, _ = roc_curve(p.label_ids[:, i], preds[:, i])
                        result[f"ks_{label_name}"] = float(np.max(tpr - fpr))
                    except ValueError as e:
                        raise RuntimeError(
                            f"无法计算标签 '{label_name}' 的 KS 指标: {e}。"
                            f"请检查验证集是否包含足够多的正负样本。"
                        ) from e
            return result
        else:
            probs = preds
            preds = np.argmax(preds, axis=1)
            result = {}
            if metric_name == "accuracy":
                result["accuracy"] = accuracy_score(p.label_ids, preds)
            elif metric_name == "auc":
                try:
                    if probs.shape[1] == 2:
                        score = roc_auc_score(p.label_ids, probs[:, 1])
                    else:
                        score = roc_auc_score(p.label_ids, probs, multi_class="ovr")
                    result["auc"] = score if not np.isnan(score) else 0.0
                except ValueError as e:
                    raise RuntimeError(
                        f"无法计算 AUC 指标: {e}。请检查验证集是否包含所有类别样本。"
                    ) from e
                # Also compute KS for binary classification
                if probs.shape[1] == 2:
                    fpr, tpr, _ = roc_curve(p.label_ids, probs[:, 1])
                    result["ks"] = float(np.max(tpr - fpr))
            elif metric_name == "ks":
                # KS = max(TPR - FPR) from ROC curve
                if probs.shape[1] == 2:
                    fpr, tpr, _ = roc_curve(p.label_ids, probs[:, 1])
                    result["ks"] = float(np.max(tpr - fpr))
                else:
                    raise ValueError("KS metric only supports binary classification.")
            else:
                raise ValueError(f"Unknown metric: {metric_name}. Supported: accuracy, auc, ks, mse, f1")
            return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif os.path.isdir(training_args.output_dir):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is not None:
                checkpoint = last_checkpoint
                logger.info(f"Resuming from latest checkpoint: {checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        if not isinstance(eval_dataset, dict):
            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        # Determine columns to exclude from output: text columns + label
        exclude_cols = {"label"}
        if data_args.text_column_names is not None:
            exclude_cols.update(c.strip() for c in data_args.text_column_names.split(","))
        if data_args.remove_columns is not None:
            exclude_cols.update(c.strip() for c in data_args.remove_columns.split(","))
        # Preprocess tokenizer columns to exclude
        tokenizer_cols = {"input_ids", "token_type_ids", "attention_mask"}

        for split_name, predict_dataset in predict_datasets.items():
            # Removing the `label` columns if exists because it might contains -1 and Trainer won't like that.
            if "label" in predict_dataset.features:
                predict_dataset = predict_dataset.remove_columns("label")
            # Save original row data (non-tokenizer, non-excluded columns) before prediction
            all_cols = list(predict_dataset.features.keys())
            keep_cols = [c for c in all_cols if c not in exclude_cols and c not in tokenizer_cols]
            rows = [{c: predict_dataset[i][c] for c in keep_cols} for i in range(len(predict_dataset))]

            predict_result = trainer.predict(predict_dataset, metric_key_prefix="predict")
            raw_predictions = predict_result.predictions
            if is_regression:
                predictions = np.squeeze(raw_predictions)
            elif is_multi_label:
                predictions = scipy.special.expit(raw_predictions)  # sigmoid probabilities
            else:
                # For binary classification, output probability of positive class (class 1)
                # For multi-class, output argmax label
                if raw_predictions.shape[1] == 2:
                    # Binary: softmax to get probabilities, output class 1 probability
                    probs = scipy.special.softmax(raw_predictions, axis=1)
                    predictions = probs[:, 1]
                else:
                    predictions = np.argmax(raw_predictions, axis=1)
            # Output filename: single test -> predict_results.txt, multiple -> predict_results_N.txt
            if len(predict_datasets) == 1:
                output_predict_file = os.path.join(training_args.output_dir, "predict_results.txt")
            else:
                output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{split_name}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results for {split_name} *****")
                    header = "\t".join(keep_cols + ["prediction"])
                    writer.write(header + "\n")
                    for index, item in enumerate(predictions):
                        cols = [str(rows[index][c]) for c in keep_cols]
                        if is_regression:
                            cols.append(f"{item:3.3f}")
                        elif is_multi_label:
                            probs = {label_list[i]: f"{item[i]:.6f}" for i in range(len(item))}
                            cols.append(str(probs))
                        elif raw_predictions.shape[1] == 2:
                            cols.append(f"{item:.6f}")
                        else:
                            cols.append(label_list[item])
                        writer.write("\t".join(cols) + "\n")
            logger.info(f"Predict results saved at {output_predict_file}")
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger(__name__).exception("Training failed with exception")
        raise
