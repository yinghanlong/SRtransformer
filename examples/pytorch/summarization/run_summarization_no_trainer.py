#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import random
import time

import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datetime import datetime

import transformers
from accelerate import Accelerator
from filelock import FileLock
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.utils.versions import require_version

date= datetime.now().strftime('%Y-%m-%d-%H-%M')
logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--evaluation_only",
        type=bool,
        default=False,
        help="Do not train the model, only evaluate",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help="The maximum total sequence length for validation "
        "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
        "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
        "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    
    parser.add_argument('--gpus', metavar='DEV_ID', default='0',
                        help='Comma-separated list of GPU device IDs to be used (default is to use all available devices)')
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,#8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,#8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    args.output_dir+=date+"/" 
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        filename=args.output_dir+'results.log', #save to a log file
        filemode='a',
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.info(accelerator.device)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
   

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.config_name:
        tokenizer = AutoTokenizer.from_pretrained(args.config_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(args.dataset_name, None)
    if args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    #model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    #    model, optimizer, train_dataloader, eval_dataloader)

    #TODO:set up data parallelism
    if not torch.cuda.is_available():
        # Set GPU index to -1 if using CPU
        args.device = 'cpu'
        args.gpus = -1
    else:
        args.device = 'cuda'
        if args.gpus is not None:
            try:
                args.gpus = [int(s) for s in args.gpus.split(',')]
            except ValueError:
                logging.error('ERROR: Argument --gpus must be a comma-separated list of integers only')
                exit(1)
            available_gpus = torch.cuda.device_count()
            for dev_id in args.gpus:
                if dev_id >= available_gpus:
                    logging.error('ERROR: GPU device ID {0} requested, but only {1} devices available'
                                    .format(dev_id, available_gpus))
                    exit(1)
            # Set default device in case the first one on the list != 0
            torch.cuda.set_device(args.gpus[0])
    model = torch.nn.DataParallel(model, device_ids=args.gpus)
    model = model.to(args.device)
    print(model) #show layer names

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Metric
    metric = load_metric("rouge")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=False)#not accelerator.is_local_main_process)
    completed_steps = 0
    if args.evaluation_only:
        args.num_train_epochs=1

    log_step= 200
    eval_log_step= 100
    use_mem = False #TODO: set to true for spiking networks
    ffn_spike= False
    attn_spike= False
    cross_spike = False
    ffn_id = 1
    
    for epoch in range(args.num_train_epochs):
        logger.info(f"Splitting encoded features in cross attention. Padded to N*split_size. Baseline no spiking")
        logger.info(f"Use RNN for cross-attention; SIZE=8. Concat its output with originial input")
        if use_mem:
            #logger.info(f"Using pos/neg spikes; half spiking FFN T5 blocks without selfattention. Threshold init=0.1, leak=0.5")
            #logger.info(f"LinearSpike func: x=1 if x>1, x=x if 0<x<1. Init thre=0.1. global SNN with a linear layer outside of loop Before proj on key/value with leak and reset")
            #logger.info(f"SNN in FFN with leak and reset.  With sparse attention (size=32, SELECT KEY AND VALUE BY ITSELF)")
            #logger.info(f"SNN in FFN with leak and reset. Output=act(mem_thre), update new Vmem=hidden(Vmem/thre-1)")
            #logger.info(f"Using a mem network with ONE linear layers and tanh  before proj layers for k/v. thresholds with size=inner_dim. Training shift mem_out by 1 each step. Apply spiking equation 3 with init thre=1. q is not changed")
            #logger.info(f"Using separate mem gates after proj layers for keys/values")
            #logger.info(f"Using maxpooling to make attention sparse! Keep current key/values")
            #logger.info(f"New:  scores= q_exp*k_exp, k_exp=exp(k-mean), q_exp=exp(q-mean),  compute (q_exp+position)*k_exp)*v/sum(k_exp)*(q_exp+position)")
            #logger.info(f"Implemented backward gradients Using linear transformer, phi(attnout)=ELU, grad K=SV, grad V=S*K from N-1 to 0, position bias(q,1), torch.div(attn,qz)")#position bias size(1,k)
            #logger.info(f"Using linear transformer, phi(k)=linearspike(x/thre-1.0)")#position bias size(1,k)
            #logger.info(f"Put mem_func ahead of spiking. Attn_out=(Q*Ks)*Vs+ rnn(Q).Timestep=1. Using seq-based sparse activation on keys/values! set size=32. Use an act_sparse_func")
            logger.info(f"Connect type={model.module.decoder.block[0].layer[0].SelfAttention.connect_type}")
            #logger.info(f"Using {model.module.decoder.block[0].layer[0].SelfAttention.window_size} recurrent spiking mem gate!")
        avg_spiking_rate= torch.zeros(config.num_layers)
        if (args.evaluation_only==False):
            model.train()
            total_steps=0
            avg_loss=0
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                avg_loss+=loss
                loss.backward() #accelerator.backward(loss)
                
                #TODO: clipping gradients 
                #torch.nn.utils.clip_grad_norm_(model.module.parameters(), 5.0)

                #for i in range(config.num_layers):
                #    torch.nn.utils.clip_grad_norm_(model.module.decoder.block[i].layer[0].SelfAttention.threshold, 1.0)
                
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= args.max_train_steps:
                    break        
                #calculate average spiking rates

                if step%log_step==1:
                    for i in range(0,int(config.num_layers)):
                        print('loss ', loss)
                        if ffn_spike==True:
                                
                            print('spiking rate of ffn layer ',i,model.module.decoder.block[i].layer[ffn_id].DenseReluDense.spiking_rate.cpu())
                            print('ffn threshold=',torch.mean(model.module.decoder.block[i].layer[ffn_id].DenseReluDense.threshold))
                            print('ffn leak=',torch.mean(model.module.decoder.block[i].layer[ffn_id].DenseReluDense.leak)) 
                            avg_spiking_rate[i]+=model.module.decoder.block[i].layer[ffn_id].DenseReluDense.spiking_rate.cpu()
                        if cross_spike==True:
                            print('spiking rate of ffn layer ',i,model.module.decoder.block[i].layer[1].EncDecAttention.spiking_rate.cpu())
                            print('ffn threshold=',torch.mean(model.module.decoder.block[i].layer[1].EncDecAttention.threshold))
                            print('ffn leak=',torch.mean(model.module.decoder.block[i].layer[1].EncDecAttention.leak)) 
                            avg_spiking_rate[i]+=model.module.decoder.block[i].layer[1].EncDecAttention.spiking_rate.cpu()
                        if attn_spike==True:
                            print('key spiking rate of Layer ',i,model.module.decoder.block[i].layer[0].SelfAttention.spiking_rate.cpu())
                            print('value spiking rate of Layer ',i,model.module.decoder.block[i].layer[0].SelfAttention.v_spiking_rate.cpu())
                            print('threshold=',torch.mean(model.module.decoder.block[i].layer[0].SelfAttention.threshold))
                            print('attention leak=',torch.mean(model.module.decoder.block[i].layer[0].SelfAttention.leak))
                            #print("reduce length to=",model.module.decoder.block[i].layer[0].SelfAttention.key_length, model.module.decoder.block[i].layer[0].SelfAttention.real_key_length)
                            avg_spiking_rate[i]+=model.module.decoder.block[i].layer[0].SelfAttention.spiking_rate.cpu()
                            avg_spiking_rate[i]+=model.module.decoder.block[i].layer[0].SelfAttention.v_spiking_rate.cpu()
                total_steps= step
            avg_spiking_rate/= total_steps/log_step
            logger.info(f"average spiking rate={avg_spiking_rate}")
            logger.info(f"Avg Loss={avg_loss/total_steps}")

        model.eval()
        logging.info("Evaluating model")
        eval_time = time.time()
        if args.val_max_target_length is None:
            args.val_max_target_length = args.max_target_length

        #TODO:use cached past key/value states to speed up decoding or not. By default, set to True
        use_cache = True
        gen_kwargs = {
            "max_length": args.val_max_target_length if args is not None else config.max_length,
            "num_beams": args.num_beams,
            "use_cache": use_cache,
        }
        avg_spiking_rate= torch.zeros(config.num_layers)
        total_steps=0

        for i in range(0,int(config.num_layers)):
            if ffn_spike:
                logger.info(f"threshold={model.module.decoder.block[i].layer[ffn_id].DenseReluDense.threshold}")
                logger.info(f"leak={model.module.decoder.block[i].layer[ffn_id].DenseReluDense.leak}")
            if cross_spike:
                logger.info(f"threshold={model.module.decoder.block[i].layer[1].EncDecAttention.threshold}")
                logger.info(f"leak={model.module.decoder.block[i].layer[1].EncDecAttention.leak}")
            if attn_spike:
                logger.info(f"threshold={model.module.decoder.block[i].layer[0].SelfAttention.threshold}")
                logger.info(f"leak={model.module.decoder.block[i].layer[0].SelfAttention.leak}")
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = model.module.generate(#accelerator.unwrap_model(model).generate(
                    batch["input_ids"].to(args.device),
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )
                #calculate average spiking rates
                if step%eval_log_step==0:
                    end_time = time.time()
                    logger.info(f"time to eval {step} steps from start={end_time-eval_time}")
                    for i in range(0,int(config.num_layers)):
                        #if use_mem==True:
                            #logger.info(f"reduce length to={model.module.decoder.block[i].layer[0].SelfAttention.key_length, model.module.decoder.block[i].layer[0].SelfAttention.real_key_length}")
                        
                        if ffn_spike:
                            print('spiking rate of ffn layer ',i,model.module.decoder.block[i].layer[ffn_id].DenseReluDense.spiking_rate.cpu())
                            print('ffn threshold=',torch.mean(model.module.decoder.block[i].layer[ffn_id].DenseReluDense.threshold))
                            avg_spiking_rate[i]+=model.module.decoder.block[i].layer[ffn_id].DenseReluDense.spiking_rate.cpu()
                        if cross_spike:
                            print('spiking rate of cross layer ',i,model.module.decoder.block[i].layer[1].EncDecAttention.spiking_rate.cpu())
                            print('cross threshold=',torch.mean(model.module.decoder.block[i].layer[1].EncDecAttention.threshold))
                            avg_spiking_rate[i]+=model.module.decoder.block[i].layer[1].EncDecAttention.spiking_rate.cpu()
                        if attn_spike:
                            print('spiking rate of Layer ',i,model.module.decoder.block[i].layer[0].SelfAttention.spiking_rate.cpu())
                            print('value spiking rate of Layer ',i,model.module.decoder.block[i].layer[0].SelfAttention.v_spiking_rate.cpu())
                            print('threshold=',model.module.decoder.block[i].layer[0].SelfAttention.threshold)
                            avg_spiking_rate[i]+=model.module.decoder.block[i].layer[0].SelfAttention.spiking_rate.cpu()
                            avg_spiking_rate[i]+=model.module.decoder.block[i].layer[0].SelfAttention.v_spiking_rate.cpu()
                    

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"].to(args.device)
                if not args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                metric.add_batch(predictions=decoded_preds, references=decoded_labels)
                total_steps= step/log_step
        avg_spiking_rate/=total_steps
        logger.info(f"average spiking rate={avg_spiking_rate}")
        #logger.info(f"average spiking rate={torch.mean(avg_spiking_rate)}")
        end_time = time.time()
        logger.info(f"eval time={end_time-eval_time}")
        result = metric.compute(use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        result = {k: round(v, 4) for k, v in result.items()}

        logger.info(result)
        #only run evaluataion once
        if (args.evaluation_only):
            break

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            logger.info("Saving model",args.output_dir)
            #unwrapped_model = accelerator.unwrap_model(model)
            #unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            model.module.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
