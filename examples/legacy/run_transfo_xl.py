#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" PyTorch Transformer XL model evaluation script.
    Adapted from https://github.com/kimiyoung/transformer-xl.
    In particular https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/eval.py

    This script with default values evaluates a pretrained Transformer-XL on WikiText 103
"""


import argparse
import logging
import math
import time

from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
import torch

from transformers import TransfoXLCorpus, TransfoXLLMHeadModel

from datetime import datetime
import os


def main():
    parser = argparse.ArgumentParser(description="PyTorch Transformer Language Model")
    parser.add_argument("--model_name", type=str, default="transfo-xl-wt103", help="pretrained model name")
    parser.add_argument(
        "--split", type=str, default="all", choices=["all", "valid", "test"], help="which split to evaluate"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--tgt_len", type=int, default=384, help="number of tokens to predict")
    parser.add_argument("--ext_len", type=int, default=0, help="length of the extended context")
    parser.add_argument("--mem_len", type=int, default=384, help="length of the retained previous heads")
    parser.add_argument("--eval_tgt_len", type=int, default=128, help="number of tokens to predict")
    parser.add_argument("--eval_mem_len", type=int, default=1600, help="length of the retained previous heads")
    parser.add_argument("--clamp_len", type=int, default=1000, help="max positional embedding index")
    parser.add_argument("--no_cuda", action="store_true", help="Do not use CUDA even though CUA is available")
    #parser.add_argument("--work_dir", type=str, required=True, help="path to the work_dir")
    parser.add_argument("--no_log", action="store_true", help="do not log the eval result")
    parser.add_argument("--same_length", action="store_true", help="set same length attention with masking")
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="max train steps to perform.")
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
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,#5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument('--gpus', metavar='DEV_ID', default='0',
                        help='Comma-separated list of GPU device IDs to be used (default is to use all available devices)')
    
    args = parser.parse_args()
    assert args.ext_len >= 0, "extended context length must be non-negative"

    date= datetime.now().strftime('%Y-%m-%d-%H-%M')
    args.output_dir+=date+"/" 
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        filename=args.output_dir+'results.log',format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    #Set up data parallelism
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
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info("device: {}".format(device))

    # Load a pre-processed dataset
    # You can also build the corpus yourself using TransfoXLCorpus methods
    # The pre-processing involve computing word frequencies to prepare the Adaptive input and SoftMax
    # and tokenizing the dataset
    # The pre-processed corpus is a convertion (using the conversion script )
    corpus = TransfoXLCorpus.from_pretrained(args.model_name)
    train_iter = corpus.get_iterator("train", args.batch_size, args.tgt_len, device=device, ext_len=args.ext_len)

    va_iter = corpus.get_iterator("valid", args.batch_size, args.eval_tgt_len, device=device, ext_len=args.ext_len)
    te_iter = corpus.get_iterator("test", args.batch_size, args.eval_tgt_len, device=device, ext_len=args.ext_len)

    # Load a pre-trained model
    config = AutoConfig.from_pretrained(args.model_name)
    model = TransfoXLLMHeadModel.from_pretrained(args.model_name, config=config)


    model = torch.nn.DataParallel(model, device_ids=args.gpus)
    model.to(args.device)

    
    logger.info(
        "Training with bsz {} tgt_len {} ext_len {} mem_len {} clamp_len {}".format(
            args.batch_size, args.tgt_len, args.ext_len, args.mem_len, args.clamp_len
        )
    )

    model.module.reset_memory_length(args.mem_len)
    if args.clamp_len > 0:
        model.module.clamp_len = args.clamp_len
    if args.same_length:
        model.module.same_length = True

    ###############################################################################
    # TODO: Train code
    ###############################################################################
    log_step = 500
    enable_spiking = False
    def train(train_iter, eval_iter):
        min_loss= 100.0
        total_len, total_loss = 0, 0.0
        for i in range(args.num_epochs):
            logger.info(f"Epoch {i}")
            model.train()
            mems = None
            total_len, total_loss = 0, 0.0
            start_time = time.time()
            for idx, (data, target, seq_len) in enumerate(train_iter):
                ret = model(data, mems=mems, labels=target)
                loss = ret.losses
                mems = ret.mems
                loss = loss.mean()
                #backpropagation
                loss.backward()
                total_loss += seq_len * loss.item()
                total_len += seq_len
                if idx%log_step == 1:
                    logger.info("Avg train Loss: {:.2f}".format(total_loss/total_len))
                    if enable_spiking is True:
                        for i in range(0,int(config.n_layer),1):
                            logger.info(f'spiking rate of layer {model.module.transformer.layers[i].dec_attn.spiking_rate.cpu()}')
                            logger.info(f'threshold={torch.mean(model.module.transformer.layers[i].dec_attn.threshold)}')
                            logger.info(f'leak={torch.mean(model.module.transformer.layers[i].dec_attn.leak)}') 
                            
                #optimize weights
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            total_time = time.time() - start_time
            logger.info("Train Time : {:.2f}s, {:.2f}ms/segment".format(total_time, 1000 * total_time / (idx + 1)))
            #evaluate every epoch
            model.eval()
            eval_loss = evaluate(eval_iter)
            logger.info("Eval Loss: {:.2f}".format(eval_loss))
            log_str = ""
            #if eval_loss is not None:
            #    log_str += format_log(eval_loss, "valid")
            #    logging.ifo(log_str)
            if eval_loss< min_loss:
                #save model
                logger.info("Saving model",args.output_dir)
                model.module.save_pretrained(args.output_dir)

                    

        return total_loss / total_len

    ###############################################################################
    # Evaluation code
    ###############################################################################
    def evaluate(eval_iter):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_len, total_loss = 0, 0.0
        start_time = time.time()

        log_step = 100
        with torch.no_grad():
            mems = None
            for idx, (data, target, seq_len) in enumerate(eval_iter):
                ret = model(data, mems=mems, labels=target)
                loss = ret.losses
                mems = ret.mems
                loss = loss.mean()
                total_loss += seq_len * loss.item()
                total_len += seq_len
                if idx%log_step==1:
                    if enable_spiking is True:
                        for i in range(0,int(config.n_layer),1):
                            logger.info(f'spiking rate of layer {model.module.transformer.layers[i].dec_attn.spiking_rate.cpu()}')
                            logger.info(f'threshold={torch.mean(model.module.transformer.layers[i].dec_attn.threshold)}')
                            logger.info(f'leak={torch.mean(model.module.transformer.layers[i].dec_attn.leak)}') 
            total_time = time.time() - start_time

        logger.info("Time : {:.2f}s, {:.2f}ms/segment".format(total_time, 1000 * total_time / (idx + 1)))
        return total_loss / total_len

    

    # Run on test data.
    if args.split == "all":
        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = train_iter.get_n_batch()
        if args.max_train_steps is None:
            args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
        logger.info(f"out_mems = self.leak * mems. Use SNN in RelPartialLearnableMultiHeadAttn to update mems, init thre=0.1, leak=0.8")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps))
        #TODO:add training optimizer and lr scheduler
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,#args.num_epochs,
        )


        train_loss = train(train_iter, va_iter)

        logger.info(
            "Evaluating with bsz {} tgt_len {} ext_len {} mem_len {} clamp_len {}".format(
                args.batch_size, args.eval_tgt_len, args.ext_len, args.eval_mem_len, args.clamp_len
            )
        )

        model.module.reset_memory_length(args.eval_mem_len)
        test_loss = evaluate(te_iter)
        valid_loss = evaluate(va_iter)
    elif args.split == "valid":
        logger.info(
            "Evaluating with bsz {} tgt_len {} ext_len {} mem_len {} clamp_len {}".format(
                args.batch_size, args.eval_tgt_len, args.ext_len, args.eval_mem_len, args.clamp_len
            )
        )

        model.module.reset_memory_length(args.eval_mem_len)
        valid_loss = evaluate(va_iter)
        test_loss = None
    elif args.split == "test":
        logger.info(
            "Evaluating with bsz {} tgt_len {} ext_len {} mem_len {} clamp_len {}".format(
                args.batch_size, args.eval_tgt_len, args.ext_len, args.eval_mem_len, args.clamp_len
            )
        )

        model.module.reset_memory_length(args.eval_mem_len)
        test_loss = evaluate(te_iter)
        valid_loss = None

    def format_log(loss, split):
        log_str = "| {0} loss {1:5.2f} | {0} ppl {2:9.3f} ".format(split, loss, math.exp(loss))
        return log_str

    log_str = ""
    if valid_loss is not None:
        log_str += format_log(valid_loss, "valid")
    if test_loss is not None:
        log_str += format_log(test_loss, "test")

    logger.info("=" * 100)
    logger.info(log_str)
    logger.info("=" * 100)


if __name__ == "__main__":
    main()
