<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# About the project
This repository includes the codes of SRformer: [Segmented Recurrent Transformer](https://arxiv.org/abs/2305.16340)

This project is built on 🤗 Transformers. We modified T5 and BART models for text summarization. They are both encoder-decoder models and we use the proposed segmented recurrent attention in their decoders' cross attention blocks. Please see [huggingface transformers](https://github.com/huggingface/transformers) for general instructions.

## Get started

In ```/examples/pytorch/summarization/```, you can find run_commands.sh that lists example commands.
For example, to run T5-small on CNN-Dailymail,
```
$ python examples/pytorch/summarization/run_summarization_no_trainer.py     --model_name_or_path t5-small   --dataset_name ccdv/cnn_dailymail     --dataset_config "3.0.0"   --source_prefix "summarize: "     --output_dir ./results/t5-base/     --per_device_train_batch_size=16     --per_device_eval_batch_size=16     --num_train_epochs=25  --gpus=3
```
To use a pretrained BART model, set model_name_or_path to ainize/bart-base-cnn.
To run on other datasets, change dataset_name to XSUM, ccdv/arxiv-summarization, or ccdv/mediasum.

In run_summarization_no_trainer.py, you can change ```cache_dir``` to your own directory. It is where the dataset will be downloaded to.

After training your own model, you can load the model by setting model_name_or_path to the local path. Please remember to set config_name and tokenizer_name. If you only want to evaluate the model, set --evaluation_only=True.

Some configurations such as the segment size can be set in the source file of models such as```src/transformers/models/t5/modeling_t5.py```. We did not add those parameters to a specific configuration file so pretrained models and their configuration file can be loaded directly.


## Citation

If you use this repo, please use the following citation:
@misc{long2023segmented,
      title={Segmented Recurrent Transformer: An Efficient Sequence-to-Sequence Model}, 
      author={Yinghan Long and Sayeed Shafayet Chowdhury and Kaushik Roy},
      year={2023},
      eprint={2305.16340},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
