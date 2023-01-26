python run_summarization_no_trainer.py \
    --model_name_or_path t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ~/tmp/tst-summarization
    --per_device_train_batch_size 16
    --per_device_eval_batch_size 16

accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path t5-base \
    --config_name config.json \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ./results/t5-base/ \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --num_train_epochs=10 

#evaluation
python run_summarization_no_trainer.py \
    --model_name_or_path ./tmp/1-layer/pytorch_model.bin \
    --config_name config.json \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ./t5-sum/ \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --evaluation_only=true \
    --tokenizer_name=t5-small

python examples/pytorch/summarization/run_summarization_no_trainer.py     --model_name_or_path t5-base     --dataset_name cnn_dailymail     --dataset_config "3.0.0"     --source_prefix "summarize: "     --output_dir ./t5-base/     --per_device_train_batch_size=16     --per_device_eval_batch_size=16     --num_train_epochs=10

python examples/pytorch/summarization/run_summarization_no_trainer.py  \
   --model_name_or_path ./results/t5-base/2021-11-18/pytorch_model.bin \
     --config_name ./results/t5-base/2021-11-18/config.json   \
         --tokenizer_name=t5-base --dataset_name cnn_dailymail     --dataset_config "3.0.0"  \
            --source_prefix "summarize: "     --output_dir ./results/t5-base/     --per_device_train_batch_size=8     --per_device_eval_batch_size=8     --num_train_epochs=10
#evaluate only
python examples/pytorch/summarization/run_summarization_no_trainer.py     --model_name_or_path ./results/t5-base/2021-11-30/pytorch_model.bin      --config_name ./results/t5-base/2021-11-30/config.json   --dataset_name cnn_dailymail     --dataset_config "3.0.0"              --source_prefix "summarize: "     --output_dir ./results/t5-base/     --per_device_train_batch_size=8     --per_device_eval_batch_size=8     --num_train_epochs=10 --tokenizer_name=t5-base --evaluation_only=True --gpus=1

ssh long273@cbric-gpu18.ecn.purdue.edu
cd Documents/transformer-spiking/
source env/bin/activate
python examples/pytorch/summarization/run_summarization_no_trainer.py     --model_name_or_path ./results/t5-base/direct-connect/pytorch_model.bin      --config_name ./results/t5-base/direct-connect/config.json   --dataset_name cnn_dailymail     --dataset_config "3.0.0"              --source_prefix "summarize: "     --output_dir ./results/t5-base/     --per_device_train_batch_size=8     --per_device_eval_batch_size=8     --num_train_epochs=10 --tokenizer_name=t5-base  --gpus=3 
#note: set use_mem=True, reset, and connect_type

github token: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
gpustat -up

python examples/pytorch/summarization/run_summarization_no_trainer.py  \
   --model_name_or_path t5-small \
   --dataset_name cnn_dailymail     --dataset_config "3.0.0"  \
 --source_prefix "summarize: "     --output_dir ./results/t5-small/     --per_device_train_batch_size=16     --per_device_eval_batch_size=16     --num_train_epochs=10 \
 --gpus =2

python examples/pytorch/summarization/run_summarization_no_trainer.py     --model_name_or_path ./results/t5-small/2023-01-raf-8/pytorch_model.bin  --config_name ./results/t5-small/2023-01-raf-8/config.json --tokenizer_name=t5-small --dataset_name ccdv/cnn_dailymail     --dataset_config "3.0.0"   --source_prefix "summarize: "     --output_dir ./results/t5-small/     --per_device_train_batch_size=8     --per_device_eval_batch_size=8     --num_train_epochs=15  --gpus=3 --evaluation_only=True
python examples/pytorch/summarization/run_summarization_no_trainer.py     --model_name_or_path ./results/t5-small/2023-01-base16/pytorch_model.bin  --config_name ./results/t5-small/2023-01-base16/config.json --tokenizer_name=t5-small --dataset_name ccdv/cnn_dailymail     --dataset_config "3.0.0"   --source_prefix "summarize: "     --output_dir ./results/t5-small/     --per_device_train_batch_size=8     --per_device_eval_batch_size=8     --num_train_epochs=15  --gpus=2 

python examples/pytorch/summarization/run_summarization_no_trainer.py     --model_name_or_path ./results/t5-small/cross-attn-base64/pytorch_model.bin --config_name ./results/t5-small/cross-attn-base64/config.json --tokenizer_name=t5-small    --dataset_name ccdv/cnn_dailymail     --dataset_config "3.0.0"   --source_prefix "summarize: "     --output_dir ./results/t5-small/     --per_device_train_batch_size=16     --per_device_eval_batch_size=16     --num_train_epochs=12  --gpus=1 --evaluation_only=True