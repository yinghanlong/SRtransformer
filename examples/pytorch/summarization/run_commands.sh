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
python examples/pytorch/summarization/run_summarization_no_trainer.py     --model_name_or_path ./results/t5-small/2023-03-raf64/pytorch_model.bin  --config_name ./results/t5-small/2023-03-raf64/config.json --tokenizer_name=t5-small --dataset_name ccdv/cnn_dailymail     --dataset_config "3.0.0"   --source_prefix "summarize: "     --output_dir ./results/t5-small/     --per_device_train_batch_size=16     --per_device_eval_batch_size=16     --num_train_epochs=15  --gpus=2 

python examples/pytorch/summarization/run_summarization_no_trainer.py     --model_name_or_path ./results/t5-small/cross-attn-base64/pytorch_model.bin --config_name ./results/t5-small/cross-attn-base64/config.json --tokenizer_name=t5-small    --dataset_name ccdv/cnn_dailymail     --dataset_config "3.0.0"   --source_prefix "summarize: "     --output_dir ./results/t5-small/     --per_device_train_batch_size=16     --per_device_eval_batch_size=16     --num_train_epochs=12  --gpus=1 --evaluation_only=True
#T5-base
python examples/pytorch/summarization/run_summarization_no_trainer.py     --model_name_or_path t5-base   --dataset_name ccdv/cnn_dailymail     --dataset_config "3.0.0"   --source_prefix "summarize: "     --output_dir ./results/t5-base/     --per_device_train_batch_size=4     --per_device_eval_batch_size=4     --num_train_epochs=25  --gpus=3

python examples/pytorch/summarization/run_summarization_no_trainer.py     --model_name_or_path t5-small   --dataset_name ccdv/cnn_dailymail     --dataset_config "3.0.0"   --source_prefix "summarize: "     --output_dir ./results/t5-small/     --per_device_train_batch_size=16     --per_device_eval_batch_size=16    --num_train_epochs=25  --gpus=3
#Run on other datasets
python examples/pytorch/summarization/run_summarization_no_trainer.py     --model_name_or_path ./results/t5-small/2023-03-raf16/pytorch_model.bin  --config_name ./results/t5-small/2023-03-raf16/config.json --tokenizer_name=t5-small --dataset_name xsum      --source_prefix "summarize: "     --output_dir ./results/t5-small/     --per_device_train_batch_size=16     --per_device_eval_batch_size=16     --num_train_epochs=25  --gpus=3 
python examples/pytorch/summarization/run_summarization_no_trainer.py     --model_name_or_path ./results/t5-small/2023-03-raf16/pytorch_model.bin  --config_name ./results/t5-small/2023-03-raf16/config.json --tokenizer_name=t5-small --dataset_name ./datasets/gigaword  --source_prefix "summarize: "     --output_dir ./results/t5-small/     --per_device_train_batch_size=16     --per_device_eval_batch_size=16     --num_train_epochs=25  --gpus=3 
python examples/pytorch/summarization/run_summarization_no_trainer.py     --model_name_or_path ./results/t5-small/2023-03-raf16/pytorch_model.bin  --config_name ./results/t5-small/2023-03-raf16/config.json --tokenizer_name=t5-small --dataset_name amazon_reviews_multi     --source_prefix "summarize: "     --output_dir ./results/t5-small/     --per_device_train_batch_size=16     --per_device_eval_batch_size=16     --num_train_epochs=25  --gpus=3 

python examples/pytorch/summarization/run_summarization_no_trainer.py     --model_name_or_path ./results/t5-small/2023-03-mediasum/pytorch_model.bin  --config_name ./results/t5-small/2023-03-mediasum/config.json --tokenizer_name=t5-small --dataset_name ccdv/mediasum     --source_prefix "summarize: "     --output_dir ./results/t5-small/     --per_device_train_batch_size=16     --per_device_eval_batch_size=16     --num_train_epochs=15  --gpus=3 
python examples/pytorch/summarization/run_summarization_no_trainer.py     --model_name_or_path ./results/t5-small/2023-03-arxiv/pytorch_model.bin  --config_name ./results/t5-small/2023-03-arxiv/config.json --tokenizer_name=t5-small --dataset_name ccdv/arxiv-summarization     --source_prefix "summarize: "     --output_dir ./results/t5-small/     --per_device_train_batch_size=16     --per_device_eval_batch_size=16     --num_train_epochs=10  --gpus=3 

python examples/pytorch/summarization/run_summarization_no_trainer.py     --model_name_or_path t5-small --dataset_name ccdv/mediasum     --source_prefix "summarize: "     --output_dir ./results/t5-small/     --per_device_train_batch_size=16     --per_device_eval_batch_size=16     --num_train_epochs=25  --gpus=3 
#run with Bart
python examples/pytorch/summarization/run_summarization_no_trainer.py     --model_name_or_path ainize/bart-base-cnn --dataset_name ccdv/cnn_dailymail  --dataset_config "3.0.0"    --source_prefix "summarization"     --output_dir ./results/bart/     --per_device_train_batch_size=8     --per_device_eval_batch_size=8     --num_train_epochs=25  --gpus=3 