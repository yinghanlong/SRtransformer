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

python examples/pytorch/summarization/run_summarization_no_trainer.py     --model_name_or_path ./t5-base/2021-11-11/pytorch_model.bin   --config_name ./t5-base/2021-11-11/config.json       --tokenizer_name=t5-base --dataset_name cnn_dailymail     --dataset_config "3.0.0"     --source_prefix "summarize: "     --output_dir ./results/t5-base/     --per_device_train_batch_size=8     --per_device_eval_batch_size=8     --num_train_epochs=10