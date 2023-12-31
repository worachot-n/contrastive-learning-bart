echo "= = = = = = = = = = = = = ="
echo "The project is running..."
currentDate=`date`
echo $currentDate
start=`date +%s`
echo "= = = = = = = = = = = = = ="

python3 train.py \
    --topic_prompt_input True \
    --length_prompt_input True \
    --output_dir ./output/bart-topic-length-prompt-contrastive-combine-negative \
    --train_file ./data/dialogtest/dialogsum.train.jsonl \
    --validation_file ./data/dialogtest/dialogsum.dev.jsonl \
    --test_file ./data/dialogtest/dialogsum.test.jsonl \
    --text_column prompt \
    --summary_column summary \
    --model_name_or_path facebook/bart-large \
    --model_type bart \
    --max_source_length 1024 \
    --min_target_length 1 \
    --max_target_length 128 \
    --num_beams 4 \
    --learning_rate 5e-5 \
    --weight_decay 1e-3 \
    --label_smoothing 0.1 \
    --length_penalty 1.0 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --per_device_eval_batch_size 8 \
    --per_device_test_batch_size 8 \
    --num_warmup_steps 0 \
    --cache_dir ./output/cache \
    --overwrite_cache True \
    --seed 12345 \
    --contrastive_loss True \
    --tagging no \
    --synonym_replacement True \
    --random_topic True \

echo "= = = = = = = = = = = = = ="
echo "The project is Finished..."
end=`date +%s`
runtime=$((end-start))
echo "The program takes '$((runtime / 60))' minutes."
echo "= = = = = = = = = = = = = ="