for suffix in notitledup_tfidfkws notitledup_tfidfkws_to_comment_noised
do
env CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=13243  finetune_trainer.py \
    --data_dir=./data/ \
    --train_name=train_$suffix \
    --val_name=val_$suffix \
    --test_name=test_$suffix \
    --output_dir=./cg_base_$suffix \
    --save_total_limit=5 \
    --per_gpu_train_batch_size=4 \
    --per_gpu_eval_batch_size=10 \
    --gradient_accumulation_steps=8 \
    --num_train_epochs=30 \
    --logging_steps=50 \
    --model_name_or_path=thu-coai/LongLM-base \
    --learning_rate=3e-5 \
    --n_val=-1 \
    --fp16 \
    --do_train --do_eval \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --overwrite_output_dir \
    --load_best_model_at_end
done
