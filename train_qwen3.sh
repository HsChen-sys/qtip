torchrun --nproc_per_node=1  input_hessian_llama.py  --base_model Qwen/Qwen3-1.7B --save_path hessians/qwen3_1_7b --sample_proc 40 --devset_size 8192
python quantize_finetune_llama.py \
       --save_path ckpt/q3_1_7b_4bit \
       --codebook bitshift \
       --base_model Qwen/Qwen3-1.7B \
       --in_hess_path hessians/qwen3_1_7b \
       --scale_override 0.9 \
       --ft_epochs 5 \
       --td_x 16 \
       --td_y 16 \
       --L 16 \
       --K 4 \
       --V 2 \
       --decode_mode quantlut_sym \
       --tlut_bits 9 

python hfize_llama.py --quantized_path ckpt/q3_1_7b_4bit --hf_output_path hf/q3_1_7b_4bit

# todo: fix distributed training issue
python finetune_e2e_llama.py --base_model Qwen/Qwen3-1.7B --hf_path hf/q3_1_7b_4bit --devset_size 640 --ft_valid_size 128 --ft_epochs 4 --ft_update_freq 4 --ft_bs 2 --ctx_size 4096 --ft_train_lut --hf_output_path hf/q3_1_7b_4bit_QTIP 
python interactive_gen.py --hf_path hf/q3_1_7b_4bit_QTIP --empty_model  --bench_model

