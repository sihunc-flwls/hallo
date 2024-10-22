# -------------------------------------
# pip install .
# -------------------------------------


# echo "stage1 w/ deepspeed"
# accelerate launch -m \
#   --config_file accelerate_config.yaml \
#   --machine_rank 0 \
#   --main_process_ip 0.0.0.0 \
#   --main_process_port 20055 \
#   --num_machines 1 \
#   --num_processes 1 \
#   scripts.train_stage2_hallo --config ./configs/train/hallo/stage2.yaml
# accelerate launch -m \
#   --config_file accelerate_config.yaml \
#   --machine_rank 0 \
#   --main_process_ip 0.0.0.0 \
#   --main_process_port 20055 \
#   --num_machines 1 \
#   --num_processes 1 \
#   scripts.train_stage3_emo --config ./configs/train/emo/stage3.yaml

# echo "stage1 w/o deepspeed"
# accelerate launch -m \
#   --config_file _default_config.yaml \
#   --machine_rank 0 \
#   --main_process_ip 0.0.0.0 \
#   --main_process_port 20055 \
#   --num_machines 1 \
#   --num_processes 1 \
#   scripts.train_stage1 --config ./configs/train/stage1.yaml

# accelerate launch -m scripts.train_stage1 --config ./configs/train/stage1.yaml

# echo "stage2 w/ deepspeed"
# accelerate launch -m \
#   --config_file accelerate_config.yaml \
#   --machine_rank 0 \
#   --main_process_ip 0.0.0.0 \
#   --main_process_port 20055 \
#   --num_machines 1 \
#   --num_processes 1 \
#   scripts.train_stage2 --config ./configs/train/stage2.yaml

# accelerate launch -m scripts.train_stage1_hallo --config ./configs/train/hallo/stage1.yaml
# echo "emo stage 1"
# accelerate launch -m scripts.train_stage1_emo --config ./configs/train/emo/stage1.yaml

# python -m scripts.train_stage1_emo --config ./configs/train/emo/stage1.yaml
# echo "stage1_emo w/ deepspeed"
# accelerate launch -m \
#   --config_file accelerate_config.yaml \
#   --machine_rank 0 \
#   --main_process_ip 0.0.0.0 \
#   --main_process_port 20055 \
#   --num_machines 1 \
#   --num_processes 1 \
#   scripts.train_stage1_emo --config ./configs/train/emo/stage1.yaml

# echo "stage2_emo w/ deepspeed"
# accelerate launch -m \
#   --config_file accelerate_config.yaml \
#   --machine_rank 0 \
#   --main_process_ip 0.0.0.0 \
#   --main_process_port 20055 \
#   --num_machines 1 \
#   --num_processes 1 \
#   scripts.train_stage2_emo --config ./configs/train/stage2.yaml
# echo "stage2_emo"
# python -m scripts.train_stage2_emo --config ./configs/train/stage2.yaml
# accelerate launch -m scripts.train_stage2_emo --config ./configs/train/stage2.yaml
# python -m scripts.train_stage2_hallo --config ./configs/train/stage2.yaml


# echo "stage3_emo"
# python -m scripts.train_stage3_emo --config ./configs/train/stage3.yaml
# echo "stage3_emo w/ deepspeed"
# accelerate launch -m \
#   --config_file accelerate_config.yaml \
#   --machine_rank 0 \
#   --main_process_ip 0.0.0.0 \
#   --main_process_port 20055 \
#   --num_machines 1 \
#   --num_processes 1 \
#   scripts.train_stage3_emo --config ./configs/train/stage3.yaml

echo "hallo + LoRA (MotionDirector)"
accelerate launch -m \
  --config_file accelerate_config.yaml \
  --machine_rank 0 \
  --main_process_ip 0.0.0.0 \
  --main_process_port 20055 \
  --num_machines 1 \
  --num_processes 1 \
  scripts.train_stage_exp_lora --config ./configs/train/exp/stage2.yaml