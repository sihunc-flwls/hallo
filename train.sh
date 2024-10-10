accelerate launch -m \
  --config_file accelerate_config.yaml \
  --machine_rank 0 \
  --main_process_ip 0.0.0.0 \
  --main_process_port 20055 \
  --num_machines 1 \
  --num_processes 1 \
  scripts.train_stage1 --config ./configs/train/stage1.yaml