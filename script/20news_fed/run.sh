py=/Users/fenghao/Documents/pythonWork/venv/bin/python
#py=python

mpirun -np 11 --allow-run-as-root \
${py} -m main.main_20news \
  --gpu_mapping_file "gpu/gpu_mapping.yaml" \
  --gpu_mapping_key test \
  --client_num_per_round 2 \
  --comm_round 10 \
  --ci 0 \
  --dataset "20news" \
  --data_file "data/store/20news/20news_data.h5" \
  --partition_file "data/store/20news/20news_partition.h5" \
  --partition_method niid_quantity_clients=10_beta=5.0 \
  --fl_algorithm FedAvg \
  --model_type distilbert \
  --model_name distilbert-base-uncased \
  --do_lower_case True \
  --train_batch_size 32 \
  --eval_batch_size 8 \
  --max_seq_length 256 \
  --lr 5e-5 \
  --server_lr 0.1 \
  --epochs 1 \
  --output_dir "tmp/fedavg_20news_output/"