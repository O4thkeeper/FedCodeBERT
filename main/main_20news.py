import argparse
import logging
import os
import socket
import sys

import psutil
import setproctitle
import torch
import wandb

from communicate.api.fedavg_api import FedML_init
from communicate.api.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file
from data.manager.base.base_data_manager import BaseDataManager
from data.manager.text_classification_data_manager import TextClassificationDataManager
from data.preprocess.text_classification_preprocessor import TLMPreprocessor
from main.initialize import set_seed, create_model, add_federated_args, get_fl_algorithm_initializer
from model.model_args import ClassificationArgs
from train.fed_trainer_transformer import FedTransformerTrainer
from train.tc_transformer_trainer import TextClassificationTrainer

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))


def post_complete_message(tc_args):
    pipe_path = "/tmp/fednlp_tc"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, 'w') as pipe:
        pipe.write("training is finished! \n%s" % (str(tc_args)))


if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    parser = add_federated_args(parser)
    args = parser.parse_args()

    # customize the log format
    logging.basicConfig(
        level=logging.INFO,
        format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S')
    logging.info(args)

    set_seed(args.manual_seed)

    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    # customize the process name
    str_process_name = "FedNLP-" + str(args.dataset) + ":" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    # todo wandb init
    # if process_id == 0:
    # initialize the wandb machine learning experimental tracking platform (https://wandb.ai/automl/fednlp).
    # wandb.init(project="fednlp", entity="automl", name="FedNLP-" + str(args.fl_algorithm) +
    #                                                    "-TC-" + str(args.dataset) + "-" + str(
    #     args.model_name) + "-freeze-" + args.freeze_layers if args.freeze_layers else "",
    #            config=args)

    # device: check "gpu_mapping.yaml" to see how to define the topology
    device = mapping_processes_to_gpu_device_from_yaml_file(process_id, worker_number, args.gpu_mapping_file,
                                                            args.gpu_mapping_key)
    logging.info("process_id = %d, size = %d, device=%s" % (process_id, worker_number, str(device)))
    logging.info("torch.cuda.current_device()=" + str(torch.cuda.current_device()))
    logging.info("torch.cuda.device_count()=" + str(torch.cuda.device_count()))

    # dataset attributes
    attributes = BaseDataManager.load_attributes(args.data_file_path)
    num_labels = len(attributes["label_vocab"])

    # create the model
    model_args = ClassificationArgs()
    model_args.model_name = args.model_name
    model_args.model_type = args.model_type
    model_args.load(model_args.model_name)
    model_args.num_labels = num_labels
    model_args.update_from_dict({"fl_algorithm": args.fl_algorithm,
                                 "freeze_layers": args.freeze_layers,
                                 "epochs": args.epochs,
                                 "learning_rate": args.lr,
                                 "gradient_accumulation_steps": args.gradient_accumulation_steps,
                                 "do_lower_case": args.do_lower_case,
                                 "manual_seed": args.manual_seed,
                                 # for ignoring the cache features.
                                 "reprocess_input_data": args.reprocess_input_data,
                                 "overwrite_output_dir": True,
                                 "max_seq_length": args.max_seq_length,
                                 "train_batch_size": args.train_batch_size,
                                 "eval_batch_size": args.eval_batch_size,
                                 "evaluate_during_training": False,  # Disabled for FedAvg.
                                 "evaluate_during_training_steps": args.evaluate_during_training_steps,
                                 "fp16": args.fp16,
                                 "data_file_path": args.data_file_path,
                                 "partition_file_path": args.partition_file_path,
                                 "partition_method": args.partition_method,
                                 "dataset": args.dataset,
                                 "output_dir": args.output_dir,
                                 "is_debug_mode": args.is_debug_mode,
                                 "fedprox_mu": args.fedprox_mu
                                 })
    model_args.config["num_labels"] = num_labels
    model_config, client_model, tokenizer = create_model(model_args, formulation="classification")
    logging.info("process %d :model created" % process_id)

    # trainer
    client_trainer = TextClassificationTrainer(model_args, device, client_model, None, None)
    fed_trainer = FedTransformerTrainer(client_trainer, client_model)
    logging.info("process %d :trainer created" % process_id)

    # data manager
    preprocessor = TLMPreprocessor(args=model_args, label_vocab=attributes["label_vocab"], tokenizer=tokenizer)
    dm = TextClassificationDataManager(args, model_args, preprocessor, process_id, args.client_num_per_round)
    # train_data_num, train_data_global, test_data_global, train_data_local_num_dict, \
    # train_data_local_dict, test_data_local_dict, num_clients = dm.load_federated_data(process_id=process_id)
    # train_data_num 训练资料数
    # train_data_global 训练资料BaseDataLoader
    # test_data_global 测试资料数
    # train_data_local_num_dict: none if server else dict{client id: length of BaseDataLoader}
    # train_data_local_dict: none if server else dict{client id: BaseDataLoader}
    # test_data_local_dict: none if server else dict{client id: BaseDataLoader}
    # num_clients: 10
    logging.info("process %d :data manager created" % process_id)
    fl_algorithm = get_fl_algorithm_initializer(args.fl_algorithm)
    args.client_num_in_total = worker_number - 1
    train_loader, train_data_num, test_loader = dm.load_federated_data(process_id)
    fl_algorithm(process_id, worker_number, device, comm, train_loader, train_data_num, test_loader, fed_trainer, args)

    # start FedAvg algorithm
    # for distributed algorithm, train_data_gloabl and test_data_global are required
    # if process_id == 0:
    #     client_trainer.test_dl = test_data_global
    # args.client_num_in_total = num_clients
    #
    # # fl_algorithm = get_fl_algorithm_initializer(args.fl_algorithm)
    # # logging.info("process %d :fl method created" % process_id)
    # # todo bind data to client
    # fl_algorithm(process_id, worker_number, device, comm, client_model, train_data_num,
    #              train_data_global, test_data_global, train_data_local_num_dict,
    #              train_data_local_dict, test_data_local_dict, args, fed_trainer)
