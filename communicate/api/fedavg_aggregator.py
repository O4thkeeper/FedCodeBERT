import copy
import logging
import os.path
import pickle
import random
import time

import numpy as np
import torch
import wandb

from communicate.api.utils import transform_list_to_tensor


class FedAVGAggregator(object):

    def __init__(self, worker_num, device, model_trainer, args):
        self.trainer = model_trainer

        self.args = args
        # self.val_global = self._generate_validation_set()

        self.worker_num = worker_num
        self.device = device
        # self.model_dict = dict()
        self.model_path_list = list()
        self.sample_num_list = list()
        self.receive_num = 0
        # self.flag_client_model_uploaded_dict = dict()
        # for idx in range(self.worker_num):
        #     self.flag_client_model_uploaded_dict[idx] = False

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        filename = os.path.join('cache', str(time.time()))
        with open(filename, 'wb') as f:
            pickle.dump(model_params, f)
            self.model_path_list.append(filename)
        self.sample_num_list.append(sample_num)
        self.receive_num += 1

    def check_whether_all_receive(self, client_num):
        if client_num == self.receive_num:
            self.receive_num = 0
            return True
        return False

    def aggregate(self):
        start_time = time.time()
        training_num = sum(self.sample_num_list)

        logging.info("len of self.model_path_list = " + str(len(self.model_path_list)))

        with open(self.model_path_list[0], 'rb') as f:
            averaged_params = pickle.load(f)
        for i in range(0, len(self.model_path_list)):
            local_sample_number = self.sample_num_list[i]
            with open(self.model_path_list[i], 'rb') as f:
                local_model_params = pickle.load(f)
            w = local_sample_number / training_num
            for k in averaged_params.keys():
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def test_on_server_for_all_clients(self, round_idx):
        if not self.trainer.test_on_the_server(self.device):
            logging.info("round %d not tested all" % round_idx)
        return
