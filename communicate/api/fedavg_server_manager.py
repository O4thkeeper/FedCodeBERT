import logging
import os
import sys

from communicate.api.my_message import MyMessage
from communicate.api.utils import transform_tensor_to_list, post_complete_message_to_sweep_process
from communicate.core.base.message import Message
from communicate.core.server_manager import ServerManager


class FedAVGServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI", is_preprocessed=False,
                 preprocessed_client_lists=None):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists

    def run(self):
        super().run()

    def send_first_msg(self):
        logging.info("********first round********")
        # sampling clients
        client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                         self.args.client_num_per_round)
        global_model_params = self.aggregator.get_global_model_params()
        for client_index in client_indexes:
            self.send_message_sync_model_to_client(client_index + 1, global_model_params, client_index)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                              self.handle_message_receive_model_from_client)

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number)
        b_all_received = self.aggregator.check_whether_all_receive(self.args.client_num_per_round)
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            global_model_params = self.aggregator.aggregate()
            self.aggregator.test_on_server_for_all_clients(self.round_idx)

            # start the next round
            self.round_idx += 1
            if self.round_idx == self.round_num:
                for client in range(self.args.client_num_in_total):
                    self.send_message_finish_to_client(client + 1)
                post_complete_message_to_sweep_process(self.args)
                self.finish()
                logging.info("********finish training*********")
                return
            if self.is_preprocessed:
                if self.preprocessed_client_lists is None:
                    # sampling has already been done in data preprocessor
                    client_indexes = [self.round_idx] * self.args.client_num_per_round
                else:
                    client_indexes = self.preprocessed_client_lists[self.round_idx]
            else:
                # sampling clients
                client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                                 self.args.client_num_per_round)

            logging.info("********round %d begin********" % self.round_idx)
            logging.info('indexes of clients: ' + str(client_indexes))

            # todo fix bug
            for client_index in client_indexes:
                self.send_message_sync_model_to_client(client_index + 1, global_model_params, client_index)

    # def send_message_init_config(self, receive_id, global_model_params, client_index):
    #     logging.info("init: %s"%str(client_index))
    #     message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
    #     message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
    #     message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
    #     self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index):
        logging.info("sync: %s" % str(client_index))
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_finish_to_client(self, receive_id):
        logging.info("finish: %s" % str(receive_id))
        message = Message(MyMessage.MSG_TYPE_S2C_FINISH, self.get_sender_id(), receive_id)
        self.send_message(message)
