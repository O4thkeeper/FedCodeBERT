import logging
import os
import sys

from communicate.api.my_message import MyMessage
from communicate.api.utils import transform_list_to_tensor, post_complete_message_to_sweep_process
from communicate.core.base.message import Message
from communicate.core.client_manager import ClientManager


class FedAVGClientManager(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        # self.round_idx = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        # self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
        #                                       self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_FINISH, self.handle_finish)

    # def handle_message_init(self, msg_params):
    #     global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
    #     client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
    #
    #     if self.args.is_mobile == 1:
    #         global_model_params = transform_list_to_tensor(global_model_params)
    #
    #     self.trainer.update_model(global_model_params)
    #     # todo update maybe unnecessary
    #     #self.trainer.update_dataset(int(client_index))
    #     self.__train()

    # def start_training(self):
    #     self.round_idx = 0
    #     self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.trainer.update_model(model_params)
        # todo update maybe unnecessary
        # self.trainer.update_dataset(int(client_index))
        # todo round_idx update change
        # self.round_idx += 1
        self.__train()
        # if self.round_idx == self.num_rounds:
        #     post_complete_message_to_sweep_process(self.args)
        #     self.finish()

    def handle_finish(self, msg_params):
        post_complete_message_to_sweep_process(self.args)
        self.finish()

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

    def __train(self):
        # logging.info("#######training########### round_id = %d" % self.round_idx)
        weights, local_sample_num = self.trainer.train()
        logging.info("client %d send to server" % self.rank)
        self.send_model_to_server(0, weights, local_sample_num)
