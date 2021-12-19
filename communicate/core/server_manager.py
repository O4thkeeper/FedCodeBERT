import logging
from abc import abstractmethod

from mpi4py import MPI

from communicate.core.base.observer import Observer
from communicate.core.mpi_com_manager import MpiCommunicationManager


class ServerManager(Observer):

    def __init__(self, args, comm=None, rank=0, size=0, backend="MPI"):
        self.args = args
        self.size = size
        self.rank = rank

        self.backend = backend
        if backend == "MPI":
            self.com_manager = MpiCommunicationManager(comm, rank, size, node_type="server")
        # elif backend == "MQTT":
        #     HOST = "0.0.0.0"
        #     # HOST = "broker.emqx.io"
        #     PORT = 1883
        #     self.com_manager = MqttCommManager(HOST, PORT, client_id=rank, client_num=size - 1)
        # elif backend == "GRPC":
        #     HOST = "0.0.0.0"
        #     PORT = 50000 + rank
        #     self.com_manager = GRPCCommManager(HOST, PORT, ip_config_path=args.grpc_ipconfig_path, client_id=rank, client_num=size - 1)
        # else:
        #     self.com_manager = MpiCommunicationManager(comm, rank, size, node_type="server")
        self.com_manager.add_observer(self)
        self.message_handler_dict = dict()

    def run(self):
        self.register_message_receive_handlers()
        self.com_manager.handle_receive_message()
        print('done running')

    def get_sender_id(self):
        return self.rank

    def receive_message(self, msg_type, msg_params) -> None:
        # logging.info("receive_message. rank_id = %d, msg_type = %s. msg_params = %s" % (
        #     self.rank, str(msg_type), str(msg_params.get_content())))
        handler_callback_func = self.message_handler_dict[msg_type]
        handler_callback_func(msg_params)

    def send_message(self, message):
        self.com_manager.send_message(message)

    @abstractmethod
    def register_message_receive_handlers(self) -> None:
        pass

    def register_message_receive_handler(self, msg_type, handler_callback_func):
        self.message_handler_dict[msg_type] = handler_callback_func

    def finish(self):
        logging.info("__finish server")
        if self.backend == "MPI":
            MPI.COMM_WORLD.Abort()
        elif self.backend == "MQTT":
            self.com_manager.stop_receive_message()
        elif self.backend == "GRPC":
            self.com_manager.stop_receive_message()
