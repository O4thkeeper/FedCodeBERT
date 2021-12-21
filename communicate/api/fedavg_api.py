from mpi4py import MPI

from communicate.api.fedavg_aggregator import FedAVGAggregator
from communicate.api.fedavg_client_manager import FedAVGClientManager
from communicate.api.fedavg_server_manager import FedAVGServerManager
from communicate.api.fedavg_trainer import FedAVGTrainer


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_FedAvg_distributed(process_id, worker_number, device, comm, train_loader, train_data_num, test_loader,
                             model_trainer, args):
    if process_id == 0:
        init_server(args, device, comm, 0, worker_number, train_data_num, train_loader, test_loader, model_trainer)
    else:
        init_client(args, device, comm, process_id, worker_number, train_data_num, train_loader, test_loader,
                    model_trainer)


def init_server(args, device, comm, rank, size, train_data_num, train_loader, test_loader, model_trainer):
    model_trainer.set_id(-1)
    model_trainer.model_trainer.set_data(train_loader, test_loader)
    # aggregator
    worker_num = size - 1
    aggregator = FedAVGAggregator(worker_num, device, model_trainer, args)

    # start the distributed training
    backend = args.backend
    server_manager = FedAVGServerManager(args, aggregator, comm, rank, size, backend)
    server_manager.send_first_msg()
    server_manager.run()


def init_client(args, device, comm, process_id, size, train_data_num, train_loader, test_loader, model_trainer):
    client_index = process_id - 1
    model_trainer.set_id(client_index)
    backend = args.backend
    trainer = FedAVGTrainer(client_index, train_loader, test_loader, train_data_num, device, args, model_trainer)
    client_manager = FedAVGClientManager(args, trainer, comm, process_id, size, backend)
    client_manager.run()
