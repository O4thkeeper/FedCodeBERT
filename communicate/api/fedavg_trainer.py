from .utils import transform_tensor_to_list


class FedAVGTrainer(object):

    def __init__(self, client_index, train_loader, test_loader, train_data_num, device, args, model_trainer):
        self.trainer = model_trainer
        # todo all data maybe not necessary to store
        # solved
        self.client_index = client_index
        self.train_local = train_loader
        self.local_sample_number = train_data_num
        self.test_local = test_loader

        self.device = device
        self.args = args

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    # def update_dataset(self, client_index):
    #     self.client_index = client_index
    #     self.train_local = self.train_data_local_dict[client_index]
    #     self.local_sample_number = self.train_data_local_num_dict[client_index]
    #     self.test_local = self.test_data_local_dict[client_index]

    def train(self, round_idx=None):
        # self.args.round_idx = round_idx
        self.trainer.train(self.train_local, self.device, self.args)

        weights = self.trainer.get_model_params()

        return weights, self.local_sample_number

    def test(self):
        # train data
        train_metrics = self.trainer.test(self.train_local, self.device, self.args)
        train_tot_correct, train_num_sample, train_loss = train_metrics['test_correct'], \
                                                          train_metrics['test_total'], train_metrics['test_loss']

        # test data
        test_metrics = self.trainer.test(self.test_local, self.device, self.args)
        test_tot_correct, test_num_sample, test_loss = test_metrics['test_correct'], \
                                                       test_metrics['test_total'], test_metrics['test_loss']

        return train_tot_correct, train_loss, train_num_sample, test_tot_correct, test_loss, test_num_sample
