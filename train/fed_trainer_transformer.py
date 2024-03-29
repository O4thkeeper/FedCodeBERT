import logging

from train.base.model_trainer import ModelTrainer


class FedTransformerTrainer(ModelTrainer):

    def __init__(self, trainer, model):
        super().__init__(model)
        self.model_trainer = trainer
        self.model = model

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args=None):
        logging.info("Client(%d)" % self.id + ":| Local Train Data Size = %d" % (len(train_data)))
        self.model_trainer.train_dl = train_data
        self.model_trainer.train_model(device=device)
        logging.info("Client %d train model finished" % self.id)

    def test(self, test_data, device, args=None):
        pass

    def test_on_the_server(self, device):
        self.model_trainer.eval_model(device=device)
        return True
