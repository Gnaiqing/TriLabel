import numpy as np
import torch


class NeuralNetworkTrainer:
    def __init__(self, model,
                 num_training_iterations=1000,
                 lr=1e-2,
                 batch_size=256):
        self.model = model.train()
        self.which_loss = None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.num_iterations = num_training_iterations
        self.batch_size = batch_size
        # Early stopping to avoid overfitting
        self.early_stopping_patience = None

    def compute_loss(self, network_output, y_data, which_loss):
        if which_loss == 'cross-entropy':
            loss_func = torch.nn.CrossEntropyLoss()
            loss = loss_func(network_output, y_data)
        else:
            raise NotImplementedError("Loss choice not valid or Loss not implemented!")
        return loss

    def train_model(self, x_data, y_data, which_loss='cross-entropy', device="cpu"):
        """
        :param x_data: (self.num_data_samples x 1): torch tensor containing input features.
        :param y_data: (self.num_data_samples): torch tensor containing output pairs.
        :param which_loss: (str): loss function used.
        :return: model: trained model
        """
        self.which_loss = which_loss
        x_data = x_data.to(device)
        y_data = y_data.to(device)
        for i in range(self.num_iterations):
            self.optimizer.zero_grad()
            network_output = self.model(x_data)
            loss = self.compute_loss(network_output, y_data, which_loss)

            loss = loss.mean()
            loss.backward()

            if i % 500 == 0:
                print('Iter {0}/{1}, Loss {2}'.format(i, self.num_iterations, loss.item()))

            self.optimizer.step()
        return self.model

    def train_model_with_dataloader(self, training_dataset,
                                    eval_dataset,
                                    early_stopping=True,
                                    which_loss='cross-entropy',
                                    device='cpu'):
        self.which_loss = which_loss
        self.model = self.model.to(device)
        # Validation Initialization
        if eval_dataset is not None:
            x_val, y_val = (
                eval_dataset.tensors[0].to(device),
                eval_dataset.tensors[1].to(device),
            )

        # Training Initialization
        training_dataloader = torch.utils.data.DataLoader(
            training_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        training_data_generator = iter(training_dataloader)

        best_val_loss = np.inf
        patience_counter = 0.0
        self.early_stopping_patience = 100 * len(training_dataloader)  # 100 epochs
        for i in range(self.num_iterations):
            try:
                # Samples the batch
                x_data, y_data = next(training_data_generator)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                training_data_generator = iter(training_dataloader)
                x_data, y_data = next(training_data_generator)

            x_data = x_data.to(device)
            y_data = y_data.to(device)

            self.optimizer.zero_grad()
            network_output = self.model(x_data)

            loss = self.compute_loss(network_output, y_data, which_loss)

            loss = loss.mean()
            loss.backward()
            if (i % 50 == 0) & (eval_dataset is not None):
                with torch.no_grad():
                    network_val_output = self.model(x_val)
                val_loss = self.compute_loss(network_val_output, y_val, which_loss)
                val_loss = val_loss.mean().cpu().numpy()
                print('Iter {0}/{1}, Training Loss {2}'.format(i, self.num_iterations, loss.item()))
                print('Iter {0}/{1}, Validation Loss {2}'.format(i, self.num_iterations, val_loss))
            elif i % 50 == 0:
                print('Iter {0}/{1}, Training Loss {2}'.format(i, self.num_iterations, loss.item()))

            if early_stopping:
                if val_loss > best_val_loss:
                    patience_counter = patience_counter + 1
                else:
                    patience_counter = 0
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), "state_dict_model.pt")

            if patience_counter >= self.early_stopping_patience:
                self.model.load_state_dict(torch.load("state_dict_model.pt"))
                return self.model

            self.optimizer.step()

        self.model.load_state_dict(torch.load("state_dict_model.pt"))
        return self.model