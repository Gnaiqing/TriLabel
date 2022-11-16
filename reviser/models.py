import numpy as np
import torch
from scipy.special import softmax
import torch.nn.functional as F


class BaseNetwork(torch.nn.Module):
    """
    Base Network Class. Contains shared components among all other models.
    """

    def __init__(self, n_neurons=50,
                 activation='relu',
                 input_dim=1,
                 output_dim=1):
        """
        :param n_neurons: size of network layers
        :param activation: activation function.
        :param input_dim: Single scalar representing the dimension of input features.
        :param output_dim: Single scalar representing the dimension of the output predicted vector.
        :param min_var: minimum variance. This value is added to the predicted variance to prevent NLL overflow.
        """
        super(BaseNetwork, self).__init__()
        if activation == 'sigmoid':
            act_func = torch.nn.Sigmoid()
        elif activation == 'relu':
            act_func = torch.nn.ReLU()
        elif activation == 'tanh':
            act_func = torch.nn.Tanh()
        elif activation == 'leaky_relu':
            act_func = torch.nn.LeakyReLU()
        elif activation == 'elu':
            act_func = torch.nn.ELU()
        else:
            raise (NotImplementedError,
                   'Activation Function Not Implemented. Needs to be one of: [sigmoid, relu, leaky_relu]')

        self.act_func = act_func
        self.n_neurons = n_neurons
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        raise NotImplementedError

    def inference_forward(self, x, **kwargs):
        raise NotImplementedError

    def predict(self, x):
        f = self.inference_forward(torch.tensor(x))
        y_pred = np.argmax(f, axis=1)
        return y_pred

    def predict_proba(self, x):
        f = self.inference_forward(torch.tensor(x))
        proba = softmax(f, axis=1)
        return proba


class MLPNet(BaseNetwork):
    """
    Multi-layer Perceptron
    """
    def __init__(self, n_neurons=50,
                 activation='relu',
                 input_dim=1,
                 output_dim=1):
        super(MLPNet, self).__init__(n_neurons, activation, input_dim, output_dim)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, n_neurons),
            self.act_func,
            torch.nn.Linear(n_neurons, output_dim)
        )

    def forward(self, x):
        f = self.net(x)
        return f

    def inference_forward(self, x, **kwargs):
        self.eval()
        with torch.no_grad():
            f = self.forward(x)
        return f.cpu().numpy()


class MLPTempNet(MLPNet):
    """
    MLP network with temperature scaling
    """
    def __init__(self, n_neurons=50,
                 activation='relu',
                 input_dim=1,
                 output_dim=1):
        super(MLPTempNet, self).__init__(n_neurons, activation, input_dim, output_dim)
        self.temp = 1

    def temp_scale(self, eval_dataset):
        self.eval()
        if eval_dataset is None:
            return

        x_val, y_val = eval_dataset.tensors
        min_loss = 1e6
        with torch.no_grad():
            f = self.forward(x_val)
            for t in np.logspace(-2, 2):
                f_scale = f / t
                loss_func = torch.nn.CrossEntropyLoss()
                loss = loss_func(f_scale, y_val).mean().cpu().numpy()
                if loss < min_loss:
                    min_loss = loss
                    self.temp = t

        print(f"Set temperature scale to {self.temp:.2f}")

    def inference_forward(self, x, **kwargs):
        self.eval()
        with torch.no_grad():
            f = self.forward(x) / self.temp
        return f.cpu().numpy()


class DropOutNet(BaseNetwork):
    """
    Dropout network.
    """

    def __init__(self,
                 n_neurons=50,
                 p_dropout=0.05,
                 activation='relu',
                 input_dim=1,
                 output_dim=1):
        super(DropOutNet, self).__init__(n_neurons=n_neurons,
                                         activation=activation,
                                         input_dim=input_dim,
                                         output_dim=output_dim)
        self.net = torch.nn.Sequential(torch.nn.Linear(self.input_dim, n_neurons),
                                       torch.nn.Dropout(p=p_dropout),
                                       self.act_func,
                                       torch.nn.Linear(n_neurons, self.output_dim),
                                       torch.nn.Dropout(p=p_dropout))

    def forward(self, x):
        f = self.net(x)
        return f

    def inference_forward(self, x, num_samples=100, **kwargs):
        with torch.no_grad():
            probs = []
            for i in range(num_samples):
                fi = self.forward(x)
                pi = F.softmax(fi, dim=1)
                probs.append(pi.cpu().numpy())

            probs = np.stack(probs, axis=0).mean(axis=0)
            logits = np.log(probs)

        return logits

