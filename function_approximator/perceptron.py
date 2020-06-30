import torch


class SLP:
    """A Single Layer Perception (SLP) class to approximate
    """

    def __init__(self, input_shape, output_shape, device=torch.device("cpu")):
        """
        :param input_shape: Shape/dimension of the input
        :param output_shape: Shape/dimension of the output
        :param device: the device that SLP should use to store the inputs for the forward pass
        """
        super().__init__()
        self.device = device
        self.input_shape = input_shape
        self.hidden_shape = 40
        self.linear = torch.nn.Linear(self.input_shape, self.hidden_shape)
        self.out = torch.nn.Linear(self.hidden_shape, self.output_shape)

    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        x = self.linear(x)
        x = torch.nn.functional.relu(x)
        x = self.out(x)
        return x
