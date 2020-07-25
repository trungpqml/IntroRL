import torch


def xavier(m):
    """Initialize weight of the network using method described in paper by Xavier Glorot and Yoshua Bengio
    :param self:
    :param m: nn.Module
    :return:
    """
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
