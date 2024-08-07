import torch

def multi_loss_aggregation(x, y, loss_fun, w_lambda=None):

    a_loss = loss_fun(x, y)
    t_loss = loss_fun(torch.mean(torch.stack([x[:, -1], x[:, -2]], dim=0), dim=0), torch.logical_or(y[:, -1], y[:, -2]).float())
    h_loss = loss_fun(torch.mean(torch.stack([x[:, -3], x[:, -4]], dim=0), dim=0), torch.logical_or(y[:, -3], y[:, -4]).float())
    w_loss = loss_fun(torch.mean(torch.stack([x[:, -2], x[:, -4]], dim=0), dim=0), torch.logical_or(y[:, -2], y[:, -4]).float())
    g_loss = loss_fun(torch.mean(torch.stack([x[:, -1], x[:, -3]], dim=0), dim=0), torch.logical_or(y[:, -1], y[:, -3]).float())

    if w_lambda is None: w_lambda = torch.tensor([x.shape[1], 1, 1, 1, 1], device=x.device).float()

    return torch.stack([a_loss, t_loss, h_loss, w_loss, g_loss]) @ w_lambda
