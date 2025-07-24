import torch


def reduce_ht(x, y):

    h_pred = torch.maximum(x[:, -3], x[:, -4])
    t_pred = torch.maximum(x[:, -1], x[:, -2])
    h_true = torch.logical_or(y[:, -3], y[:, -4]).float()
    t_true = torch.logical_or(y[:, -1], y[:, -2]).float()

    return h_pred, t_pred, h_true, t_true


def reduce_wg(x, y):

    w_pred = torch.maximum(x[:, -2], x[:, -4])
    g_pred = torch.maximum(x[:, -1], x[:, -3])
    w_true = torch.logical_or(y[:, -2], y[:, -4]).float()
    g_true = torch.logical_or(y[:, -1], y[:, -3]).float()

    return w_pred, g_pred, w_true, g_true


def multi_loss_aggregation(x, y, loss_fun, w_lambda=None):

    a_loss = loss_fun(x, y)
    h_pred, t_pred, h_true, t_true = reduce_ht(x, y)
    w_pred, g_pred, w_true, g_true = reduce_wg(x, y)
    h_loss = loss_fun(h_pred, h_true)
    t_loss = loss_fun(t_pred, t_true)
    w_loss = loss_fun(w_pred, w_true)
    g_loss = loss_fun(g_pred, g_true)

    if w_lambda is None: w_lambda = torch.tensor([x.shape[1], 1, 1, 1, 1], device=x.device, dtype=x.dtype) / (x.shape[1]+4+2)

    return torch.stack([a_loss, t_loss, h_loss, w_loss, g_loss]) @ w_lambda

def reduce_htgm(x, y, class_num=4):

    # negative indices to account for background class at index 0
    hwm_pred = x[:, -4]
    twm_pred = x[:, -2]
    hwm_true = y[:, -4]
    twm_true = y[:, -2]
    gm_pred = torch.maximum(x[:, -1], x[:, -3])
    gm_true = torch.logical_or(y[:, -1], y[:, -3]).float()

    pred = torch.stack([x[:, 0], hwm_pred, twm_pred, gm_pred], dim=1) if x.shape[1] == 5 else torch.stack([hwm_pred, twm_pred, gm_pred], dim=1)
    true = torch.stack([y[:, 0], hwm_true, twm_true, gm_true], dim=1) if y.shape[1] == 5 else torch.stack([hwm_true, twm_true, gm_true], dim=1)

    if class_num in (6, 7):
        # negative indices to account for background class at index 0
        iwm_pred = x[:, -6]
        iwm_true = y[:, -6]
        gm_pred = torch.maximum(gm_pred, x[:, -5])
        gm_true = torch.logical_or(gm_true, y[:, -5]).float()

        pred = torch.concat([x[:, 0][:, None], iwm_pred[:, None], pred], dim=1) if x.shape[1] == 7 else torch.concat([iwm_pred[:, None], pred], dim=1)
        true = torch.concat([y[:, 0][:, None], iwm_true[:, None], true], dim=1) if y.shape[1] == 7 else torch.concat([iwm_true[:, None], true], dim=1)

    return pred, true
