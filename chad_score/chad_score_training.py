import torch.nn.functional as F


def chad_training_step(model, batch, xcol='emb', ycol='avg_rating'):
    x = batch[xcol]
    y = batch[ycol].reshape(-1, 1)
    x_hat = model(x)
    loss = F.mse_loss(x_hat, y)
    return loss


def chad_validation_step(model, batch, xcol='emb', ycol='avg_rating'):
    x = batch[xcol]
    y = batch[ycol].reshape(-1, 1)
    x_hat = model(x)
    loss = F.mse_loss(x_hat, y)
    return loss
