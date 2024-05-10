import torch.optim as optim


def adam_opt(model, lr):
    return optim.Adam(model.parameters(), lr=lr)