import torch
import numpy as np

class EarlyStopping:
    """Stop training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

def get_scheduler(optimizer, patience=5, factor=0.5, min_lr=1e-6):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=patience, 
        factor=factor, 
        min_lr=min_lr
    )

def class_weights_from_labels(labels):
    # labels: numpy array or pandas Series of class indices
    unique, counts = np.unique(labels, return_counts=True)
    weights = len(labels) / (len(unique) * counts)
    # Return as a tensor compatible with CrossEntropyLoss
    return torch.FloatTensor(weights)

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x, target):
        # x: [B, num_classes], target: [B]
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * self.log_softmax(x), dim=-1))

if __name__ == "__main__":
    # Example usage context illustrated for copy-paste into train.py:
    # model, optimizer defined elsewhere
    optimizer = torch.optim.Adam([torch.randn(2,2, requires_grad=True)], lr=1e-3)
    scheduler = get_scheduler(optimizer)
