import abc
import os
import sys
import tqdm
import torch

from torch.utils.data import DataLoader
from types_ import *
import json


class Trainer():
    """
    A class abstracting the various tasks of training models.
    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/validate_epoch)
    - Single batch (train_batch/validate_batch)
    """
    def __init__(self, model, loss_fn, optimizer, device=None, model_fname="model"):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.model_fname = model_fname

        if self.device:
            model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_valid: DataLoader,
            num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, **kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_valid: Dataloader for the valid set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            valid set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            valid loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :return: A FitResult object containing train and valid losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, valid_loss, valid_acc = [], [], [], []

        best_acc = 0
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            verbose = False  # pass this to train/validate_epoch.
            if epoch % print_every == 0 or epoch == num_epochs-1:
                verbose = True
            self._print(f'--- EPOCH {epoch+1}/{num_epochs} ---', verbose)
            result_train = self.train_epoch(dl_train, **kw)
            result_valid = self.validate_epoch(dl_valid, **kw)
            train_loss.append((sum(result_train.losses)/(len(result_train.losses))).item())
            train_acc.append(result_train.accuracy.item())
            valid_loss.append((sum(result_valid.losses)/(len(result_valid.losses))).item())
            valid_acc.append(result_valid.accuracy.item())
            
            if best_acc >= valid_acc[-1]:
                epochs_without_improvement+=1
            else:
                best_acc = valid_acc[-1]
                epochs_without_improvement = 0
                torch.save(self.model, f"./checkpoints/{self.model_fname}.pth")
                # save accuracy
                with open('./checkpoints/accuracies.json', 'r') as f:
                    data = json.load(f)
                data[self.model_fname] = best_acc / 100
                with open('./checkpoints/accuracies.json', 'w') as f:
                    json.dump(data, f)
            
            if epochs_without_improvement == early_stopping:
                break;

        return FitResult(actual_num_epochs,
                         train_loss, train_acc, valid_loss, valid_acc)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def validate_epoch(self, dl_valid: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a valid set (single epoch).
        :param dl_valid: DataLoader for the valid set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (valid) mode
        return self._foreach_batch(dl_valid, self.validate_batch, **kw)

    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        self.optimizer.zero_grad()
        o = self.model(X)
        loss = self.loss_fn(o, y); loss.backward()
        self.optimizer.step()
        num_correct=torch.sum(o.argmax(1)==y.argmax(1))

        return BatchResult(loss, num_correct)

    def validate_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        with torch.no_grad():
            o = self.model(X)
            loss = self.loss_fn(o, y)
            num_correct=torch.sum(o.argmax(1)==y.argmax(1))

        return BatchResult(loss, num_correct)

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100. * num_correct / num_samples
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, '
                                 f'Accuracy {accuracy:.1f})')

        return EpochResult(losses=losses, accuracy=accuracy)
