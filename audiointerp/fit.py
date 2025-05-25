import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader


def plot_learning_curves(train_losses, val_losses, train_accs=None, val_accs=None):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    if val_losses:
        plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    if train_accs is not None:
        plt.plot(epochs, train_accs, label="Train Acc")
    if val_accs is not None:
        plt.plot(epochs, val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()


class Trainer:
    """Class for handling model training, validation and testing
    Also supports fine-tuning
    """

    # to be completed with gradient accumulation and amp
    # sheduler usage is limited to batches and epochs
    def __init__(self, model_cls, train_data, train_loader_kwargs,
                 criterion_cls, optimizer_cls,
                 model_kwargs=None, model_pretrain_weights_path=None,
                 optimizer_kwargs=None, device="cuda" if torch.cuda.is_available() else "cpu",
                 valid_data=None, valid_loader_kwargs=None,
                 scheduler_cls=None, scheduler_kwargs=None,
                 scheduler_update_point="epoch", verbose=True,
                 test_data=None, test_loader_kwargs=None,
                 checkpoint_path=None, use_mixup=False, mixup_alpha=None, seed=42):
        
        self.verbose = verbose

        self.seed = seed
        self._set_seed()

        # essentials
        self.device = device
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.model = self.model_cls(**self.model_kwargs)
        self.model_pretrain_weights_path = model_pretrain_weights_path
        if self.model_pretrain_weights_path is not None:
            self.model.load_base_weights(self.model_pretrain_weights_path)
        self.model.to(self.device)

        self.train_data = train_data
        self.train_loader_kwargs = train_loader_kwargs if train_loader_kwargs is not None else {}
        self.train_loader = self._make_dataloader(self.train_data, self.train_loader_kwargs)

        self.criterion = criterion_cls()

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        if self.optimizer_cls is not None:
            self.optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)
        else:
            self.optimizer = None

        # optionals
        self.valid_data = valid_data
        self.valid_loader_kwargs = valid_loader_kwargs if valid_loader_kwargs is not None else {}
        if self.valid_data is not None:
            self.valid_loader = self._make_dataloader(self.valid_data, self.valid_loader_kwargs)
        else:
            self.valid_loader = None

        self.test_data = test_data
        self.test_loader_kwargs = test_loader_kwargs if test_loader_kwargs is not None else {}
        if self.test_data is not None:
            self.test_loader = self._make_dataloader(self.test_data, self.test_loader_kwargs)
        else:
            self.test_loader = None

        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs is not None else {}
        if scheduler_cls is not None and self.optimizer is not None:
            self.scheduler = self.scheduler_cls(self.optimizer, **self.scheduler_kwargs)
        else:
            self.scheduler = None
        self.scheduler_update_point = scheduler_update_point

        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

        self.checkpoint_path = checkpoint_path


    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if self.verbose:
            tqdm.write(f"Random seed set to: {self.seed}")


    def _make_dataloader(self, data, loader_kwargs):
        kwargs = loader_kwargs.copy()

        if "generator" not in kwargs and kwargs.get("shuffle", False):
            kwargs["generator"] = torch.Generator()
            kwargs["generator"].manual_seed(self.seed)

        if "worker_init_fn" not in kwargs and kwargs.get("num_workers", 0) > 0:
            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)
            kwargs["worker_init_fn"] = seed_worker

        return DataLoader(data, **kwargs)
    

    def _mixup(self, x, y):
        if self.mixup_alpha <= 0:
            return x, y, y, 1.0

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]

        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


    def _train_step(self):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
    
        for samples, labels in self.train_loader:
            samples = samples.to(self.device)
            labels = labels.to(self.device)
        
            self.optimizer.zero_grad()

            if self.use_mixup:
                samples, labels_a, labels_b, lam = self._mixup(samples, labels)
                outputs = self.model(samples)
                loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
            else:
                outputs = self.model(samples)
                loss = self.criterion(outputs, labels)
        
            loss.backward()
            self.optimizer.step()
        
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * samples.size(0)
            if self.use_mixup:
                running_corrects += lam * torch.sum(preds == labels_a) + (1 - lam) * torch.sum(preds == labels_b)
            else:
                running_corrects += torch.sum(preds == labels)
                
            total_samples += samples.size(0)

            if self.scheduler is not None and self.scheduler_update_point == "batch":
                self.scheduler.step()
    
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
    
        return epoch_loss, epoch_acc.item()


    def _valid_step(self):
        if self.valid_loader is None:
            return None, None
                
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
    
        with torch.no_grad():
            for samples, labels in self.valid_loader:
                samples = samples.to(self.device)
                labels = labels.to(self.device)
            
                outputs = self.model(samples)
                loss = self.criterion(outputs, labels)
            
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * samples.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += samples.size(0)
    
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
    
        return epoch_loss, epoch_acc.item()


    def reset(self, new_seed=None):
        """Reset model, optimizer and scheduler
        before new training run
        """
        if new_seed is not None:
            self.seed = new_seed
        self._set_seed()

        self.model = self.model_cls(**self.model_kwargs).to(self.device)
        if self.model_pretrain_weights_path is not None:
            self.model.load_base_weights(self.model_pretrain_weights_path)

        self.train_loader = self._make_dataloader(self.train_data, self.train_loader_kwargs)

        if self.optimizer_cls is not None:
            self.optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)

        if self.valid_data is not None:
            self.valid_loader = self._make_dataloader(self.valid_data, self.valid_loader_kwargs)

        if self.test_data is not None:
            self.test_loader = self._make_dataloader(self.test_data, self.test_loader_kwargs)

        if self.scheduler_cls is not None and self.optimizer is not None:
            self.scheduler = self.scheduler_cls(self.optimizer, **self.scheduler_kwargs)


    def train(self, num_epochs=10, save_weights_path=None):
        best_acc = 0.0
        best_model = None
        start_epoch = 1

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        for epoch in tqdm(range(start_epoch, num_epochs + 1), desc="Epoch"):
    
            train_loss, train_acc = self._train_step()
            val_loss, val_acc = self._valid_step()

            if self.verbose:
                epoch_report = f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                if val_loss is not None:
                    epoch_report += f"\nValid Loss: {val_loss:.4f}, Valid Acc: {val_acc:.4f}"
                tqdm.write(epoch_report)

            if self.scheduler is not None:
                if self.scheduler_update_point == "epoch":
                    self.scheduler.step()
                elif self.scheduler_update_point == "plateau":
                    self.scheduler.step(val_loss)

            train_losses.append(train_loss)
            train_accs.append(train_acc)

            val_losses.append(val_loss)
            val_accs.append(val_acc)
    
            if val_acc is not None and val_acc > best_acc:
                best_acc = val_acc
                best_model = self.model.state_dict()

        if self.verbose:
            tqdm.write(f"Best val Acc: {best_acc:.4f}")

        if best_model is not None:
            self.model.load_state_dict(best_model)
    
        torch.save(self.model.state_dict(), save_weights_path)
        if self.verbose:
            tqdm.write(f"Модель сохранена в {save_weights_path}")

        test_loss, test_acc = self.test()

        return train_losses, train_accs, val_losses, val_accs, test_loss, test_acc


    def test(self, test_loader_custom=None):
        if self.test_loader is None and test_loader_custom is None: 
            return None, None

        if test_loader_custom is not None:
            loader = test_loader_custom
        else:
            loader = self.test_loader        
        
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
    
        with torch.no_grad():
            for samples, labels in loader:
                samples = samples.to(self.device)
                labels = labels.to(self.device)
            
                outputs = self.model(samples)
                loss = self.criterion(outputs, labels)
            
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * samples.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += samples.size(0)
    
        test_loss = running_loss / total_samples
        test_acc = running_corrects.double() / total_samples

        if self.verbose:
            tqdm.write(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc.item():.4f}")
    
        return test_loss, test_acc.item()