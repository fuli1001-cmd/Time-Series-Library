from data_provider.data_factory import data_provider
from exp.exp_classification import Exp_Classification
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
from sklearn.metrics import precision_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score # Added r2_score
import warnings
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Stock_Classification(Exp_Classification):
    def __init__(self, args):
        super(Exp_Stock_Classification, self).__init__(args)
        self.padding_mask = torch.ones(args.batch_size, args.seq_len, device=args.device)

    def _build_model(self):
        self.args.pred_len = 0 # No prediction length for classification
        self.args.num_class = self.args.c_out
        
        model = self.model_dict[self.args.model].Model(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        all_outputs = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(vali_loader):
                outputs = self.model(batch_x, self.padding_mask, None, None)

                loss = criterion(outputs, label.long().squeeze())
                total_loss.append(loss)

                all_outputs.append(outputs.detach())
                all_labels.append(label)

        avg_loss = np.average(total_loss)
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
        all_outputs = torch.cat(all_outputs, dim=0)
        all_preds = all_outputs.argmax(1).cpu().numpy()
        all_probs_positive = (torch.softmax(all_outputs, dim=1)[:, 1]).cpu().numpy()

        precision = precision_score(all_labels, all_preds, zero_division=0)
            
        auc_score_val = float('nan')
        # Ensure all_probs_positive is not empty and has correct shape for roc_auc_score
        if all_probs_positive is not None and len(all_probs_positive) > 0 and len(all_labels) == len(all_probs_positive):
            try:
                auc_score_val = roc_auc_score(all_labels, all_probs_positive)
            except ValueError: 
                auc_score_val = float('nan')

        precision_at_k = float('nan')
        # Ensure all_probs_positive is not empty for topk
        if all_probs_positive is not None and len(all_probs_positive) > 0: 
            probs_tensor = torch.tensor(all_probs_positive)
            labels_tensor = torch.tensor(all_labels)
            
            num_samples = len(probs_tensor)
            top_k = 5
            actual_k = min(top_k, num_samples)
            
            if actual_k > 0 and num_samples > 0: # Ensure num_samples > 0
                topk_indices = probs_tensor.topk(actual_k).indices
                # Check if labels_tensor is not empty and indices are valid
                if len(labels_tensor) > 0 and (len(topk_indices) == 0 or topk_indices.max() < len(labels_tensor)):
                    topk_labels = labels_tensor[topk_indices]
                    if len(topk_labels) > 0:
                        precision_at_k = topk_labels.float().mean().item()

        self.model.train()
        return avg_loss, precision, auc_score_val, precision_at_k

    def train(self, setting):
        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()

            for i, (batch_x, label) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                outputs = self.model(batch_x, self.padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            train_loss = np.average(train_loss)
            val_loss, precision, auc_score_val, precision_at_k = self.vali(None, vali_loader, criterion)

            print(f"epoch: {epoch + 1}, train Loss: {train_loss:.3f}, val Loss: {val_loss:.3f}, precision: {precision:.3f}, precision@5: {precision_at_k:.3f}, auc: {auc_score_val:.3f}")
            early_stopping(-precision, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
