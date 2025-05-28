import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import seaborn as sns

class Metrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.tp = torch.zeros(self.num_classes)  # True Positives
        self.fp = torch.zeros(self.num_classes)  # False Positives
        self.fn = torch.zeros(self.num_classes)  # False Negatives
        self.tn = torch.zeros(self.num_classes)  # True Negatives
        self.all_targets = []
        self.all_outputs = []

    def update(self, output, target):
        _, predicted = torch.max(output, 1)
        predicted = predicted.view(-1)
        target = target.view(-1)
        self.total += target.size(0)
        self.correct += (predicted == target).sum().item()
         
        # Store outputs and targets for AUC calculation
        self.all_targets.append(target.cpu())
        self.all_outputs.append(output.cpu())
        # self.all_outputs.append(output.permute(0, 2, 3, 1).reshape(-1, output.size(1)).cpu())  # 注意这里要reshape


        for i in range(self.num_classes):
            tp_mask = (predicted == i) & (target == i)
            fp_mask = (predicted == i) & (target != i)
            fn_mask = (predicted != i) & (target == i)
            tn_mask = (predicted != i) & (target != i)  # Correctly predicting non-class `i`

            self.tp[i] += tp_mask.sum().item()
            self.fp[i] += fp_mask.sum().item()
            self.fn[i] += fn_mask.sum().item()
            self.tn[i] += tn_mask.sum().item()

    def accuracy(self):
        return self.correct * 100 / self.total if self.total > 0 else 0
        # correct = (self.tp.sum() + self.tn.sum())
        # total = (self.tp + self.fp + self.fn + self.tn).sum()
        # return correct * 100 / total if total > 0 else 0.0

    def precision(self):
        # Precision = TP / (TP + FP)
        per_class_precision = self.tp * 100 / (self.tp + self.fp)
        return torch.nan_to_num(per_class_precision, nan=0.0).mean().item()

    def recall(self):
        # Recall (Sensitivity) = TP / (TP + FN)
        per_class_recall = self.tp * 100 / (self.tp + self.fn)
        return torch.nan_to_num(per_class_recall, nan=0.0).mean().item()

    def f1_score(self):
        # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        p = self.precision()
        r = self.recall()
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0

    def specificity(self):
        # Specificity = TN / (TN + FP)
        per_class_specificity = self.tn * 100 / (self.tn + self.fp)
        return torch.nan_to_num(per_class_specificity, nan=0.0).mean().item()

    def g_mean(self):
        # G-Mean = sqrt(Sensitivity * Specificity)
        per_class_sensitivity = self.tp / (self.tp + self.fn)
        per_class_specificity = self.tn / (self.tn + self.fp)
        per_class_g_mean = torch.sqrt(per_class_sensitivity * per_class_specificity)
        return torch.nan_to_num(per_class_g_mean, nan=0.0).mean().item() * 100

    def kappa(self):
        # Cohen's Kappa
        po = self.correct / self.total if self.total > 0 else 0  # Observed Accuracy
        pe = ((self.tp + self.fp) * (self.tp + self.fn)).sum() / (self.total ** 2) if self.total > 0 else 0  # Expected Accuracy
        return (po - pe) * 100 / (1 - pe) if (1 - pe) > 0 else 0

    def dice(self):
        # Dice Score = 2 * TP / (2 * TP + FP + FN)
        per_class_dice = 2 * self.tp / (2 * self.tp + self.fp + self.fn)
        return torch.nan_to_num(per_class_dice, nan=0.0).mean().item() * 100

    def auc_score(self):
        # Concatenate all targets and outputs for AUC calculation
        all_targets = torch.cat(self.all_targets).numpy()
        all_outputs = torch.cat(self.all_outputs).numpy()

        # Calculate AUC for each class (one-vs-rest approach)
        auc_scores = []
        for i in range(self.num_classes):
            # Binarize the target for the current class
            binarized_target = (all_targets == i).astype(int)
            # Use the softmax probability for class `i` as the score
            class_output = all_outputs[:, i]
            if len(np.unique(binarized_target)) > 1:  # Ensure there are both positive and negative samples
                auc = roc_auc_score(binarized_target, class_output)
                auc_scores.append(auc)

        # Return the average AUC score across all classes
        return np.mean(auc_scores) * 100 if auc_scores else 0.0
    
    def metric_variance(self):
        # Standard Deviation calculations for each metric based on per-class metrics
        precision_std = torch.nan_to_num(self.tp / (self.tp + self.fp), nan=0.0).std().item()
        recall_std = torch.nan_to_num(self.tp / (self.tp + self.fn), nan=0.0).std().item()
        
        # F1 Score Standard Deviation: Calculated for each class and then standard deviation is computed
        per_class_f1 = 2 * self.tp / (2 * self.tp + self.fp + self.fn)
        f1_std = torch.nan_to_num(per_class_f1, nan=0.0).std().item()
        
        # Specificity Standard Deviation
        per_class_specificity = self.tn / (self.tn + self.fp)
        specificity_std = torch.nan_to_num(per_class_specificity, nan=0.0).std().item()
        
        # Dice Score Standard Deviation
        per_class_dice = 2 * self.tp / (2 * self.tp + self.fp + self.fn)
        dice_std = torch.nan_to_num(per_class_dice, nan=0.0).std().item()
        
        # G-Mean Standard Deviation
        per_class_sensitivity = self.tp / (self.tp + self.fn)
        per_class_specificity = self.tn / (self.tn + self.fp)
        per_class_g_mean = torch.sqrt(per_class_sensitivity * per_class_specificity)
        g_mean_std = torch.nan_to_num(per_class_g_mean, nan=0.0).std().item()

        # Accuracy Standard Deviation
        per_class_accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)
        accuracy_std = torch.nan_to_num(per_class_accuracy, nan=0.0).std().item()

        # Cohen's Kappa Standard Deviation (Based on per-class Kappa scores)
        kappa_std = torch.nan_to_num(self.kappa(), nan=0.0).std().item()

        # AUC Standard Deviation
        all_targets = torch.cat(self.all_targets).numpy()
        all_outputs = torch.cat(self.all_outputs).numpy()
        auc_scores = []
        for i in range(self.num_classes):
            binarized_target = (all_targets == i).astype(int)
            class_output = all_outputs[:, i]
            if len(np.unique(binarized_target)) > 1:
                auc = roc_auc_score(binarized_target, class_output)
                auc_scores.append(auc)
        auc_std = np.std(auc_scores) if auc_scores else 0.0
        
        return {
            "precision_std": precision_std,
            "recall_std": recall_std,
            "f1_std": f1_std,
            "specificity_std": specificity_std,
            "dice_std": dice_std,
            "g_mean_std": g_mean_std,
            "accuracy_std": accuracy_std,
            "kappa_std": kappa_std,
            "auc_std": auc_std
        }
    

    # def plot_roc_curve(self, save_path=None):
    #     all_targets = torch.cat(self.all_targets).numpy()
    #     all_outputs = torch.cat(self.all_outputs).numpy()

    #     # Plot ROC curves for each class
    #     plt.figure(figsize=(10, 8))
    #     for i in range(self.num_classes):
    #         # Binarize the target for the current class
    #         binarized_target = (all_targets == i).astype(int)
    #         class_output = all_outputs[:, i]
            
    #         # Calculate the AUC for the current class
    #         auc_value = roc_auc_score(binarized_target, class_output)
            
    #         # Compute ROC curve
    #         fpr, tpr, _ = roc_curve(binarized_target, class_output)
    #         plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_value:.2f})')

    #     # Plot random classifier line
    #     plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    #     plt.title('ROC Curve for Multi-Class Classification')
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')

    #     # Adjust legend position and fontsize
    #     plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=8, frameon=False)
        
    #     # Remove grid
    #     plt.grid(False)

    #     # Save the plot if save_path is provided
    #     if save_path:
    #         plt.savefig(save_path, bbox_inches='tight')
    #         plt.show()
    #     else:
    #         plt.show()





    def plot_roc_curve(self, class_names, save_path=None):
        all_targets = torch.cat(self.all_targets).numpy()
        all_outputs = torch.cat(self.all_outputs).numpy()

        # Plot ROC curves for each class
        plt.figure(figsize=(10, 8))
        for i in range(self.num_classes):
            # Binarize the target for the current class
            binarized_target = (all_targets == i).astype(int)
            class_output = all_outputs[:, i]
            
            # Calculate the AUC for the current class
            auc_value = roc_auc_score(binarized_target, class_output)
            
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(binarized_target, class_output)
            
            # Use class_names to display the label instead of the class number
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc_value:.2f})')

        # Plot random classifier line
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.title('ROC Curve for Multi-Class Classification', fontsize=16)
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)

        # Adjust legend position and fontsize
        plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=8, frameon=False)
        
        # Remove grid
        plt.grid(False)

        # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.show()
        else:
            plt.show()

            

    # def plot_confusion_matrix(self, save_path=None):
    #     # Concatenate all targets and outputs for confusion matrix plotting
    #     all_targets = torch.cat(self.all_targets).numpy()
    #     all_outputs = torch.cat(self.all_outputs).numpy()

    #     # Get predicted labels (taking the index of the max probability)
    #     predicted = np.argmax(all_outputs, axis=1)

    #     # Compute confusion matrix
    #     cm = confusion_matrix(all_targets, predicted)

    #     # Normalize the confusion matrix for color scaling
    #     norm_cm = cm / cm.max()

    #     # Plot the confusion matrix
    #     plt.figure(figsize=(12, 10))  # Adjust figure size for better visibility
    #     ax = sns.heatmap(
    #         cm,
    #         annot=True,
    #         fmt='d',
    #         cmap='Blues',
    #         xticklabels=[f'{i}' for i in range(self.num_classes)],
    #         yticklabels=[f'{i}' for i in range(self.num_classes)],
    #         cbar=True,
    #         annot_kws={"size": 8, "ha": "center", "va": "center"}
    #     )

    #     # Add dynamic background colors to annotation boxes
    #     cmap = plt.cm.Blues  # Use a colormap (e.g., Reds) to map colors
    #     norm = plt.Normalize(vmin=cm.min(), vmax=cm.max())  # Normalize based on matrix values

    #     for i in range(cm.shape[0]):
    #         for j in range(cm.shape[1]):
    #             value = cm[i, j]  # Get the value of the confusion matrix at (i, j)
    #             color = cmap(norm(value))  # Map the value to a color
    #             # Set the annotation's background box color
    #             ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color, transform=ax.transData))

    #     plt.title('Confusion Matrix', fontsize=16)
    #     plt.xlabel('Predicted Label', fontsize=14)
    #     plt.ylabel('True Label', fontsize=14)

    #     # Save the plot if save_path is provided
    #     if save_path:
    #         plt.savefig(save_path, bbox_inches='tight')
    #     plt.show()


    def plot_confusion_matrix(self, save_path=None, class_labels=None):
        # Concatenate all targets and outputs for confusion matrix plotting
        all_targets = torch.cat(self.all_targets).numpy()
        all_outputs = torch.cat(self.all_outputs).numpy()

        # Get predicted labels (taking the index of the max probability)
        predicted = np.argmax(all_outputs, axis=1)

        # Compute confusion matrix
        cm = confusion_matrix(all_targets, predicted)

        # Normalize the confusion matrix for color scaling
        norm_cm = cm / cm.max()

        # Plot the confusion matrix
        plt.figure(figsize=(12, 10))  # Adjust figure size for better visibility
        ax = sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_labels,  # Use actual class labels for x-axis
            yticklabels=class_labels,  # Use actual class labels for y-axis
            cbar=True,
            annot_kws={"size": 8, "ha": "center", "va": "center"}
        )

        # Adjust font size and angle for tick labels
        plt.xticks(rotation=45, ha='right', fontsize=8)  # Rotate and adjust fontsize for x labels
        plt.yticks(rotation=0, ha='right', fontsize=8)   # No rotation for y labels, adjust fontsize
        ax.yaxis.labelpad = 10  
        # Add dynamic background colors to annotation boxes
        cmap = plt.cm.Blues  # Use a colormap (e.g., Reds) to map colors
        norm = plt.Normalize(vmin=cm.min(), vmax=cm.max())  # Normalize based on matrix values

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm[i, j]  # Get the value of the confusion matrix at (i, j)
                color = cmap(norm(value))  # Map the value to a color
                # Set the annotation's background box color
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color, transform=ax.transData))

        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)

        # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()