import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        """
        Args:
            input_dim (int): The number of input features (i.e., feature dimension from the backbone or embedding).
            num_classes (int): The number of classes for classification.
            dropout_prob (float): Dropout probability to prevent overfitting (optional, default 0.5).
        """
        super(MLP, self).__init__()
        # Define the fully connected layers
        self.fc = nn.Linear(in_features, num_classes)  # First layer with 512 neurons (change as needed)

    def forward(self, x):
        # Pass through the first fully connected layer
        x = self.fc(x) # Apply ReLU activation
        return x
