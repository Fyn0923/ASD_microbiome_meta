import copy
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from tqdm import tqdm

# Define the neural network model
class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        dropout_rate = 0.2
        hidden_size = 256
        
        self.batch_norm0 = nn.BatchNorm1d(input_size)
        self.dropout0 = nn.Dropout(dropout_rate)
        
        self.dense1 = nn.Linear(input_size, hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.dense2 = nn.Linear(hidden_size + input_size, hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.dense3 = nn.Linear(hidden_size * 2, hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.dense4 = nn.Linear(hidden_size * 2, hidden_size)
        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        self.dense5 = nn.Linear(hidden_size * 2, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.batch_norm0(x)
        x = self.dropout0(x)
        
        x1 = self.leaky_relu(self.batch_norm1(self.dense1(x)))
        x1 = self.dropout1(x1)
        x = torch.cat([x, x1], dim=1)
        
        x2 = self.leaky_relu(self.batch_norm2(self.dense2(x)))
        x2 = self.dropout2(x2)
        x = torch.cat([x1, x2], dim=1)
        
        x3 = self.leaky_relu(self.batch_norm3(self.dense3(x)))
        x3 = self.dropout3(x3)
        x = torch.cat([x2, x3], dim=1)
        
        x4 = self.leaky_relu(self.batch_norm4(self.dense4(x)))
        x4 = self.dropout4(x4)
        x = torch.cat([x3, x4], dim=1)
        
        x = self.dense5(x)
        return torch.sigmoid(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0.1)

# Calculate evaluation metrics
def calculate_metrics(true_labels, predictions):
    preds_flat = [item[0] for item in predictions]
    pred_labels = [1 if p >= 0.5 else 0 for p in preds_flat]
    auc = roc_auc_score(true_labels, preds_flat)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    mcc = matthews_corrcoef(true_labels, pred_labels)
    return auc, accuracy, precision, recall, f1, mcc

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

# Load and preprocess data
df = pd.read_csv('your_data.csv', header=0, index_col=0)
df['label'] = df['label'].replace({'ASD': 1, 'TD': 0})
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split data
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, stratify=y_temp)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=32)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize model, loss, optimizer, and scheduler
input_size = X_train.shape[1]
model = Model(input_size)
model.initialize_weights()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=0.000001)
early_stopping = EarlyStopping(patience=10, delta=0.001)

# Training loop
num_epochs = 100
best_auc = 0.0
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    print(f"------------------------- Epoch {epoch} -------------------------")
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    predictions, true = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item() * inputs.size(0)
            predictions.extend(outputs.tolist())
            true.extend(labels.tolist())
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        auc, accuracy, precision, recall, f1, mcc = calculate_metrics(true, predictions)
        scheduler.step(auc)
        print(f"Val AUC: {auc:.4f}")
        print(f"Val: acc:{accuracy:.4f}, pre:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}, mcc:{mcc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_metrics = (auc, accuracy, precision, recall, f1, mcc)
            torch.save(model.state_dict(), "./best_model.pth")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

# Load best model and evaluate on test set
model.load_state_dict(torch.load("./best_model.pth"))
print(f"Best: auc:{best_metrics[0]:.4f}, acc:{best_metrics[1]:.4f}, pre:{best_metrics[2]:.4f}, "
      f"recall:{best_metrics[3]:.4f}, f1:{best_metrics[4]:.4f}, mcc:{best_metrics[5]:.4f}")

model.eval()
predictions, true = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predictions.extend(outputs.tolist())
        true.extend(labels.tolist())

    auc, accuracy, precision, recall, f1, mcc = calculate_metrics(true, predictions)
    print(f"Test AUC: {auc:.4f}")
    print(f"Test: acc:{accuracy:.4f}, pre:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}, mcc:{mcc:.4f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(true, [p[0] for p in predictions])
    plt.figure(figsize=(3.35, 3), dpi=300)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}', color='#283c63', lw=2)
    plt.fill_between(fpr, tpr, color='#d8e9f0', alpha=0.3)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.savefig('roc_curve_R.pdf', bbox_inches='tight')
    plt.close()

    # Confusion Matrix
    pred_labels = [1 if p[0] >= 0.5 else 0 for p in predictions]
    conf_matrix = confusion_matrix(true, pred_labels)
    plt.figure(figsize=(3.35, 3), dpi=300)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('Confusion_Matrix_R.pdf', bbox_inches='tight')
    plt.close()
