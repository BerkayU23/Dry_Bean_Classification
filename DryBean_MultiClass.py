import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Dry_Bean_Dataset.csv")

print(df.info())

print(df[df.duplicated()])

df = df.drop_duplicates()
print(df.info())

plt.figure(figsize=(8,5))
sns.histplot(df["Class"])
plt.show()

X = df.drop("Class", axis = 1).values
y = df["Class"].values

print(X.shape, y.shape)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test,  dtype=torch.long)

from torch import nn

class MultiClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_stack_layer = nn.Sequential(
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,7)
        )

    def forward(self, x):
        return self.linear_stack_layer(x)
    
torch.manual_seed(23)
from torchmetrics.classification import MulticlassAccuracy
accuracy = MulticlassAccuracy(num_classes=7)

model = MultiClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr = 0.003)

epochs = 501

for epoch in range(epochs):
    model.train()

    logits = model(X_train)
    loss = loss_fn(logits, y_train)

    pred = torch.softmax(logits, dim=1).argmax(dim=1)
    acc = accuracy(pred, y_train).item() * 100

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()

    with torch.inference_mode():

        test_logits = model(X_test)
        test_loss = loss_fn(test_logits, y_test)

        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_acc = accuracy(test_pred, y_test).item() * 100

    if epoch % 20 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.5f}, Accuracy: {acc:.3f}, Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.3f}")


from torchmetrics.classification import MulticlassConfusionMatrix
cm = MulticlassConfusionMatrix(num_classes=7)
matrix = cm(test_pred, y_test)
print(matrix)

from torchmetrics.utilities.plot import plot_confusion_matrix
print(plot_confusion_matrix(matrix))

from pathlib import Path

MODEL_PATH = Path("DryBeanMC")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "DryBeanClassifier.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

