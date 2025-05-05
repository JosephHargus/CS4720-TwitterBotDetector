
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load your data
df = pd.read_csv('bot_detection_data.csv')
# Convert 'Verified' True/False into 0/1
df['Verified'] = df['Verified'].astype(int)


# 2. Extract and engineer features
features = []
labels = []

for _, row in df.iterrows():
    try:
        username = row['Username']
        text = row['Tweet']
        hashtags = row['Hashtags']
        rt_count = int(row['Retweet Count'])
        mentions = int(row['Mention Count'])
        followers = int(row['Follower Count'])

        # skip rows with missing critical info
        if pd.isnull(username) or pd.isnull(text): continue

        # feature engineering
        char_count = len(text)
        word_count = str(text).count(' ')
        punct_count = sum(1 for c in str(text) if not c.isalnum() and c.isascii())

        username_length = len(username)

        from textblob import TextBlob
        import textstat
        # Calculate sentiment polarity, subjectivity, and readability
        sentiment = TextBlob(text).sentiment
        polarity = sentiment.polarity
        subjectivity = sentiment.subjectivity
        readability = textstat.flesch_reading_ease(text)

        # append features
        features.append([
            rt_count, mentions, followers,
            char_count, word_count, punct_count,
            username_length,
            polarity, subjectivity, readability
        ])

        # append label
        labels.append(int(row['Bot Label']))
    except Exception as e:
        pass

# # Plot feature distributions for each class
# features = np.array(features)
# labels = np.array(labels)

# num_features = features.shape[1]
# feature_names = [
#     "Retweet Count", "Mention Count", "Follower Count",
#     "Char Count", "Word Count", "Punct Count",
#     "Username Length",
#     "Polarity", "Subjectivity", "Readability"
# ]

# for i in range(num_features):
#     plt.figure(figsize=(8, 4))
#     bot_feature = features[labels == 1, i]
#     human_feature = features[labels == 0, i]
#     sns.kdeplot(bot_feature, label="Bot", shade=True, color="red")
#     sns.kdeplot(human_feature, label="Human", shade=True, color="blue")
#     plt.title(f"Feature Distribution: {feature_names[i]}")
#     plt.xlabel(feature_names[i])
#     plt.ylabel("Density")
#     plt.legend()
#     plt.show()

X = np.array(features, dtype=np.float32)
y = np.array(labels, dtype=np.float32)

# 3. Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3.1 Perform PCA
pca = PCA(n_components=0.95) # keep 95% of variance
X = pca.fit_transform(X)

# 4. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# split into test/validation
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 5. Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# 6. Define your model
class BotDetectorNN(nn.Module):
    def __init__(self, input_size):
        super(BotDetectorNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)  # 2 outputs: bot or human
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = len(X[0])
model = BotDetectorNN(input_size)

# 7. Set up loss and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 8. Train the model
test_losses = []
val_losses = []
epochs = 25
for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    test_losses.append(total_loss / len(train_loader))

    # Validation step
    with torch.no_grad():
        val_loss = 0
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = loss_function(outputs, batch_y)
            val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

# display learning curves
plt.plot(test_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curves')
plt.legend()
plt.show()

# 9. Test the model
correct = 0
total = 0
all_preds = []
all_true = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_true.extend(batch_y.numpy())
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
cm = confusion_matrix(all_true, all_preds)
sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()
print('Confusion matrix:\n' + str(cm))
