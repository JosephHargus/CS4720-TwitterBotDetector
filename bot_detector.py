
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
        verified = int(row['Verified'])

        # skip rows with missing critical info
        if pd.isnull(username) or pd.isnull(text): continue

        # feature engineering
        char_count = len(text)
        word_count = str(text).count(' ')
        punct_count = sum(1 for c in str(text) if not c.isalnum() and c.isascii())
        hashtag_count = str(hashtags).count(' ')

        rt_follower_ratio = rt_count / followers if followers != 0 else 1e9
        mentions_follower_ratio = mentions / followers if followers != 0 else 1e9

        username_length = len(username)
        username_num_digits = sum(1 for c in str(username) if c.isnumeric())

        # append features
        features.append([
            rt_count, mentions, followers, verified,
            char_count, word_count, punct_count, hashtag_count,
            rt_follower_ratio, mentions_follower_ratio,
            username_length, username_num_digits
        ])

        # append label
        labels.append(int(row['Bot Label']))
    except Exception as e:
        pass


X = np.array(features, dtype=np.float32)
y = np.array(labels, dtype=np.float32)

# 3. Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3.1 Perform PCA
pca = PCA(n_components=0.95) # keep 95% of variance
X = pca.fit_transform(X)

# 4. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 5. Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 6. Define your model
class BotDetectorNN(nn.Module):
    def __init__(self, input_size):
        super(BotDetectorNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 48)
        self.fc2 = nn.Linear(48, 24)
        self.fc3 = nn.Linear(24, 2)  # 2 outputs: bot or human
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
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

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
print('Confusion matrix:\n' + str(cm))
