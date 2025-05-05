import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics

df = pd.read_csv('bot_detection_data.csv')
# Convert 'Verified' True/False into 0/1
df['Verified'] = df['Verified'].astype(int)

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
            rt_follower_ratio, mentions_follower_ratio
            #username_length, username_num_digits
        ])

        # append label
        labels.append(int(row['Bot Label']))
    except Exception as e:
        pass

X = np.array(features, dtype=np.float32)
y = np.array(labels, dtype=np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train[0:20])

print("training...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print(metrics.classification_report(y_test, y_pred))