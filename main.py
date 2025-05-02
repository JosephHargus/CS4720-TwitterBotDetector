import numpy

# read data to numpy arrays
data = numpy.loadtxt("bot_detection_data.csv", delimiter=',', dtype=str, skiprows=1)
print(data.shape)

# construct feature vector array
# features: [username, tweet, retweet count, mention count, follower count, verified]
input_features = numpy.array(data[:,1:7])
print(input_features.shape)

# construct label array
labels = numpy.array(data[:,7])
print(labels.shape)

# convert data types, remove null entries, engineer new features
features = []
for fvec in input_features:
    # check for None entries
    isNone = False
    for feature in fvec:
        if feature is None: isNone = True
    if isNone: continue

    # convert data types
    rt_count = int(fvec[2])
    mentions = int(fvec[3])
    followers = int(fvec[4])
    verified = 1 if bool(fvec[5]) else 0

    # engineer new and additional features
    char_count = len(fvec[1])
    word_count = str(fvec[1]).count(' ')
    punct_count = 0
    for c in str(fvec[1]):
        if not c.isalnum and c.isascii:
            punct_count += 1

    hashtag_count = str(fvec[5]).count(' ')
    if followers == 0:
        rt_follower_ratio = 1000000000
        mentions_follower_ratio = 1000000000
    else:
        rt_follower_ratio = rt_count / followers
        mentions_follower_ratio = mentions / followers
    username_length = len(fvec[0])
    username_num_digits = 0
    for c in str(fvec[0]):
        if c.isnumeric: username_num_digits += 1

    features.append([rt_count, mentions, followers, 
                            verified, char_count, word_count, 
                            punct_count, hashtag_count,
                            rt_follower_ratio,mentions_follower_ratio,
                            username_length,username_num_digits])
    
features = numpy.array(features)
print(features.shape)

# perform PCA to retain only features which capture the most variance
