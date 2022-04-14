import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from preprocessing import *
from tqdm import tqdm

#Processed Data
#df = pd_data()

#Not processed Data

'''Testing
df = pd.read_csv('../collected_news.csv')
grouped = df.groupby('tweet_id')
for tweet_id, group in grouped:
    tweet_politifact_id = LabelEncoder().fit_transform(group.politifactid)
    group = group.reset_index(drop=True)
    group['tweet_politifact_id'] = tweet_politifact_id
    print(tweet_id)
    featuers = group.loc[group['tweet_id'] == tweet_id, ['tweet_politifact_id','politifactid', 'author_id','tweet_text', 'news_url']].drop_duplicates().values
    print(featuers)

    #Getting tweet ID
    target_nodes = int(group.values[:,1])
    
    #Connecting to News_URL
    source_nodes = group.values[:,-1]

    print("Target:", target_nodes)
    print("Source:", source_nodes)
    print([target_nodes, source_nodes[0]])
    edge_index = torch.tensor([target_nodes, source_nodes[0]], dtype=torch.long)
    break
'''

#print(df.head())

test_class = DataFakeNews(root="data/")
test_class = test_class.shuffle()

print(len(test_class))

#test_dataset = FakeNewsBinaryDataset(root='data/')
#dataset = test_dataset.shuffle()


cut_data_length = int(len(test_class) * 0.1)
train_dataset = test_class[:cut_data_length * 8]
val_dataset = test_class[cut_data_length*8:cut_data_length*9]
test_dataset = test_class[cut_data_length*9:]

print("Training DataSet", len(train_dataset))
print("Val DataSet", len(val_dataset))
print("Test DataSet", len(test_dataset))
for x in train_dataset:
    print(x['edge_index'])
'''
df = pd.read_csv('../collected_news.csv')
#Normalizing the tweet ids
item_encoder = LabelEncoder()
df['tweet_id'] = item_encoder.fit_transform(df.tweet_id)
grouped = df.groupby('tweet_id')
for tweet_id, group in tqdm(grouped):
    print(group)
    break
'''