import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

import torch 
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

import sys


def pd_data():
    df = pd.read_csv('data/raw/collected_news.csv')
    #df.columns=['politifactid','tweet_id','author_id','tweet_text','news_url','label']
    item_encoder = LabelEncoder()
    df['tweet_id'] = item_encoder.fit_transform(df.tweet_id)

    #Encoding 
    #tweet_politifact_id = LabelEncoder().fit_transform(df.politifactid)
    encoded_url_id = LabelEncoder().fit_transform(df.news_url)
    encoded_politifact_id = LabelEncoder().fit_transform(df.politifactid)
    encoded_author_id = LabelEncoder().fit_transform(df.author_id)
    df['encoded_politifact_id'] = encoded_politifact_id
    df['encoded_news_url'] = encoded_url_id
    df['encoded_author_id'] = encoded_author_id
    return df


def hello_world():
    print("hello world")


class DataFakeNews(InMemoryDataset):
    def __init__(self, root, transform =None, pre_transform=None):
        super(DataFakeNews, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'collected_news.csv'

    @property
    def processed_file_names(self):
        return 'graph_collected_news.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. No download allowed')
    
    def process(self):
        data_list = []
        counter = 0
        df = pd_data()
        #print(df.head())
        grouped = df.groupby('tweet_id')
        #tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        #print(grouped.head())
        #print(grouped)
        for tweet_id, group in tqdm(grouped):
            
            #tokens_politifactid = tokenizer.tokenize(group.politifactid)
            #politifactid_tokens = tokenizer.tokenizer.convert_tokens_to_ids(tokens_tweet)

            #tweet_politifact_id = LabelEncoder().fit_transform(group.politifactid)
            #encoded_url_id = LabelEncoder().fit_transform(group.news_url)
            #encoded_politifact_id = LabelEncoder().fit_transform(group.politifactid)
            #encoded_author_id = LabelEncoder().fit_transform(group.author_id)
            #print(encoded_url_id)
            group = group.reset_index(drop=True)
            #group['tweet_politifact_id'] = tweet_politifact_id
            #print(group['author_id'].drop_duplicates().values[0] == 'author_id')
            if (group['author_id'].drop_duplicates().values)[0] != 'author_id':
                group['author_id'] = group['author_id'].astype(int)
            else:
                group['author_id'] = 0
            
            #group['encoded_politifact_id'] = encoded_politifact_id
            #group['encoded_news_url'] = encoded_url_id
            #group['encoded_author_id'] = encoded_author_id
            #print(tweet_id)
            features = group.loc[group['tweet_id'] == tweet_id, ['encoded_politifact_id', 'author_id', 'encoded_news_url']].drop_duplicates().values

            features = torch.FloatTensor(features).unsqueeze(1)
            
            #Getting tweet ID
            target_nodes = group['tweet_id'].values.astype(int)
            source_nodes = group['encoded_news_url'].values.astype(int)
            #Connecting to News_URL
            #print("Y value: ", group.label.values.astype(int))
            if (group.label.values[0] != 'label'):
                label_array = np.array(group.label.values.astype(int))
                y = torch.FloatTensor(label_array)
            else:
                source_nodes = [1]
                y = torch.FloatTensor([1])

            '''
            print(group)
            #print(politifactid_tokens)
           
            if counter == 100:
                sys.exit("BREAKING")
            
            counter+=1
            '''
            #print(source_nodes)
            #print("Target Node Type: ", type(target_nodes[0]))
            #print("Source Node Type: ", type(source_nodes[0]))
            #print("Feature 1 Type:", type(features[0][0]))
            #print("Feature 2 Type:", type(features[0][1]))
            #print("Feature 3 Type:", type(features[0][2]))
            #print("Feature 4 Type:", type(features[0][3]))

            #print(group['encoded_news_url'])
            target_source_array = np.array([target_nodes, source_nodes])
            edge_index = torch.tensor(target_source_array, dtype=torch.long)
            #x = node_features
            data = Data(x=features, edge_index=edge_index, y=y)
            data_list.append(data)
        
        #print(len(data_list))
        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])


class FakeNewsBinaryDataset(InMemoryDataset):
    def __init___(self, root, transform=None, pre_transform=None):
        super(LOL, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


        @property
        def raw_file_names(self):
            return 'collected_news.csv'

        @property
        def processed_file_names(self):
            return 'graph_collected_news.pt'

        def download(self):
            pass
        
        def process(self):
            data_list = []
            df = pd_data()
            #sys.exit("Error message")
            grouped = df.groupby('tweet_id')
            #print(grouped)
            for tweet_id, group in tqdm(grouped):
                tweet_politifact_id = LabelEncoder().fit_transform(group.politifactid)
                group = group.reset_index(drop=True)
                group['tweet_politifact_id'] = tweet_politifact_id
                node_features = group.loc[group.tweet_id==tweet_id, ['tweet_politifact_id', 'author_id', 'news_url', 'label']].sort_values('tweet_politifact_id').tweet_id.drop_duplicates().values

                node_features = torch.LongTensor(node_features).unsqueeze(1)
                target_nodes = group.tweet_politifact_id.values[1:]
                source_nodes = group.tweet_politifact_id.values[:-1]

                edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
                #x = node_features

                #y = torch.FloatTensor([group.label.values[0]])

                data = Data(x=node_features, edge_index=edge_index, y=torch.FloatTensor([group.label.values[0]]))
                data_list.append(data)

            data, slices = self.collate(data_list)
            
            torch.save((data, slices), self.processed_paths[0])

if __name__ == "main":
    pass