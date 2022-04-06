'''
Python 3.9.7 (default, Sep 16 2021, 13:09:58) 
[GCC 7.5.0] :: Anaconda, Inc. on linux
'''
import os
from time import sleep
from datetime import datetime
import concurrent.futures
import json
from requests_oauthlib import OAuth1Session
import csv

begin = datetime.now()

BASE_DATASETS_PATH='../FakeNews_Dataset/CSV'
POLITIFACT_REAL_DATASET='sampled_real_news.csv'
POLITIFACT_FAKE_DATASET='sampled_fake_news.csv'
TWITTER_API_CONSUMER_KEY=os.environ.get('TWITTER_API_CONSUMER_KEY')
TWITTER_API_CONSUMER_KEY_SECRET=os.environ.get('TWITTER_API_CONSUMER_KEY_SECRET')
TWITTER_API_ACCESS_TOKEN=os.environ.get('TWITTER_API_ACCESS_TOKEN')
TWITTER_API_ACCESS_TOKEN_SECRET=os.environ.get('TWITTER_API_ACCESS_TOKEN_SECRET')
TWITTER_BASE_URL='https://api.twitter.com/2/tweets'
TWITTER_QUERY='{{tweet_id}}?tweet.fields=author_id'
FAKE_NEWS=0
REAL_NEWS=1
REAL_COLLECTED_DATA=[['politifactid','author_id','tweet_text','news_url','label']]
FAKE_COLLECTED_DATA=[['politifactid','author_id','tweet_text','news_url','label']]

# author_id,tweet_text,news_url,label

# working tweet id: 974104473168695296
# private tweet id: 937696540453466112

# load datasets
with open(f'{BASE_DATASETS_PATH}/{POLITIFACT_REAL_DATASET}','r') as real_ds:
    politifact_real = (real_ds.read().splitlines())
with open(f'{BASE_DATASETS_PATH}/{POLITIFACT_FAKE_DATASET}','r') as fake_ds:
    politifact_fake = (fake_ds.read().splitlines())

def perform_request(tid):
    url = f'{TWITTER_BASE_URL}/{TWITTER_QUERY}'.replace('{{tweet_id}}', tid)
    twitter_session = OAuth1Session(client_key=TWITTER_API_CONSUMER_KEY,
        client_secret=TWITTER_API_CONSUMER_KEY_SECRET,
        resource_owner_key=TWITTER_API_ACCESS_TOKEN,
        resource_owner_secret=TWITTER_API_ACCESS_TOKEN_SECRET)
    req = twitter_session.get(url, timeout=60)
    print(f"TweetID: {tid} - StatusCode: {req.status_code}")
    sleep(5)
    return req

for real_news_row in politifact_real[1:]:
    splitted_row = real_news_row.split(',')
    politifact_id = splitted_row[0]
    news_url = splitted_row[1]
    tweet_ids_to_collect = splitted_row[3].split('\t')

    if tweet_ids_to_collect[0]:
        # perform http request
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_tweet = {executor.submit(perform_request, tid): tid for tid in tweet_ids_to_collect}
            for future in concurrent.futures.as_completed(future_to_tweet):
                tweet_data = future_to_tweet[future]
                print('REAL_COLLECTED len', len(REAL_COLLECTED_DATA))
                try:
                    data = future.result()
                    if data.status_code == 200:
                        resp = json.loads(data.content)
                        try:
                            data = resp['data']
                        except:
                            pass
                        else:
                            tweet_id = data['id']
                            tweet_author = data['author_id']
                            tweet_text = data['text'].replace('\n', ' ')
                            REAL_COLLECTED_DATA.append([politifact_id, tweet_id, tweet_author, tweet_text, news_url, REAL_NEWS])
                    elif data.status_code == 429:
                        print(data.headers)
                        sleep(10)
                    else:
                        pass
                except Exception as e:
                    print('%r generated an exception %s' % (tweet_data,e))

with open(f'collected_real_news.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(REAL_COLLECTED_DATA)

for fake_news_row in politifact_fake[1:]:
    splitted_row = fake_news_row.split(',')
    politifact_id = splitted_row[0]
    news_url = splitted_row[1]
    tweet_ids_to_collect = splitted_row[3].split('\t')

    if tweet_ids_to_collect[0]:
        # perform http request
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_tweet = {executor.submit(perform_request, tid): tid for tid in tweet_ids_to_collect}
            for future in concurrent.futures.as_completed(future_to_tweet):
                tweet_data = future_to_tweet[future]
                print('FAKE_COLLECTED len', len(FAKE_COLLECTED_DATA))
                try:
                    data = future.result()
                    if data.status_code == 200:
                        resp = json.loads(data.content)
                        try:
                            data = resp['data']
                        except:
                            pass
                        else:
                            tweet_id = data['id']
                            tweet_author = data['author_id']
                            tweet_text = data['text'].replace('\n', ' ')
                            FAKE_COLLECTED_DATA.append([politifact_id, tweet_id, tweet_author, tweet_text, news_url, FAKE_NEWS])
                    elif data.status_code == 429:
                        print(data.headers)
                        sleep(10)
                    else:
                        pass
                except Exception as e:
                    print('%r generated an exception %s' % (tweet_data,e))

with open(f'collected_fake_news.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(FAKE_COLLECTED_DATA)

end = datetime.now()

print("Started:", begin.strftime("%H:%M:%S"))
print("Ended:", end.strftime("%H:%M:%S"))