import spacy
import pandas as pd
from torchtext.data import Field, TabularDataset, Iterator , BucketIterator
from sklearn.model_selection import train_test_split

english_txt = open('C:\\Users\\yatha\\Desktop\\dataset\\finalrepo\\train\\alt\\en-hi\\train.en' , encoding='utf-8').read().split('\n')
hindi_txt = open('C:\\Users\\yatha\\Desktop\\dataset\\finalrepo\\train\\alt\\en-hi\\train.hi' , encoding='utf-8').read().split('\n')

raw_data = {'en' : [line for line in english_txt[1:10]] , 
            'hi' : [line for line in hindi_txt[1:10]]}

df = pd.DataFrame(raw_data , columns=['en' , 'hi'])

train , test = train_test_split(df , test_size=0.2)
train.to_csv('dataset_en_hi/train.csv' , index=False)
test.to_csv('dataset_en_hi/test.csv' , index=False)
