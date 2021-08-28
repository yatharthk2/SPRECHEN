from numpy import save
import pandas as pd
from torchtext.data import Field, TabularDataset, Iterator , BucketIterator
from sklearn.model_selection import train_test_split
from inltk.inltk import setup
from inltk.inltk import tokenize
import spacy

english_txt = open('EngtoHin/dataset/train.en' , encoding='utf-8').read().split('\n')
hindi_txt = open('EngtoHin/dataset/train.hi' , encoding='utf-8').read().split('\n')

raw_data = {'english' : [line for line in english_txt[1:20000]] , 
            'hindi' : [line for line in hindi_txt[1:20000]]}

df = pd.DataFrame(raw_data , columns=['english' , 'hindi'])

train , test = train_test_split(df , test_size=0.2)
train.to_csv('EngtoHin/dataset/train.csv' , index=False)
test.to_csv('EngtoHin/dataset/test.csv' , index=False)

spacy_eng = spacy.load("en_core_web_sm")
inltk_hindi = setup('hi')



def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]
def tokenize_hin(text):
    return  tokenize(text ,'hi')


hindi = Field(tokenize=tokenize_hin, init_token="<sos>", eos_token="<eos>" , 
        sequential=True )

english = Field(tokenize=tokenize_eng, init_token="<sos>", eos_token="<eos>" 
        , sequential=True , lower=True)

fields = {'english' : ('src' , english) , 'hindi' : ('trg' , hindi)}

train_data , test_data = TabularDataset.splits(path='EngtoHin/dataset/' ,
                 train='train.csv' , test='test.csv' , format='csv' , fields=fields)

english.build_vocab(train_data , min_freq=1 , max_size=20000)
hindi.build_vocab(train_data , min_freq=1 , max_size=20000)

def save_vocab_eng(vocab = english.vocab, path = 'EngtoHin/saved_vocab/train.en'):
    with open(path, 'w+', encoding='utf-8') as f:     
        for token, index in vocab.stoi.items():
            f.write(f'{index}\t{token}\n')
def save_vocab_hin(vocab = hindi.vocab, path = 'EngtoHin/saved_vocab/train.hi'):
    with open(path, 'w+', encoding='utf-8') as f:     
        for token, index in vocab.stoi.items():
            f.write(f'{index}\t{token}\n')

def read_vocab_eng(path = 'EngtoHin/saved_vocab/train.en'):
    vocab = dict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            index, token = line.split('\t')
            vocab[token] = int(index)
    return vocab

def read_vocab_hin(path = 'EngtoHin/saved_vocab/train.hi'):
    vocab = dict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            index, token = line.split('\t')
            vocab[token] = int(index)
    return vocab

save_vocab_eng()
save_vocab_hin()
english.vocab = read_vocab_eng()
hindi.vocab =  read_vocab_hin()

src_vocab_size = len(english.vocab)
trg_vocab_size = len(hindi.vocab)
print(src_vocab_size)
print(trg_vocab_size)
