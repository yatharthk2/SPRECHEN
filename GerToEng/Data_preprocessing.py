import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")

english = Field(
    tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
)

train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english) , root="GerToEng/dataset"
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

def save_vocab_eng(vocab = english.vocab, path = 'GerToEng/saved_vocab/train.en'):
    with open(path, 'w+', encoding='utf-8') as f:     
        for token, index in vocab.stoi.items():
            f.write(f'{index}\t{token}\n')
def save_vocab_ger(vocab = german.vocab, path = 'GerToEng/saved_vocab/train.de'):
    with open(path, 'w+', encoding='utf-8') as f:     
        for token, index in vocab.stoi.items():
            f.write(f'{index}\t{token}\n')

def read_vocab_eng(path = 'GerToEng/saved_vocab/train.en'):
    vocab = dict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            index, token = line.split('\t')
            vocab[token] = int(index)
    return vocab

def read_vocab_ger(path = 'GerToEng/saved_vocab/train.de'):
    vocab = dict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            index, token = line.split('\t')
            vocab[token] = int(index)
    return vocab

save_vocab_eng()
save_vocab_ger()
english.vocab = read_vocab_eng()
german.vocab =  read_vocab_ger()

src_vocab_size = len(english.vocab)
trg_vocab_size = len(german.vocab)
