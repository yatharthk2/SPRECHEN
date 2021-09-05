from model import Transformer
import torch
import spacy 
from utils import translate_sentence
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
from utils import load_checkpoint
import torch.optim as optim
sentence = "Jungen tanzen mitten in der Nacht auf Pfosten."

spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")
num_epochs = 10000
learning_rate = 3e-4
batch_size = 32




'''def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")

english = Field(
    tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
)

train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english) , root="GerToEng\\dataset"
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)'''
def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]
english = torch.load('GerToEng/saved_vocab/english_obj.pth')
german = torch.load('GerToEng/saved_vocab/german_obj.pth')

src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100
forward_expansion = 4
src_pad_idx = english.vocab.stoi["<pad>"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
load_checkpoint(torch.load("GerToEng/checkpoints/my_checkpoint.pth.tar"), model, optimizer)


with torch.no_grad():
    model.eval()
    translated_sentence = translate_sentence(
            model, sentence, german, english, device, max_length=50
        )
    print(f"Translated example sentence: \n {translated_sentence}")
