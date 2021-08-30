from model import Transformer
import torch
import spacy 
from utils import translate_sentence
from torchtext.data import Field, BucketIterator , TabularDataset
from torchtext.datasets import Multi30k
from utils import load_checkpoint
import torch.optim as optim
from inltk.inltk import setup
from inltk.inltk import tokenize
sentence = "Hello how do you do"

spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")
num_epochs = 10000
learning_rate = 3e-4
batch_size = 32


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

english.build_vocab(train_data , min_freq=1 , max_size=200)
hindi.build_vocab(train_data , min_freq=1 , max_size=200)

src_vocab_size = len(english.vocab)
trg_vocab_size = len(hindi.vocab)
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10

forward_expansion = 4
src_pad_idx = hindi.vocab.stoi["<pad>"]
max_len = 200
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
load_checkpoint(torch.load("EngToHin/checkpoints/my_checkpoint.pth.tar"), model, optimizer)


with torch.no_grad():
    model.eval()
    translated_sentence = translate_sentence(
            model, sentence, english, hindi , device, max_length=50
        )
    print(f"Translated example sentence: \n {translated_sentence}")
