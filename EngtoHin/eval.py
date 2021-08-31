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
num_epochs = 10000
learning_rate = 3e-4
batch_size = 32


spacy_eng = spacy.load("en_core_web_sm")
inltk_hindi = setup('hi')

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]
def tokenize_hin(text):
    return  tokenize(text ,'hi')
english = torch.load('EngtoHin/saved_vocab/english_obj.pth')
hindi = torch.load('EngtoHin/saved_vocab/hindi_obj.pth')
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
