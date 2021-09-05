import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
#from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator , TabularDataset
from sklearn.model_selection import train_test_split
from inltk.inltk import setup
from inltk.inltk import tokenize
import pandas as pd
from model import Transformer
from hyperparam import *

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

english.build_vocab(train_data , min_freq=1 , max_size=10000)
hindi.build_vocab(train_data , min_freq=1 , max_size=10000)

torch.save(english , 'EngtoHin/saved_vocab/english_obj.pth')
torch.save(hindi , 'EngtoHin/saved_vocab/hindi_obj.pth')
src_vocab_size = len(english.vocab)
trg_vocab_size = len(hindi.vocab)
src_pad_idx = hindi.vocab.stoi["<pad>"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Tensorboard to get nice loss plot
writer = SummaryWriter("EngtoHin/runs/loss_plot")
step = 0

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

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

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

pad_idx = hindi.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("EngtoHin/checkpoints/my_checkpoint.pth.tar"), model, optimizer)

sentence = "Zelaya asked his people to be able to practice his rights peacefully after his arrival in the Central American country."

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, english, hindi , device, max_length=50
    )

    print(f"Translated example sentence: \n {translated_sentence}")
    model.train()
    losses = []

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # Forward prop
        output = model(inp_data, target[:-1, :])

        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        losses.append(loss.item())

        # Back prop
        loss.backward()
        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)

# running on entire test data takes a while
score = bleu(test_data[1:100], model, english , hindi, device)
print(f"Bleu score {score * 100:.2f}")