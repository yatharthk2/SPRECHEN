from GermanToEnglish import *
from Data_preprocessing import *
import torch
from utils import translate_sentence
sentence = "ein pferd geht unter einer brücke neben einem boot."


with torch.no_grad():
    model.eval()
    translated_sentence = translate_sentence(
            model, sentence, german, english, device, max_length=50
        )
    print(f"Translated example sentence: \n {translated_sentence}")