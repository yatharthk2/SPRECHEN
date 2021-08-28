from EnglishToHindi import model , english , hindi , device
from utils_en_to_hi import translate_sentence
sentence = "hello, how are you?"
translated_sentence = translate_sentence(
        model, sentence, english, hindi , device, max_length=50
    )

print(f"Translated example sentence: \n {translated_sentence}")