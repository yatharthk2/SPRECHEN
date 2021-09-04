<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
<!--[![Stargazers][stars-shield]][stars-url]-->
[![Issues][issues-shield]][issues-url]




<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/yatharthk2/SPRECHEN">
    <img src="https://github.com/yatharthk2/SPRECHEN/blob/master/IVG/head_image.png" alt="Logo" width="1080" height="500">
  </a>

  <p align="center">
    <h3 align="center">A state-of-the-art Translator to translate EnglishToHindi and GermanToEnglish </h3>
    <br />
    <a href="https://github.com/yatharthk2/SPRECHEN"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/yatharthk2/SPRECHEN/blob/master/result.jpg">View Demo</a>
    ·
    <a href="https://github.com/yatharthk2/SPRECHEN/issues">Report Bug</a>
    ·
    <a href="https://github.com/yatharthk2/SPRECHEN/issues">Request Feature</a>
  </p>
</p>

<!-- ABOUT THE PROJECT -->
## About The Project



Implemented takeaways from the original transformer paper "Attention is all you need" and deployed an idea to code a language translator /converter that extracts data from the epochs entered in one language  , trains it and convenes the translation of one language into another. 
The proposed automated English-to-local-language system architecture is designed according to the transfer-based approach to automating translation. At the first layer of the architecture is a natural language processing tool that performs morphological analysis: sentence tokenization, part-of-speech tagging, phrase formation and figure of speech tagging. The second layer of the architecture comprises of a grammar generator that is responsible for converting English language structure to target local language structures. At the 
third layer of the application, results produced by the grammar generator are mapped with matching terms in the bilingual dictionary. The  architectures  for a transfer-based 
model and automated German-English and English-Hindi translator are trained respectively as per as the above mentioned procedure. 

### Architecture: The Image depicts flowchart for working of Encoder and Decoder based architecture from Vaswani et. al.
Image taken from 
<a href="https://arxiv.org/pdf/1706.03762.pdf"><strong>Research Paper »</strong></a>

<img src="https://github.com/yatharthk2/SPRECHEN/blob/master/IVG/arch_image.png" alt="Logo" width="800" height="800">

### GermanToEnglish : 
Finding libraries to tokenise and setup vocab was fairly simple . We used Spacy for the same purpose and trained the model on Multi30k dataset for 150 epochs .
One important thing to note is 
value of some parameters such as HeadCount and Encoder-Decoder iterations where reduced in comparison to original paper 
considering limited computational power we had at our disposal.
. Prerained weights for GerToEng can be found 
<a href="https://drive.google.com/drive/folders/1Mqy2gOyDwlv-MeqNqAFeRnVsGg7obLYg?usp=sharing"><strong>here</strong></a>

### EnglishToHindi : 
Tokenising and setting up the vocab was quite difficult given the complexities of Hindi grammers , but task was done using 
<a href="https://github.com/goru001/inltk"><strong>INLTK lib</strong></a>.
The model was trained on english & hindi parallel corpus from 
<a href="https://www.kaggle.com/aiswaryaramachandran/hindienglish-corpora"><strong>Dataset</strong></a>
and more 80k lines from 24 lakh parallel corpus from 
<a href="http://lotus.kuee.kyoto-u.ac.jp/WAT/indic-multilingual/index.html"><strong>Dataset</strong></a> , preprocessed and cleaned version can be found 
<a href="https://drive.google.com/drive/folders/1ZezM4OWqsdPhYHQdbgmzaIP2BzDSNaKM?usp=sharing"><strong>here</strong></a>.
Same goes here too ,Value of some parameters such as HeadCount and Encoder-Decoder 
iterations were reduced in comparison to original paper considering limited computational power we had at our disposal.
Prerained weights for EngToHin can be found 
<a href="https://drive.google.com/drive/folders/1dtSJcRFfTNLR0x4VmKUR32vbRQUZejwg?usp=sharing"><strong>here</strong></a>

### Built With
1) TorchText
2) Spacy
3) INLTK
4) Nvidia cuda toolkit



<!-- GETTING STARTED -->
## Getting Started

Step 1. Clone the repository.

Step 2. Download the dataset from <a href="https://drive.google.com/drive/folders/1ZezM4OWqsdPhYHQdbgmzaIP2BzDSNaKM?usp=sharing">Here</a> and place it in the respective data  file. Remember both the translation pipelines have diffterent data folder

### Installation

* Python 3.7
  
* Install python libraries
  ```sh
  conda install -r requirements.txt
  ```
### Testing
* Run training file (engtohindi.py or gertoeng.py) , it will build tokenised vocab for you.(this process is needed to be done only once) 

* Add check points to the Folder:
   Download the checkpoints from <a href="https://drive.google.com/drive/folders/1ZezM4OWqsdPhYHQdbgmzaIP2BzDSNaKM?usp=sharing">Here</a> and place it in the respective checkpoints   file. Remember both the translation pipelines have diffterent checkpoints folder
 
* Edit the sentence variable in the eval.py file , for the sentce you want to translate .
* run eval.py file 
  ```sh
  python GerToEng/eval.py
  ```
  

### Training

* Start from scratch
  ```sh
  python GerToEng/GermanToEnglish.py
  ```
* To resume training : change load parameter in hyperparameter file to true , the model will automatically load the checkpoints 
  ```sh
  python GerToEng/GermanToEnglish.py
  ```
<!-- CONTRIBUTING -->
## References
1. <a href="https://arxiv.org/abs/1706.03762"><strong>Attention is all you need  paper»</strong></a> 
3. <a href="https://analyticsindiamag.com/top-nlp-libraries-datasets-for-indian-languages/"><strong>dataset for English and Hindi parallel corpus »</strong></a> 
4. <a href="https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/seq2seq_transformer/seq2seq_transformer.py"><strong>Transformers base implementation »</strong></a> 
<!-- CONTACT -->
## MadeBy
* Contact Yatharth Kapadia @yatharthk2.nn@gmail.com 
* Contact Abhinav Chandra @abhinavchandra0526@gmail.com
* contact Sidarth Jain @...




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/yatharthk2/SPRECHEN?color=red&logo=github&logoColor=green&style=for-the-badge
[contributors-url]: https://github.com/yatharthk2/SPRECHEN/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/yatharthk2/sprechen?color=red&logo=github&logoColor=green&style=for-the-badge
[forks-url]: https://github.com/yatharthk2/Sprechen/network/members
<!--[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge-->
<!--[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers-->
[issues-shield]: https://img.shields.io/bitbucket/issues/yatharthk2/sprechen?color=red&logo=github&logoColor=green&style=for-the-badge
[issues-url]:https://github.com/yatharthk2/SPRECHEN/issues
