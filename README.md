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
[![MIT License][license-shield]][license-url]
<!--[![LinkedIn][linkedin-shield]][linkedin-url]-->



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/yatharthk2/Inpainting">
    <img src="https://github.com/yatharthk2/SPRECHEN/blob/master/IVG/head_image.png" alt="Logo" width="1080" height="500">
  </a>

  <p align="center">
    <h3 align="center">A state-of-the-art Translator to translate EnglishToHindi and GermanToEnglish </h3>
    <br />
    <a href="https://github.com/yatharthk2/Inpainting"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/yatharthk2/Inpainting/blob/master/result.jpg">View Demo</a>
    ·
    <a href="https://github.com/yatharthk2/Inpainting/issues">Report Bug</a>
    ·
    <a href="https://github.com/yatharthk2/Inpainting/issues">Request Feature</a>
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

<img src="https://github.com/yatharthk2/SPRECHEN/blob/master/IVG/arch_image.png" alt="Logo" width="1080" height="300">

### GermanToEnglish : 
Finding libraries to tokenise and setup vocab was fairly simple . We used Spacy for the same purpose and trained the model on Multi30k dataset for 150 epochs .
One important thing to note is 
Value of some parameters such as HeadCount and Encoder-Decoder iterations where reduced in comparison to original paper 
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
iterations where reduced in comparison to original paper considering limited computational power we had at our disposal.
Prerained weights for EngToHin can be found 
<a href="https://drive.google.com/drive/folders/1dtSJcRFfTNLR0x4VmKUR32vbRQUZejwg?usp=sharing"><strong>here</strong></a>

### Built With
1) Pytorch
2) Opencv
3) Nvidia cuda toolkit



<!-- GETTING STARTED -->
## Getting Started

Step 1. Clone the repository.

Step 2. Download the dataset and pretrained weights from <a href="https://drive.google.com/drive/folders/1E482OOOe_xYWVE9nKCnF_hrh0aLHgZIN?usp=sharing">Here</a> and place it in the same directory.

### Installation

* Python 3.6+
* Install Pytorch

  (for cude 10.2 – GPU)
  ```sh
  pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
  ```
  (for CPU)
  ```sh
  pip install torch torchvision torchaudio
  ```
  
* Install python libraries
  ```sh
  pip install -r requirements.txt
  ```

### Training

* Start from scratch
  ```sh
  python train.py
  ```
* Resume training
  ```sh
  python train.py --resume <weights_path>
  ```
### Testing

* Run the command line
  ```sh
  python run.py --photo <test_image_path>
  ```
* Draw Mask
* Press "s"

Output will be saved in the root directory in ```result.jpg``` format. 


<!-- CONTRIBUTING -->
## References
1. <a href="https://arxiv.org/pdf/1804.07723.pdf"><strong>Partial convolution research paper »</strong></a> 
2. <a href="https://openaccess.thecvf.com/content_ICCV_2017/papers/Harley_Segmentation-Aware_Convolutional_Networks_ICCV_2017_paper.pdf"><strong>Segmentation aware convolution research paper »</strong></a>
3. <a href="https://github.com/NVIDIA/partialconv"><strong>reference code Implementation »</strong></a> 
4. <a href="https://github.com/naoto0804/pytorch-inpainting-with-partial-conv"><strong>Base code Implementation »</strong></a> 
5. <a href="https://github.com/spmallick/learnopencv/blob/master/Image-Inpainting/inpaint.py"><strong>Manual mask generator code reference »</strong></a> 

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact
* Contact Yatharth Kapadia @yatharthk2.nn@gmail.com
* Contact Poojan Panchal @ pdavpoojan@gmail.com 




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/yatharthk2/Inpainting?color=red&label=contributors&logo=github&logoColor=green&style=for-the-badge
[contributors-url]: https://github.com/yatharthk2/Inpainting/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/yatharthk2/Inpainting?color=red&logo=github&logoColor=green&style=for-the-badge
[forks-url]: https://github.com/yatharthk2/Inpainting/network/members
<!--[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge-->
<!--[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers-->
[issues-shield]: https://img.shields.io/bitbucket/issues-raw/yatharthk2/Inpainting?color=red&logo=github&logoColor=green&style=for-the-badge
[issues-url]:https://github.com/yatharthk2/Inpainting/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/yatharthk2/Inpainting/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: C:\Users\yatha\OneDrive\Desktop\projects\Inpainting_project\Inpainting\train_video.gif
