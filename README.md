
# Generating ART using a Simple GAN architecture
## Introduction
After being inspired by thousand Text-To-Art models I took it upon myself to learn more about the field.\
 So as a First step I made this project which was for educational purposes only.\
It consists of :
- Finding a good and free Dataset
- Training a Simple **GAN architecture** On the dataset
- Viewing the model's progress in training
- And finally being able to run inference using the trained weights

## Training process
Going from generating random noise to somewhat of an art.
<p align="center">
  <img src="https://github.com/houssemeddinebayoudha/GAN-art-portraits/blob/main/Gif/animation_5.gif" alt="animated" />
</p>
## What's a GAN ?
# GAN (Generative Adversarial Network)

A Generative Adversarial Network (GAN) is a type of deep learning model that is used for generating new, synthetic data that is similar to a training dataset. It consists of two neural networks: a generator and a discriminator. The generator produces synthetic data, while the discriminator determines whether the data is real or fake.

The generator and discriminator are trained together in an adversarial process, where the generator tries to produce synthetic data that is indistinguishable from real data, and the discriminator tries to correctly identify whether the data is real or fake. The goal is to find a balance where the generator is able to produce high-quality synthetic data, while the discriminator is only able to correctly identify real data a small percentage of the time.

GANs have been used for a wide range of tasks, including generating synthetic images, synthesizing speech, and even translating one language to another. They are a powerful tool for generating synthetic data, and have led to significant advances in the field of machine learning.

## How GANs work

The training process for a GAN involves alternating between training the generator and discriminator.

First, the generator is given a random noise vector as input, and produces a synthetic data sample. The discriminator is then presented with both the synthetic data sample and a real data sample, and tries to correctly classify them as either real or fake. The generator is then updated based on the performance of the discriminator, in order to try and produce synthetic data that is more similar to the real data.

This process is repeated until the generator is able to produce synthetic data that is indistinguishable from real data, and the discriminator is only able to correctly classify real data a small percentage of the time.

## Applications of GANs

GANs have been used for a wide range of applications, including:

- Generating synthetic images: GANs have been used to generate realistic images of faces, landscapes, and even animals.

- Synthesizing speech: GANs have been used to synthesize speech in a variety of languages, allowing for the generation of new, synthetic audio data.

- Language translation: GANs have been used to translate between languages, allowing for the generation of synthetic translations of text.

Overall, GANs are a powerful tool for generating synthetic data, and have the potential to revolutionize a wide range of applications in machine learning and artificial intelligence.\
\
*This information was provided by an AI trained by [OpenAI](https://openai.com).*

## Process
To begin I started searching for a good dataset that I can train the model on\
After countless 
hours I was able to find a good Hand Drawn Portraits [Dataset](https://www.kaggle.com/datasets/karnikakapoor/art-portraits)\
which was perfect for the kind of art I was hoping to generate.\
Now it came down to modeling:
As explained before I used a simple GAN architecture that was very much inspired by [This implementation](https://www.kaggle.com/code/trungthanhnguyen0502/gan-anime/notebook)
.\
The rest of the process can be found in the Colab Notebook. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/185WY6I6Z5CcvQID7hZryb357O95AHYi9?usp=sharing)






## Results
After Hours of training (315 epochs) I got the following results : 



![](https://i.imgur.com/18ajqrF.png)
![](https://i.imgur.com/3khaQIL.png)

Even tho the results are far from perfect I'm really happy with the way it turned out.\
There is no denying that there is a lot of room for improvement and I'm happy to hear your suggestions.

## Acknowledgements

 - [Awesome Free kaggle Dataset](https://www.kaggle.com/datasets/karnikakapoor/art-portraits)
 - [The kaggle notebook that inspired me](https://www.kaggle.com/code/trungthanhnguyen0502/gan-anime/notebook)
