# Project for the course of the MMML at UCU

Done by Iryna Kokhan, Artem Kraievskyi, Anna Yaremko

---
This project investigates the task of video frame interpolation using motion-
aware latent diffusion models (https://arxiv.org/pdf/2404.13534), building on the MADIFF framework. Video frame
interpolation aims to generate intermediate frames between existing ones, often
needed in applications like slow-motion video, video compression, and frame rate
conversion. Traditional methods often struggle with issues like temporal inconsis-
tency and generating realistic frames in the presence of complex motion or occlu-
sions. To address these challenges, the proposed method incorporates motion priors
into the diffusion process, enhancing the model’s ability to generate temporally
coherent and visually accurate intermediate frames. By leveraging motion-aware
latent representations, this approach ensures smooth transitions between frames
and effectively handles difficult video sequences. We reimplement the MADIFF
method and evaluate its performance, demonstrating its effectiveness in producing
high-quality interpolations even in challenging video contexts.

---

All the information is available here: https://www.overleaf.com/read/nsqrkbjssnfq#2264a9

Some additional files, i.e. folders with results, weights of models, etc. are available here: https://drive.google.com/drive/folders/16LvYv-lL6beBFVVXew9GUL1xoxDFqDEG?usp=sharing
