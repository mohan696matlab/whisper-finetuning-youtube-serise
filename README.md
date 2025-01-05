# Whisper Model Fine-Tuning and Speech-to-Text Generation

## Overview

This repository contains a video series on fine-tuning the OpenAI Whisper model for speech-to-text tasks. The series covers the basics of the Whisper model, how to fine-tune it for a custom dataset, and how to use it to generate text from audio (multilingual).

- [Fine-tuning Whisper to learn my mother tongue ODIA || PART-4](https://youtu.be/lNj7RkOms2U)
- [Fine tuning  Whisper with Pytorch (Simplest way)  || PART-3](https://youtu.be/vMGSExIql8w)
- [Word Error Rate || Evaluation of Whisper  || PART-2](https://youtu.be/NOEJo3miOec)
- [Master Fine-Tuning OpenAI Whisper with PyTorch for Custom ASR Tasks || PART-1](https://youtu.be/iGEJkvu0Qrg)


## Table of Contents

* [Introduction](#introduction)
* [Fine-Tuning the Whisper Model](#fine-tuning-the-whisper-model)
* [Speech-to-text Generation](#speech-to-text-generation)
* [Code](#code)
* [Installation](#installation)

## Introduction

The Whisper model is a speech-to-text model that can generate audio from text. This repository provides a comprehensive introduction to the Whisper model and how to use it for speech-to-text tasks.

## Fine-Tuning the Whisper Model

This section covers the process of fine-tuning the Whisper model on a custom dataset. We will demonstrate how to extract audio features from the dataset, convert them into input features for the Whisper model, and fine-tune the model using the Adam optimizer.

### Step-by-Step Guide

1. **Extract Audio Features**: Extract audio features from the dataset using a library such as Librosa or PyAudio.
2. **Convert to Input Features**: Convert the extracted audio features into input features for the Whisper model.
3. **Fine-Tune the Model**: Fine-tune the Whisper model using the Adam optimizer and a learning rate of 1e-5 for 5 epochs.

## speech-to-text Generation

This section covers the process of using the fine-tuned Whisper model to generate audio from text.

### Step-by-Step Guide

1. **Load the Fine-Tuned Model**: Load the fine-tuned Whisper model using the TorchSA library.
2. **Prepare the Input Text**: Prepare the input text for the Whisper model.
3. **Generate Audio**: Use the Whisper model to generate audio from the input text.

## Code

The code for this repository is written in Python and uses the following libraries:

* PyTorch
* Transformers
* Scipy
* PyAudio
* Numpy
* Matplotlib
* evaluate
* jiwar

## Installation

To install the required libraries, run the following command:

```bash
git clone https://github.com/mohan696matlab/whisper-finetuning-youtube-serise.git
```

## Evaluation

To evaluate the performance of the Whisper model, we can use the Word Error Rate (WER) metric. The WER measures the substitution, insertion, deletion, and total number of words in the reference and predicted transcripts.

### Step-by-Step Guide

1. **Compute the WER**: Compute the WER using the Evaluate library.
2. **Evaluate the Model**: Evaluate the performance of the Whisper model using the WER metric.

## Conclusion

This repository provides a comprehensive introduction to the Whisper model and how to use it for speech-to-text tasks. We hope that this repository will be helpful for researchers and developers who want to fine-tune the Whisper model for their specific use cases.
