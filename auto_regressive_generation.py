from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio
from scipy.signal import resample
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


from transformers import WhisperTokenizer
from transformers import WhisperFeatureExtractor
from transformers import WhisperForConditionalGeneration


model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to('cuda')



model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to('cuda')
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small",language='en',task='transcribe')
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small",language='en',task='transcribe')


from scipy.io import wavfile
sample_rate, audio_data = wavfile.read('Recording.wav')
audio_data = audio_data / np.iinfo(np.int16).max

input_feature = feature_extractor(raw_speech=audio_data,
                                sampling_rate=16000,
                                return_tensors='pt').input_features

with torch.no_grad():
    op = model.generate(input_feature.to('cuda'), language='en', task='transcribe')

print(tokenizer.batch_decode(op,skip_special_tokens=True))

# Autoregressive: keep on appending newly generated token to decoder_input_ids
with torch.no_grad():
    op = model(input_feature.to('cuda'),decoder_input_ids=torch.LongTensor([[50258, 50259, 50359, 50363,2425,1518,11,286,669,16123]]).to('cuda'))#torch.LongTensor([[50258, 50259, 50359, 50363]]).to('cuda')
activations = torch.nn.functional.softmax(op.logits,dim=-1)
pred_token_ids = torch.argmax(activations,dim=-1)

print(tokenizer.batch_decode(pred_token_ids))