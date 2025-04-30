from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio
from scipy.signal import resample
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

import os
from transformers import WhisperTokenizer
from transformers import WhisperFeatureExtractor
from transformers import WhisperForConditionalGeneration

import evaluate
import re

def normalize_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove only ", ', and ,
    text = re.sub(r'[",\']', '', text)
    # Optional: remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

os.makedirs(name='runs/lora_adapter',exist_ok=True)
wer  = evaluate.load('wer')



def down_sample_audio(audio_original, original_sample_rate):
    target_sample_rate = 16000

    # Calculate the number of samples for the target sample rate
    num_samples = int(len(audio_original) * target_sample_rate / original_sample_rate)

    # Resample the audio array to the target sample rate
    downsampled_audio = resample(audio_original, num_samples)

    return downsampled_audio

from datasets import load_dataset,concatenate_datasets

asr_dataset = load_dataset("Mohan-diffuser/odia-english-ASR")

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small",language='bengali',task='translate')
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small",language='bengali',task='translate')
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to('cuda')

class whisper_training_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len):#daatset is huggingface dataset object
        self.dataset = dataset
        self.max_len = max_len
        self.bos_token = model.config.decoder_start_token_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        audio_data = down_sample_audio(item['audio']["array"], item['audio']["sampling_rate"])
        input_features = feature_extractor(audio_data, sampling_rate=16000,return_tensors='pt').input_features[0]

        # Process the transcription
        transcription = item['eng_translation']

        # Create labels
        labels = tokenizer(transcription, padding="max_length", max_length=self.max_len, truncation=True, return_tensors="pt")
        labels = labels["input_ids"].masked_fill(labels['attention_mask'].ne(1), -100)
        labels = labels[0][1:]


        return {
            "input_features": input_features,
            "labels": labels
        }
        
def print_predictions(step):
    model.eval()
    model.config.use_cache=True
    output_lines = []

    for idx in range(5):

        target = normalize_text(asr_dataset['validation'][idx]['eng_translation'])
        audio_original = asr_dataset['validation'][idx]['audio']['array']
        original_sample_rate = asr_dataset['validation'][idx]['audio']['sampling_rate']

        audio_16000 = down_sample_audio(audio_original, original_sample_rate)

        input_feature = feature_extractor(raw_speech=audio_16000,
                                        sampling_rate=16000,
                                        return_tensors='pt').input_features

        with torch.no_grad():
            op = model.generate(input_feature.to('cuda'), language='bengali', task='translate')
            
        text_pred = tokenizer.batch_decode(op, skip_special_tokens=True)[0]
        
        line = (
            f'-------{idx}------\n'
            f'true : {target} \n'
            f'pred : {text_pred}\n\n'
        )
        output_lines.append(line)
        
        # Save to file
        with open(f'runs/predictions_step_{step}.txt', 'w', encoding='utf-8') as f:
            f.writelines(output_lines)
        
dataset = whisper_training_dataset(dataset=asr_dataset['train'], max_len=60)

train_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,  # Adjust batch size as needed
    shuffle=True,  # Shuffle data during training
)

test_dataset = whisper_training_dataset(dataset=asr_dataset['validation'], max_len=60)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=8,  # Adjust batch size as needed
    shuffle=True,  # Shuffle data during training
)

def evaluation(model):

    device='cuda'

    test_dataset = whisper_training_dataset(dataset=asr_dataset['validation'], max_len=60)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,  # Adjust batch size as needed
        shuffle=True,  # Shuffle data during training
    )

    model.eval()

    predictions=[]
    references=[]

    for batch in tqdm(test_dataloader,total=len(test_dataloader)):
        

        model.eval()  # Set model to training mode
        model.config.use_cache = True

        input_features, labels = batch["input_features"].to(device), batch["labels"].to(device)

        with torch.no_grad():
            generated_tokens = model.generate(input_features=input_features,language='bengali', task='translate')
                        
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        predictions.extend(decoded_preds)
        references.extend(decoded_labels)

    WER = wer.compute(predictions=predictions, references=references) * 100

    return WER

from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

config = LoraConfig(r=64, lora_alpha=64, target_modules=["q_proj", "v_proj", "q_proj", "out_proj"], lora_dropout=0.05, bias="none")

model = get_peft_model(model, config)
model.print_trainable_parameters()

torch.cuda.empty_cache()

model.config.use_cache = False
model.train()

device='cuda'

# Filter parameters with requires_grad=True
requires_grad_params = filter(lambda x: x[1].requires_grad, model.parameters())
optimizer=torch.optim.AdamW(requires_grad_params, lr=5e-4) # Only for LoRA Training

gradient_accumulation_steps = 4
eval_steps=2
max_epochs=10
global_step=0


running_wer=[]
running_loss_buffer=[]
val_losses=[]
train_losses=[]



for epoch in range(max_epochs):

    for step, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader), leave=False)):
        global_step += 1

        model.train()  # Set model to training mode

        input_features, labels = batch["input_features"].to(device), batch["labels"].to(device)

        # Forward pass
        outputs = model(input_features, labels=labels)  # Assuming your model takes these inputs
        loss = outputs.loss
        running_loss_buffer.append(loss.item())
        loss = loss / gradient_accumulation_steps  # Scale the loss
        loss.backward()
  
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        
        if global_step % eval_steps ==0:  # Print loss every 50 batches
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in tqdm(test_dataloader,total=len(test_dataloader), leave=False):
                    val_input, val_labels = val_batch["input_features"].to(device), val_batch["labels"].to(device)
                    val_outputs = model(val_input, labels=val_labels)
                    val_loss += val_outputs.loss.item()
            val_loss /= len(test_dataloader)
            val_losses.append(val_loss)
            train_losses.append(np.mean(running_loss_buffer[-int(eval_steps*gradient_accumulation_steps):]))
            
            
            plt.plot(train_losses, label='train loss', color='blue')
            plt.plot(val_losses, label='test loss', color='red')
            plt.xlabel('steps')
            plt.ylabel('loss')
            plt.savefig('runs/loss.png')
            plt.close()

            model.save_pretrained('runs/lora_adapter')

    torch.cuda.empty_cache()
    print(evaluation(model))