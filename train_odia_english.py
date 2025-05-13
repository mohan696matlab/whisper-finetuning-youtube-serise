from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio
from scipy.signal import resample
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

import os,json
from transformers import WhisperTokenizer, get_scheduler
from transformers import WhisperFeatureExtractor
from transformers import WhisperForConditionalGeneration
import wandb
import evaluate
import re
import time
from jiwer import cer

device='cuda'

MAX_STEPS=5000
BATCH_SIZE=8
eval_steps=100
gradient_accumulation_steps = 4
LR=5e-4
warmup_steps=20



wandb.init(
    project="your_project_name",  # e.g., "whisper-odia-asr"
    name=f"run-{time.strftime('%Y%m%d-%H%M%S')}",  # or a custom run name
    config={
        "max_steps": MAX_STEPS,
        "eval_steps": eval_steps,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
    }
)


def normalize_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove only ", ', and ,
    text = re.sub(r'[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\»\«]', '', text)
    # Optional: remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

os.makedirs(name='runs',exist_ok=True)




def down_sample_audio(audio_original, original_sample_rate):
    target_sample_rate = 16000

    # Calculate the number of samples for the target sample rate
    num_samples = int(len(audio_original) * target_sample_rate / original_sample_rate)

    # Resample the audio array to the target sample rate
    downsampled_audio = resample(audio_original, num_samples)

    return downsampled_audio

from datasets import load_dataset,concatenate_datasets

asr_dataset = load_dataset("Mohan-diffuser/odia-english-ASR")

model_id="openai/whisper-base"
tokenizer = WhisperTokenizer.from_pretrained(model_id,language='bengali',task='transcribe')
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id,language='bengali',task='transcribe')
model = WhisperForConditionalGeneration.from_pretrained(model_id).to('cuda')

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
        transcription = item['transcription']

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

        target = normalize_text(asr_dataset['validation'][idx]['transcription'])
        audio_original = asr_dataset['validation'][idx]['audio']['array']
        original_sample_rate = asr_dataset['validation'][idx]['audio']['sampling_rate']

        audio_16000 = down_sample_audio(audio_original, original_sample_rate)

        input_feature = feature_extractor(raw_speech=audio_16000,
                                        sampling_rate=16000,
                                        return_tensors='pt').input_features

        with torch.no_grad():
            op = model.generate(input_feature.to('cuda'), language='bengali', task='transcribe')
            
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

from datasets import concatenate_datasets
# Combine train and test splits
combined_dataset = concatenate_datasets([asr_dataset['train'], asr_dataset['test']])

# Now use the combined dataset
dataset = whisper_training_dataset(dataset=combined_dataset, max_len=400)

train_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,  # Adjust batch size as needed
    shuffle=True,  # Shuffle data during training
)

test_dataset = whisper_training_dataset(dataset=asr_dataset['validation'], max_len=400)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,  # Adjust batch size as needed
    shuffle=True,  # Shuffle data during training
)

def evaluation(model):

    device='cuda'

    test_dataset = whisper_training_dataset(dataset=asr_dataset['validation'], max_len=400)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,  # Adjust batch size as needed
        shuffle=True,  # Shuffle data during training
    )

    model.eval()

    predictions=[]
    references=[]

    batch_counter=0

    for batch in tqdm(test_dataloader,total=len(test_dataloader)):
        
        if batch_counter >= 5:
            break
        
        batch_counter+=1

        model.eval()  # Set model to training mode
        model.config.use_cache = True

        input_features, labels = batch["input_features"].to(device), batch["labels"].to(device)

        with torch.no_grad():
            generated_tokens = model.generate(input_features=input_features,language='bengali', task='transcribe')
                        
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        predictions.extend(decoded_preds)
        references.extend(decoded_labels)

    CER = cer(references, predictions) * 100

    return CER

from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

config = LoraConfig(r=256, lora_alpha=256, target_modules=["q_proj", "v_proj", "q_proj", "out_proj"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, config)
# model = PeftModel.from_pretrained(model, "runs/lora_adapter", is_trainable=True, device_map={"": 0})
model.print_trainable_parameters()

torch.cuda.empty_cache()

model.config.use_cache = False
model.train()



# Filter parameters with requires_grad=True
requires_grad_params = filter(lambda x: x[1].requires_grad, model.parameters())
optimizer=torch.optim.AdamW(requires_grad_params, lr=LR) # Only for LoRA Training

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=MAX_STEPS,
)


global_step=0


running_cer=[]
running_loss_buffer=[]
val_losses=[]
train_losses=[]


pbar = tqdm(np.arange(MAX_STEPS), total=MAX_STEPS, leave=False)

while global_step < MAX_STEPS:

    for step, batch in enumerate(train_dataloader):
        if global_step >= MAX_STEPS:
            break  # Exit the loop early if max steps reached

        model.config.use_cache = False
        model.train()

        input_features, labels = batch["input_features"].to(device), batch["labels"].to(device)

        # Forward pass
        outputs = model(input_features, labels=labels)
        loss = outputs.loss
        running_loss_buffer.append(loss.item())

        # Show current loss in progress bar
        global_step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        wandb.log({"train/loss": loss.item(), "step": global_step})


        loss = loss / gradient_accumulation_steps  # Scale the loss
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            

        if global_step % eval_steps == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in tqdm(test_dataloader, total=len(test_dataloader), leave=False):
                    val_input = val_batch["input_features"].to(device)
                    val_labels = val_batch["labels"].to(device)
                    val_outputs = model(val_input, labels=val_labels)
                    val_loss += val_outputs.loss.item()
            val_loss /= len(test_dataloader)
            val_losses.append(val_loss)
            train_losses.append(np.mean(running_loss_buffer[-int(eval_steps * gradient_accumulation_steps):]))

            wandb.log({
                        "eval/loss": val_loss,
                        "train/loss_avg": train_losses[-1],
                        "step": global_step
                    })

            plt.plot(train_losses, label='train loss', color='blue')
            plt.plot(val_losses, label='test loss', color='red')
            plt.xlabel('steps')
            plt.ylabel('loss')
            plt.legend()
            plt.tight_layout()
            plt.savefig('runs/loss.png')
            plt.close()

            print_predictions(global_step)

            os.makedirs(name=f'runs/lora_adapter_{global_step}_steps',exist_ok=True)
            model.save_pretrained(f'runs/lora_adapter_{global_step}_steps')

            current_cer = evaluation(model)
            wandb.log({"eval/cer": current_cer, "step": global_step})
            running_cer.append({"step": global_step, "cer": current_cer})

            # Later save:
            with open('runs/running_cer.json', 'w') as f:
                json.dump(running_cer, f, indent=2)



    torch.cuda.empty_cache()
