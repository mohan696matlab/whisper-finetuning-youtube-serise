
# Whisper Model Tutorials: From Dataset Preparation to Fine-Tuning (A Youtube serise)

Welcome to the **Whisper Model Tutorials** repository! This series takes you through a step-by-step journey of working with OpenAI’s Whisper model, starting from dataset preparation, evaluating, and fine-tuning it effectively. Each tutorial is designed to guide you with detailed explanations and practical implementations.

---

## 🚀 What You'll Learn
1. **Understanding the Whisper model:**
   - Dataset selectionfor Whisper-based ASR tasks.
   - Using the Hugging Face `transformers` library for tokenization.
   - Extracting and analyzing audio features like log-mel spectrograms.

2. **Evaluating Model Performance:**
   - Generating predictions using the Whisper model.
   - Calculating loss and performance metrics like WER (Word Error Rate).

3. **Fine-Tuning Whisper for Custom Use-Cases:**
   - Preparing datasets for fine-tuning, including generating input features and target labels.
   - Training the Whisper model on domain-specific data.

4. **LoRA finetuning of Whisper Model on low resource language (ODIA - a regional Indian language):**



---

## 📂 Repository Structure

```plaintext
.
├── notebooks/
│   ├── 01_data_preparation.ipynb       # Preparing and loading ASR datasets
│   ├── 02_feature_extraction.ipynb     # Extracting log-mel spectrogram features
│   ├── 03_tokenization.ipynb           # Tokenizing text for Whisper
│   ├── 04_training_and_fine_tuning.ipynb # Fine-tuning the Whisper model
│   ├── 05_evaluation_and_loss.ipynb    # Evaluating the model and calculating loss
│   ├── 06_deployment.ipynb             # Deploying and testing the Whisper model
```



## 🛠️ Setup Instructions
1. Clone the Repository
```
git clone https://github.com/your-username/whisper-tutorials.git
cd whisper-tutorials
```

2. Install Dependencies
Ensure you have Python 3.10+ installed. Then run:

```
pip install -r requirements.txt
```

3. Download the Dataset
Download the Hugging Face ASR dataset used in the tutorials:

Move the .parquet files to the data/datasets/ directory.
Ensure the dataset is accessible in the notebooks.

4. Run the Jupyter Notebooks
Start Jupyter Notebook and navigate to the working directory:

## 📚 Key Topics Covered in the Codebase
1. Data Preparation
- Loading and processing .parquet files using Hugging Face’s datasets library.
- Handling audio data: resampling, normalizing, and converting to NumPy arrays.

2. Feature Extraction
- Extracting log-mel spectrograms using Whisper’s FeatureExtractor.
- Visualizing audio waveforms and spectrograms.

3. Tokenization
- Tokenizing and encoding ASR targets with WhisperTokenizer.
- Handling padding and creating attention masks.

4. Fine-Tuning
- Initializing the WhisperForConditionalGeneration model.
- Training the model on GPU with PyTorch and Hugging Face.

5. Evaluation
- Generating predictions using model.generate().
- Comparing predicted outputs with target labels.
- Calculating loss and performance metrics.


## 🔗 Resources
- Whisper Model Documentation
- Hugging Face Datasets
- Hugging Face Transformers Library
