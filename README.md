## Fine-tuning MARBERTv2 Model for ARABIC Text Classification

This project fine-tunes the `MARBERTv2` model for text classification on a provided dataset. The dataset consists of text samples with corresponding labels. The goal is to train the BERT model to classify the text into predefined categories such as positive, negative, or mixed sentiments.

### Model Used

The model used for this task is `MARBERTv2`, a pre-trained BERT model specifically trained for Arabic text. This model has been fine-tuned on a large corpus of Arabic text data and has been shown to perform well on various NLP tasks, including text classification.

### Dataset

The dataset used for training and evaluation is a tab-separated file (`Dataset.tsv`) containing two columns: `text` and `label`. The `text` column contains the text samples to be classified, while the `label` column contains the corresponding class labels.

### Tokenizer

The tokenizer used for this task is the tokenizer associated with the `MARBERTv2` model. This tokenizer is specifically designed to tokenize Arabic text in a way that is compatible with the `MARBERTv2` model architecture.

### Workflow

1. **Data Loading and Preprocessing**: The dataset is loaded from the provided file and preprocessed to remove special characters.

2. **Tokenization**: The text data is tokenized using the `MARBERTv2` tokenizer. The maximum sequence length is determined to ensure that no samples are truncated during training.

3. **Model Training**: The pre-trained `MARBERTv2` model is fine-tuned on the tokenized text data. Training is performed for one epoch with a specified batch size and gradient accumulation steps.

4. **Model Evaluation**: The fine-tuned model is evaluated on a separate test set to assess its performance. Metrics such as accuracy, precision, recall, and F1-score are computed and reported.

5. **Model Inference**: The trained model is used to make predictions on custom text samples to demonstrate its classification capabilities.

### Results

The project provides insights into the performance of the fine-tuned BERT model for text classification in Arabic. It includes metrics such as accuracy, precision, recall, and F1-score, as well as a confusion matrix to visualize the model's performance on the test set. Additionally, the model's ability to classify custom text samples is demonstrated through inference.
