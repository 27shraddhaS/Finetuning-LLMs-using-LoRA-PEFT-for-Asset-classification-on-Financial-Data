# Finetuning-LLMs-using-LoRA-PEFT-for-Asset-classification-on-Financial-Data
Finetuning LLMs using LoRA-PEFT for Asset classification on Financial Data

This repository demonstrates the fine-tuning of pretrained Large Language Models (LLMs) for various text classification tasks. We use models such as distilBERT, finBERT, and flan-t5, and apply Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning.

## Overview

### Tasks

1. **Sentiment Analysis**: Classifying the sentiment of text.
2. **Classifying Aspect Categories**: Categorizing text into predefined aspect categories.
3. **Detecting Target from Sentence**: Identifying the target within a sentence.

### Models Used

- **distilBERT**: A lightweight version of BERT for efficient text classification.
- **finBERT**: Specifically designed for financial sentiment analysis.
- **flan-t5**: A sequence-to-sequence model for diverse NLP tasks.

## Data Preparation

1. **Exploratory Data Analysis (EDA)**:
   - Visualize and clean the data.
   - Encode the data and convert targets into embeddings.

2. **Tokenization**:
   - Use the appropriate tokenizers for each model.

## Fine-Tuning and Evaluation

### Task 1: Sentiment Analysis

**Approach 1**
- Fine-tuned distilBERT for sentiment classification.
- Fine-tuned finBERT, which did not yield satisfactory results.

**Approach 2**
- Added aspect categories to the text for improved context.
- Resulted in a 5.33% increase in accuracy.

**Evaluation Metrics**

| Metric    | Before Fine-Tuning | After Fine-Tuning | Improved Model |
|-----------|---------------------|-------------------|----------------|
| Accuracy  | 26%                 | 59.34%            | 64.67%         |
| Precision | 26%                 | 59.34%            | 64.67%         |
| Recall    | 26%                 | 59.34%            | 64.67%         |
| F1 Score  | 26%                 | 59.34%            | 64.67%         |

### Task 2: Classifying Aspect Categories

**Approach 1**
- Fine-tuned distilBERT for aspect category classification.
- Fine-tuned finBERT, which did not yield satisfactory results.

**Approach 2**
- Simplified category levels from 6 to 2 for improved classification.
- Achieved a 26% increase in accuracy.

**Evaluation Metrics**

| Metric    | Before Fine-Tuning | After Fine-Tuning | Improved Model |
|-----------|---------------------|-------------------|----------------|
| Accuracy  | 36.67%              | 41.34%            | 67.34%         |
| Precision | 36.67%              | 41.34%            | 67.34%         |
| Recall    | 36.67%              | 41.34%            | 67.34%         |
| F1 Score  | 36.67%              | 41.34%            | 67.34%         |

### Task 3: Detecting Target from Sentence

**Approach**
- Fine-tuned the flan-t5 sequence-to-sequence model for target prediction.

**Evaluation Metrics**

| Metric    | Fine-Tuned Model |
|-----------|------------------|
| Accuracy  | 82.1%            |
| Precision | 82.1%            |
| Recall    | 82.1%            |
| F1 Score  | 82.1%            |

## Conclusion

**distilBERT** exhibited the highest performance in sentiment analysis and aspect classification tasks. The **flan-t5** model achieved high accuracy in detecting targets from sentences.

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- Transformers library from Hugging Face
- Required datasets

### Installation

To install the required packages, run:

```bash
pip install torch transformers datasets
