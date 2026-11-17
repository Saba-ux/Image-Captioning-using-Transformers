# Image Captioning Using CNN and Transformers

An image captioning project that generates natural-language descriptions of images using a **ResNet-101 CNN encoder and Transformer decoder**.

The model extracts visual features from an input image and uses Transformer-based sequence generation to produce a meaningful caption.

## Project Overview

Image captioning combines computer vision and natural language processing to understand visual content and describe it in human-readable language.

This project implements an encoder-decoder architecture in which:
- A pretrained ResNet-101 extracts visual features from an image.
- The visual feature representations are passed to a Transformer decoder.
- The decoder uses attention mechanisms to generate a sequence of words.
- Beam search is used during inference to generate captions.

## Model Architecture

**CNN Encoder: ResNet-101**
- Extracts high-level visual features.
- Input image size: 256 × 256.
- Feature map size: 14 × 14.
- Produces 196 spatial feature vectors.
- Feature embedding dimension: 512.

**Transformer Decoder**
- Model dimension (`d_model`): 512.
- Encoder layers: 2.
- Decoder layers: 4.
- Attention heads: 8.
- Dropout: 0.1.
- Maximum caption length: 52.

## Dataset

The project uses the MS COCO image captioning dataset.

Dataset split used for model training and evaluation:

| Split | Number of images |
|---|---:|
| Training | 25,000 |
| Validation | 2,000 |
| Testing | 2,000 |

Each image is associated with five human-written reference captions.

The dataset files and trained model checkpoints are not included in this repository because of their large size.

## Model Evaluation

The trained model was evaluated on the complete test set of 2,000 images.

### Quantitative Results

| Evaluation Metric | Mean Score |
|---|---:|
| BLEU-1 | 0.6942 |
| BLEU-2 | 0.4814 |
| BLEU-3 | 0.3495 |
| BLEU-4 | 0.2639 |
| GLEU | 0.2581 |
| METEOR | 0.4701 |

*Note: These values are averages of per-image metric scores, not corpus-level BLEU scores.*

### Qualitative Evaluation

Generated captions were compared against the five human-written reference captions for individual test images.

The qualitative evaluation examines:
- Object identification.
- Activity and action recognition.
- Scene understanding.
- Missing visual details.
- Caption relevance and accuracy.

**[View Qualitative Evaluation Report (PDF)](qualitative_results/qualitative_evaluation.pdf)**

The report contains 10 test examples, generated captions, reference captions, and individual evaluation scores.

## Technologies Used

- Python
- PyTorch
- ResNet-101
- Transformer Architecture
- Attention Mechanism
- NumPy
- Pandas
- HDF5
- NLTK
- Matplotlib

## Project Structure

```text
Image-Captioning-using-Transformers/
│
├── code/
│   ├── dataset/
│   ├── models/
│   ├── utils/
│   ├── config.json
│   ├── config_mac.json
│   ├── inference_single.py
│   ├── inference_test.py
│   └── run_train.py
│
├── qualitative_results/
│   └── qualitative_evaluation.pdf
│
├── qualitative_evaluation.py
├── README.md
└── .gitignore
```

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/Saba-ux/Image-Captioning-using-Transformers.git
cd Image-Captioning-using-Transformers
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**macOS/Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

Install the project dependencies using the available requirements file, if provided.

The dataset, GloVe embeddings, and trained checkpoint must be downloaded or prepared separately.

## Running Inference

After preparing the dataset, embeddings, and trained checkpoint, configure the appropriate paths in the configuration file.

Run single-image inference:

```bash
python code/inference_single.py
```

Run test-set evaluation:

```bash
python code/inference_test.py \
  --dataset_dir data/mscoco_h5 \
  --save_dir test_results_colab \
  --config_path code/config_mac.json \
  --checkpoint_name 0310.1454/checkpoint.pth.tar \
  --device cpu
```

For a quick test on a limited number of images, add:

```bash
--limit 20
```

## Limitations and Future Improvements

- Improve fine-grained object recognition and object counting.
- Capture more detailed scene context and relationships.
- Improve recognition of actions and complex activities.
- Explore stronger visual feature extractors and attention mechanisms.
- Develop an interactive web application for image caption generation.

## Author

**Saba**

M.Tech – Data Science  
Delhi Technological University

[GitHub Profile](https://github.com/Saba-ux)
