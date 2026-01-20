# Guitar Tabs AI

A PyTorch-based guitar audio transcription system that converts guitar audio recordings into ASCII tablature using onset and frame detection with a CNN-RNN architecture.

## Demo

Try the live demo on [Hugging Face Spaces](https://huggingface.co/spaces/JonathanJH/guitar-tabs-ai)

## Dataset

This project uses the [IDMT-SMT-GUITAR_V2 dataset](https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html) for training, which contains guitar recordings with annotated notes including string numbers and fret positions.

## Installation

```bash
pip install -r requirements.txt
```

## Training

To train the model:

```bash
cd onsets_and_frames
python train.py
```

The model will train with early stopping and save the best model to `tab_model_best.pt`.

## Usage

Run the Gradio web interface:

```bash
cd onsets_and_frames
python app.py
```

Then upload a guitar audio file (WAV format) to get ASCII tabs output.

## Model Architecture

- CNN feature extraction (16 channels)
- Bidirectional GRU (128 hidden dimensions)
- Dual outputs: onset detection and fret prediction
- 6 strings, 22 frets (0-21)


