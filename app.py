import torch
import librosa
import numpy as np
import gradio as gr

from train import GuitarTabModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "tab_model_best.pt"

model = GuitarTabModel(cnn_channels=16, rnn_dim=128)

dummy_input = torch.zeros(1, 100, 512, 6)
with torch.no_grad():
    _ = model(dummy_input)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
model.to(DEVICE)

HOP_LENGTH = 512
SR = 44100
NUM_STRINGS = 6
MAX_FRET = 20

def audio_to_tensor(y, hop_length=HOP_LENGTH):
    audio_frames = librosa.util.frame(y, frame_length=hop_length, hop_length=hop_length)
    if audio_frames.ndim == 1:
        audio_frames = audio_frames[None, :]
    else:
        audio_frames = audio_frames.T
    
    audio_tensor = torch.tensor(audio_frames, dtype=torch.float32)
    audio_tensor = audio_tensor.unsqueeze(-1).expand(-1, -1, NUM_STRINGS)  # (T, hop, 6)
    audio_tensor = audio_tensor.unsqueeze(0)  # (1, T, hop, 6)
    audio_tensor = audio_tensor.to(DEVICE)
    return audio_tensor

def tensor_to_ascii(onset, frame):
    T, strings = onset.shape
    tab_lines = ["e|", "B|", "G|", "D|", "A|", "E|"]  # Standard tuning, string 0 = high E

    frame_frets = (frame * 21.0).round().int()
    
    onset_detected = onset > 0.04
    
    for s in range(NUM_STRINGS):
        line = tab_lines[s]
        for t in range(T):
            if onset_detected[t, s]:
                fret = frame_frets[t, s].item()
                if fret > 0:
                    line += f"{fret}-"
                else:
                    line += "0-"
            else:
                line += "--"
        tab_lines[s] = line
    return "\n".join(tab_lines)

def transcribe(file):
    y, sr = librosa.load(file, sr=SR)
    audio_tensor = audio_to_tensor(y)

    with torch.no_grad():
        onset_pred, frame_pred = model(audio_tensor) # (1, T, strings)
        onset_pred = onset_pred.squeeze(0).cpu()
        frame_pred = frame_pred.squeeze(0).cpu()

    # Debug
    max_onset = onset_pred.max().item()
    mean_onset = onset_pred.mean().item()
    num_onsets = (onset_pred > 0.04).sum().item()
    
    debug_info = f"Notes detected: {num_onsets}\n"
    debug_info += f"Audio length: {len(y)/SR:.2f}s, Frames: {onset_pred.shape[0]}\n\n"
    
    ascii_tab = tensor_to_ascii(onset_pred, frame_pred)
    return debug_info + ascii_tab


# Interface
iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Upload Guitar Audio"),
    outputs=gr.Textbox(
        label="Predicted ASCII Tabs",
        lines=10,
        max_lines=15,
        elem_classes="tab-output"
    ),
    title="Guitar Tabs AI",
    description="Upload a guitar solo WAV file to turn it to ASCII TABS",
    css="""
    .tab-output textarea {
        font-family: 'Courier New', monospace !important;
        font-size: 14px !important;
        overflow-x: auto !important;
        overflow-y: auto !important;
        white-space: pre !important;
        word-wrap: normal !important;
    }
    """
)

if __name__ == "__main__":
    iface.launch()