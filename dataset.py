import os
import glob
import torch
import librosa
import numpy as np
import xml.etree.ElementTree as ET

class GuitarTabDataset2:

    def __init__(self, dataset_path, sr=16000, hop_length=512, max_fret=21):
        self.audio_path = os.path.join(dataset_path, "audio")
        self.annot_path = os.path.join(dataset_path, "annotation")
        self.sr = sr
        self.hop_length = hop_length
        self.max_fret = max_fret

        self.files = self._collect_files()

    def _collect_files(self):
        xmls = glob.glob(os.path.join(self.annot_path, "*.xml"))
        print(f"Looking for XML files in: {self.annot_path}")
        
        valid_files = []
        for xml in xmls:
            name = os.path.splitext(os.path.basename(xml))[0]
            wav = os.path.join(self.audio_path, name + ".wav")
            if os.path.exists(wav):
                valid_files.append(name)
            else:
                print(f"Skipping {name}: audio file not found")
        
        print(f"Found {len(valid_files)} valid pairs")
        return valid_files

    def __len__(self):
        return len(self.files)

    def _parse_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        events = []
        for ev in root.findall(".//event"):
            onset = float(ev.find("onsetSec").text)
            offset = float(ev.find("offsetSec").text)
            string = int(ev.find("stringNumber").text) - 1
            fret = int(ev.find("fretNumber").text)
            events.append((onset, offset, string, fret))
        return events

    def __getitem__(self, idx):
        name = self.files[idx]

        wav = os.path.join(self.audio_path, name + ".wav")
        xml = os.path.join(self.annot_path, name + ".xml")

        y, sr = librosa.load(wav, sr=self.sr)

        n_frames = librosa.samples_to_frames(len(y), hop_length=self.hop_length)
        onset = torch.zeros(n_frames, 6, dtype=torch.long)
        frame = torch.zeros(n_frames, 6, dtype=torch.long)

        events = self._parse_xml(xml)

        for onset_s, offset_s, string, fret in events:
            on = librosa.time_to_frames(onset_s, sr=self.sr, hop_length=self.hop_length)
            off = librosa.time_to_frames(offset_s, sr=self.sr, hop_length=self.hop_length)

            if fret > self.max_fret:
                fret = self.max_fret

            if on < n_frames:
                onset[on, string] = fret

            off = min(off, n_frames)
            if on < off:
                frame[on:off, string] = fret

        audio_frames = librosa.util.frame(
            y,
            frame_length=self.hop_length,
            hop_length=self.hop_length
        )

        if audio_frames.ndim == 1:
            audio_frames = audio_frames[None, :]
        elif audio_frames.shape[0] != audio_frames.shape[1]:
            audio_frames = audio_frames.T

        if audio_frames.ndim != 2:
            raise RuntimeError(f"Unexpected audio shape: {audio_frames.shape}")

        audio = torch.tensor(audio_frames, dtype=torch.float32)
        audio = audio.unsqueeze(-1).expand(-1, -1, 6)
        
        t = audio_frames.shape[0]
        onset = onset[:t]
        frame = frame[:t]

        return audio, onset, frame
