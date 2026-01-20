import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import GuitarTabDataset2


# =====================================
# Collate (padding batch)
# =====================================
def pad_collate(batch):
    audios, onsets, frames = zip(*batch)

    # find max length
    T = max(x.shape[0] for x in audios)

    padded_audio = []
    padded_onset = []
    padded_frame = []

    for a, o, f in batch:
        t = a.shape[0]
        pad = T - t
        if pad > 0:
            a = F.pad(a, (0,0,0,0,0, pad))
            o = F.pad(o, (0,0,0, pad))
            f = F.pad(f, (0,0,0, pad))
        padded_audio.append(a)
        padded_onset.append(o)
        padded_frame.append(f)

    return torch.stack(padded_audio), torch.stack(padded_onset), torch.stack(padded_frame)


# =====================================
# Model
# =====================================
class GuitarTabModel(nn.Module):
    def __init__(self, n_strings=6, n_frets=21, cnn_channels=32, rnn_dim=256):
        super().__init__()
        self.n_strings = n_strings
        self.n_frets = n_frets
        self.cnn_channels = cnn_channels
        self.rnn_dim = rnn_dim
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, 3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Conv2d(cnn_channels, cnn_channels, 3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
        )
        
        # RNN will be initialized lazily after first forward pass
        self.rnn = None
        
        self.onset_head = nn.Linear(rnn_dim*2, n_strings)
        self.frame_head = nn.Linear(rnn_dim*2, n_strings)
    
    def forward(self, x):
        B, T, freq, S = x.shape
        # Reshape to (B*S, 1, T, freq) to process each string independently
        x = x.permute(0,3,1,2).reshape(B*S, 1, T, freq)
        x = self.cnn(x)
        _, C, T2, F2 = x.shape
        # Reshape back to (B, S, T2, C*F2) then (B, T2, S*C*F2)
        x = x.reshape(B, S, T2, C*F2).permute(0, 2, 1, 3)
        x = x.reshape(B, T2, S*C*F2)
        
        # Initialize RNN lazily with correct input size
        if self.rnn is None:
            rnn_input_size = S * C * F2
            self.rnn = nn.GRU(
                input_size=rnn_input_size,
                hidden_size=self.rnn_dim,
                batch_first=True,
                bidirectional=True
            ).to(x.device)
        
        out,_ = self.rnn(x)
        onset = torch.sigmoid(self.onset_head(out))
        frame = torch.sigmoid(self.frame_head(out))  # Use sigmoid for normalized [0,1] output
        return onset, frame


# =====================================
# Loss
# =====================================
class TabLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.reg = nn.SmoothL1Loss()
    
    def forward(self, p_onset, p_frame, gt_onset, gt_frame):
        lo = self.bce(p_onset, gt_onset)
        lf = self.bce(p_frame, gt_frame)  # Use BCE since frame is now normalized [0,1]
        return lo + lf, lo, lf


# =====================================
# Train step
# =====================================
def train_epoch(model, loader, optim, loss_fn, device):
    model.train()
    tot = 0
    for audio, onset, frame in loader:
        audio, onset, frame = audio.to(device), onset.to(device).float(), frame.to(device).float()
        optim.zero_grad()
        po, pf = model(audio)
        
        # Downsample labels to match model output (due to pooling in CNN)
        T_out = po.shape[1]
        onset = onset[:, ::2, :][:, :T_out, :]  # downsample by 2 and trim
        frame = frame[:, ::2, :][:, :T_out, :]
        
        # Convert onset to binary (0/1) and normalize frame values
        onset_binary = (onset > 0).float()  # 1 if fret is played, 0 otherwise
        frame_norm = frame / 21.0  # normalize to [0, 1]
        
        loss,_,_ = loss_fn(po, pf, onset_binary, frame_norm)
        loss.backward()
        optim.step()
        tot += loss.item()
    return tot / len(loader)


# =====================================
# Valid step
# =====================================
@torch.no_grad()
def valid_epoch(model, loader, loss_fn, device):
    model.eval()
    tot = 0
    for audio, onset, frame in loader:
        audio, onset, frame = audio.to(device), onset.to(device).float(), frame.to(device).float()
        po, pf = model(audio)
        
        # Downsample labels to match model output (due to pooling in CNN)
        T_out = po.shape[1]
        onset = onset[:, ::2, :][:, :T_out, :]  # downsample by 2 and trim
        frame = frame[:, ::2, :][:, :T_out, :]
        
        # Convert onset to binary (0/1) and normalize frame values
        onset_binary = (onset > 0).float()  # 1 if fret is played, 0 otherwise
        frame_norm = frame / 21.0  # normalize to [0, 1]
        
        loss,_,_ = loss_fn(po, pf, onset_binary, frame_norm)
        tot += loss.item()
    return tot / len(loader)


# =====================================
# Runner
# =====================================
def run_training():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    ds = GuitarTabDataset2(
        dataset_path="data/IDMT-SMT-GUITAR_V2/dataset2",
        sr=44100,
        hop_length=512
    )
    print(f"Total files: {len(ds)}")

    loader = DataLoader(
        ds, batch_size=1, shuffle=True,
        num_workers=0, collate_fn=pad_collate
    )

    model = GuitarTabModel(cnn_channels=16, rnn_dim=128).to(device)
    loss_fn = TabLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Starting training")
    best_val_loss = float('inf')
    patience = 5 
    patience_counter = 0
    
    for epoch in range(50):
        tr = train_epoch(model, loader, optim, loss_fn, device)
        va = valid_epoch(model, loader, loss_fn, device)
        print(f"Epoch {epoch+1}/50 | Train {tr:.4f} | Valid {va:.4f}")
        
        if va < best_val_loss:
            best_val_loss = va
            patience_counter = 0
            torch.save(model.state_dict(), "tab_model_best.pt")
            print(f"  → Validation improved! Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  → No improvement for {patience} epochs. Stopping early!")
                break

    # Load best model
    model.load_state_dict(torch.load("tab_model_best.pt"))
    torch.save(model.state_dict(), "tab_model.pt")


if __name__ == "__main__":
    run_training()
