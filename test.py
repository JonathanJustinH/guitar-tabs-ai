import torch
from dataset import GuitarTabDataset2
from torch.utils.data import DataLoader

def full_sanity():
    print("Loading dataset")
    dataset = GuitarTabDataset2(
        dataset_path="data/IDMT-SMT-GUITAR_V2/dataset2",
        sr=44100,
        hop_length=512
    )
    print(f"Files loaded: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader))

    audio = batch[0]
    onset = batch[1]
    frame = batch[2]

    print("Shapes:")
    print(" audio:", audio.shape)
    print(" onset:", onset.shape)
    print(" frame:", frame.shape)

    assert audio.dim() == 4, "Audio should be [B, T, F, strings]"
    assert onset.dim() == 3, "Onset should be [B, T, strings]"
    assert frame.dim() == 3, "Frame should be [B, T, strings]"

    B, T, F, S = audio.shape
    print(f"Time frames: {T}, Strings: {S}")

    print("Max fret in onset:", onset.max().item())
    print("Max fret in frame:", frame.max().item())

    poly = frame.gt(0).sum(dim=-1)
    print("Polyphony min/max per frame:", poly.min().item(), "/", poly.max().item())

    subset_violations = ((onset > 0) & (frame == 0)).sum().item()
    print("Subset violations:", subset_violations)
    assert subset_violations == 0, "onset must be subset of frame"

    sample_rate = 44100
    hop_length = 512
    expected_secs = T * hop_length / sample_rate

    print("Frame duration:", expected_secs, "sec")

    assert expected_secs > 0.1, "Expected duration too small"
    assert expected_secs < 60.0, "Expected duration too large"

    print("Test Completed")

if __name__ == "__main__":
    full_sanity()
