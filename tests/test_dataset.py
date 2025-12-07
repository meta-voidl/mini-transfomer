"""
Tests für TextDataset & create_dataloader
"""

import torch
from src.dataset import TextDataset, create_dataloader


def test_dataset_shapes():
    """
    Prüft, ob Input- und Target-Shape stimmen.
    """
    token_ids = list(range(20))
    seq_len = 5

    ds = TextDataset(token_ids, seq_len)

    inp, tgt = ds[0]
    assert inp.shape == (seq_len,)
    assert tgt.shape == (seq_len,)


def test_target_shift_by_one():
    """
    Prüft, ob Target um genau 1 Schritt nach rechts verschoben ist.
    """
    token_ids = [10, 11, 12, 13, 14, 15]
    seq_len = 4

    ds = TextDataset(token_ids, seq_len)
    inp, tgt = ds[0]

    # Beispiel:
    # token_ids = [10,11,12,13,14,15]
    # Sample 0:
    #   input: [10,11,12,13]
    #   target:[11,12,13,14]
    for i in range(seq_len):
        assert tgt[i].item() == inp[i].item() + 1


def test_dataloader_batching():
    """
    Prüft, ob der DataLoader korrekt batcht.
    """
    token_ids = list(range(30))
    seq_len = 6
    batch_size = 4

    dl = create_dataloader(token_ids, seq_len, batch_size)

    for batch_inp, batch_tgt in dl:
        # Batch-Shape = (batch_size, seq_len)
        assert batch_inp.shape == (batch_size, seq_len)
        assert batch_tgt.shape == (batch_size, seq_len)
        break  # Nur ersten Batch testen


def test_dataset_works_on_cuda_if_available():
    """
    Optionaler Test: Tensoren lassen sich auf CUDA verschieben,
    falls verfügbar.
    """
    token_ids = list(range(50))
    seq_len = 10

    ds = TextDataset(token_ids, seq_len)
    inp, tgt = ds[0]

    if torch.cuda.is_available():
        inp_cuda = inp.to("cuda")
        tgt_cuda = tgt.to("cuda")
        assert inp_cuda.device.type == "cuda"
        assert tgt_cuda.device.type == "cuda"
    else:
        assert inp.device.type == "cpu"
