"""DataLoader factory for VCM.

Batches variable-length mels into padded tensors and builds the targets
the CTC loss needs (packed token ids + token lengths).
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

import config
from dataset import VCMDataset


def collate_fn(batch):
    """Pad mels to the longest in the batch and build CTC targets.

    Returns
    -------
    mels        : (B, T_max, n_mels)
    intents     : (B,) int64
    transcripts : list[str]
    ctc_targets : (sum of token lengths,) int64  -- flattened, no blanks
    ctc_lengths : (B,) int64 -- number of tokens per sample
    """
    mels, intents, transcripts = zip(*batch)

    t_max = max(m.shape[0] for m in mels)
    n_mels = mels[0].shape[1]
    padded = torch.zeros(len(mels), t_max, n_mels)
    for i, m in enumerate(mels):
        padded[i, :m.shape[0]] = m

    token_lists = [config.transcript_to_tokens(t) for t in transcripts]
    ctc_targets = torch.tensor(
        [tid for toks in token_lists for tid in
         (config.CTC_TOKEN_TO_ID.get(t, config.CTC_UNK) for t in toks)],
        dtype=torch.long)
    ctc_lengths = torch.tensor([len(toks) for toks in token_lists],
                               dtype=torch.long)

    return (padded,
            torch.tensor(intents, dtype=torch.long),
            transcripts,
            ctc_targets,
            ctc_lengths)


def make_dataloader(dataset: VCMDataset,
                    batch_size: int = 32,
                    shuffle: bool = False,
                    num_workers: int = 0,
                    pin_memory: bool = False,
                    persistent_workers: bool | None = None) -> DataLoader:
    """DataLoader with the VCM collate function.

    num_workers: keep 0 on the Pi (RAM); 4-8 is fine on a dev machine.
    persistent_workers: default True when num_workers > 0 (avoids respawning
        workers every epoch); forced False when num_workers == 0.
    """
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
