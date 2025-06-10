import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

PAD_ID = 0  # your PAD token ID (make sure it matches SentencePiece pad_id)


class CustomDataset(Dataset):
    def __init__(self, df):
        self.src = df['src_ids'].tolist()
        self.trg = df['trg_ids'].tolist()

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return {'src_ids': self.src[idx], 'trg_ids': self.trg[idx]}

    @staticmethod
    def collate_fn(batch):
        src_batch = [torch.tensor(item['src_ids']) for item in batch]
        trg_batch = [torch.tensor(item['trg_ids']) for item in batch]

        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_ID)
        trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=PAD_ID)

        return {'src_ids': src_batch, 'trg_ids': trg_batch}


def get_dataloaders(df, batch_size=64, shuffle=True):
    dataset = CustomDataset(df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=CustomDataset.collate_fn)
    return loader


if __name__ == "__main__":
    # Example: load preprocessed CSV
    df = pd.read_csv('data/processed/cleaned_dataset.csv')

    train_loader = get_dataloaders(df)

    # Check one batch
    batch = next(iter(train_loader))
    print(batch['src_ids'].shape)  # (batch_size, seq_len)
    print(batch['trg_ids'].shape)
