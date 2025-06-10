import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.training.preprocess import CustomDataset
from src.models.seq2seq import Seq2Seq
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.training.evaluate import evaluate
import yaml
import os

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        src = batch['src_ids'].to(device)
        trg = batch['trg_ids'].to(device)

        optimizer.zero_grad()
        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['device'])

    train_dataset = CustomDataset(config['train_data_path'])
    val_dataset = CustomDataset(config['val_data_path'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=val_dataset.collate_fn)

    encoder = Encoder(config['src_vocab_size'], config['embedding_dim'], config['hidden_dim'], config['num_layers'], config['dropout'])
    decoder = Decoder(config['trg_vocab_size'], config['embedding_dim'], config['hidden_dim'], config['num_layers'], config['dropout'])
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=config['pad_id'])


    for epoch in range(config['epochs']):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
