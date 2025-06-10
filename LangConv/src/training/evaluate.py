import torch

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch['src_ids'].to(device)
            trg = batch['trg_ids'].to(device)

            output = model(src, trg, teacher_forcing_ratio=0)  # no teacher forcing in eval

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            total_loss += loss.item()

    return total_loss / len(dataloader)
