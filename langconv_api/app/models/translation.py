import torch
import os
import torch.nn as nn
import sentencepiece as spm
from app.models.encoder import Encoder
from app.models.decoder import Decoder
from app.models.seq2seq import Seq2Seq


def translate(sentence: str, convert_to: str, max_len: int = 50) -> str:
    """
    Translates an English sentence to Nepali or German using pre-trained models.

    Args:
        sentence (str): Input English sentence.
        convert_to (str): Target language ('nepali' or 'german').
        max_len (int): Max length of the output sentence.

    Returns:
        str: Translated sentence.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # points to app/models

    config = {
        'nepali': {
            'input_dim': 4000,
            'output_dim': 4000,
            'sp_src': os.path.join(BASE_DIR, "spm_eng_n.model"),
            'sp_trg': os.path.join(BASE_DIR, "spm_npi_e.model"),
            'model_path': os.path.join(BASE_DIR, "seq2seq_gru_eng_npi_model.pt"),
        },
        'german': {
            'input_dim': 16000,
            'output_dim': 16000,
            'sp_src': os.path.join(BASE_DIR, "spm_en.model"),
            'sp_trg': os.path.join(BASE_DIR, "spm_de.model"),
            'model_path': os.path.join(BASE_DIR, "seq2seq_gru_model.pt"),
        }
    }

    if convert_to.lower() not in config:
        raise ValueError("convert_to must be either 'nepali' or 'german'")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load sentencepiece models
    sp_en = spm.SentencePieceProcessor()
    sp_trg = spm.SentencePieceProcessor()

    sp_en.load(config[convert_to]['sp_src'])
    sp_trg.load(config[convert_to]['sp_trg'])

    # Load encoder and decoder
    INPUT_DIM = config[convert_to]['input_dim']
    OUTPUT_DIM = config[convert_to]['output_dim']
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HIDDEN_DIM = 512

    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HIDDEN_DIM)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Load pre-trained weights
    model.load_state_dict(torch.load(config[convert_to]['model_path'], map_location=device))
    model.eval()

    # Tokenize and encode input sentence
    tokens = [2] + sp_en.encode(sentence, out_type=int) + [3]  # BOS + tokens + EOS
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)  # [1, seq_len]

    # Encode input
    with torch.no_grad():
        hidden = model.encoder(src_tensor)

    # Decode step-by-step
    trg_indexes = [2]  # BOS
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == 3:  # EOS
            break

    # Decode output to text
    translated_text = sp_trg.decode(trg_indexes[1:-1])  # exclude BOS and EOS
    return translated_text
