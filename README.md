# Neural Machine Translation

A beginner-friendly, end-to-end language translation project built from scratch using PyTorch. This project supports translation from **English → German** and **English → Nepali** using a custom-trained Seq2Seq model with GRU.

## Features

- Sequence-to-Sequence model with GRU
- Trained on English-German and English-Nepali datasets
- Preprocessing using SentencePiece (subword tokenizer)
- Evaluation with BLEU Score
- Inference API using FastAPI
- UI built with Next.js
- Dockerized backend for deployment

## Project Structure

```
neural-machine-translation/
│
├── data/                    # Datasets and preprocessing scripts
├── models/                  # Saved PyTorch models
├── src/
│   ├── encoder_rnn.py
│   ├── decoder_rnn.py
│   └── seq2seq_rnn.py      # Core Seq2Seq logic
│
├── api/                    # FastAPI app
│   └── main.py
│
├── frontend/               # Next.js UI app
│   └── pages/index.tsx
│
├── Dockerfile              # Backend Dockerfile
├── requirements.txt
└── README.md
```

## Model Information

| Property        | English-German | English-Nepali |
| --------------- | -------------- | -------------- |
| Batch Size      | 64             | 32             |
| Vocab (English) | 16,000         | 4,000          |
| Vocab (Target)  | 16,000         | 4,000          |
| Embedding Dim   | 256            | 256            |
| Hidden Dim      | 512            | 512            |
| Epochs          | 15             | 15             |
| Training Time   | ~3h 4m         | ~1 min         |

## BLEU Score Evaluation

### English → German

- 1-gram: 0.5732
- 2-gram: 0.3976
- 3-gram: 0.2997
- 4-gram: 0.2322

### English → Nepali (Low-resource)

- 1-gram: 0.1262
- 2-gram: 0.0612
- 3-gram: 0.0431
- 4-gram: 0.0319

## Setup and Usage

### Prerequisites

- Python 3.8+
- PyTorch
- Node.js (for frontend)
- Docker (optional)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/neural-machine-translation.git
cd neural-machine-translation
```

2. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

3. **Train the model**

Run the colab .ipynb file `03_training.ipynb` and `03_training_np.ipynb`

4. **Run FastAPI server**

```bash
uvicorn api.main:app --reload
```

5. **Run the Next.js frontend**

```bash
cd frontend
npm install
npm run dev
```

The API will be available at `http://localhost:8000` and the frontend at `http://localhost:3000`.

## Docker Setup

### Build and run the API container

```bash
# Build Docker image
docker build -t neural-translation-api .

# Run container
docker run -p 8000:8000 neural-translation-api
```

## API Usage

### Translation Endpoint

**POST** `/translate/{target_language}`

Request body:

```json
{
  "text": "Hello, how are you?",
}
```

Response:

```json
{
  "translated_text": "Hallo, wie geht es dir?",
  "language":"german"
}
```

## Deployment

- **Frontend**: Deploy on Vercel or Netlify
- **Backend**: Deploy on Render, Railway, or any cloud platform supporting Docker

## Architecture

The project uses a Sequence-to-Sequence architecture with:

- **Encoder**: GRU-based encoder that processes the input sequence
- **Decoder**: GRU-based decoder that generates the output sequence
- **Attention**: Basic attention mechanism for better translation quality
- **Tokenization**: SentencePiece for subword tokenization

## Dataset

The model is trained on:

- **English-German**: Standard parallel corpus
- **English-Nepali**: Limited parallel corpus (low-resource scenario)

## Evaluation

The model performance is evaluated using BLEU scores:

- Higher scores indicate better translation quality
- English-German shows reasonable performance
- English-Nepali shows lower scores due to limited training data

## Learnings and Challenges

- Built and trained a Seq2Seq model from scratch
- Implemented subword tokenization with SentencePiece
- Hardware limitations with GTX 3050 4GB GPU
- Long training times, especially for English-German pairs
- BLEU score improvements limited by dataset size and computational resources

## Future Improvements

- Use larger and higher-quality datasets
- Implement Transformer-based models (attention is all you need)
- Add more sophisticated attention mechanisms
- Experiment with bidirectional GRU and LSTM variants
- Hyperparameter optimization for better performance
- Support for more language pairs
- Implement beam search for better decoding

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- SentencePiece for tokenization
- FastAPI for the web framework
- Next.js for the frontend framework
