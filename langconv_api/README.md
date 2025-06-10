# Language Translation API

A FastAPI-based REST API for translating English sentences into **Nepali** or **German**, using pre-trained RNN-based sequence-to-sequence models.

---

## Features

- Translate English to Nepali
- Translate English to German
- SentencePiece tokenization
- FastAPI backend for low-latency inference
- Docker support for containerized deployment

---

## Project Structure

langconv_api/
├── app/
│ ├── main.py # FastAPI application entry point
│ ├── models/
│ │ ├── encoder.py # Encoder RNN
│ │ ├── decoder.py # Decoder RNN
│ │ ├── seq2seq.py # Seq2Seq wrapper
│ │ └── translation.py # Inference logic
│ └── schemas/
│ ├── input_validation.py # Input request schema
│ └── predection_response.py # Output response schema
├── data/
│ ├── spm_en.model
│ ├── spm_de.model
│ ├── spm_eng_n.model
│ └── spm_npi_e.model
├── seq2seq_gru_model.pt # English → German model weights
├── seq2seq_gru_eng_npi_model.pt # English → Nepali model weights
├── requirements.txt
├── Dockerfile
└── README.md



---

## API Endpoint

### `POST /convert/{convert_to}`

Translate an English sentence to Nepali or German.

**Path Parameter:**
- `convert_to`: Either `nepali` or `german`

**Request Body:**
```json
{
  "text": "Hello, how are you?"
}


Response:
{
  "message": "Translation successful",
  "english_text": "Hello, how are you?",
  "translated_text": "Hallo, wie geht es dir?",
  "language": "german"
}

Installation:
Local Setup

git clone https://github.com/your-username/langconv_api.git
cd langconv_api
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload

Visit: http://localhost:8000

Docker Setup
Build the Docker Image
bash
Copy
Edit
docker build -t iamsaurav/langconv-api .
Run the Container
bash
Copy
Edit
docker run -p 8000:8000 iamsaurav/langconv-api
Access the API at: http://localhost:8000

CORS Configuration
If you are accessing the API from a frontend application, ensure CORS is enabled. Add this to main.py:

python
Copy
Edit
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, use specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Install using:

bash
Copy
Edit
pip install -r requirements.txt
Notes
Ensure model weights (.pt files) and SentencePiece model files are present in the correct paths as shown in the project structure.

Update paths in translation.py if your directory layout changes.

To reduce Docker build time, consider using .dockerignore to exclude unnecessary files.