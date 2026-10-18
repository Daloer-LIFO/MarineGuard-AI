# MarineGuard AI

**An Intelligent Maritime Safety and Incident Decision-Support System**

## Overview

MarineGuard AI is an AI-powered prototype that supports maritime information retrieval, incident classification, and incident visualization.

The application combines Retrieval-Augmented Generation (RAG), a machine learning text classifier, REST APIs, a web interface, and structured incident storage.

## Features

- PDF-based maritime knowledge assistant with retrieved source references
- AI-assisted maritime incident classification
- Incident record storage and retrieval
- Interactive incident dashboard and map
- FastAPI backend and Streamlit frontend

## Technology Stack

- Python
- Streamlit
- FastAPI
- Google Gemini API
- Sentence Transformers
- FAISS
- Scikit-learn
- SQLite
- Plotly and Folium

## Project Structure

- `frontend/`: User interface
- `backend/`: REST API and database
- `backend/services/`: RAG and classification modules
- `data/`: Local data and uploaded documents
- `tests/`: Automated tests
- `docs/`: Project documentation

## Installation

1. Install Python 3.11 or 3.12 and Git.
2. Clone the repository and open the project folder.
3. Create and activate a Python virtual environment.
4. Install dependencies using `pip install -r requirements.txt`.
5. Create a `.env` file with the required Gemini API configuration.
6. Start FastAPI and Streamlit in separate terminals.

## Running the Application

Start the backend:

`python -m uvicorn backend.main:app --reload --port 8000`

Start the frontend:

`python -m streamlit run frontend/app.py`

The frontend is normally available at `http://localhost:8501`.

FastAPI documentation is available at `http://127.0.0.1:8000/docs`.

## Evaluation

The incident classifier will be evaluated using a held-out labeled dataset, precision, recall, F1-score, and a confusion matrix.

The RAG assistant will be assessed using predefined questions, answer relevance, source accuracy, and unsupported-answer checks.

## Limitations

The initial version uses a small illustrative classification dataset, text-extractable PDFs, and sample incident records. Its predictions are not official maritime warnings or operational instructions.

## Security

Never commit API keys, credentials, confidential documents, or sensitive operational information.

## Future Improvements

- Expand and validate the incident dataset
- Improve document retrieval and answer evaluation
- Add OCR for scanned PDFs
- Add document-management features
- Introduce persistent vector storage
- Integrate verified public maritime datasets where appropriate

## Deployment

The deployed application URL and demonstration screenshots will be added after testing and deployment.