
# Jatin AI Assistant 🤖

## 🚀 Live Demo

Try the app here: [Jatin AI Assistant](https://jatin-ai-assistant.streamlit.app/)

A personal AI assistant built with Python, Streamlit, and Groq API.

## Features

- AI chat using Groq
- PDF and TXT file upload
- Image upload and description
- Basic RAG using TF-IDF and cosine similarity
- Chat history saved locally
- Multiple conversations with New Chat and Recents

## Tech Stack

- Python
- Streamlit
- Groq API
- PyPDF
- Scikit-learn


## Installation

1. Clone the repository:

   git clone https://github.com/Jatingulia9192/Jatin-AI-Assistant.git

2. Open the project folder:

   cd Jatin-AI-Assistant

3. Create a virtual environment:

   python -m venv .venv

4. Activate it on Windows:

   .venv\Scripts\Activate.ps1

5. Install dependencies:

   pip install -r requirements.txt

6. Create `.streamlit/secrets.toml` and add your Groq API key:

   GROQ_API_KEY = "your_groq_api_key"

7. Run the application:

   streamlit run app.py

## Security

Keep your API key private. Never commit your secrets file to GitHub.

## Author

Jatin Gulia
