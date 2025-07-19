
# Voice RAG App

A web and phone-based Question Answering system using Retrieval-Augmented Generation (RAG) with Gemini, Whisper, and Google Cloud Speech/TTs APIs. Users can ask questions via browser voice or outbound phone calls, and get answers from a knowledge base built from a PDF or URL.

## Features
- Web-based voice chat (Whisper transcription, Gemini RAG, TTS answer)
- Outbound phone call Q&A (Twilio, Google STT, Gemini RAG, TTS answer)
- Upload PDF or enter URL to set the knowledge base
- Both text and audio answers

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/voice-rag-app.git
   cd voice-rag-app
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project root (see below for contents), or set these in your deployment environment (e.g., Render dashboard).

4. **Google Cloud Credentials:**
   - Create a Google Cloud service account with Speech-to-Text and Text-to-Speech permissions.
   - Download the JSON key and **do not commit it to GitHub**.
   - Set the path in the `.env` file as `GOOGLE_APPLICATION_CREDENTIALS`.

5. **Run the app locally:**
   ```bash
   uvicorn main:app --reload
   ```

6. **Deploy to Render:**
   - Connect your GitHub repo to Render.
   - Set all environment variables in the Render dashboard.
   - Add a build command if needed: `pip install -r requirements.txt`
   - Set the start command: `uvicorn main:app --host 0.0.0.0 --port 10000`

## Environment Variables (`.env` example)

```
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_APPLICATION_CREDENTIALS=/full/path/to/your-google-credentials.json
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=your_twilio_phone_number
```

- **GEMINI_API_KEY**: Gemini/Google Generative AI API key
- **GOOGLE_APPLICATION_CREDENTIALS**: Path to your Google Cloud service account JSON
- **TWILIO_ACCOUNT_SID**: Twilio account SID
- **TWILIO_AUTH_TOKEN**: Twilio auth token
- **TWILIO_PHONE_NUMBER**: Your Twilio phone number (E.164 format, e.g., +15551234567)

## Usage

- Visit `/` for web-based voice chat (Whisper)
- Visit `/call` to make outbound phone calls
- Upload a PDF or enter a URL to set the knowledge base

## Security
- **Never commit your `.env` or Google credentials JSON to GitHub!**
- Add them to your `.gitignore`.

## License
MIT

