from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import os
import uuid
import tempfile
from typing import Optional

# --- RAG, Whisper, Google STT, and TTS imports ---
import whisper
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from gtts import gTTS
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import numpy as np
from google.cloud import speech, texttospeech

# --- Twilio imports ---
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse

# --- API Key Setup ---
import google.generativeai as genai
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Google Cloud credentials must be set as env var GOOGLE_APPLICATION_CREDENTIALS
speech_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()

# --- Twilio Setup ---
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER")
twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

app = FastAPI()

# Set up templates and static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# In-memory knowledge base (replace with persistent storage as needed)
knowledge_base_path = os.path.join(BASE_DIR, "faiss_index")
kb_type = None  # 'pdf' or 'url'
kb_source = None
vectorstore = None
qa_chain = None

# --- Helper: Crawl site for URL KB ---
def crawl_site(base_url, max_depth=2, visited=None):
    if visited is None:
        visited = set()
    docs = []
    def crawl(url, depth):
        if depth > max_depth or url in visited:
            return
        visited.add(url)
        try:
            response = requests.get(url, verify=False, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            if text:
                docs.append({"page_content": text, "metadata": {"source": url}})
            for link_tag in soup.find_all("a", href=True):
                href = link_tag['href']
                full_url = urljoin(url, href)
                if urlparse(full_url).netloc == urlparse(base_url).netloc:
                    crawl(full_url, depth + 1)
        except Exception as e:
            print(f"❌ Failed to fetch {url}: {e}")
    crawl(base_url, 0)
    return docs

# --- Helper: Build KB from PDF or URL ---
def build_knowledge_base_from_pdf(pdf_path):
    global vectorstore, qa_chain
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local(knowledge_base_path)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

def build_knowledge_base_from_url(url):
    global vectorstore, qa_chain
    docs_raw = crawl_site(url, max_depth=3)
    from langchain.schema import Document
    docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in docs_raw]
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local(knowledge_base_path)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

# --- Helper: Download Twilio recording ---
def download_recording(recording_url, save_path, retries=5, delay=2):
    for attempt in range(retries):
        print(f"🔁 Attempt {attempt + 1} to download recording...")
        try:
            response = requests.get(recording_url, stream=True, verify=False, 
                                  auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
            if response.status_code == 200:
                if len(response.content) > 10240:  # Check if content size is reasonable (> 10KB)
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    print("✅ Recording downloaded successfully.")
                    return True
                else:
                    print(f"⚠️ Downloaded file too small (size: {len(response.content)} bytes). Retrying...")
            else:
                print(f"⚠️ HTTP Status Code: {response.status_code}. Response: {response.text[:100]}... Retrying...")
        except Exception as e:
            print(f"⚠️ Error during download: {e}")
        import time
        time.sleep(delay)
    print("❌ Failed to download a valid recording after retries.")
    return False

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/set_knowledge_base")
def set_knowledge_base(pdf: Optional[UploadFile] = File(None), url: Optional[str] = Form(None)):
    global kb_type, kb_source
    if pdf:
        pdf_path = os.path.join(tempfile.gettempdir(), f"kb_{uuid.uuid4()}.pdf")
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(pdf.file, f)
        kb_type = 'pdf'
        kb_source = pdf_path
        build_knowledge_base_from_pdf(pdf_path)
        return {"status": "ok", "message": "Knowledge base set from PDF."}
    elif url:
        kb_type = 'url'
        kb_source = url
        build_knowledge_base_from_url(url)
        return {"status": "ok", "message": "Knowledge base set from URL."}
    else:
        return {"status": "error", "message": "Provide a PDF or URL."}

# --- Whisper Transcription ---
def transcribe_with_whisper(audio_path):
    model = whisper.load_model("medium")
    result = model.transcribe(audio_path)
    return result["text"]

# --- Google STT Transcription ---
def transcribe_with_google(audio_path):
    # Convert to 16kHz mono wav if needed
    import subprocess
    wav_path = audio_path
    if not audio_path.endswith(".wav"):
        wav_path = audio_path.replace(".mp3", ".wav")
        subprocess.run([
            "ffmpeg", "-i", audio_path, "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", wav_path
        ], check=True)
    with open(wav_path, "rb") as audio_file:
        content = audio_file.read()
    audio_config = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )
    response = speech_client.recognize(config=config, audio=audio_config)
    transcript = " ".join([result_alt.alternatives[0].transcript for result_alt in response.results])
    return transcript.strip()

# --- TTS with gTTS (for Whisper) ---
def synthesize_speech_gtts(text, output_path):
    tts = gTTS(text)
    tts.save(output_path)

# --- TTS with Google Cloud (for Google STT) ---
def synthesize_speech_google(text, output_path):
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    with open(output_path, "wb") as out:
        out.write(response.audio_content)

@app.post("/ask_whisper")
def ask_whisper(audio: UploadFile = File(...)):
    global qa_chain
    if qa_chain is None:
        return JSONResponse({"text": "Knowledge base not set.", "audio_url": None})
    audio_path = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4()}.wav")
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    transcribed_text = transcribe_with_whisper(audio_path)
    result = qa_chain.invoke(transcribed_text)
    rag_answer = result["answer"]
    # Fallback to LLM if needed (optional, can add Gemini self-eval here)
    # Synthesize answer
    audio_answer_path = os.path.join(BASE_DIR, "static", f"reply_{uuid.uuid4()}.mp3")
    synthesize_speech_gtts(rag_answer, audio_answer_path)
    audio_url = f"/static/{os.path.basename(audio_answer_path)}"
    return JSONResponse({"text": rag_answer, "audio_url": audio_url})

@app.post("/ask_google")
def ask_google(audio: UploadFile = File(...)):
    global qa_chain
    if qa_chain is None:
        return JSONResponse({"text": "Knowledge base not set.", "audio_url": None})
    audio_path = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4()}.wav")
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    transcribed_text = transcribe_with_google(audio_path)
    result = qa_chain.invoke(transcribed_text)
    rag_answer = result["answer"]
    # Fallback to LLM if needed (optional, can add Gemini self-eval here)
    # Synthesize answer
    audio_answer_path = os.path.join(BASE_DIR, "static", f"reply_{uuid.uuid4()}.mp3")
    synthesize_speech_google(rag_answer, audio_answer_path)
    audio_url = f"/static/{os.path.basename(audio_answer_path)}"
    return JSONResponse({"text": rag_answer, "audio_url": audio_url})

# --- Twilio Voice Webhook Endpoint ---
@app.post("/voice")
async def voice_webhook(request: Request):
    response = VoiceResponse()
    try:
        form_data = await request.form()
        audio_url = form_data.get('RecordingUrl')
        
        if not audio_url:
            # First call - ask user to speak
            response.say("Please ask your question after the beep. Press any key when done.")
            response.record(
                action="/voice",
                method="POST",
                max_length=30,
                play_beep=True,
                timeout=5,
                finish_on_key="*"
            )
            return Response(str(response), mimetype="application/xml")
        
        print(f"🔊 Received audio URL: {audio_url}")
        
        # Download the recording
        input_file = f"input_{uuid.uuid4()}.mp3"
        input_path = os.path.join(tempfile.gettempdir(), input_file)
        download_success = download_recording(audio_url, input_path)
        
        if not download_success:
            response.say("Sorry, we couldn't process your voice message.")
            return Response(str(response), mimetype="application/xml")
        
        # Process with Google STT (you can change to Whisper if preferred)
        global qa_chain
        if qa_chain is None:
            response.say("Knowledge base not set. Please set up the knowledge base first.")
            return Response(str(response), mimetype="application/xml")
        
        transcribed_text = transcribe_with_google(input_path)
        if not transcribed_text:
            response.say("I'm sorry, I couldn't understand that.")
            return Response(str(response), mimetype="application/xml")
        
        # Run RAG
        result = qa_chain.invoke(transcribed_text)
        rag_answer = result["answer"]
        
        # Synthesize and play response
        reply_file = f"reply_{uuid.uuid4()}.mp3"
        reply_path = os.path.join(BASE_DIR, "static", reply_file)
        synthesize_speech_google(rag_answer, reply_path)
        
        # For Twilio, we need to host the audio file publicly
        # You'll need to set up a public URL for your static files
        # For now, we'll use text-to-speech directly
        response.say(rag_answer)
        
        return Response(str(response), mimetype="application/xml")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        response.say("Sorry, an application error occurred.")
        return Response(str(response), mimetype="application/xml")

# --- Outbound Call Endpoint ---
@app.post("/call")
def make_outbound_call(to_number: str = Form(...)):
    if not twilio_client:
        return {"error": "Twilio not configured"}
    
    try:
        # You'll need to set up a public URL for your /voice endpoint
        # For Render, this would be your app URL + /voice
        voice_url = "https://your-app-name.onrender.com/voice"  # Update this with your actual URL
        
        call = twilio_client.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            url=voice_url
        )
        return {"status": "success", "message": f"Call initiated to {to_number}", "call_sid": call.sid}
    except Exception as e:
        return {"error": str(e)}

# --- Call Management Page ---
@app.get("/call", response_class=HTMLResponse)
def call_page(request: Request):
    return templates.TemplateResponse("call.html", {"request": request}) 