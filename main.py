from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from pypdf import PdfReader
from dotenv import load_dotenv
import os
import uuid

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Initialise clients ───────────────────────────────────────────────────────
app = FastAPI(title="RAG Chatbot")
groq_client = Groq(api_key=GROQ_API_KEY)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="documents")


# ── Helper: split text into chunks ──────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ── Helper: extract text from file ──────────────────────────────────────────
def extract_text(file_bytes: bytes, filename: str) -> str:
    if filename.endswith(".pdf"):
        import io
        reader = PdfReader(io.BytesIO(file_bytes))
        return " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    else:
        return file_bytes.decode("utf-8")


# ── Route 1: Health check ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ── Route 2: Upload a document ───────────────────────────────────────────────
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    contents = await file.read()
    text = extract_text(contents, file.filename)

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from file")

    chunks = chunk_text(text)
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [str(uuid.uuid4()) for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

    return {
        "message": "Document uploaded successfully",
        "filename": file.filename,
        "chunks_stored": len(chunks)
    }


# ── Route 3: Ask a question ──────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(request: QuestionRequest):
    # Step 1: embed the question
    question_embedding = embedding_model.encode([request.question]).tolist()[0]

    # Step 2: retrieve top 3 relevant chunks from ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    if not results["documents"][0]:
        raise HTTPException(status_code=404, detail="No documents uploaded yet")

    context = "\n\n".join(results["documents"][0])

    # Step 3: send context + question to Groq
    prompt = f"""You are a helpful assistant. Answer the question based only on the context below.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {request.question}
"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    answer = response.choices[0].message.content

    return {
        "question": request.question,
        "answer": answer,
        "chunks_used": len(results["documents"][0])
    }


# ── Route 4: Frontend UI ─────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def frontend():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>RAG Chatbot</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Inter', sans-serif;
      background: #0f0f13;
      color: #e2e2e9;
      height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-start;
      padding: 40px 16px;
    }

    .header {
      text-align: center;
      margin-bottom: 36px;
    }

    .header h1 {
      font-size: 28px;
      font-weight: 600;
      background: linear-gradient(135deg, #a78bfa, #60a5fa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 6px;
    }

    .header p {
      font-size: 14px;
      color: #6b7280;
      font-weight: 300;
    }

    .container {
      width: 100%;
      max-width: 720px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    .upload-card {
      background: #1a1a24;
      border: 1px solid #2a2a3a;
      border-radius: 16px;
      padding: 24px;
    }

    .upload-card h2 {
      font-size: 14px;
      font-weight: 500;
      color: #9ca3af;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 14px;
    }

    .upload-row {
      display: flex;
      gap: 10px;
      align-items: center;
    }

    .file-label {
      flex: 1;
      display: flex;
      align-items: center;
      gap: 10px;
      background: #111118;
      border: 1px dashed #3a3a50;
      border-radius: 10px;
      padding: 12px 16px;
      cursor: pointer;
      transition: border-color 0.2s;
      font-size: 13px;
      color: #6b7280;
    }

    .file-label:hover { border-color: #a78bfa; color: #a78bfa; }
    .file-label input { display: none; }
    .file-label span.icon { font-size: 18px; }

    .btn {
      padding: 12px 22px;
      border-radius: 10px;
      border: none;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
      font-family: 'Inter', sans-serif;
    }

    .btn-primary {
      background: linear-gradient(135deg, #7c3aed, #3b82f6);
      color: white;
    }

    .btn-primary:hover { opacity: 0.88; transform: translateY(-1px); }
    .btn-primary:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

    .upload-status {
      margin-top: 10px;
      font-size: 13px;
      min-height: 18px;
      color: #6b7280;
    }

    .upload-status.success { color: #34d399; }
    .upload-status.error { color: #f87171; }

    .chat-card {
      background: #1a1a24;
      border: 1px solid #2a2a3a;
      border-radius: 16px;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .chat-card h2 {
      font-size: 14px;
      font-weight: 500;
      color: #9ca3af;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .chat-window {
      background: #111118;
      border: 1px solid #2a2a3a;
      border-radius: 12px;
      padding: 16px;
      min-height: 280px;
      max-height: 380px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .chat-window::-webkit-scrollbar { width: 4px; }
    .chat-window::-webkit-scrollbar-track { background: transparent; }
    .chat-window::-webkit-scrollbar-thumb { background: #2a2a3a; border-radius: 4px; }

    .empty-state {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 8px;
      color: #3a3a50;
      font-size: 13px;
      text-align: center;
      padding: 40px 0;
    }

    .empty-state .big { font-size: 32px; }

    .message { display: flex; flex-direction: column; gap: 4px; max-width: 88%; }
    .message.user { align-self: flex-end; align-items: flex-end; }
    .message.bot { align-self: flex-start; align-items: flex-start; }

    .bubble {
      padding: 10px 14px;
      border-radius: 12px;
      font-size: 14px;
      line-height: 1.6;
    }

    .message.user .bubble {
      background: linear-gradient(135deg, #7c3aed, #3b82f6);
      color: white;
      border-bottom-right-radius: 4px;
    }

    .message.bot .bubble {
      background: #1f1f2e;
      border: 1px solid #2a2a3a;
      color: #e2e2e9;
      border-bottom-left-radius: 4px;
    }

    .message .meta {
      font-size: 11px;
      color: #4b5563;
    }

    .typing .bubble {
      display: flex;
      gap: 5px;
      align-items: center;
      padding: 12px 16px;
    }

    .dot {
      width: 7px; height: 7px;
      background: #6b7280;
      border-radius: 50%;
      animation: bounce 1.2s infinite;
    }
    .dot:nth-child(2) { animation-delay: 0.2s; }
    .dot:nth-child(3) { animation-delay: 0.4s; }

    @keyframes bounce {
      0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
      40% { transform: translateY(-5px); opacity: 1; }
    }

    .input-row {
      display: flex;
      gap: 10px;
    }

    .input-row input {
      flex: 1;
      background: #111118;
      border: 1px solid #2a2a3a;
      border-radius: 10px;
      padding: 12px 16px;
      font-size: 14px;
      color: #e2e2e9;
      font-family: 'Inter', sans-serif;
      outline: none;
      transition: border-color 0.2s;
    }

    .input-row input:focus { border-color: #7c3aed; }
    .input-row input::placeholder { color: #3a3a50; }
  </style>
</head>
<body>

  <div class="header">
    <h1>✦ RAG Chatbot</h1>
    <p>Upload a document, then ask anything about it</p>
  </div>

  <div class="container">

    <div class="upload-card">
      <h2>📄 Document</h2>
      <div class="upload-row">
        <label class="file-label">
          <span class="icon">⬆</span>
          <span id="file-name">Choose a .txt or .pdf file</span>
          <input type="file" id="file-input" accept=".txt,.pdf" onchange="updateFileName(this)"/>
        </label>
        <button class="btn btn-primary" onclick="uploadFile()" id="upload-btn">Upload</button>
      </div>
      <div class="upload-status" id="upload-status"></div>
    </div>

    <div class="chat-card">
      <h2>💬 Chat</h2>
      <div class="chat-window" id="chat-window">
        <div class="empty-state" id="empty-state">
          <span class="big">🔍</span>
          <span>Upload a document to start chatting</span>
        </div>
      </div>
      <div class="input-row">
        <input
          type="text"
          id="question-input"
          placeholder="Ask a question about your document..."
          onkeydown="if(event.key==='Enter') askQuestion()"
        />
        <button class="btn btn-primary" onclick="askQuestion()" id="ask-btn">Ask</button>
      </div>
    </div>

  </div>

  <script>
    function updateFileName(input) {
      const name = input.files[0]?.name || 'Choose a .txt or .pdf file';
      document.getElementById('file-name').textContent = name;
    }

    async function uploadFile() {
      const fileInput = document.getElementById('file-input');
      const status = document.getElementById('upload-status');
      const btn = document.getElementById('upload-btn');

      if (!fileInput.files[0]) {
        status.textContent = 'Please select a file first.';
        status.className = 'upload-status error';
        return;
      }

      btn.disabled = true;
      status.textContent = 'Uploading...';
      status.className = 'upload-status';

      const formData = new FormData();
      formData.append('file', fileInput.files[0]);

      try {
        const res = await fetch('/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (res.ok) {
          status.textContent = `✓ "${data.filename}" uploaded — ${data.chunks_stored} chunk(s) stored`;
          status.className = 'upload-status success';
          document.getElementById('empty-state').style.display = 'none';
        } else {
          status.textContent = `Error: ${data.detail}`;
          status.className = 'upload-status error';
        }
      } catch (e) {
        status.textContent = 'Upload failed. Is the server running?';
        status.className = 'upload-status error';
      }

      btn.disabled = false;
    }

    async function askQuestion() {
      const input = document.getElementById('question-input');
      const question = input.value.trim();
      if (!question) return;

      const btn = document.getElementById('ask-btn');
      btn.disabled = true;
      input.value = '';

      addMessage('user', question);
      const typing = addTyping();

      try {
        const res = await fetch('/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question })
        });
        const data = await res.json();
        removeTyping(typing);
        if (res.ok) {
          addMessage('bot', data.answer, `${data.chunks_used} chunk(s) used`);
        } else {
          addMessage('bot', `Error: ${data.detail}`);
        }
      } catch (e) {
        removeTyping(typing);
        addMessage('bot', 'Something went wrong. Please try again.');
      }

      btn.disabled = false;
      input.focus();
    }

    function addMessage(role, text, meta = '') {
      const window = document.getElementById('chat-window');
      const div = document.createElement('div');
      div.className = `message ${role}`;
      div.innerHTML = `
        <div class="bubble">${text}</div>
        ${meta ? `<span class="meta">${meta}</span>` : ''}
      `;
      window.appendChild(div);
      window.scrollTop = window.scrollHeight;
    }

    function addTyping() {
      const window = document.getElementById('chat-window');
      const div = document.createElement('div');
      div.className = 'message bot typing';
      div.innerHTML = `<div class="bubble"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>`;
      window.appendChild(div);
      window.scrollTop = window.scrollHeight;
      return div;
    }

    function removeTyping(el) { el.remove(); }
  </script>

</body>
</html>
"""