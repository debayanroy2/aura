from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os, uuid, re
import logging
from google import genai
from google.genai import types
import chromadb
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import pymupdf
import hashlib
from functools import lru_cache
import base64

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    logging.warning("GOOGLE_API_KEY not set - AI features will be disabled")
    client = None
else:
    client = genai.Client(api_key=GOOGLE_API_KEY)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

KB_PATH = os.getenv("CHROMA_PATH", "/tmp/aura_chroma")
os.makedirs(KB_PATH, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=KB_PATH)
collection = chroma_client.get_or_create_collection("aura_docs")

logging.info(f"Chroma initialized at: {KB_PATH}")

system_prompt = """
You are AURA — Academic Unified Research Agent.
Core behavior:
    - Be direct and helpful. No greetings, no filler.
    - Always format output in Markdown with headings, bold, and bullet points.
    - Use LaTeX for equations using $$...$$.
    - If the user uploads context (PDF/image text), use it as the primary source.
    - If context is missing or insufficient, answer using general knowledge and clearly mention: "Answered using general knowledge (context insufficient)."
    - Do NOT repeat the uploaded text unless the user asks to extract it.
    - If the user asks to solve a question paper, solve step-by-step and give final answers clearly.

Capabilities:
    - Explain concepts
    - Solve problems
    - Summarize notes
    - Evaluate answers
    - Provide career roadmaps.

Output rules:
    - Keep answers structured and readable.
    - For numericals: formula → substitution → final answer with unit.
"""

MODEL = "gemini-2.0-flash-exp"
EMBED_MODEL = "text-embedding-004"


def safe_extract_text(response) -> str:
    """Extract text from google-genai response."""
    try:
        return response.text if hasattr(response, 'text') else str(response)
    except Exception as e:
        logging.error(f"Error extracting text: {e}")
        return ""


def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.encode("utf-8", "ignore").decode("utf-8", "ignore")
    t = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def chunk_text(text, chunk_size=900, overlap=150):
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


def encode_jpeg(img: Image.Image) -> bytes:
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=60, optimize=True)
    return buf.getvalue()


def _hash_text(text: str) -> str:
    return hashlib.sha1(text[:8000].encode("utf-8", "ignore")).hexdigest()


@lru_cache(maxsize=8000)
def _cached_embed_by_hash(text_hash: str, text: str):
    if not client:
        raise RuntimeError("API client not initialized")
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[text[:8000]]
    )
    return result.embeddings[0].values


def embed_texts(texts):
    vecs = []
    for t in texts:
        t = t or ""
        h = _hash_text(t)
        vecs.append(_cached_embed_by_hash(h, t))
    return vecs


@lru_cache(maxsize=256)
def _cached_embed_single(text: str):
    if not client:
        raise RuntimeError("API client not initialized")
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[text[:8000]]
    )
    return result.embeddings[0].values


def ocr_single_page(img_bytes: bytes) -> str:
    try:
        if not client:
            raise RuntimeError("API client not initialized")
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                "Extract ALL text EXACTLY as it appears. Preserve line breaks and symbols.",
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
            ]
        )
        return clean_text(safe_extract_text(response))
    except Exception as e:
        logging.exception(f"OCR failed: {e}")
        return ""


def process_pdf_fast(pdf_bytes, filename, doc_id):
    pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    total_pages = pdf.page_count
    total = min(total_pages, 10)

    logging.info(f"Processing {total}/{total_pages} pages")

    def extract_text(i):
        try:
            return i, (pdf[i].get_text("text") or "")
        except:
            return i, ""

    all_text = [""] * total

    with ThreadPoolExecutor(max_workers=6) as exe:
        futures = [exe.submit(extract_text, i) for i in range(total)]
        for f in as_completed(futures):
            i, txt = f.result()
            all_text[i] = txt

    has_text_layer = any(t.strip() for t in all_text)

    if not has_text_layer:
        logging.info("No text layer → Running OCR (FAST MODE)")

        def ocr_page(i):
            page = pdf[i]
            pix = page.get_pixmap(dpi=120)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img.thumbnail((1600, 1600))
            return i, clean_text(ocr_single_page(encode_jpeg(img)))

        ocr_results = [""] * total
        with ThreadPoolExecutor(max_workers=2) as exe:
            futures = [exe.submit(ocr_page, i) for i in range(total)]
            for f in as_completed(futures):
                i, txt = f.result()
                ocr_results[i] = txt

        merged = "\n".join(ocr_results)
    else:
        merged = "\n".join(clean_text(t) for t in all_text if t.strip())

    final_text = clean_text(merged)
    chunks = [clean_text(c) for c in chunk_text(final_text) if clean_text(c)]
    chunks = [c for c in chunks if len(c.split()) >= 30]

    if not chunks:
        logging.warning("No valid chunks extracted from document.")
        return

    ids = [f"{doc_id}_{uuid.uuid4().hex}" for _ in chunks]
    metas = [{"source": filename, "doc_id": doc_id} for _ in chunks]
    vecs = embed_texts(chunks)

    collection.add(ids=ids, documents=chunks, metadatas=metas, embeddings=vecs)
    pdf.close()

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint required by Replit deployments."""
    return jsonify({"status": "healthy", "service": "AURA"}), 200


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        filename = (file.filename or "").lower()
        ext = os.path.splitext(filename)[1]
        file_bytes = file.read()

        if not file_bytes:
            return jsonify({"error": "Empty file"}), 400

        doc_id = uuid.uuid4().hex

        if ext == ".pdf":
            process_pdf_fast(file_bytes, filename, doc_id)
            return jsonify({"message": "PDF processed", "file_id": doc_id})

        return jsonify({"error": "Unsupported file type"}), 400

    except Exception as e:
        logging.exception("UPLOAD FAILED")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if request.is_json:
            data = request.get_json(silent=True) or {}
        else:
            data = request.form.to_dict()

        user_input = (data.get("user_input") or "").strip()
        doc_id = (data.get("file_id") or "").strip() or None

        if not user_input:
            return jsonify({"message": "Please enter a valid question."})

        if not client:
            return jsonify({"message": "Error: GOOGLE_API_KEY not configured. Please set it in Secrets."}), 500

        try:
            qvec = _cached_embed_single(user_input)
        except Exception as e:
            logging.exception("Embedding failed")
            return jsonify({"message": f"Embedding error: {str(e)}"}), 500

        try:
            if doc_id:
                res = collection.query(
                    query_embeddings=[qvec],
                    n_results=6,
                    where={"doc_id": {"$eq": doc_id}}
                )
            else:
                res = collection.query(query_embeddings=[qvec], n_results=6)

            docs = []
            if isinstance(res, dict):
                docs_list = res.get("documents")
                if docs_list and len(docs_list) > 0 and docs_list[0]:
                    docs = docs_list[0][:4]

            context = "\n\n".join((d or "")[:6000] for d in docs).strip()

        except Exception as e:
            logging.exception("Chroma query failed")
            return jsonify({"message": f"KB query error: {str(e)}"}), 500

        if context:
            prompt = f"""
User Query: {user_input}
Context: {context}
Rules:
    - The Context contains the question paper / notes.
    - Your job is to ANSWER/SOLVE the questions provided.
    - If the Context is a question paper, solve each question step-by-step and give final answers clearly.
    - If a question is missing required data, state what is missing and still provide the general method/formula.
    - Format the response in Markdown with headings and numbering.
    - Use LaTeX for equations like $$...$$.
"""
        else:
            prompt = f"""
User Query: {user_input}
Rules:
    - No uploaded context is available.
    - Answer from general knowledge.
    - Format in Markdown with headings, bullet points, and practical steps.
    - If it's a career question, give a roadmap and skills list.
    - Use LaTeX for equations ($$...$$) if needed.
"""

        try:
            # Use new google-genai API
            response = client.models.generate_content(
                model=MODEL,
                contents=[system_prompt, prompt]
            )

            msg = safe_extract_text(response)

            if not msg:
                msg = "No response text returned by the model (possibly blocked or empty output)."

            return jsonify({"message": msg})

        except Exception as e:
            logging.exception("MODEL.generate_content failed")
            return jsonify({"message": f"Model error: {str(e)}"}), 500

    return render_template("index.html")


@app.route("/reset", methods=["POST"])
def reset():
    global collection
    try:
        chroma_client.delete_collection("aura_docs")
    except Exception:
        logging.exception("delete_collection failed (ignored)")

    collection = chroma_client.get_or_create_collection("aura_docs")
    return jsonify({"message": "Knowledge base reset"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
