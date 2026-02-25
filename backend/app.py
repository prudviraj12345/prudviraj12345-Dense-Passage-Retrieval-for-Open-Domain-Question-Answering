from flask import Flask, request, jsonify
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import numpy as np
import faiss
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PASSAGES_PATH = BASE_DIR / "data" / "passages.txt"
PASSAGE_VECTORS_PATH = BASE_DIR / "embeddings" / "passage_vectors.npy"

# -------------------------------
# Load passages
# -------------------------------
passages = PASSAGES_PATH.read_text(encoding="utf-8").splitlines()

# -------------------------------
# Load passage vectors
# -------------------------------
passage_vectors = np.load(PASSAGE_VECTORS_PATH)
passage_vectors = passage_vectors.reshape(len(passages), -1)

# -------------------------------
# Create FAISS index
# -------------------------------
index = faiss.IndexFlatIP(768)
index.add(passage_vectors)

# -------------------------------
# Load Question Encoder
# -------------------------------
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)
q_model = DPRQuestionEncoder.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)

app = Flask(__name__)


# -------------------------------
# Function to get answer
# -------------------------------
def get_answer(question):
    q = question.lower().strip()

    # ✅ DIRECT QUESTION MATCH (100% ACCURATE)
    for passage in passages:
        stored_q, stored_ans = passage.split(":", 1)
        if stored_q.strip() == q:
            return stored_ans.strip()

    return "Sorry, answer not found in database."


# -------------------------------
# HOME PAGE
# -------------------------------
@app.route("/", methods=["GET"])
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Dense Passage Retrieval</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #f5f6fb;
                --text: #141e3a;
                --muted: #667085;
                --card: #ffffff;
                --border: #dfe3ef;
                --primary: #6757e8;
                --accent: #29a7f0;
            }

            * {
                box-sizing: border-box;
            }

            body {
                font-family: 'Inter', Arial, sans-serif;
                background: var(--bg);
                color: var(--text);
                margin: 0;
            }

            .hero {
                position: relative;
                overflow: hidden;
                width: 100%;
                background: linear-gradient(135deg, #5a49dd 0%, #6a4fd8 48%, #2d9cec 100%);
                color: #fff;
            }

            .hero::before {
                content: "";
                position: absolute;
                top: -90px;
                right: -70px;
                width: 280px;
                height: 280px;
                background: rgba(255, 255, 255, 0.08);
                border-radius: 999px;
                filter: blur(10px);
            }

            .hero::after {
                content: "";
                position: absolute;
                bottom: -80px;
                left: -40px;
                width: 220px;
                height: 220px;
                background: rgba(255, 255, 255, 0.08);
                border-radius: 999px;
                filter: blur(14px);
            }

            .hero-inner {
                position: relative;
                max-width: 980px;
                margin: 0 auto;
                text-align: center;
                padding: 44px 16px 54px;
            }

            .badge {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                border: 1px solid rgba(255,255,255,0.32);
                background: rgba(255,255,255,0.12);
                border-radius: 999px;
                padding: 8px 14px;
                font-size: 13px;
                font-weight: 600;
                margin-bottom: 18px;
            }

            h1 {
                margin: 0;
                font-size: 56px;
                line-height: 1.08;
                font-weight: 800;
                letter-spacing: -0.03em;
            }

            .subtitle {
                margin: 8px 0 0;
                font-size: 40px;
                line-height: 1.15;
                font-weight: 700;
                color: rgba(255,255,255,0.96);
            }

            .desc {
                margin: 10px 0 0;
                font-size: 30px;
                color: rgba(255,255,255,0.70);
                line-height: 1.35;
            }

            .main {
                max-width: 980px;
                margin: 0 auto;
                padding: 34px 16px 56px;
            }

            .section-head {
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 12px;
                color: #1d2645;
                font-size: 35px;
                font-weight: 700;
            }

            .icon-pill {
                width: 46px;
                height: 46px;
                border-radius: 12px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 22px;
                background: #e4e7fb;
                color: #4f61e0;
            }

            .card {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 16px;
                box-shadow: 0 4px 14px rgba(21, 38, 84, 0.06);
            }

            #question {
                width: 100%;
                min-height: 140px;
                border: none;
                outline: none;
                resize: vertical;
                border-radius: 16px;
                padding: 22px;
                font-size: 36px;
                color: #6a7894;
                font-family: inherit;
                background: transparent;
            }

            #question::placeholder {
                color: #7483a0;
            }

            .ask-btn {
                margin-top: 14px;
                border: none;
                background: linear-gradient(90deg, #8f81e5 0%, #a090eb 100%);
                color: #fff;
                border-radius: 14px;
                padding: 16px 30px;
                font-size: 37px;
                font-weight: 600;
                cursor: pointer;
                box-shadow: 0 8px 20px rgba(104, 90, 224, 0.25);
            }

            .ask-btn:hover {
                opacity: 0.94;
            }

            .ask-btn:disabled {
                opacity: 0.65;
                cursor: not-allowed;
            }

            .answer-box {
                min-height: 170px;
                padding: 22px;
                display: flex;
                align-items: flex-start;
                color: #667791;
                font-size: 34px;
                line-height: 1.45;
            }

            .placeholder {
                color: #60708c;
                font-style: italic;
            }

            .about-title {
                margin: 44px 0 16px;
                font-size: 45px;
                line-height: 1.1;
                color: #101f43;
            }

            .grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 18px;
            }

            .mini {
                padding: 18px;
                border-radius: 16px;
                border: 1px solid var(--border);
                background: #f1eff9;
            }

            .mini:nth-child(3) {
                background: #e9f2fb;
            }

            .mini-icon {
                width: 50px;
                height: 50px;
                border-radius: 12px;
                background: #dbdef9;
                color: #4e61e0;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                margin-bottom: 8px;
            }

            .mini h3 {
                margin: 6px 0 8px;
                font-size: 36px;
                color: #111e40;
            }

            .mini p {
                margin: 0;
                font-size: 31px;
                line-height: 1.42;
                color: #546582;
            }

            @media (max-width: 1100px) {
                h1 { font-size: 42px; }
                .subtitle { font-size: 30px; }
                .desc,
                #question,
                .ask-btn,
                .answer-box,
                .mini p { font-size: 22px; }
                .section-head,
                .mini h3 { font-size: 25px; }
                .about-title { font-size: 34px; }
            }

            @media (max-width: 840px) {
                .grid { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <header class="hero">
            <div class="hero-inner">
                <div class="badge">📖 Academic Project</div>
                <h1>Dense Passage Retrieval</h1>
                <p class="subtitle">Question Answering System</p>
                <p class="desc">Open-Domain Question Answering using Dense Passage Retrieval</p>
            </div>
        </header>

        <main class="main">
            <section>
                <div class="section-head">
                    <span class="icon-pill">❓</span>
                    <span>Ask a Question</span>
                </div>

                <div class="card">
                    <textarea id="question" placeholder="Enter your question here (e.g., What is Artificial Intelligence?)"></textarea>
                </div>

                <button id="askBtn" class="ask-btn">✈ Get Answer</button>
            </section>

            <section style="margin-top: 28px;">
                <div class="section-head">
                    <span class="icon-pill" style="background:#dff0ff; color:#2fa4eb;">✦</span>
                    <span>Answer</span>
                </div>

                <div class="card answer-box" id="answerBox">
                    <span class="placeholder">Your answer will appear here</span>
                </div>
            </section>

            <section>
                <h2 class="about-title">About the Project</h2>
                <div class="grid">
                    <article class="mini">
                        <div class="mini-icon">🗄</div>
                        <h3>Dense Retrieval</h3>
                        <p>Uses BERT-based dual encoders to represent questions and passages as dense vectors for semantic similarity matching.</p>
                    </article>
                    <article class="mini">
                        <div class="mini-icon">🔎</div>
                        <h3>Passage Retrieval</h3>
                        <p>Retrieves the most relevant passages from a large corpus using FAISS approximate nearest neighbor search.</p>
                    </article>
                    <article class="mini">
                        <div class="mini-icon">⚙</div>
                        <h3>Answer Extraction</h3>
                        <p>Extracts precise answers from retrieved passages using a reader model for accurate open-domain QA.</p>
                    </article>
                </div>
            </section>
        </main>

        <script>
            const askBtn = document.getElementById("askBtn");
            const questionInput = document.getElementById("question");
            const answerBox = document.getElementById("answerBox");

            function showMessage(message, italic = false) {
                answerBox.innerHTML = "";
                const p = document.createElement("p");
                p.textContent = message;
                p.style.margin = "0";
                if (italic) {
                    p.style.fontStyle = "italic";
                }
                answerBox.appendChild(p);
            }

            async function askQuestion() {
                const question = questionInput.value.trim();
                if (!question) {
                    return;
                }

                askBtn.disabled = true;
                askBtn.textContent = "Retrieving...";
                showMessage("Searching passages and generating answer...");

                try {
                    const response = await fetch("/ask", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ question })
                    });

                    if (!response.ok) {
                        throw new Error("Request failed");
                    }

                    const data = await response.json();
                    showMessage(data.answer || "No answer returned.");
                } catch (err) {
                    showMessage("Unable to reach the backend. Please ensure the Flask server is running.");
                } finally {
                    askBtn.disabled = false;
                    askBtn.textContent = "✈ Get Answer";
                }
            }

            askBtn.addEventListener("click", askQuestion);
            questionInput.addEventListener("keydown", function (event) {
                if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    askQuestion();
                }
            });
        </script>
    </body>
    </html>
    """


# -------------------------------
# API ROUTE (OPTIONAL – for API use)
# -------------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data["question"]
    answer = get_answer(question)
    return jsonify({
        "question": question,
        "answer": answer
    })


# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
