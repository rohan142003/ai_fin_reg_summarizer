import streamlit as st
from transformers import pipeline
from bs4 import BeautifulSoup
import requests
import pdfplumber
import tempfile
from docx import Document
from sentence_transformers import SentenceTransformer, util
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from newsapi import NewsApiClient

# ------------------------------
# CONFIG / KEYS
# ------------------------------
NEWS_API_KEY = "1cd0a79be8274af49361298d6347b7db"  # <<< UPDATE THIS
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# ------------------------------
# MODELS & TOPICS
# ------------------------------
TOPICS = [
    "banking", "insurance", "stock market", "fintech",
    "regulation", "macroeconomy", "payments",
    "cryptocurrency", "other"
]

@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return summarizer, classifier, embedder

summarizer, classifier, embedder = load_models()

# ------------------------------
# DATABASE
# ------------------------------
def init_db():
    conn = sqlite3.connect("results.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    topic TEXT,
                    confidence REAL,
                    summary TEXT,
                    insight TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )''')
    conn.commit()
    conn.close()

def save_to_db(title, topic, confidence, summary, insight):
    conn = sqlite3.connect("results.db")
    c = conn.cursor()
    c.execute("INSERT INTO analysis (title, topic, confidence, summary, insight) VALUES (?, ?, ?, ?, ?)",
              (title, topic, confidence, summary, insight))
    conn.commit()
    conn.close()

init_db()

# ------------------------------
# TEXT EXTRACTOR
# ------------------------------
def extract_text_from_url(url):
    try:
        headers={"User-Agent":"Mozilla/5.0"}
        res=requests.get(url,headers=headers,timeout=10)
        soup=BeautifulSoup(res.text,"html.parser")
        article=soup.find("article")
        if article:
            return article.get_text(separator="\n").strip()
        else:
            return "\n".join(p.get_text() for p in soup.find_all("p")).strip()
    except Exception as e:
        st.error(f"❌ URL Error: {e}")
        return ""

# ------------------------------
# EVIDENCE (RAG-style)
# ------------------------------
def extract_evidence_sentences(article_text, query, k=3):
    sentences=[s.strip() for s in article_text.split('.') if len(s.split())>6]
    if not sentences:
        return []
    sent_emb=embedder.encode(sentences,convert_to_tensor=True)
    query_emb=embedder.encode(query,convert_to_tensor=True)
    hits=util.semantic_search(query_emb,sent_emb,top_k=k)[0]
    return [(sentences[h['corpus_id']],float(h['score'])) for h in hits]


# ------------------------------
# SIDEBAR NAV
# ------------------------------
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to:", ["Analyze Article", "Fetch News", "Dashboard"])


# ==========================================================
# PAGE 1 — ANALYZE ARTICLE
# ==========================================================
if choice == "Analyze Article":
    st.title("📑 Analyze Financial Article")

    url = st.text_input("🌐 Enter Article URL")
    uploaded = st.file_uploader("📎 Upload Document (PDF, DOCX, TXT)", ["pdf", "docx", "txt"])
    text_input = st.text_area("📝 Paste Article Text", height=180)

    article_text = ""

    if url:
        article_text = extract_text_from_url(url)
    elif uploaded:
        if uploaded.type=="application/pdf":
            with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp:
                tmp.write(uploaded.getbuffer()); tmp.flush()
                with pdfplumber.open(tmp.name) as pdf:
                    article_text="\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        elif uploaded.type=="application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(uploaded)
            article_text = "\n".join(p.text for p in doc.paragraphs)
        else:
            article_text = uploaded.getvalue().decode("utf-8")
    elif text_input:
        article_text = text_input

    if article_text:
        st.success("✅ Text extracted!")
        st.write(article_text[:2000])

        if st.button("🔍 Run Full AI Analysis"):
            with st.spinner("Analyzing..."):
                summary=summarizer(article_text[:4000],max_length=200,min_length=60,do_sample=False)[0]["summary_text"]
                result=classifier(article_text[:1200],TOPICS)
                topic,conf=result["labels"][0],result["scores"][0]
                evidence=extract_evidence_sentences(article_text,summary,3)

            st.subheader("📌 Summary")
            st.write(summary)

            st.subheader("🏷️ Topic")
            st.write(f"**{topic.upper()}** ({conf:.2f})")

            st.subheader("📎 Evidence Sentences")
            if evidence:
                for sent,score in evidence:
                    st.markdown(f"- {sent} _(score {score:.2f})_")

            insight=f"Article relates to **{topic}**. Key point: {summary[:200]}..."
            st.subheader("💡 Insight")
            st.write(insight)

            save_to_db(url if url else "Uploaded File",topic,conf,summary,insight)
            st.success("✅ Saved to database")

    else:
        st.info("Enter URL / upload file / paste text.")


# ==========================================================
# PAGE 2 — FETCH LATEST FINANCIAL NEWS
# ==========================================================
elif choice == "Fetch News":
    st.title("📰 Latest Business & Finance News")

    if st.button("Fetch Business Headlines"):
        with st.spinner("Fetching..."):
            articles = newsapi.get_top_headlines(
                category='business', language='en', country='in'
            )["articles"]

        if not articles:
            st.warning("No news found.")
        else:
            for art in articles[:5]:
                st.write(f"### {art['title']}")
                st.write(art["description"] or "")
                st.markdown(f"[Read More]({art['url']})")
                st.write("---")


# ==========================================================
# PAGE 3 — DASHBOARD
# ==========================================================
else:  # Dashboard
    st.title("📊 Insights Dashboard")

    conn=sqlite3.connect("results.db")
    df=pd.read_sql("SELECT * FROM analysis",conn)
    conn.close()

    if df.empty:
        st.info("No analysis records yet.")
    else:
        st.dataframe(df)

        if st.button("📥 Export CSV"):
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "financial_insights.csv", "text/csv")

        topic_counts=df["topic"].value_counts()
        fig,ax=plt.subplots()
        topic_counts.plot(kind="bar",ax=ax)
        ax.set_title("Topic Frequency")
        ax.set_xlabel("Topic"); ax.set_ylabel("Count")
        st.pyplot(fig)
