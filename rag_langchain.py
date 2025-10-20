import os, json
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found!")


TEXTS_PATH = Path("data") / "texts.json"
if not TEXTS_PATH.exists():
    print("data/texts.json not found — creating empty list until upload.")
    texts = []
else:
    with open(TEXTS_PATH, "r", encoding="utf-8") as f:
        texts = json.load(f)


embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_texts(texts, embedding=embeddings) if texts else None



llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) if vectorstore else None
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff") if retriever else None


def ask_question(question: str, top_k: int = 3):
    """
    Return answer + top-k relevant chunks with similarity scores.
    Agent suggestion: if avg_score > 0.5, recommend web search.
    """
    if not texts:
        return {"answer": "No documents uploaded!", "results": [], "used_web": False}

    try:
        answer = qa_chain.run(question) if qa_chain else "Not enough documents for RAG."
        docs_with_scores = vectorstore.similarity_search_with_score(question, k=top_k) if vectorstore else []
        
        results = [{"chunk": d.page_content, "score": float(s)} for d, s in docs_with_scores]

        avg_score = sum(r["score"] for r in results)/len(results) if results else 1.0
        used_web = False
        if avg_score > 0.5:
            answer = "Information not found in uploaded documents."
            used_web = True

    except Exception as e:
        answer = f"Error: {e}"
        results = []
        used_web = False

    return {"answer": answer, "results": results, "used_web": used_web}
