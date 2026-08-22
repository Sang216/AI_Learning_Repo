# PART-1 - Load the PDF & index it once

# pip install langchain langchain-openai langchain-chroma langchain-huggingface \
#             langchain-community pypdf sentence-transformers gradio python-dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def build_index(urls):
    # ① Read multiple web pages
    loader = WebBaseLoader(urls)
    pages = loader.load()

    # ② Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(pages)

    # ③ Create embeddings
    embedder = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # ④ Store in Chroma
    db = Chroma.from_documents(
        chunks,
        embedder
    )

    return db


# PART-2 - Answer with retrieval + citations

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are a helpful PDF assistant. Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I couldn't find that in the document."
After your answer, list the page numbers you used as: Sources: page X, page Y.

Context:
{context}

Question: {question}
""")

def ask(db, question):
    chunks  = db.similarity_search(question, k=4)
    context = "\n\n".join(
        f"[{c.metadata.get('source', 'Unknown')}] {c.page_content}" for c in chunks)
    chain = prompt | model
    return chain.invoke({"context": context, "question": question}).content


# PART-3 - Give it a face with Gradio

import gradio as gr

state = {"db": None}                    # ① remember the index across turns

def load_urls(url_text):
    # Parse URLs (comma-separated or newline-separated)
    urls = [url.strip() for url in url_text.replace(',', '\n').split('\n') if url.strip()]
    if not urls:
        return "❌ Please enter at least one URL"
    try:
        state["db"] = build_index(urls)   # ② re-index with new URLs
        return f"✅ Loaded {len(urls)} URL(s)! Ask me anything about in AI sessions."
    except Exception as e:
        return f"❌ Error loading URLs: {str(e)}"

def chat(message, history):
    if state["db"] is None:
        return "Please load URLs first 🔗"
    return ask(state["db"], message)

with gr.Blocks(title="🔗 Chat with Web Pages of 4 AI Session Contents") as demo:
    gr.Markdown("## 🔗 Chat with Web Pages of 4 AI Session Contents (powered by RAG)")
    urls   = gr.Textbox(label="Enter URLs (comma or newline separated)", lines=3, placeholder="https://example.com\nhttps://example.com/page2")
    status = gr.Markdown()
    load_btn = gr.Button("Load URLs")
    load_btn.click(load_urls, inputs=urls, outputs=status)
    gr.ChatInterface(fn=chat)

demo.launch(share=True)                   # share=True → public link!