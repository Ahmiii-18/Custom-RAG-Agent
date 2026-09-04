import os
import tempfile
import base64

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------------------
# Page setup
# ----------------------------------------
st.set_page_config(page_title="Multimodal AI RAG Agent", page_icon="🔮", layout="wide")
st.title("🔮 Multimodal AI RAG Agent")
st.caption("Upload text, PDFs, Images, Audio, or Videos — get contextual answers with source tracking.")

# ----------------------------------------
# Sidebar: API key + settings
# ----------------------------------------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password",
                             value=os.getenv("OPENAI_API_KEY", ""))
    # Restrict to models that natively support multimodal inputs
    model_name = st.selectbox("Multimodal Model", ["gpt-4o", "gpt-4o-mini"])
    chunk_size = st.slider("Text Chunk size", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)
    top_k = st.slider("Retrieved chunks (k)", 2, 10, 4)

if not api_key:
    st.info("Enter your OpenAI API key in the sidebar to begin.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# ----------------------------------------
# Helper Functions for File Handling
# ----------------------------------------
def process_text_and_pdf(file_bytes: bytes, file_name: str, size: int, overlap: int):
    """Processes text-based PDFs or raw text files into vector chunks."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if file_name.lower().endswith('.pdf'):
            docs = PyPDFLoader(tmp_path).load()
        else:
            # Handle plain text file
            text_content = file_bytes.decode("utf-8", errors="ignore")
            from langchain_core.documents import Document
            docs = [Document(page_content=text_content, metadata={"source": file_name, "page": 1})]
    finally:
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)

def extract_media_insights(file_bytes: bytes, file_name: str) -> str:
    """Uses OpenAI's native audio transcriptions or vision to document media files."""
    suffix = os.path.splitext(file_name)[1].lower()
    client = ChatOpenAI(model="gpt-4o", temperature=0)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # --- Handle Images ---
        if suffix in ['.png', '.jpg', '.jpeg', '.webp']:
            base64_image = base64.b64encode(file_bytes).decode('utf-8')
            msg = client.invoke([
                {"role": "user", "content": [
                    {"type": "text", "text": "Provide an incredibly detailed structural, textual, and context breakdown of this image for a search database index."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ])
            return f"[Image Source: {file_name}] {msg.content}"

        # --- Handle Audio / Video ---
        elif suffix in ['.mp3', '.wav', '.m4a', '.mp4', '.mpeg', '.avi']:
            from openai import OpenAI
            native_client = OpenAI()
            
            # If it's a video, we extract audio or route directly to whisper if supported
            # Whisper handles audio files up to 25MB natively
            with open(tmp_path, "rb") as audio_file:
                transcript = native_client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    response_format="text"
                )
            
            media_type = "Video Transcript" if suffix in ['.mp4', '.mpeg', '.avi'] else "Audio Transcript"
            return f"[{media_type}: {file_name}] {transcript}"
            
    except Exception as e:
        return f"Error extracting info from {file_name}: {str(e)}"
    finally:
        os.unlink(tmp_path)
    return ""

# ----------------------------------------
# Multimodal File Upload Interface
# ----------------------------------------
uploaded_files = st.file_uploader(
    "Upload Documents or Media Assets", 
    type=["pdf", "txt", "png", "jpg", "jpeg", "mp3", "wav", "mp4"],
    accept_multiple_files=True
)

if uploaded_files:
    # Build unique tracking fingerprint for the file pool
    pool_sig = [(f.name, f.size) for f in uploaded_files] + [chunk_size, chunk_overlap]
    
    if st.session_state.get("pool_sig") != pool_sig:
        all_chunks = []
        
        with st.spinner("Processing multi-format asset pool..."):
            for f in uploaded_files:
                name_lower = f.name.lower()
                file_bytes = f.getvalue()
                
                if name_lower.endswith(('.pdf', '.txt')):
                    chunks = process_text_and_pdf(file_bytes, f.name, chunk_size, chunk_overlap)
                    all_chunks.extend(chunks)
                else:
                    # Parse image, video audio context into text embedding space
                    media_description = extract_media_insights(file_bytes, f.name)
                    from langchain_core.documents import Document
                    all_chunks.append(Document(page_content=media_description, metadata={"source": f.name, "page": "Media System Target"}))
            
            if all_chunks:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                st.session_state.vectorstore = FAISS.from_documents(all_chunks, embeddings)
                st.session_state.pool_sig = pool_sig
                st.session_state.messages = []
                st.success("All multimodal contexts successfully aligned and indexed!")

# ----------------------------------------
# Unified Multimodal RAG Chain
# ----------------------------------------
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an advanced multimodal AI assistant that answers questions strictly using the "
     "provided file system contexts (which include transcripts, image analysis data, and text chunks).\n"
     "Rules:\n"
     "1. Answer ONLY using facts directly stated or visible within the context below.\n"
     "2. If the answer cannot be confidently verified by the context, say: "
     "\"I couldn't find that explicit information across your uploaded media.\"\n"
     "3. Always cite the exact source asset file name and structural detail location, e.g. (Source: presentation.mp4) or (Source: manual.pdf, p. 3).\n\n"
     "Context Data Pool:\n{context}"),
    ("human", "{question}"),
])

def format_docs(docs) -> str:
    return "\n\n".join(
        f"[Source Asset: {d.metadata.get('source', 'Unknown')} | Location Reference: {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )

def get_chain(vectorstore: FAISS, model: str, k: int):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model=model, temperature=0)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    ), retriever

# ----------------------------------------
# Chat interface
# ----------------------------------------
if "vectorstore" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask about text, images, visual scripts, or media transcripts...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        chain, retriever = get_chain(st.session_state.vectorstore, model_name, top_k)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing media context ecosystem..."):
                answer = chain.invoke(question)
                st.markdown(answer)

                with st.expander("🔍 Multimodal Sources (retrieved nodes)"):
                    for doc in retriever.invoke(question):
                        source = doc.metadata.get('source', 'Unknown')
                        loc = doc.metadata.get('page', '?')
                        st.markdown(f"**Asset:** `{source}` (Ref: {loc})")
                        st.text(doc.page_content[:600])
                        st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👆 Upload text, layout PDFs, audio, or video files to explore cross-media RAG.")
