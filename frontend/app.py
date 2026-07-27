import os
import streamlit as st
import requests

st.set_page_config(page_title="Hybrid RAG Pipeline", page_icon="🔍", layout="wide")

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("🔍 Hybrid RAG Pipeline")
st.markdown("A portfolio-grade Hybrid Retrieval-Augmented Generation system using Dense + Sparse vectors, RRF, and Cross-Encoder reranking.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for document ingestion
with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=['pdf', 'txt'])
    
    if st.button("Ingest Document"):
        if uploaded_file is not None:
            with st.spinner("Ingesting and processing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "application/octet-stream")}
                    response = requests.post(f"{API_URL}/ingest", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Success! {data['chunks_processed']} chunks processed.")
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
        else:
            st.warning("Please upload a file first.")
            
    st.divider()
    st.header("⚙️ Search Settings")
    top_k = st.slider("Top-K Context Chunks", min_value=1, max_value=10, value=5)
    use_hybrid = st.checkbox("Enable Hybrid Search (Dense + Sparse)", value=True)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display citations if available
        if "citations" in message and message["citations"]:
            with st.expander("📚 View Sources & Citations"):
                for idx, cite in enumerate(message["citations"]):
                    st.markdown(f"**Source {idx+1}** (Confidence Score: {cite.get('score', 0):.4f})")
                    st.markdown(f"*{cite.get('metadata', {}).get('source', 'Unknown')}*")
                    st.info(cite.get("content", ""))

# React to user input
if prompt := st.chat_input("Ask a question based on the ingested documents..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call API to get response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Searching and generating response..."):
            try:
                payload = {
                    "query": prompt,
                    "top_k": top_k,
                    "use_hybrid": use_hybrid
                }
                
                # We use a standard POST request here. In a true production app, we might stream this.
                response = requests.post(f"{API_URL}/query", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer generated.")
                    citations = data.get("citations", [])
                    
                    message_placeholder.markdown(answer)
                    
                    if citations:
                        with st.expander("📚 View Sources & Citations"):
                            for idx, cite in enumerate(citations):
                                st.markdown(f"**Source {idx+1}** (Confidence Score: {cite.get('score', 0):.4f})")
                                st.markdown(f"*{cite.get('metadata', {}).get('source', 'Unknown')}*")
                                st.info(cite.get("content", ""))
                                
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "citations": citations
                    })
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
