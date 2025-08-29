import streamlit as st
import logging

# Support both package and standalone execution
try:
    from .utils import MemoryManager
    from .qa_engine import QAEngine
    from .runtime_docs import handle_uploaded_documents
except Exception:
    from utils import MemoryManager
    from qa_engine import QAEngine
    from runtime_docs import handle_uploaded_documents

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


st.set_page_config(
    page_title="Document Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Reuse the styling philosophy from root app
st.markdown("""
<style>
h1 { text-align: center !important; color: #667eea !important; font-size: 2.2rem !important; font-weight: 700 !important; margin: 1rem 0 1.5rem 0 !important; }
.stTextInput > div > div > input { background-color: #2D2D2D !important; border: 1px solid #363636 !important; color: white !important; border-radius: 10px !important; padding: 0.75rem 1rem !important; height: auto !important; font-size: 1rem !important; min-height: 45px !important; }
.stButton > button { background-color: #4a9eff !important; color: white !important; border: none !important; padding: 0.75rem 1.25rem !important; border-radius: 8px !important; font-weight: 600 !important; }
.chat-message-container { margin-bottom: 1.2rem; display: flex; width: 100%; }
.user-message-bubble { background-color: #2D2D2D; padding: 1rem 1.25rem; border-radius: 12px; color: white; max-width: 75%; border-left: 4px solid #2E7BF6; margin-left: auto; }
.assistant-message-bubble { background-color: #363636; padding: 1rem 1.25rem; border-radius: 12px; color: white; max-width: 75%; border-left: 4px solid #4CAF50; margin-right: auto; }
.message-source { font-size: 0.8rem; color: #888; margin-top: 0.5rem; font-style: italic; padding-top: 0.4rem; border-top: 1px solid #404040; }
</style>
""", unsafe_allow_html=True)


if 'memory' not in st.session_state:
    st.session_state.memory = MemoryManager()
if 'qa' not in st.session_state:
    st.session_state.qa = QAEngine(st.session_state.memory)
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'processing_docs' not in st.session_state:
    st.session_state.processing_docs = False
if 'files_to_process' not in st.session_state:
    st.session_state.files_to_process = []
if 'docs_added' not in st.session_state:
    st.session_state.docs_added = False
if 'uploader_version' not in st.session_state:
    st.session_state.uploader_version = 0
if 'last_uploaded_fingerprints' not in st.session_state:
    st.session_state.last_uploaded_fingerprints = []
if 'last_uploaded_names' not in st.session_state:
    st.session_state.last_uploaded_names = []
if 'last_upload_message' not in st.session_state:
    st.session_state.last_upload_message = ""


def clear_chat():
    st.session_state.messages = []
    if "question_input" in st.session_state:
        st.session_state.question_input = ""


st.title("Document Extractor")

st.markdown('<div class="upload-header"><h3>Upload Documents (PDF/DOCX)</h3></div>', unsafe_allow_html=True)

has_data = st.session_state.memory.get_status().get('has_data', False)

# Always show the uploader; auto-upload on selection (no extra button)
uploaded_files = st.file_uploader(
    "Upload PDF or DOCX files (max 30MB each)",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    key=f"file_uploader_{st.session_state.uploader_version}",
    label_visibility="collapsed",
    disabled=st.session_state.processing_docs,
)

if uploaded_files and not st.session_state.processing_docs:
    # Create a fingerprint of the selection to avoid duplicate processing
    fingerprints = []
    for uf in uploaded_files:
        try:
            pos = uf.tell()
            uf.seek(0, 2)
            size = uf.tell()
            uf.seek(pos)
        except Exception:
            size = 0
        fingerprints.append(f"{uf.name}:{size}")
    if fingerprints and fingerprints != st.session_state.last_uploaded_fingerprints:
        st.session_state.files_to_process = uploaded_files
        st.session_state.processing_docs = True
        st.session_state.last_uploaded_fingerprints = fingerprints

if st.session_state.processing_docs:
    st.info("Uploading and indexing documents. Please wait...")
    files = st.session_state.get('files_to_process', [])
    try:
        ok, skipped = handle_uploaded_documents(files, st.session_state.memory)
        if ok:
            names_list = [getattr(f, 'name', 'document') for f in files]
            names = ", ".join(names_list)
            msg = f"Uploaded: {names}. Indexed successfully. Search is now enabled."
            st.success(msg)
            st.session_state.last_uploaded_names = names_list
            st.session_state.last_upload_message = msg
            st.session_state.docs_added = True
            st.session_state.qa = QAEngine(st.session_state.memory)
        else:
            st.warning("No valid documents were uploaded.")
        if skipped:
            st.warning(f"Skipped files: {', '.join(skipped)}")
    except Exception as e:
        st.error(f"Error processing documents: {e}")
    st.session_state.processing_docs = False
    st.session_state.files_to_process = []
    # Reset uploader so the same selection doesn't auto-trigger again
    st.session_state.uploader_version += 1
    st.rerun()


# Persistently show last upload message (filenames + success)
if st.session_state.last_upload_message:
    st.success(st.session_state.last_upload_message)

# Display chat history only if we have any messages
if st.session_state.messages:
    for message in st.session_state.messages:
        role_class = "user-message-bubble" if message["role"] == "user" else "assistant-message-bubble"
        # Render assistant as HTML if it's valid HTML; else as plain text
        content = message["content"]
        if message["role"] == "assistant" and isinstance(content, str) and content.strip().startswith("<"):
            html = f'<div class="chat-message-container"><div class="{role_class}">{content}</div></div>'
        else:
            html = f'<div class="chat-message-container"><div class="{role_class}">{content}</div></div>'
        st.markdown(html, unsafe_allow_html=True)


if not has_data:
    st.info("Upload at least one document to enable search.")
else:
    with st.form(key="chat_form", clear_on_submit=True):
        cols = st.columns([0.8, 0.2])
        with cols[0]:
            user_query = st.text_input(
                "Type your question here...",
                placeholder="Type your question here...",
                label_visibility="collapsed",
                key="question_input",
                disabled=st.session_state.processing_docs,
            )
        with cols[1]:
            send_clicked = st.form_submit_button("Send", use_container_width=True, disabled=st.session_state.processing_docs)

        if send_clicked and user_query.strip():
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.info("Searching documents and generating an answer...")
            resp = st.session_state.qa.answer(user_query)
            # Prepare assistant message, prefer HTML fragment from LLM
            assistant_text = resp['answer']
            # If answer came from docs and at least one source exists, append a single document reference line
            if resp.get('source') == 'document_rag' and resp.get('sources'):
                if isinstance(assistant_text, str) and assistant_text.strip().startswith("<"):
                    assistant_text += f"<p><strong>Document:</strong> {resp['sources'][0]}</p>"
                else:
                    assistant_text = assistant_text + f"\n\nDocument: {resp['sources'][0]}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_text,
                "source": resp.get('source', ''),
            })
            st.rerun()


controls = st.columns([1,1])
with controls[0]:
    st.button("🗑️ Clear Chat", on_click=clear_chat, use_container_width=True, key="clear_chat_button")
with controls[1]:
    if st.button("🧹 Clear Uploaded Index", use_container_width=True, key="clear_index_button"):
        st.session_state.memory.clear_all()
        st.session_state.qa = QAEngine(st.session_state.memory)
        st.session_state.messages = []
        st.session_state.docs_added = False
        st.session_state.last_uploaded_names = []
        st.session_state.last_upload_message = ""
        st.session_state.last_uploaded_fingerprints = []
        st.session_state.processing_docs = False
        st.session_state.files_to_process = []
        # bump uploader key so previous selection is cleared
        st.session_state.uploader_version += 1
        st.success("Index cleared. Upload documents to start again.")
        st.rerun()


