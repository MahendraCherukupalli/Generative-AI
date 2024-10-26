import os
import re
import dotenv


from langchain_openai import OpenAI
from langchain.llms import HuggingFaceHub

from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from pypdf import PdfReader

# from langchain_openai.chat_models import ChatOpenAI
# from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

import gradio as gr


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
model_id = "tiiuae/falcon-7b-instruct"

llm_hub = HuggingFaceHub(repo_id=model_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 300, "max_length": 600})

def summarize_pdf(chunk_size, chunk_overlap, pdf_file=None, pdf_file_path=None):
    llm=llm_hub
    
    docs_raw = []  
    
    if pdf_file_path :
        loader = PyPDFLoader(pdf_file_path)
        docs_raw = loader.load()
    elif pdf_file:
        reader = PdfReader(pdf_file)
        i = 1
        for page in reader.pages:
            docs_raw.append(Document(page_content=page.extract_text(), metadata={'page': i}))
            i += 1
    else:
        raise ValueError("Either pdf_file_path or pdf_file must be provided.")
    
    docs_raw_text = {doc.page_content for doc in docs_raw}
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size,chunk_overlap = chunk_overlap)
    docs_chunks = text_splitter.create_documents(docs_raw_text)
    
    chain = load_summarize_chain(llm,chain_type='stuff')
    
    summary = chain.invoke(docs_chunks[:4],return_only_outputs = True)
    return summary['output_text'].split('CONCISE SUMMARY:')[1].strip()


iface = gr.Interface(
    fn=summarize_pdf, 
    inputs=[
        gr.Number(label="Chunk Size",value = 1000),
        gr.Number(label="Chunk Overlap",value = 20),
        gr.File(label="Upload PDF File", type="filepath"),
        gr.Textbox(label="PDF File Path", placeholder="Enter file path here")
    ], 
    outputs="text",
    title="Text Summarization",
    description="<div style='text-align: center;'>This is a simple web app for Text Summarization.<br><br>Upload a file or provide a file path.</div>"
)

iface.launch()