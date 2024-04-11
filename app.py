import streamlit as st

from PyPDF2 import PdfReader
from langchain_community.llms import Ollama

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain


st.set_page_config('preguntaDOC')

# App title
st.header("Pregunta a tu PDF")

# loader PDF file
pdf_file = st.file_uploader("Carga tu documento", type="pdf",
on_change=st.cache_resource.clear)

# create embeddings with the PDF file 
@st.cache_resource
def create_embeddings(pdf):

    # read the PDF file
    pdf_reader = PdfReader(pdf)
    text = ""

    # extract the text from the PDF
    for page in pdf_reader.pages:
        text += page.extract_text()

    # split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
        )

    # join text chunks
    join_chunks = text_splitter.split_text(text)

    # call embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # create the vectorial data base with the embeddings
    vectorial_data_base = FAISS.from_texts(join_chunks, embeddings)

    return vectorial_data_base

if pdf_file:
    # create the vectorial_data_base with the PDF file
    knowledge_data_base = create_embeddings(pdf_file)

    # show text to user in UI
    user_question = st.text_input("Haz una pregunta sobre tu PDF:")

    if user_question:
        # search the context for the question
        context = knowledge_data_base.similarity_search(user_question, 3)

        # call LLAMA2 model
        llm = Ollama(model="llama2", base_url='http://localhost:11434')

        # create the chain with the model
        chain = load_qa_chain(llm, chain_type="stuff")

        # run the chain with the context and the question
        answer = chain.run(input_documents=context, question=user_question)

        st.write(answer)
