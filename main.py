import os
from langchain.llms import LlamaCpp
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

db = Chroma(embedding_function=embedding_function)
def add_documents(loader, db):
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators= ["\n\n", "\n", ".", ";", ",", " ", ""])
    texts = text_splitter.split_documents(documents)
    db.add_documents(texts)

loader = TextLoader("./data/cards.csv")
add_documents(loader, db)

pdf_names = ['./data/doc1.pdf', './data/doc2.pdf']
for pdf_name in pdf_names:
    loader = UnstructuredPDFLoader(pdf_name)
    add_documents(loader, db)


template = """Твоя роль - это консультант Тинькофф банка,
Твоя задача - это отвечать на вопросы клиентов, на основе базы данных,
Избегай придумывания ответов,
Предостовляй только релевантную информацию на основе базы данных(контекста),
Не задавай клиенту вопросов и не веди с ним диалог, только отвечай на вопрос

Передаю базу данных и вопрос клиента:

База данных: {context}

Вопрос клиента: {question}

Ответ:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
    temperature=0.75,
    max_tokens=5000,
    top_p=1,
    f16_kv=True,
    callback_manager=callback_manager, 
    verbose=True, # Verbose is required to pass to the callback manager
    n_ctx=4048,
)
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm,
                                chain_type="stuff",
                                retriever=retriever,
                                chain_type_kwargs={"prompt": prompt})


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class Input(BaseModel):
    human_input: str

class Output(BaseModel):
    output: str

app=FastAPI()

@app.post("/llm")
async def input(input: Input):
    output = Output(output=qa.run(input.human_input))
    return output



