import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain 

app = FastAPI()

#api keys (read from .env file or environment variables)
load_dotenv()
google_api_key = os.environ["GOOGLE_API_KEY"]          # still needed for embeddings
openrouter_api_key = os.environ["OPENROUTER_API_KEY"]  # used for the LLM
openrouter_model = os.getenv("OPENROUTER_MODEL", "inclusionai/ling-3.2-flash:free")

#embedding (must stay Gemini - the vector db was built with it)

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    api_key=google_api_key
)

#load vector db + retriever
# built per request so re-running ingest.py never leaves the server holding a stale collection
def get_retriever():
    db = Chroma(
        persist_directory="vector db",
        embedding_function=embedding
    )
    return db.as_retriever(search_kwargs={"k":5})

#LLM
llm = ChatOpenAI(
    model=openrouter_model,
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_retries=3,
)

#prompt 
prompt = ChatPromptTemplate.from_template("""
You are a helpful AI Assistant. Provide brief answer to the provided question.
Answer only from the context. If you don't find answer in the context then reply i dont know .

Context : 

{context}

Question : {input}

Answer : 


""")
#rag chain 
document_chain= create_stuff_documents_chain(
    llm,
    prompt
)

#request_format
class Question(BaseModel):
    question:str


#home route 
@app.get ("/")
def home():
    return{
        "message":"100 page so ml API is running"
    }

#predict rag route 
@app.post("/ask")
def ask_question(data:Question):
    try:
        retrieval_chain = create_retrieval_chain(get_retriever(), document_chain)
        response = retrieval_chain.invoke({
            "input":data.question
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")

    return{
        "question":data.question,
        "answer":response['answer']
    }
