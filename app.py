from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline, HuggingFaceHub
from transformers import pipeline
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()


index_name = "medicalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = OpenAI(temperature=0.4, max_tokens=500)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


#@app.route("/get", methods=["GET", "POST"])
#def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    #response = rag_chain.invoke({"input": msg})
    #print("Response : ", response["answer"])
   # return str(response["answer"])
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    try:
        # 1) Try your existing OpenAI-backed chain first (unchanged)
        response = rag_chain.invoke({"input": msg})
        print("Response : ", response["answer"])
        return str(response["answer"])

    except Exception as e:
        # 2) On any OpenAI error (e.g., insufficient_quota), fall back to Hugging Face
        print("OpenAI failed, switching to Hugging Face fallback. Error:", e)

        # Prefer Hugging Face Inference API if a token is available (free; no big downloads)
        hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN")
        if hf_token:
            # Remote model via Hugging Face Hub (good balance of quality & speed)
            hf_llm = HuggingFaceHub(
                repo_id="google/flan-t5-large",
                model_kwargs={"temperature": 0.3, "max_length": 512}
            )
        else:
            # Last-resort local model (fully free/offline; smaller to avoid huge downloads)
            gen = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                max_new_tokens=256
            )
            hf_llm = HuggingFacePipeline(pipeline=gen)

        # Rebuild the chain with HF LLM (retriever & prompt stay the same)
        hf_question_answer_chain = create_stuff_documents_chain(hf_llm, prompt)
        hf_rag_chain = create_retrieval_chain(retriever, hf_question_answer_chain)

        response = hf_rag_chain.invoke({"input": msg})
        print("HF Response : ", response["answer"])
        return str(response["answer"])




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
