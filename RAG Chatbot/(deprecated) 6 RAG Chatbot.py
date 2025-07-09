# Databricks notebook source
# DBTITLE 1,Install Dependencies
# MAGIC %pip install mlflow==2.10.1 langchain==0.1.5 databricks-vectorsearch==0.22 databricks-sdk==0.18.0 mlflow[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Set needed parameters
import os

host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ['DATABRICKS_TOKEN'] = ""

index_name="bluetab.rag.idx"
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

VECTOR_SEARCH_ENDPOINT_NAME="doc_vector_endpoint"

# COMMAND ----------

# DBTITLE 1,Build Retriever
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings

embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="text", embedding=embedding_model
    )
    return vectorstore.as_retriever()



# COMMAND ----------

# DBTITLE 1,Create the RAG Langchain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks

chat_model = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", max_tokens = 200)

# TEMPLATE = """You are an assistant for home appliance users. You are answering how to, maintenance and troubleshooting questions regarding the appliances you have data on. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. If the question appears to be for an appliance you don't have data on, say so.  Keep the answer as concise as possible.  Provide all answers only in English.
# Use the following pieces of context to answer the question at the end:
# {context}
# Question: {question}
# Answer:
# """
TEMPLATE = """You are an internal assistant chatbot for Bluetab employees. You answer questions related to Bluetab’s company information, products, technologies, employee guidelines, and internal processes. If the question is outside of these topics, politely decline to answer. If you don't know the answer, clearly state that you don't know and avoid inventing answers. If the question is about products or services not related to Bluetab, say so. Keep your answers clear and concise. Provide all answers only in Spanish.

Use the following pieces of context to answer the question at the end:
{context}
Pregunta: {question}
Respuesta:
"""

prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)



# COMMAND ----------

# DBTITLE 1,Test Langchain
question = {"query": "dime los productos de bluetab?"}
answer = chain.run(question)
print(answer)

# COMMAND ----------

# DBTITLE 1,Register our Chain as a model to Unity Catalog
from mlflow.models import infer_signature
import mlflow
import langchain

mlflow.set_registry_uri("databricks-uc")
model_name = "llm.rag_bluetab.appliance_chatbot_model_bluetab"

my_conda_env = {
    "name": "my_env",
    "channels": ["defaults"],  # usa solo el canal defaults, no conda-forge
    "dependencies": [
        "python=3.10.12",
        "pip<=22.3.1",
        {
            "pip": [
                "mlflow==" + mlflow.__version__,
                "langchain==" + langchain.__version__,
                "databricks-vectorsearch",
            ]
        }
    ]
}

with mlflow.start_run(run_name="appliance_chatbot_bluetab_run") as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        input_example=question,
        signature=signature,
        pip_requirements="requirements.txt",
    )

# COMMAND ----------

import mlflow.pyfunc

class CustomModel(mlflow.pyfunc.PythonModel):
    def __init__(self, chain):
        self.chain = chain

    def predict(self, context, model_input):
        # Aquí defines cómo usar tu chain para predecir,
        # model_input es el input que recibe predict
        return self.chain.run(model_input)

with mlflow.start_run(run_name="appliance_chatbot_bluetab_run") as run:
    signature = infer_signature(question, answer)
    
    # Guardas tu modelo pyfunc personalizado
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=CustomModel(chain),
        registered_model_name=model_name,
        input_example=question,
        signature=signature,
        pip_requirements="requirements.txt"  # para evitar conda
    )

# COMMAND ----------


