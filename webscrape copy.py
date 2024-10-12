"""
Utilizing RAG (retrieval and generation)-> feeds information to LLM
- RAG-> 2 parts -> storing data in indexed format, retrieval acts as search engine finding specific info related to user query, generates more specified response
- Process: intial data split using text splitter into chunks, then embedded chunks, and index is then stored in vector database(Pinecone),
model picks specific embedded chunk related to the user's query
Hugging Face: provides LLM
Langchain: implements model

Goal: Webscrape data from a website to have actual source rather than a plain text file to draw from
Problem: Package Installation in external env 
Solution: Install virtual environment

Problem: Missing module
Solution: Install langchain

Problem: Attribute Error: 'Runnable Sequence'
Solution: LangChain uses 'Runnable Sequence' to process data by retrieving, prompt processing and generating response;
-> You must call it like a function and Must pass input as a dictionary with keys matching inut variable defined in prompt template: context,question
"""

from dotenv import load_dotenv  # Ensure this is imported
import requests # to make HTTP request to fetch info on web
from bs4 import BeautifulSoup #python package for parsing HTML
from langchain.text_splitter import CharacterTextSplitter #splitting text into chunkss
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
import pinecone #vector database
#from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnableSequence

from langchain.schema.output_parser import StrOutputParser
import os


load_dotenv()  # Ensure this is at the top

# Check if keys are loaded
print("Pinecone API Key:", os.getenv('PINECONE_API_KEY'))
print("Hugging Face API Key:", os.getenv('HUGGINGFACE_API_KEY'))

#Part 1: Getting the Data
# Step 1: Fetch the webpage content
url = "https://www.parkinson.org/understanding-parkinsons/statistics"
response = requests.get(url) #GET request sent to url to get info

# Check if the request was successful
if response.status_code == 200:
    #print("Request successful. Status code:", response.status_code)
    
    # Step 2: Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser') #using Beautiful soup to parse HTML content
    
    
    #print("\nParsed HTML snippet:\n", soup.prettify()[:500])  # Print first 500 characters to check if code worked
    
    # Step 3: Extract the text data
    statistics_data = "" #empty string to store extracted data
    
    for paragraph in soup.find_all(['p', 'h2', 'h3']):
        text = paragraph.get_text().strip() #get rid of white space
        if text:  # Filter out empty text
            statistics_data += text + "\n" #only add to string if actual text
    
    # Optional: Print the extracted text (to check if the right content is being extracted)
    #print("\nExtracted Text Snippet:\n", statistics_data[:500])  # Print first 500 characters
    
    # Step 4: Process the extracted data
    # Splitting the text into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=4) #initializes requirements such as chunk size and overlap
    docs = text_splitter.split_text(statistics_data) #chunks the text according to requirements in the previous line

    # Print the chunks (to check splitting)
    #print("\nFirst Text Chunk:\n", docs[0] if docs else "No text chunks found")

    # Embedding the text
    #embeddings = HuggingFaceEmbeddings()
    #embeddings = HuggingFaceEmbeddings(model_name="bert-base-uncased")
    embeddings = HuggingFaceEmbeddings()
    # Print number of chunks to embed
    #print(f"\nNumber of chunks to embed: {len(docs)}")
    
    # Save chunks to a file
    # with open("statistics_chunks.txt", "w") as f:
    #     for chunk in docs:
    #         f.write(chunk + "\n")
    
else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")

#Part 2: Initialize Pinecone and index data
#Intialize Pinecone client
pinecone.init(
    api_key= os.getenv('PINECONE_API_KEY'),
    environment = 'gcp-starter'
)
#define Index Name
index_name = "langchain-demo"

# Checking Index
if index_name not in pinecone.list_indexes():
  # Create new Index
  pinecone.create_index(name=index_name, metric="cosine", dimension=768)
  docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
else:
  # Link to the existing index
  docsearch = Pinecone.from_existing_index(index_name, embeddings)


# Define the repo ID and connect to Mixtral model on Huggingface -> actul preexisting LLM
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceEndpoint(
  repo_id=repo_id, 
  top_k=50,  # Specify top_k directly
  temperature=0.8,
  #model_kwargs={"temperature": 0.8, "top_k": 50}, 
  huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
)

template = """
You are a researcher. These Human will ask you a questions about Parkinson's disease. 
Use following piece of context to answer the question. 
If you don't know the answer, just say you don't know. 
Keep the answer within 2 sentences and concise.

Context: {context}
Question: {question}
Answer: 

"""

prompt = PromptTemplate(
  template=template, 
  input_variables=["context", "question"]
)


# rag_chain = (
#   {"context": docsearch.as_retriever(),  "question": RunnablePassthrough()} #docsearch pulls relevant documents for contenxt; RunnablePassthrough makes sure query is unchanged 
#   | prompt  # refines and modifies query
#   | llm #actual model
#   | StrOutputParser() #model response turned into text
# )

# Create a RunnableSequence
rag_chain = RunnableSequence(
    steps=[
        docsearch.as_retriever(),  # Retrieves context based on the query
        prompt,                    # Applies the prompt template
        llm,                       # Calls the Hugging Face model
        StrOutputParser()          # Parses the output to a string format
    ]
)


# if __name__ == "__main__":
#     while True:
#         question = input("Ask me anything about Parkinson's disease (type 'exit' to stop): ")
#         if question.lower() == 'exit':
#             break
#         response = rag_chain({"context": "", "question": question})# Use the RAG chain to get a response
#         print("Response:", response)

# Adjust the way you handle input in the main function
if __name__ == "__main__":
    while True:
        question = input("Ask me anything about Parkinson's disease (type 'exit' to stop): ")
        if question.lower() == 'exit':
            break
        # Use the RunnableSequence to process the input and get a response
        response = rag_chain.invoke({"context": "", "question": question})  # 'invoke' runs the chain
        print("Response:", response)