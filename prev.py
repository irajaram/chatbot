import requests # to make HTTP request to fetch info on web
from bs4 import BeautifulSoup #python package for parsing HTML
from langchain.text_splitter import CharacterTextSplitter #splitting text into chunkss
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone #vector database
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os


class Chatbot():
    self.url = "https://www.parkinson.org/understanding-parkinsons/statistics"
    self.response = requests.get(url) #GET request sent to url to get info

# Check if the request was successful
    if self.response.status_code == 200:
    #print("Request successful. Status code:", response.status_code)
    
    # Step 2: Parse the HTML content
        self.soup = BeautifulSoup(response.text, 'html.parser') #using Beautiful soup to parse HTML content
        
        
        #print("\nParsed HTML snippet:\n", soup.prettify()[:500])  # Print first 500 characters to check if code worked
        
        # Step 3: Extract the text data
        self.statistics_data = "" #empty string to store extracted data
        
        for paragraph in self.soup.find_all(['p', 'h2', 'h3']):
            text = paragraph.get_text().strip() #get rid of white space
            if text:  # Filter out empty text
                self.statistics_data += text + "\n" #only add to string if actual text
        
        # Optional: Print the extracted text (to check if the right content is being extracted)
        #print("\nExtracted Text Snippet:\n", statistics_data[:500])  # Print first 500 characters
        
        # Step 4: Process the extracted data
        # Splitting the text into manageable chunks
        self.text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=4) #initializes requirements such as chunk size and overlap
        self.docs = text_splitter.split_text(self.statistics_data) #chunks the text according to requirements in the previous line

        # Print the chunks (to check splitting)
        #print("\nFirst Text Chunk:\n", docs[0] if docs else "No text chunks found")

        # Embedding the text
        #embeddings = HuggingFaceEmbeddings()
        #embeddings = HuggingFaceEmbeddings(model_name="bert-base-uncased")
        self.embeddings = HuggingFaceEmbeddings()
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
        self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)
    else:
  # Link to the existing index
        self.docsearch = Pinecone.from_existing_index(self.index_name, self.embeddings)


# Define the repo ID and connect to Mixtral model on Huggingface -> actul preexisting LLM
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    self.llm = HuggingFaceHub(
        repo_id=repo_id, 
        model_kwargs={"temperature": 0.8, "top_k": 50}, 
        huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
    )

    self.template = """
    You are a researcher. These Human will ask you a questions about Parkinson's disease. 
    Use following piece of context to answer the question. 
    If you don't know the answer, just say you don't know. 
    Keep the answer within 2 sentences and concise.

    Context: {context}
    Question: {question}
    Answer: 

    """

    self.prompt = PromptTemplate(
        template=self.template, 
        input_variables=["context", "question"]
    )


    self.rag_chain = (
        {"context": self.docsearch.as_retriever(),  "question": RunnablePassthrough()} #docsearch pulls relevant documents for contenxt; RunnablePassthrough makes sure query is unchanged 
        | self.prompt  # refines and modifies query
        | self.llm #actual model
        | StrOutputParser() #model response turned into text
    )

    def get_response(self, question):
        return self.rag_chain.run(question=question)

if __name__ == "__main__":
    bot = ChatBot()
    response = bot.get_response("What are the symptoms of Parkinson's disease?")
    print(response)
#complete later