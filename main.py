import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import os



class ChatBot:
    def __init__(self):
        load_dotenv()
        # loader = TextLoader()
        # documents = loader.load()
        # Step 1: Fetch and process web page content
        self.url = "https://www.parkinson.org/understanding-parkinsons/statistics"
        self.response = requests.get(self.url)

        if self.response.status_code == 200:
            print("Request successful. Status code:", self.response.status_code)
            
            # Parse the HTML content
            self.soup = BeautifulSoup(self.response.text, 'html.parser')
            
            # Extract the text data
            self.statistics_data = ""
            for paragraph in self.soup.find_all(['p', 'h2', 'h3']):
                text = paragraph.get_text().strip()
                if text:
                    self.statistics_data += text + "\n"
            
            # Split the text into chunks
            self.text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=4)
            self.docs = self.text_splitter.split_text(self.statistics_data)

            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings()

            # Step 2: Initialize Pinecone and index data
            print("Pinecone API Key:", os.getenv('PINECONE_API_KEY'))
            pinecone.init(
                api_key=os.getenv('PINECONE_API_KEY'),
                environment='gcp-starter'
            )
            
            self.index_name = "langchain-demo"
            
            if self.index_name not in pinecone.list_indexes():
                pinecone.create_index(name=self.index_name, metric="cosine", dimension=768)
                self.docsearch = Pinecone.from_documents(self.docs, self.embeddings, index_name=self.index_name)
            else:
                self.docsearch = Pinecone.from_existing_index(self.index_name, self.embeddings)

            # Define the repo ID and connect to Mixtral model on Huggingface
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
                {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            print(f"Failed to retrieve the page. Status code: {self.response.status_code}")

    # Define a method to handle user queries
    def get_response(self, question):
        return self.rag_chain.run(question=question)

# Example usage
if __name__ == "__main__":
    bot = ChatBot()
    input = input("Ask me anything: ")
    response = bot.get_response(input)
    print(response)
