from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os
import time
# Load environment variables (for API keys, etc.)
load_dotenv()


#Final Project
class ChatBot:
    def __init__(self):
        # Load your Pinecone and Hugging Face API keys from .env
        print("Pinecone API Key:", os.getenv('PINECONE_API_KEY'))
        print("Hugging Face API Key:", os.getenv('HUGGINGFACE_API_KEY'))
        
        # Part 1: Scraping and preparing data
        print("Starting data scraping and preparation...")
        self.scrape_data()

        # Part 2: Initialize Pinecone and LLM chain
        print("Initializing LLM chain...")
        self.rag_chain = self.initialize_chain()

    def scrape_data(self):
        url = "https://www.parkinson.org/understanding-parkinsons/statistics"
        print(f"Fetching data from {url}")
        response = requests.get(url)
        if response.status_code == 200:
            print("Successfully retrieved the webpage")
            soup = BeautifulSoup(response.text, 'html.parser')
            statistics_data = "" 
            for paragraph in soup.find_all(['p', 'h2', 'h3']):
                text = paragraph.get_text().strip()
                if text:
                    statistics_data += text + "\n"

            print(f"Debug - Scraped data length: {len(statistics_data)} characters")
            print("Debug - First 500 characters of scraped data:", statistics_data[:500])

            # Modify text splitting parameters
            text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
            docs = text_splitter.split_text(statistics_data)
            
            print(f"Debug - Number of chunks after splitting: {len(docs)}")
            print("Debug - First chunk:", docs[0] if docs else "No chunks created")

            self.embeddings = HuggingFaceEmbeddings()
            print("Created HuggingFaceEmbeddings instance")

            # Initialize Pinecone
            print("Initializing Pinecone...")
            pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment='gcp-starter')
            index_name = "langchain-demo"

            # Force recreation of the index
            if index_name in pinecone.list_indexes():
                print(f"Debug - Deleting existing Pinecone index: {index_name}")
                pinecone.delete_index(index_name)

            print(f"Debug - Creating new Pinecone index: {index_name}")
            pinecone.create_index(name=index_name, metric="cosine", dimension=768)
            print("Pinecone index created. Adding documents...")
            
            # Add documents to the index
            self.docsearch = Pinecone.from_texts([doc for doc in docs], self.embeddings, index_name=index_name)
            print(f"Debug - Added {len(docs)} documents to Pinecone index")
            
            print("Debug - Pinecone index stats:", pinecone.describe_index(index_name))
            
            # Test query to ensure the index is working
            print("Testing Pinecone index with a sample query...")
            test_query = "Parkinson's disease statistics"
            test_results = self.docsearch.similarity_search(test_query, k=1)
            print(f"Test query results: {test_results}")
            
        else:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")

    # ... (rest of the code remains the same)
    def initialize_chain(self):
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        llm = HuggingFaceEndpoint(
            repo_id=repo_id, 
            task="text-generation",
            top_k=50,  
            temperature=0.7,
            max_new_tokens=500,
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

        template = """
        You are a helpful assistant specialized in Parkinson's disease information. Use the following context to answer the question. If you don't know the answer or can't find relevant information in the context, just say you don't have enough information to answer accurately.

        Context: {context}
        
        Question: {question}
        
        Answer:
        """

        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        # Chain construction
        chain = prompt | llm  # Passes directly from prompt to LLM without additional complexity
        return chain


    def generate_response(self, question):
        try:
            # Get relevant documents using the question
            context_docs = self.docsearch.similarity_search(question, k=3)
            print(f"Debug - Retrieved docs: {context_docs}")

            if not context_docs:
                return "I'm sorry, but I couldn't find any relevant information to answer your question about Parkinson's disease. Could you try rephrasing your question or asking about a different aspect of Parkinson's?"

            # Extract the page_content from each Document object and join into a single string
            context = " ".join([doc.page_content for doc in context_docs])
            print(f"Debug - Constructed context: {context[:500]}...")  # Print first 500 chars for brevity

            # Prepare input for the PromptTemplate
            input_data = {"context": context, "question": question}

            # Generate response using RAG chain
            response = self.rag_chain.invoke(input_data)

            # Process response and return the result
            if isinstance(response, str):
                return response
            elif isinstance(response, dict) and 'result' in response:
                return response['result']
            else:
                return f"Unexpected response format: {type(response)}. Please try rephrasing your question."

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}. Type of context_docs: {type(context_docs)}"
            print(f"Debug - Error in generate_response: {error_msg}")
            return "I apologize, but I encountered an error while processing your question. Could you please try asking again or rephrasing your question?"


# Terminal-based chatbot interface
def run_terminal_chatbot():
    print("Initializing ChatBot...")
    bot = ChatBot()
    print("ChatBot initialization complete. Waiting for 5 seconds to ensure all processes are complete...")
    time.sleep(5)  # Add a 5-second delay
    print("Welcome to the Parkinson's Disease Information Bot!")
    print("Type 'exit' at any time to stop the conversation.")
    
    while True:
        question = input("\nYou: ")
        if question.lower() == 'exit':
            print("Goodbye!")
            break

        response = bot.generate_response(question)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    run_terminal_chatbot()
