import os
import shutil
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain.schema.document import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.load import dumps, loads



load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = 'true'

os.environ["LANGCHAIN_ENDPOINT"] = 'https://api.smith.langchain.com'

os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')

os.environ["LANGCHAIN_PROJECT"] = "LAW"

os.environ["GEMINI_API_KEY"] = os.getenv('GEMINI_API_KEY')


embeddings = GoogleGenerativeAIEmbeddings(model= "models/text-embedding-004", google_api_key=os.getenv("GEMINI_API_KEY"))
# LLM
llm = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0)

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def create_retriever():
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    return retriever


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)



#########################################
### RAG FUSION
#########################################

generate_queries_template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""

# hyde_fusion_template = """You are an AI assistant specialized in information retrieval and synthesis. \n
# Given a user query, generate 4 hypothetical, highly relevant document snippets that would ideally answer the query. \n
# User query: {question} \n
# """

prompt_rag_fusion = ChatPromptTemplate.from_template(generate_queries_template)
# prompt_rag_fusion = ChatPromptTemplate.from_template(hyde_fusion_template)


def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal Rank Fusion that takes multiple lists of ranked documents
        and an optional parameter k used in the RRF formula. """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate over each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its  rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the documet to a string format to use as  a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with a score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any 
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1/(rank * k)
            fused_scores[doc_str] += 1 / (rank + k)
        
    # Sort the documents by their fused scores in descending order to get the final ranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


################################################



################################################################
########## FINAL CHAIN
################################################################

final_template = """Answer the question based only on the following context
{context}

Question: {question}
"""

final_prompt =  ChatPromptTemplate.from_template(final_template)


#######################################################################



global_stream_response = None

def main():
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    retriever = create_retriever()

    generate_queries = (
        prompt_rag_fusion
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion

    final_rag_chain = (
        {
            "context": retrieval_chain_rag_fusion,
            "question": RunnablePassthrough()
        }
        | final_prompt
        | llm
        | StrOutputParser()
    )

    def stream_response(conversation):
        # prompt = ChatPromptTemplate.from_template(template)
        # chain = prompt | llm | StrOutputParser()

        for chunk in final_rag_chain.stream({"question": conversation}):
            yield chunk

    global global_stream_response
    global_stream_response = stream_response

main()