import operator
import uuid
from typing import Annotated, TypedDict

from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import HTMLSectionSplitter, RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph


class InputState(TypedDict):
    """Input state - only what the user needs to provide"""

    urls: list[str]


class RAGPipelineState(TypedDict):
    """Complete internal state for the RAG processing pipeline"""

    # Input (from InputState) - immutable after initialization
    urls: list[str]
    current_url_index: int

    # Raw data - written by single nodes (overwritten each URL)
    raw_html_docs: list[Document]
    text_docs: list[Document]

    # Chunks for this URL only - overwritten each iteration (no accumulation)
    coarse_chunks_separate: list[Document]
    granular_chunks_separate: list[Document]

    # For ParentDocumentRetriever approach
    parent_retriever_docs: list[Document]

    # For MultiVectorRetriever with granular->coarse (no accumulation)
    mvr_coarse_chunks: list[Document]
    mvr_coarse_chunk_ids: list[str]
    mvr_granular_chunks: list[Document]

    # For MultiVectorRetriever with summaries (no accumulation)
    summary_coarse_chunks: list[Document]
    summary_chunk_ids: list[str]
    summaries: list[Document]

    # Collections and retrievers - set once during initialization
    granular_collection: Chroma
    coarse_collection: Chroma
    parent_doc_retriever: ParentDocumentRetriever
    multi_vector_retriever_granular: MultiVectorRetriever
    multi_vector_retriever_summaries: MultiVectorRetriever

    # Error handling - last write wins
    error: str


class OutputState(TypedDict):
    """Output state - only what gets returned to the user"""

    urls_processed: int
    granular_collection: Chroma
    coarse_collection: Chroma
    parent_doc_retriever: ParentDocumentRetriever
    multi_vector_retriever_granular: MultiVectorRetriever
    multi_vector_retriever_summaries: MultiVectorRetriever
    error: str


def load_html(state: RAGPipelineState) -> dict:
    """Load HTML from current URL"""
    try:
        current_url = state["urls"][state["current_url_index"]]
        print(
            f"\nProcessing URL {state['current_url_index'] + 1}/{len(state['urls'])}: {current_url}"
        )

        html_loader = AsyncHtmlLoader(current_url)
        raw_html_docs = html_loader.load()

        return {"raw_html_docs": raw_html_docs, "error": None}
    except Exception as e:
        return {"error": f"HTML loading failed: {str(e)}"}


class TransformToText:
    def __init__(self):
        self.html2text_transformer = Html2TextTransformer()

    def transform_to_text(self, state):
        """Transform HTML to text"""
        try:
            text_docs = self.html2text_transformer.transform_documents(
                state["raw_html_docs"]
            )

            return {"text_docs": text_docs, "error": None}
        except Exception as e:
            return {"error": f"HTML transformation failed: {str(e)}"}


# def transform_to_text(state: RAGPipelineState) -> dict:
#     """Transform HTML to text"""
#     try:
#         text_docs = html2text_transformer.transform_documents(state["raw_html_docs"])

#         return {"text_docs": text_docs, "error": None}
#     except Exception as e:
#         return {"error": f"HTML transformation failed: {str(e)}"}


def create_separate_coarse_chunks(state: RAGPipelineState) -> dict:
    """Create coarse chunks for separate collection approach"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000, chunk_overlap=300
        )
        coarse_chunks = text_splitter.split_documents(state["text_docs"])

        return {
            "coarse_chunks_separate": coarse_chunks,
        }
    except Exception as e:
        return {"error": f"Separate coarse chunking failed: {str(e)}"}


def create_separate_granular_chunks(state: RAGPipelineState) -> dict:
    """Create granular chunks using HTML section splitter"""
    try:
        headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2")]
        html_section_splitter = HTMLSectionSplitter(
            headers_to_split_on=headers_to_split_on
        )

        all_chunks = []
        for doc in state["raw_html_docs"]:
            html_string = doc.page_content
            temp_chunks = html_section_splitter.split_text(html_string)
            all_chunks.extend(temp_chunks)

        return {
            "granular_chunks_separate": all_chunks,
        }
    except Exception as e:
        return {"error": f"Separate granular chunking failed: {str(e)}"}


def add_to_parent_retriever(state: RAGPipelineState) -> dict:
    """Add documents to ParentDocumentRetriever"""
    try:
        state["parent_doc_retriever"].add_documents(state["text_docs"], ids=None)

        return {}  # No state updates needed
    except Exception as e:
        return {"error": f"ParentDocumentRetriever failed: {str(e)}"}


def create_mvr_chunks(state: RAGPipelineState) -> dict:
    """Create coarse and granular chunks for MultiVectorRetriever (granular approach)"""
    try:
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=500)

        coarse_chunks = parent_splitter.split_documents(state["text_docs"])
        coarse_chunks_ids = [str(uuid.uuid4()) for _ in coarse_chunks]

        all_granular_chunks = []
        for i, coarse_chunk in enumerate(coarse_chunks):
            coarse_chunk_id = coarse_chunks_ids[i]

            granular_chunks = child_splitter.split_documents([coarse_chunk])

            for granular_chunk in granular_chunks:
                granular_chunk.metadata["doc_id"] = coarse_chunk_id

            all_granular_chunks.extend(granular_chunks)

        return {
            "mvr_coarse_chunks": coarse_chunks,
            "mvr_coarse_chunk_ids": coarse_chunks_ids,
            "mvr_granular_chunks": all_granular_chunks,
        }
    except Exception as e:
        return {"error": f"MVR chunking failed: {str(e)}"}


def generate_summaries(state: RAGPipelineState) -> dict:
    """Generate summaries for MultiVectorRetriever (summary approach)"""
    try:
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)
        coarse_chunks = parent_splitter.split_documents(state["text_docs"])

        llm = ChatOllama(model="gemma2:2b", temperature=0)
        summarization_chain = (
            {"document": lambda x: x.page_content}
            | ChatPromptTemplate.from_template(
                "Summarize the following document concisely in 2-3 sentences:\n\n{document}"
            )
            | llm
            | StrOutputParser()
        )

        coarse_chunks_ids = [str(uuid.uuid4()) for _ in coarse_chunks]
        all_summaries = []

        for i, coarse_chunk in enumerate(coarse_chunks):
            coarse_chunk_id = coarse_chunks_ids[i]

            summary_text = summarization_chain.invoke(coarse_chunk)
            summary_doc = Document(
                page_content=summary_text,
                metadata={**coarse_chunk.metadata, "doc_id": coarse_chunk_id},
            )

            all_summaries.append(summary_doc)

        return {
            "summary_coarse_chunks": coarse_chunks,
            "summary_chunk_ids": coarse_chunks_ids,
            "summaries": all_summaries,
        }
    except Exception as e:
        return {"error": f"Summary generation failed: {str(e)}"}


def barrier_before_store(state: RAGPipelineState) -> dict:
    """Barrier node to ensure all parallel branches complete before storing"""
    print(
        f"  → All parallel processing complete for URL {state['current_url_index'] + 1}"
    )
    return {}


def store_all_embeddings(state: RAGPipelineState) -> dict:
    """Store all embeddings in their respective collections"""
    try:
        print(f"  → Storing embeddings for URL {state['current_url_index'] + 1}")

        # 1. Store in separate collections
        if state.get("coarse_chunks_separate"):
            state["coarse_collection"].add_documents(state["coarse_chunks_separate"])
            print(f"    ✓ Stored {len(state['coarse_chunks_separate'])} coarse chunks")

        if state.get("granular_chunks_separate"):
            state["granular_collection"].add_documents(
                state["granular_chunks_separate"]
            )
            print(
                f"    ✓ Stored {len(state['granular_chunks_separate'])} granular chunks"
            )

        # 2. MultiVectorRetriever with granular chunks
        if state.get("mvr_granular_chunks") and state.get("mvr_coarse_chunks"):
            state["multi_vector_retriever_granular"].vectorstore.add_documents(
                state["mvr_granular_chunks"]
            )
            state["multi_vector_retriever_granular"].docstore.mset(
                list(zip(state["mvr_coarse_chunk_ids"], state["mvr_coarse_chunks"]))
            )
            print(
                f"    ✓ Stored {len(state['mvr_granular_chunks'])} MVR granular chunks"
            )

        # 3. MultiVectorRetriever with summaries
        if state.get("summaries") and state.get("summary_coarse_chunks"):
            state["multi_vector_retriever_summaries"].vectorstore.add_documents(
                state["summaries"]
            )
            state["multi_vector_retriever_summaries"].docstore.mset(
                list(zip(state["summary_chunk_ids"], state["summary_coarse_chunks"]))
            )
            print(f"    ✓ Stored {len(state['summaries'])} summaries")

        # Increment the URL index
        new_index = state["current_url_index"] + 1
        print(f"  → Incrementing index: {state['current_url_index']} → {new_index}")

        return {
            "current_url_index": new_index,
            "urls_processed": new_index,
            "error": None,
        }
    except Exception as e:
        print(f"  ✗ Storage error: {str(e)}")
        return {"error": f"Storage failed: {str(e)}"}


def check_more_urls(state: RAGPipelineState) -> str:
    """Check if there are more URLs to process"""
    if state.get("error"):
        print(f"  ✗ Error detected: {state['error']}")
        return "error"

    current_index = state["current_url_index"]
    total_urls = len(state["urls"])

    print(f"\n→ Checking progress: URL {current_index}/{total_urls}")

    if current_index < total_urls:
        print("  → Continuing to next URL\n")
        return "continue"

    print("  ✓ All URLs processed!\n")
    return "done"
