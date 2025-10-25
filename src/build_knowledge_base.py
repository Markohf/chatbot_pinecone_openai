# Goal:
# - Load source documents
# - Split them into chunks
# - Generate embeddings
# - Upsert them into Pinecone
# Run this script ONLY to (re)build Pinecone. Streamlit will not call this.

from pathlib import Path
import hashlib
from typing import Dict

import toml
from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.utils.logger_config import setup_logger

logger = setup_logger(__name__, "build_knowledge_base.log")


def load_secrets() -> Dict[str, str]:
    """
    Load credentials and configuration values from .streamlit/secrets.toml.
    """
    root_dir = Path(__file__).resolve().parent.parent
    secrets_path = root_dir / ".streamlit" / "secrets.toml"
    secrets = toml.load(secrets_path)

    return {
        "OPENAI_API_KEY": secrets["OPENAI_API_KEY"],
        "PINECONE_API_KEY": secrets["PINECONE_API_KEY"],
        "INDEX_NAME": secrets["INDEX_NAME"],
        "NAMESPACE": secrets["NAMESPACE"],
        "EMBEDDING_MODEL": secrets["EMBEDDING_MODEL"],
    }


def load_raw_docs() -> list[Document]:
    """
    Load all text-based documents from the 'data_rag/' directory and return
    a list of LangChain Document objects.

    Supported formats:
    - .txt
    - .md
    - .csv
    - .pdf

    PDF -> one Document per page
    others -> one Document per file
    """
    data_dir = Path(__file__).resolve().parent.parent / "data_rag"
    docs: list[Document] = []

    supported_exts = [".txt", ".md", ".csv", ".pdf"]

    for file_path in data_dir.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in supported_exts:
            logger.warning(f"⚠️ Skipped unsupported file: {file_path.name}")
            continue

        try:
            if file_path.suffix.lower() == ".pdf":
                reader = PdfReader(file_path)
                for i, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    if text and text.strip():
                        docs.append(
                            Document(
                                page_content=text.strip(),
                                metadata={
                                    "filename": file_path.name,
                                    "page": i,
                                },
                            )
                        )
            else:
                text = file_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
                if text and text.strip():
                    docs.append(
                        Document(
                            page_content=text.strip(),
                            metadata={
                                "filename": file_path.name,
                                "page": 1,
                            },
                        )
                    )

        except Exception as e:
            logger.error(f"❌ Failed to read {file_path.name}: {e}")

    if docs:
        all_files = list(data_dir.glob("*"))
        logger.info(
            f"✅ Loaded {len(docs)} documents from data_rag/ "
            f"({len(all_files)} files total)"
        )
    else:
        logger.warning("⚠️ No valid text-based documents found in data_rag/")

    return docs


def chunk_docs(docs: list[Document]) -> list[Document]:
    """
    Split long documents into overlapping text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    logger.info(
        f"✅ Generated {len(chunks)} chunks from {len(docs)} documents"
    )

    return chunks


def stable_id_for_chunk(chunk: Document) -> str:
    """
    Deterministic ID for each chunk based on content + metadata.
    Used so we don't insert duplicates across runs.
    """
    base_str = (
        chunk.page_content
        + "|"
        + str(chunk.metadata.get("filename", ""))
        + "|"
        + str(chunk.metadata.get("page", ""))
    )

    return hashlib.sha256(base_str.encode("utf-8")).hexdigest()


def ensure_pinecone_index(pc: Pinecone, index_name: str, dimension: int) -> None:
    """
    Make sure the Pinecone index exists.
    If it does not exist, create it (serverless).
    """
    existing = [idx["name"] for idx in pc.list_indexes()]
    if index_name in existing:
        logger.info(f"ℹ️ Pinecone index '{index_name}' already exists.")
        return

    logger.info(
        f"Creating Pinecone index '{index_name}' "
        f"(dimension={dimension}, metric='cosine', serverless us-east-1)..."
    )

    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

    logger.info(f"✅ Pinecone index '{index_name}' created.")


def upsert_chunks_to_pinecone(
    chunks: list[Document],
    cfg: Dict[str, str],
) -> None:
    """
    Generate embeddings for each chunk and upsert them into Pinecone.
    """
    try:
        # Pinecone client
        pc = Pinecone(api_key=cfg["PINECONE_API_KEY"])

        # Embedding model
        embedding_model_name = cfg["EMBEDDING_MODEL"]
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Get embedding dimension
        test_vec = embeddings.embed_query("dimension probe")
        dim = len(test_vec)

        # Ensure index exists (create if 404 scenario)
        ensure_pinecone_index(
            pc=pc,
            index_name=cfg["INDEX_NAME"],
            dimension=dim,
        )

        # Connect to the Pinecone index
        index = pc.Index(cfg["INDEX_NAME"])

        # Wrap in a LangChain vector store
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            namespace=cfg["NAMESPACE"],
        )

        # Deterministic IDs
        ids = [stable_id_for_chunk(ch) for ch in chunks]

        # Upsert
        vectorstore.add_documents(documents=chunks, ids=ids)

        logger.info(
            f"✅ Upserted {len(chunks)} chunks into Pinecone "
            f"(index='{cfg['INDEX_NAME']}', namespace='{cfg['NAMESPACE']}')"
        )

    except Exception as e:
        logger.error(f"❌ Error while uploading chunks to Pinecone: {e}")
        raise


def main() -> None:
    cfg = load_secrets()

    raw_docs = load_raw_docs()
    if not raw_docs:
        logger.warning("⚠️ No documents to process. Exiting.")
        return

    chunks = chunk_docs(raw_docs)
    upsert_chunks_to_pinecone(chunks, cfg)

    logger.info("✅ Knowledge base build completed successfully")


if __name__ == "__main__":
    main()
