# Streamlit application for Question Answering (QA) using RAG with Pinecone.
#
# Flow:
# 1. User submits a question.
# 2. We retrieve the most relevant chunks from Pinecone.
# 3. We send the question + retrieved context to an OpenAI model.
# 4. We display only the final answer (in Spanish).
#
# Notes:
# - The embeddings and index are pre-built with build_knowledge_base.py.
# - This script is only for retrieval + generation at query time.

import streamlit as st
from pathlib import Path
import toml
import traceback

from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# Load credentials and configuration
@st.cache_resource
def load_secrets():
    """
    Load configuration values from .streamlit/secrets.toml.
    We assume .streamlit/ is at the project root, not inside src/.
    """
    root_dir = Path(__file__).resolve().parent.parent
    secrets_path = root_dir / ".streamlit" / "secrets.toml"
    secrets = toml.load(secrets_path)

    cfg = {
        "OPENAI_API_KEY": secrets["OPENAI_API_KEY"],
        "PINECONE_API_KEY": secrets["PINECONE_API_KEY"],
        "INDEX_NAME": secrets["INDEX_NAME"],
        "NAMESPACE": secrets["NAMESPACE"],
        "EMBEDDING_MODEL": secrets["EMBEDDING_MODEL"],
        "OPENAI_MODEL": secrets["OPENAI_MODEL"],
        "K_RETRIEVE": int(secrets.get("K_RETRIEVE", 8)),
        "TEMPERATURE": float(secrets.get("TEMPERATURE", 0.0)),
    }
    return cfg


# Build retriever from Pinecone index
@st.cache_resource
def build_retriever(cfg: dict):
    """
    Initialize the Pinecone retriever.

    Steps:
    - Connect to Pinecone using API key.
    - Recreate the same embedding function we used during indexing.
    - Wrap the Pinecone index in a VectorStore.
    - Expose a Retriever interface (similarity search with top-k results).
    """
    pc = Pinecone(api_key=cfg["PINECONE_API_KEY"])
    index = pc.Index(cfg["INDEX_NAME"])

    embeddings = HuggingFaceEmbeddings(model_name=cfg["EMBEDDING_MODEL"])

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=cfg["NAMESPACE"],
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": cfg["K_RETRIEVE"]},
    )

    return retriever


# Build the RetrievalQA chain (RAG)
def build_qa_chain(cfg: dict, retriever):
    """
    Build the RetrievalQA chain:
    - Use a prompt that forces the model to answer ONLY using context.
    - Use ChatOpenAI as the generator.
    - Return a RetrievalQA chain that we can call with {"query": "..."}.
    """

    prompt = PromptTemplate.from_template(
        """
        Eres un asistente experto en la documentación proporcionada.
        Usa EXCLUSIVAMENTE la información del contexto para construir tu respuesta.
        Si la respuesta exacta no está, puedes:
        - resumir o parafrasear el contexto
        - inferir detalles obvios que se desprenden directamente del contexto

        Si de verdad no hay nada relacionado, responde:
        "No encuentro esa información en los documentos".

        Pregunta del usuario: {question}
        Contexto relevante: {context}

        Respuesta en español, breve y precisa:
        """.strip()
    )

    llm = ChatOpenAI(
        model=cfg["OPENAI_MODEL"],
        openai_api_key=cfg["OPENAI_API_KEY"],
        temperature=cfg["TEMPERATURE"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )

    return qa_chain


# Streamlit UI
def main():
    # Page config
    st.set_page_config(
        page_title="Knowledge Chatbot",
        page_icon="💬",
        layout="centered",
    )

    # App header
    st.title("Knowledge Chatbot")
    st.caption("Haz una pregunta sobre los documentos cargados.")

    # Sidebar with simple usage instructions
    with st.sidebar:
        st.header("ℹ️ Cómo usar")
        st.write(
            "- Escribe tu pregunta sobre la documentación interna.\n"
            "- Presiona Enter.\n"
            "- La respuesta usa solo el contenido cargado en la base de conocimiento."
        )
        st.divider()
        st.write("Si no se encuentra la respuesta se dara un mensaje acorde.")

    # Load config, retriever, and chain once
    try:
        cfg = load_secrets()
        retriever = build_retriever(cfg)
        qa_chain = build_qa_chain(cfg, retriever)
    except Exception as e:
        st.error("No se pudo inicializar el motor de búsqueda / LLM.")
        st.code(str(e))
        st.stop()

    # Form block
    with st.form(key="qa_form", clear_on_submit=False):
        user_question = st.text_input(
            "Tu pregunta:",
            placeholder="Ejemplo: ¿De donde proviene el nombre de la ciudad Chiclayo?",
        )
        submitted = st.form_submit_button("Preguntar")

    # Only run inference if the user actually submitted
    if submitted:
        if not user_question.strip():
            st.warning("Por favor ingresa una pregunta válida.")
        else:
            try:
                with st.spinner("Buscando en la base de conocimiento..."):
                    result = qa_chain.invoke({"query": user_question})

                # RetrievalQA with return_source_documents=False returns just "result"
                answer = result["result"] if isinstance(result, dict) else result

                # Show only the final answer, no headers/icons/sources
                st.write(answer)

            except Exception as e:
                st.error("Ocurrió un error al procesar tu pregunta.")
                st.exception(e)
                st.text("Traceback:")
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
