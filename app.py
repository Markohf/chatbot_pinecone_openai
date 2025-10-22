import os
import hashlib
import streamlit as st

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Sidebar: Config
st.set_page_config(page_title="Chatbot RAG", page_icon="💬", layout="wide")
st.title("💬 Chatbot RAG (Pinecone + LangChain + OpenAI)")

with st.sidebar:
    st.header("⚙️ Configuración")
    # Lee desde secrets (preferido) o variables de entorno
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
    PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")

    INDEX_NAME = st.secrets.get("INDEX_NAME")
    NAMESPACE = st.secrets.get("NAMESPACE")
    MODEL_NAME = st.secrets.get("EMBEDDING_MODEL")

    pinecone_cloud = st.secrets.get("PINECONE_CLOUD")
    pinecone_region = st.secrets.get("PINECONE_REGION")

    k = st.slider("k (documentos recuperados)", 3, 8, 4)
    use_mmr = st.checkbox("Usar MMR (más diversidad)", value=True)
    temperature = st.slider("Creatividad (temperature)", 0.0, 1.0, 0.0, 0.1)

    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        st.error("Falta configurar OPENAI_API_KEY y/o PINECONE_API_KEY en `secrets.toml`.")
        st.stop()

# Cache de recursos
@st.cache_resource(show_spinner=True)
def load_embeddings(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name)

@st.cache_resource(show_spinner=True)
def ensure_pinecone_index(api_key: str, index_name: str, cloud: str, region: str, dimension: int):
    pc = Pinecone(api_key=api_key)
    existing = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
    return pc

def get_vectorstore(index_name: str, namespace: str, embeddings, api_key: str):
    return PineconeVectorStore(
        index_name=index_name,
        namespace=namespace,
        embedding=embeddings,
        pinecone_api_key=api_key
    )

def get_llm(openai_api_key: str, temperature: float = 0.0):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

# Prompt Template
RAG_PROMPT = PromptTemplate.from_template(
    """
    Eres un asistente experto en la documentación proporcionada relacionada al 
    Perú y Chiclayo. Responde SOLO con la información del contexto. Si la 
    respuesta no está en el contexto, di: 
    "No encuentro esa información en los documentos".
    Pregunta: {question}
    Contexto: {context}
    Respuesta (breve y precisa, en español):
"""
)

# Inicialización de recursos
embeddings = load_embeddings(MODEL_NAME)
dimension = embeddings.client.get_sentence_embedding_dimension()
ensure_pinecone_index(PINECONE_API_KEY, INDEX_NAME, pinecone_cloud, pinecone_region, dimension)
vectorstore = get_vectorstore(INDEX_NAME, NAMESPACE, embeddings, PINECONE_API_KEY)
llm = get_llm(OPENAI_API_KEY, temperature=temperature)

retriever = vectorstore.as_retriever(
    search_type="mmr" if use_mmr else "similarity",
    search_kwargs={"k": k, "fetch_k": max(12, k * 3), "lambda_mult": 0.5} if use_mmr else {"k": k}
)

# Conversational chain (memoria de historial)
@st.cache_resource(show_spinner=True)
def get_conv_chain(llm, retriever):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": RAG_PROMPT},
    )

conv_chain = get_conv_chain(llm, retriever)

# Estado de conversación
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role":"user"/"assistant", "content":"..."}]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # lista simple de pares (user_text, ai_text)

# Render del historial
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Entrada del usuario
user_input = st.chat_input("Escribe tu pregunta...")
if user_input:
    # Muestra el mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Llama a la chain
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            res = conv_chain.invoke({"question": user_input, "chat_history": st.session_state.chat_history})
            answer = res.get("answer", "").strip()
            sources = res.get("source_documents", [])

            # Construye un pie de fuentes
            if sources:
                foot = "\n\n**Fuentes:**\n" + "\n".join(
                    f"- {d.metadata.get('filename')} (pág {d.metadata.get('page','?')})" for d in sources
                )
            else:
                foot = ""

            final_text = answer + foot
            st.markdown(final_text)

    # Actualiza historial conversacional
    st.session_state.chat_history.append((user_input, answer))
    st.session_state.messages.append({"role": "assistant", "content": final_text})

# Ingesta de PDFs (opcional, desde la app)
with st.sidebar:
    st.subheader("📄 Ingesta de PDFs")
    data_dir = st.text_input("Carpeta local con PDFs", value="./data_rag")
    chunk_size = st.number_input("chunk_size", min_value=200, max_value=2000, value=800, step=100)
    chunk_overlap = st.number_input("chunk_overlap", min_value=0, max_value=400, value=120, step=10)
    do_reingest = st.button("Reingestar PDFs en Pinecone")

def _stable_id(text: str) -> str:
    # ID estable en base al contenido (evita duplicados si reingestas)
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def _load_and_split(dirpath: str, chunk_size: int, chunk_overlap: int):
    loader = DirectoryLoader(dirpath, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    # enriquecer metadata + IDs estables
    for ch in chunks:
        src = ch.metadata.get("source", "")
        ch.metadata["filename"] = os.path.basename(src)
        page = str(ch.metadata.get("page", ""))
        ch.metadata["stable_id"] = f'{ch.metadata["filename"]}|p{page}|{_stable_id(ch.page_content)}'
    return chunks

if do_reingest:
    with st.status("Procesando y subiendo documentos…", expanded=True) as status:
        try:
            st.write("📥 Cargando PDFs y generando chunks…")
            chunks = _load_and_split(data_dir, chunk_size, chunk_overlap)
            st.write(f"✅ Cargados {len(chunks)} chunks desde `{data_dir}`.")

            st.write("🔼 Haciendo upsert en Pinecone (idempotente con stable_id)…")
            ids = [ch.metadata["stable_id"] for ch in chunks]
            # reutilizamos el vectorstore ya creado
            vectorstore.add_documents(documents=chunks, ids=ids)

            st.write("🧠 Listo. Los nuevos chunks ya están disponibles para las búsquedas.")
            status.update(label="Reingesta completada", state="complete", expanded=False)
            st.success("Reingesta exitosa. Ya puedes hacer preguntas.")
        except Exception as e:
            status.update(label="Error durante la reingesta", state="error", expanded=True)
            st.error(f"Ocurrió un error: {e}")
