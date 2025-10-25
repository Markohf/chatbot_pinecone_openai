"""
# 💬 Chatbot de Conocimiento con RAG + Pinecone + OpenAI

Este proyecto implementa un **chatbot de recuperación aumentada (RAG)** que permite hacer preguntas sobre documentos internos cargados localmente.  
El sistema combina **LangChain**, **Pinecone** y **OpenAI** para indexar documentos, generar embeddings y responder consultas con base en esos datos.

---

## 🧩 Arquitectura general

El flujo completo del proyecto se divide en dos partes:

### 1. **Construcción de la base de conocimiento (`build_knowledge_base.py`)**
- Lee los archivos almacenados en `data_rag/`.
- Los divide en fragmentos (`chunks`) de texto.
- Genera embeddings usando un modelo HuggingFace.
- Los sube a **Pinecone**, creando o actualizando el índice correspondiente.

### 2. **Aplicación interactiva (`app.py`)**
- Desarrollada con **Streamlit**.
- Permite al usuario hacer preguntas en español.
- Recupera los fragmentos más relevantes desde Pinecone.
- Envía la pregunta + contexto al modelo de OpenAI configurado.
- Muestra únicamente la respuesta final, sin fuentes ni detalles técnicos.

---

## 📂 Estructura del proyecto
```
despliegue_exam/
│
├── .streamlit/
│ └── secrets.toml # Variables de entorno y claves de API
│
├── data_rag/ # Carpeta con los documentos fuente
│ ├── ejemplo1.pdf
│ ├── ejemplo2.txt
│ └── ...
│
├── src/
│ ├── __init__.py
│ ├── app.py # Aplicación principal (Streamlit)
│ ├── build_knowledge_base.py # Script para indexar documentos en Pinecone
│ └── utils/
│   ├── __init__.py
│   └── logger_config.py # Configuración de logging
│
├── requirements.txt # Librerías necesarias
└── README.md # Este archivo
```

---

## ⚙️ Configuración inicial

### 1. Crear y activar entorno virtual

Desde la raíz del proyecto:

#### En Windows (PowerShell):
```
python -m venv venv
venv\Scripts\Activate.ps1
```

#### En Linux / macOS / Git Bash:
```
python -m venv venv
source venv/bin/activate
```

### 2. Instalar dependencias

Con el entorno activado:
```
pip install -r requirements.txt
```

### 3. Configurar variables en .streamlit/secrets.toml

Crea el archivo `.streamlit/secrets.toml` (si no existe) en la raíz del proyecto con el siguiente formato:
```
OPENAI_API_KEY = ""
PINECONE_API_KEY = ""

INDEX_NAME = ""
NAMESPACE = ""
EMBEDDING_MODEL = ""

PINECONE_CLOUD = ""
PINECONE_REGION = ""

OPENAI_MODEL = ""
```

⚠️ Reemplaza los valores vacíos por tus credenciales y parámetros reales.

- INDEX_NAME: nombre del índice que se creará en Pinecone.
- NAMESPACE: etiqueta o agrupador dentro del índice.
- EMBEDDING_MODEL: nombre del modelo HuggingFace usado para generar embeddings (por ejemplo "sentence-transformers/all-MiniLM-L6-v2").
- PINECONE_CLOUD / REGION: configuración del entorno donde se crea el índice (por ejemplo "aws" y "us-east-1").
- OPENAI_MODEL: modelo de chat usado para responder (por ejemplo "gpt-4o-mini").

---

## Cargar y actualizar documentos

Coloca tus archivos de referencia dentro de la carpeta data_rag/.

Formatos soportados:
- .pdf
- .txt
- .md
- .csv

Cada vez que agregues o actualices documentos, reconstruye la base de conocimiento con:
```
python -m src.build_knowledge_base
```
El script:
- Lee los documentos nuevos.
- Los divide en fragmentos.
- Calcula embeddings.
- Los envía a Pinecone (creando el índice si no existe).

---

## 🚀 Ejecutar la aplicación

Desde la raíz del proyecto:
```
streamlit run src/app.py
```
Luego abre el enlace local que Streamlit mostrará, por ejemplo:
```
# Ejemplo
http://localhost:9999
```

---

## 🧱 Detalles técnicos

### build_knowledge_base.py
- Usa pypdf para extraer texto de archivos PDF.
- Implementa RecursiveCharacterTextSplitter (de LangChain) para generar fragmentos superpuestos.
- Crea embeddings mediante HuggingFaceEmbeddings.
- Verifica si el índice existe en Pinecone y, si no, lo crea automáticamente usando ServerlessSpec.
- Subida de documentos a Pinecone con IDs deterministas (hash SHA-256) para evitar duplicados.

### app.py
- Desarrollado con Streamlit para interfaz web.
- Conecta a Pinecone y reconstruye el retriever dinámicamente.
- Implementa un pipeline RetrievalQA (de LangChain) con un prompt personalizado en español.
- Procesa preguntas con solo presionar Enter (sin botón manual).
- Muestra únicamente la respuesta, sin detalles ni fuentes.

---

## 🧩 Dependencias principales

Las versiones se encuentran en requirements.txt.
Incluye entre otras:
```
streamlit
langchain
langchain-community
langchain-openai
langchain-pinecone
pinecone
sentence-transformers
torch
huggingface-hub
pypdf
toml
python-dotenv
httpx
```

---

## 🧪 Ejecución típica

### Construir base de conocimiento:
```
python -m src.build_knowledge_base
```

### Ejecutar la app:
```
streamlit run src/app.py
```