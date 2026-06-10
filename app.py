import os
import streamlit as st
import tempfile
from supabase import create_client, Client
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. CONFIGURACIÓN VISUAL (MODO ADMIN)
st.set_page_config(page_title="Administración RAG - HDD", layout="centered")

# 2. CREDENCIALES
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    GOOGLE_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
    GOOGLE_KEY = os.environ.get("GEMINI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not GOOGLE_KEY:
    st.error("⚠️ Faltan credenciales en los Secrets.")
    st.stop()

# Inicialización de clientes
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
# Solo necesitamos el modelo de Embeddings para vectorizar los PDFs
embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=GOOGLE_KEY, output_dimensionality=768)

# 3. PANEL DE ADMINISTRADOR ÚNICO
st.title("⚙️ Motor de Ingesta RAG: Manuales HDD")
    
# Sistema de seguridad
clave_admin = st.text_input("🔑 Ingresa la contraseña de administrador:", type="password")

if clave_admin != "juanfernandog":
    if clave_admin: 
        st.error("❌ Contraseña incorrecta. Acceso denegado.")
    st.stop()
    
st.success("Acceso concedido.")
st.subheader("Sube un nuevo manual en PDF para vectorizarlo")

marca = st.selectbox("Marca del equipo o categoría:", ["Ditch Witch", "Vermeer", "Fluidos/Lodos", "Seguridad/Sistemas"])
archivo_subido = st.file_uploader("Arrastra aquí el manual en formato PDF", type=["pdf"])

if archivo_subido is not None:
    if st.button("🚀 Vectorizar y Subir a Supabase"):
        with st.spinner("Fragmentando y generando embeddings..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(archivo_subido.getvalue())
                    tmp_ruta = tmp_file.name

                loader = PyPDFLoader(tmp_ruta)
                paginas = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
                chunks = text_splitter.split_documents(paginas)
                
                progreso = st.progress(0)
                status_text = st.empty()
                
                for i, chunk in enumerate(chunks):
                    texto_limpio = chunk.page_content
                    num_pagina = chunk.metadata.get("page", 0) + 1
                    vector = embeddings.embed_query(texto_limpio)
                    
                    datos_fila = {
                        "contenido": texto_limpio,
                        "embedding": vector,
                        "metadata": {
                            "fuente": archivo_subido.name,
                            "pagina": num_pagina,
                            "marca": marca
                        }
                    }
                    supabase.table("documentos_hdd").insert(datos_fila).execute()
                    
                    porcentaje = int((i + 1) / len(chunks) * 100)
                    progreso.progress(porcentaje)
                    status_text.text(f"Subiendo fragmento {i+1} de {len(chunks)} (Pág. {num_pagina})")
                
                st.success(f"🏁 ¡Éxito! El manual '{archivo_subido.name}' fue guardado permanentemente.")
                os.unlink(tmp_ruta)
                
            except Exception as e:
                st.error(f"Error crítico durante el procesamiento: {e}")
