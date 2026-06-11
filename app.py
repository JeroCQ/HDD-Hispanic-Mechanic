import os
import streamlit as st
import tempfile
from supabase import create_client, Client
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. CONFIGURACIÓN VISUAL (MODO ADMIN)
st.set_page_config(page_title="Administración RAG - HDD", layout="centered")

# 2. CREDENCIALES (Manteniendo intacta tu configuración original)
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

# Dividimos el espacio de trabajo en dos pestañas limpias
tab_subir, tab_gestionar = st.tabs(["📤 Cargar Manual", "🗑️ Gestionar Biblioteca"])

# ==========================================
# PESTAÑA 1: INGESTA ORIGINAL INTACTA
# ==========================================
with tab_subir:
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

# ==========================================
# PESTAÑA 2: NUEVO SISTEMA DE GESTIÓN DIRECTA
# ==========================================
with tab_gestionar:
    st.subheader("Manuales activos en la base de datos")
    
    with st.spinner("Leyendo registros indexados desde Supabase..."):
        try:
            # Traemos la columna metadata de la tabla para identificar qué archivos existen
            respuesta = supabase.table("documentos_hdd").select("metadata").execute()
            registros = respuesta.data
        except Exception as e:
            st.error(f"Error al conectar con Supabase: {e}")
            registros = []

    if registros:
        # Extraemos los nombres únicos de los documentos guardados en el JSONB
        fuentes_unicas = set()
        for r in registros:
            meta = r.get("metadata", {})
            if meta and "fuente" in meta:
                fuentes_unicas.add(meta["fuente"])
        
        lista_fuentes = sorted(list(fuentes_unicas))
        
        if lista_fuentes:
            st.write(f"Se encontraron **{len(lista_fuentes)}** documentos guardados en `documentos_hdd`:")
            
            for documento in lista_fuentes:
                col_txt, col_btn = st.columns([3, 1])
                col_txt.markdown(f"📄 **{documento}**")
                
                # Acción de borrado directo usando filtros del formato JSONB de Supabase
                if col_btn.button("Eliminar", key=documento, help=f"Eliminar todos los vectores de {documento}"):
                    with st.spinner(f"Borrando {documento}..."):
                        try:
                            supabase.table("documentos_hdd").delete().eq("metadata->>fuente", documento).execute()
                            st.toast(f"Manual '{documento}' eliminado con éxito de Supabase.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"No se pudo completar la eliminación: {e}")
        else:
            st.info("No se encontraron metadatos de fuente válidos en los vectores almacenados.")
    else:
        st.info("La tabla `documentos_hdd` está completamente vacía.")
