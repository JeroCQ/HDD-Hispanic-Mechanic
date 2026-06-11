import os
import streamlit as st
import tempfile
import time
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
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", google_api_key=GOOGLE_KEY, output_dimensionality=768)

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
# PESTAÑA 1: INGESTA OPTIMIZADA CON GEMINI 2 PREVIEW
# ==========================================
with tab_subir:
    st.subheader("Sube un nuevo manual en PDF para vectorizarlo")
    marca = st.selectbox("Marca del equipo o categoría:", ["Vermeer", "Ditch Witch", "Fluidos/Lodos", "Seguridad/Sistemas"])
    archivo_subido = st.file_uploader("Arrastra aquí el manual en formato PDF", type=["pdf"])

    if archivo_subido is not None:
        if st.button("🚀 Vectorizar y Subir a Supabase"):
            with st.spinner("Fragmentando y generando embeddings con Gemini 2..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(archivo_subido.getvalue())
                        tmp_ruta = tmp_file.name

                    loader = PyPDFLoader(tmp_ruta)
                    paginas = loader.load()
                    
                    # Divisor adaptado para mantener un buen contexto técnico
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
                    chunks = text_splitter.split_documents(paginas)
                    
                    progreso = st.progress(0)
                    status_text = st.empty()
                    
                    # Forzamos la inicialización con el modelo verificado Y recortamos a 768 dimensiones
                    embeddings_preview = GoogleGenerativeAIEmbeddings(
                        model="gemini-embedding-2-preview", 
                        google_api_key=GOOGLE_KEY,
                        output_dimensionality=768  # <--- ESTA LÍNEA SOLUCIONA EL ERROR
                    )
                    
                    # Agrupamos en lotes pequeños de 15 fragmentos para cuidar la cuota del Free Tier
                    TAMAÑO_LOTE = 15
                    
                    for i in range(0, len(chunks), TAMAÑO_LOTE):
                        lote_chunks = chunks[i:i + TAMAÑO_LOTE]
                        textos_lote = [chunk.page_content for chunk in lote_chunks]
                        
                        try:
                            # Generación masiva usando el modelo correcto
                            vectores_lote = embeddings_preview.embed_documents(textos_lote)
                            
                            filas_supabase = []
                            for j, chunk in enumerate(lote_chunks):
                                num_pagina = chunk.metadata.get("page", 0) + 1
                                datos_fila = {
                                    "contenido": chunk.page_content,
                                    "embedding": vectores_lote[j],
                                    "metadata": {
                                        "fuente": archivo_subido.name,
                                        "pagina": num_pagina,
                                        "marca": marca
                                    }
                                }
                                filas_supabase.append(datos_fila)
                            
                            # Subida en bloque a Supabase
                            supabase.table("documentos_hdd").insert(filas_supabase).execute()
                            
                            # Cálculo visual del progreso
                            ultimo_indice_procesado = min(i + TAMAÑO_LOTE, len(chunks))
                            porcentaje = int(ultimo_indice_procesado / len(chunks) * 100)
                            progreso.progress(porcentaje)
                            status_text.text(f"⚡ Procesados {ultimo_indice_procesado} de {len(chunks)} fragmentos (Pág. {num_pagina})...")
                            
                            # Pausa obligatoria de 3 segundos para resetear la cuota por minuto de la API
                            time.sleep(3)
                            
                        except Exception as error_api:
                            # Si la API se satura temporalmente, el sistema se defiende solo y espera
                            if "429" in str(error_api) or "RESOURCE_EXHAUSTED" in str(error_api):
                                status_text.text("⚠️ Límite de ráfaga alcanzado. Enfriando motor por 12 segundos...")
                                time.sleep(12)
                                # Decrementamos el índice para volver a intentar este bloque exacto
                                i -= TAMAÑO_LOTE
                            else:
                                raise error_api
                    
                    st.success(f"🏁 ¡Éxito rotundo! El manual '{archivo_subido.name}' está completamente indexado.")
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
