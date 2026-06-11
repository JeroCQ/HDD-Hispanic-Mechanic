import os
import streamlit as st
from pypdf import PdfReader
from supabase import create_client, Client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. CONFIGURACIÓN VISUAL DEL ADMINISTRADOR
st.set_page_config(page_title="Admin - Copiloto HDD", layout="centered")
st.title("🗄️ Panel de Control de Datos: Copiloto HDD")
st.caption("Administra, segmenta y elimina los manuales técnicos conectados a Supabase.")

# 2. CONFIGURACIÓN DE CREDENCIALES (Variables de Entorno o Inputs)
google_key = os.environ.get("GEMINI_API_KEY") or st.sidebar.text_input("Gemini API Key", type="password")
supabase_url = os.environ.get("SUPABASE_URL") or st.sidebar.text_input("Supabase URL")
supabase_key = os.environ.get("SUPABASE_KEY") or st.sidebar.text_input("Supabase Service Role / Anon Key", type="password")

if not google_key or not supabase_url or not supabase_key:
    st.info("Por favor, introduce las credenciales requeridas en la barra lateral para acceder a la base de datos.")
    st.stop()

# Inicializar cliente de Supabase
supabase_client: Client = create_client(supabase_url, supabase_key)

# Inicializar el modelo oficial de Embeddings de Google (Dimensión 768)
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=google_key
)

# Estructurar la consola en dos pestañas de gestión
tab_subir, tab_gestionar = st.tabs(["📤 Cargar Nuevo Manual", "🗑️ Gestionar Documentos Activos"])

# ==========================================
# PESTAÑA 1: CARGAR Y PROCESAR MANUALES
# ==========================================
with tab_subir:
    st.subheader("Subir manual técnico (PDF o TXT)")
    marca_maquina = st.selectbox("Marca de la Maquinaria", ["Ditch Witch", "Vermeer", "Genérico / Fluidos"])
    archivo_subido = st.file_uploader("Selecciona el archivo", type=["pdf", "txt"])

    if archivo_subido is not None:
        nombre_archivo = archivo_subido.name
        texto_completo = ""
        paginas_datos = []

        if st.button("Procesar y Vectorizar Documento"):
            with st.spinner("Extrayendo texto del archivo..."):
                if archivo_subido.type == "application/pdf":
                    lector_pdf = PdfReader(archivo_subido)
                    for num_pag, pagina in enumerate(lector_pdf.pages):
                        texto_pag = pagina.extract_text()
                        if texto_pag:
                            paginas_datos.append({"texto": texto_pag, "pagina": num_pag + 1})
                else:
                    # Procesamiento básico si es archivo de texto plano TXT
                    texto_completo = archivo_subido.read().decode("utf-8")
                    paginas_datos.append({"texto": texto_completo, "pagina": 1})

            if paginas_datos:
                with st.spinner("Fragmentando texto y generando embeddings con Gemini..."):
                    # Configuramos el divisor de texto idéntico a las mejores prácticas de RAG
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
                    
                    registros_a_insertar = []
                    
                    for item in paginas_datos:
                        chunks = text_splitter.split_text(item["texto"])
                        for chunk in chunks:
                            # 1. Generar el embedding numérico real (768 dimensiones)
                            vector = embeddings_model.embed_query(chunk)
                            
                            # 2. Armar la estructura exacta de la fila para tu tabla 'documentos_hdd'
                            fila = {
                                "contenido": chunk,
                                "embedding": vector,
                                "metadata": {
                                    "fuente": nombre_archivo,
                                    "pagina": item["pagina"],
                                    "marca": marca_maquina
                                }
                            }
                            registros_a_insertar.append(fila)

                with st.spinner(f"Subiendo {len(registros_a_insertar)} fragmentos a Supabase..."):
                    try:
                        # Inserción masiva directamente en tu tabla existente
                        resultado = supabase_client.table("documentos_hdd").insert(registros_a_insertar).execute()
                        st.success(f"¡Éxito! El documento '{nombre_archivo}' fue procesado e indexado correctamente en Supabase.")
                    except Exception as e:
                        st.error(f"Error al guardar datos en Supabase: {e}")
            else:
                st.error("No se pudo extraer texto válido del documento proporcionado.")

# ==========================================
# PESTAÑA 2: VISUALIZAR Y ELIMINAR DOCUMENTOS
# ==========================================
with tab_gestionar:
    st.subheader("Manuales indexados en el sistema")
    
    with st.spinner("Consultando estado de la base de datos..."):
        try:
            # Traemos solo la columna metadata para procesar los nombres únicos en Python
            respuesta = supabase_client.table("documentos_hdd").select("metadata").execute()
            registros = respuesta.data
        except Exception as e:
            st.error(f"No se pudo conectar con Supabase: {e}")
            registros = []

    if registros:
        # Extraer fuentes únicas del objeto JSONB de metadatos
        fuentes_unicas = set()
        for r in registros:
            meta = r.get("metadata", {})
            if meta and "fuente" in meta:
                fuentes_unicas.add(meta["fuente"])
        
        lista_fuentes = list(fuentes_unicas)
        
        if lista_fuentes:
            st.write(f"Actualmente tienes **{len(lista_fuentes)}** documento(s) cargado(s) en la base de datos:")
            
            # Mostrar los archivos en formato de lista interactiva
            for documento in lista_fuentes:
                col_nombre, col_accion = st.columns([3, 1])
                col_nombre.markdown(f"📄 **{documento}**")
                
                # Botón de eliminación dedicado por archivo
                if col_accion.button("Eliminar", key=documento, help=f"Borrar todos los vectores de {documento}"):
                    with st.spinner(f"Eliminando {documento}..."):
                        try:
                            # Filtro avanzado sobre la clave del campo JSONB de Supabase
                            supabase_client.table("documentos_hdd").delete().eq("metadata->>fuente", documento).execute()
                            st.toast(f"Documento '{documento}' eliminado correctamente.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"No se pudo eliminar el registro: {e}")
        else:
            st.info("No se encontraron metadatos de fuente válidos en los registros.")
    else:
        st.info("La tabla `documentos_hdd` está vacía actualmente. Sube un archivo en la pestaña anterior.")
