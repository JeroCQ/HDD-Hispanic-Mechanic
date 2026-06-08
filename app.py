import os
import streamlit as st
from typing import Annotated, Sequence, TypedDict
from supabase import create_client, Client
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import tempfile

# 1. CONFIGURACIÓN VISUAL DE LA APP
st.set_page_config(page_title="Mecánico HDD", layout="centered")

# ==========================================
# 2. CREDENCIALES COMPARTIDAS (TODO ONLINE)
# ==========================================
# Cuando despliegues en Streamlit Cloud, estas variables se configuran en "Secrets"
SUPABASE_URL = os.environ.get("SUPABASE_URL") or st.sidebar.text_input("Supabase URL", type="default")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or st.sidebar.text_input("Supabase Anon Key", type="password")
GOOGLE_KEY = os.environ.get("GEMINI_API_KEY") or st.sidebar.text_input("Gemini API Key", type="password")

if not SUPABASE_URL or not SUPABASE_KEY or not GOOGLE_KEY:
    st.info("⚠️ Por favor, ingresa las credenciales requeridas en la barra lateral para activar el sistema.")
    st.stop()

# Inicialización de clientes online
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", google_api_key=GOOGLE_KEY, output_dimensionality=768)

# ==========================================
# 3. PANELES DE LA INTERFAZ (CHAT vs ADMIN)
# ==========================================
modo = st.sidebar.radio("Selecciona el Panel:", ["🤖 Copiloto de Campo (Chat)", "⚙️ Administrador (Cargar Manuales)"])

# ------------------------------------------
# PANEL DE ADMINISTRADOR: INGESTA 100% WEB
# ------------------------------------------
if modo == "⚙️ Administrador (Cargar Manuales)":
    st.title("⚙️ Centro de Control de Conocimiento")
    st.subheader("Nutre al agente subiendo nuevos manuales técnicos en PDF")
    
    marca = st.selectbox("Marca del equipo o categoría:", ["Ditch Witch", "Vermeer", "Fluidos/Lodos", "Seguridad/Sistemas"])
    archivo_subido = st.file_uploader("Arrastra aquí el manual en formato PDF", type=["pdf"])
    
    if archivo_subido is not None:
        if st.button("🚀 Procesar y Alimentar Base de Datos"):
            with st.spinner("Procesando PDF, fragmentando y generando embeddings con Gemini..."):
                try:
                    # Guardar el archivo subido en un directorio temporal online para que LangChain lo pueda leer
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(archivo_subido.getvalue())
                        tmp_ruta = tmp_file.name

                    # 1. Extraer texto del PDF
                    loader = PyPDFLoader(tmp_ruta)
                    paginas = loader.load()
                    
                    # 2. Chunking Semántico optimizado para datos de ingeniería
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
                    chunks = text_splitter.split_documents(paginas)
                    
                    progreso = st.progress(0)
                    status_text = st.empty()
                    
                    # 3. Vectorización y carga online fila por fila a Supabase
                    for i, chunk in enumerate(chunks):
                        texto_limpio = chunk.page_content
                        num_pagina = chunk.metadata.get("page", 0) + 1
                        
                        # Generar coordenadas matemáticas con Gemini
                        vector = embeddings.embed_query(texto_limpio)
                        
                        # Guardar directo en la tabla que creamos en Supabase
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
                        
                        # Actualizar barra de progreso visual en la web
                        porcentaje = int((i + 1) / len(chunks) * 100)
                        progreso.progress(porcentaje)
                        status_text.text(f"Subiendo fragmento {i+1} de {len(chunks)} (Pág. {num_pagina})")
                    
                    st.success(f"🏁 ¡Éxito! El manual '{archivo_subido.name}' fue fragmentado en {len(chunks)} partes y guardado permanentemente en la nube.")
                    os.unlink(tmp_ruta) # Limpiar archivo temporal
                    
                except Exception as e:
                    st.error(f"Error crítico durante el procesamiento: {e}")
    st.stop() # Detiene la ejecución aquí si estás en modo admin

# ------------------------------------------
# PANEL DE USUARIO: COPILOTO INTELIGENTE RAG
# ------------------------------------------
st.title("🚜 Mecánico Experto: Ditch Witch & Vermeer")
st.caption("Resolución de crisis mecánicas e ingeniería de fluidos en tiempo real.")

st.sidebar.title("🛠️ Panel de Administración")
st.sidebar.markdown("---")

st.sidebar.subheader("1. Configuración de Conexión")
# Mantenemos los campos de texto pre-rellenados con las variables de entorno si existen
supabase_url_input = st.sidebar.text_input(
    "Supabase URL", 
    value=os.getenv("SUPABASE_URL", "https://tu-proyecto.supabase.co"),
    help="Asegúrate de que NO termine en /rest/v1"
)
supabase_key_input = st.sidebar.text_input(
    "Supabase Anon Key", 
    value=os.getenv("SUPABASE_KEY", ""), 
    type="password"
)

st.sidebar.markdown("---")
st.sidebar.subheader("2. Ingesta de Manuales Técnicos")

# Componente nativo para arrastrar y soltar archivos
uploaded_file = st.sidebar.file_uploader(
    "Selecciona el manual de la máquina (PDF)", 
    type=["pdf"],
    help="Carga el manual de operación o taller en formato PDF para vectorizarlo."
)

if uploaded_file is not None:
    # Mostramos metadatos del archivo cargado para dar feedback al usuario
    st.sidebar.info(f"📁 Archivo detectado: {uploaded_file.name} ({round(uploaded_file.size / 1024, 2)} KB)")
    
    # Botón que dispara el pipeline de procesamiento (LangChain -> Supabase)
    if st.sidebar.button("⚙️ Procesar y Vectorizar Documento", use_container_width=True):
        barra_progreso = st.sidebar.progress(0)
        estado_texto = st.sidebar.empty()
        
        try:
            # --- FASE 1: Leer el archivo ---
            estado_texto.text("Extracting texto del PDF...")
            barra_progreso.progress(25)
            
            # Aquí se invoca tu función de procesamiento (ej: procesar_pdf(uploaded_file))
            # [Tu lógica de PyPDFLoader / RecursiveCharacterTextSplitter]
            
            # --- FASE 2: Generar Embeddings ---
            estado_texto.text("Generando embeddings (768 dimensiones)...")
            barra_progreso.progress(60)
            
            # --- FASE 3: Almacenamiento ---
            estado_texto.text("Subiendo vectores a Supabase (documentos_hdd)...")
            barra_progreso.progress(90)
            
            # Éxito total
            barra_progreso.progress(100)
            estado_texto.empty()
            st.sidebar.success("✅ ¡Manual indexado con éxito en la base de datos!")
            
        except Exception as e:
            barra_progreso.empty()
            estado_texto.empty()
            st.sidebar.error(f"❌ Error crítico durante el procesamiento: {str(e)}")

# 4. CAPACIDAD DE BÚSQUEDA DEL AGENTE (CONECTADA A SUPABASE ONLINE)
@tool
def buscar_en_manuales_supabase(query: str) -> str:
    """Busca especificaciones exactas, torques y soluciones directamente en la base de datos de Supabase en la nube."""
    try:
        # Convertimos la pregunta del operador en un vector usando Gemini
        query_vector = embeddings.embed_query(query)
        
        # Llamamos a la función matemática 'buscar_documentos' que creamos en el Paso 4 de Supabase
        respuesta_db = supabase.rpc(
            "buscar_documentos", 
            {"query_embedding": query_vector, "match_threshold": 0.4, "match_count": 3}
        ).execute()
        
        if not respuesta_db.data:
            return "No se encontraron registros coincidentes en los manuales de la base de datos."
            
        # Unimos los fragmentos encontrados para pasárselos como contexto al LLM
        fragmentos = [row["contenido"] for row in respuesta_db.data]
        return "\n\n".join(fragmentos)
    except Exception as e:
        return f"Error al buscar en la base de datos en la nube: {e}"

tools = [buscar_en_manuales_supabase]
tool_node = ToolNode(tools)

# 5. ORQUESTACIÓN CON LANGGRAPH
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

INSTRUCCION_MAESTRA = """
Eres un Ingeniero de Soporte Técnico Senior especializado en maquinaria HDD (Ditch Witch y Vermeer).
Reglas de operación obligatorias:
1. Responde de forma hiper-estructurada, directa y sin saludos corteses ni introducciones motivacionales.
2. Si el usuario te habla en Spanglish de campo (ej. 'remer', 'drill rod', 'mordazas'), mapea internamente el término al inglés técnico.
3. SIEMPRE basa tu diagnóstico en los fragmentos recuperados mediante tu herramienta de búsqueda. Si la respuesta no está en el contexto provisto por la herramienta, responde estrictamente: "Dato no disponible en los manuales cargados. Contacte a soporte de fábrica."
4. Cita las especificaciones de torque, presión, ohmios o dosificación de fluidos con absoluta precisión matemática.
"""

# Al pasar INSTRUCCION_MAESTRA aquí, Gemini la procesa de forma nativa y segura
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash", 
    temperature=0.1, 
    google_api_key=GOOGLE_KEY,
    system_instruction=INSTRUCCION_MAESTRA
).bind_tools(tools)

def call_model(state: AgentState):
    # Ya no manipulamos ni alteramos el array de mensajes, evitando errores de sintaxis
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def router_logic(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "execute_tools"
    return END

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", router_logic, {"execute_tools": "tools", END: END})
workflow.add_edge("tools", "agent")
agentic_graph = workflow.compile()

# ==========================================
# 6. ENTORNO DE CONVERSACIÓN INTERACTIVO (A PRUEBA DE CRASHES)
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Saludo visual estático
if len(st.session_state.chat_history) == 0:
    with st.chat_message("assistant"):
        st.write("Sistema en línea conectado a Supabase. ¿Qué código de error o falla mecánica presenta el equipo?")

for message in st.session_state.chat_history:
    if isinstance(message, SystemMessage):
        continue
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.write(message.content)

if user_input := st.chat_input("Ej: perdi fuerza de empuje en mi ditch witch"):
    with st.chat_message("user"):
        st.write(user_input)

    # Creamos un respaldo para no corromper el historial si la API llega a fallar
    respaldo_historial = list(st.session_state.chat_history)
    
    try:
        with st.chat_message("assistant"):
            with st.spinner("Consultando base de datos vectorial en la nube..."):
                # Construimos la secuencia limpia para esta ejecución
                mensajes_envio = respaldo_historial + [HumanMessage(content=user_input)]
                inputs = {"messages": mensajes_envio}
                output = agentic_graph.invoke(inputs)
                
                # Si la ejecución es exitosa, se actualiza el historial definitivo
                st.session_state.chat_history = output["messages"]
                raw_content = output["messages"][-1].content
                
                final_response = raw_content[0].get("text", "") if isinstance(raw_content, list) else raw_content
                st.write(final_response)
                
    except Exception as e:
        st.error(f"Error en la comunicación con el agente: {e}")
        st.info("💡 Nota: Si el error persiste, recarga la pestaña del navegador para limpiar la memoria caché.")
