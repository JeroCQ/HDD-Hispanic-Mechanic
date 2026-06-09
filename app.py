import os
import streamlit as st
import tempfile
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

# IMPORTANTE: Importamos genai puro para procesar el audio directamente con Gemini
import google.generativeai as genai

# 1. CONFIGURACIÓN VISUAL DE LA APP
st.set_page_config(page_title="Mecánico HDD", layout="centered")

# ==========================================
# 2. CREDENCIALES COMPARTIDAS (OCULTAS AL USUARIO)
# ==========================================
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    GOOGLE_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
    GOOGLE_KEY = os.environ.get("GEMINI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not GOOGLE_KEY:
    st.error("⚠️ El sistema no encuentra las credenciales. Configura los 'Secrets' en Streamlit Cloud.")
    st.stop()

# Inicialización de clientes
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", google_api_key=GOOGLE_KEY, output_dimensionality=768)
genai.configure(api_key=GOOGLE_KEY) # Configuración para el motor de audio

# ==========================================
# 3. PANELES DE LA INTERFAZ
# ==========================================
modo = st.sidebar.radio("Selecciona el Panel:", ["🤖 Copiloto de Campo (Chat)", "⚙️ Administrador (Cargar Manuales)"])

# ------------------------------------------
# PANEL DE ADMINISTRADOR CON CONTRASEÑA
# ------------------------------------------
if modo == "⚙️ Administrador (Cargar Manuales)":
    st.title("⚙️ Centro de Control de Conocimiento")
    
    # --- SISTEMA DE SEGURIDAD ---
    clave_admin = st.text_input("🔑 Ingresa la contraseña de administrador:", type="password")
    
    if clave_admin != "juanfernandog":
        if clave_admin: # Si escribió algo pero está mal
            st.error("❌ Contraseña incorrecta. Acceso denegado.")
        st.stop() # Detiene la ejecución aquí. No dibuja el resto de la interfaz.
        
    # Si la contraseña es correcta, mostramos el uploader
    st.success("Acceso concedido.")
    st.subheader("Nutre al agente subiendo nuevos manuales técnicos en PDF")
    
    marca = st.selectbox("Marca del equipo o categoría:", ["Ditch Witch", "Vermeer", "Fluidos/Lodos", "Seguridad/Sistemas"])
    archivo_subido = st.file_uploader("Arrastra aquí el manual en formato PDF", type=["pdf"])
    
    if archivo_subido is not None:
        if st.button("🚀 Procesar y Alimentar Base de Datos"):
            with st.spinner("Procesando PDF, fragmentando y generando embeddings con Gemini..."):
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
                    
                    st.success(f"🏁 ¡Éxito! El manual '{archivo_subido.name}' fue guardado en la nube.")
                    os.unlink(tmp_ruta)
                    
                except Exception as e:
                    st.error(f"Error crítico durante el procesamiento: {e}")
    st.stop()

# ------------------------------------------
# PANEL DE USUARIO: COPILOTO INTELIGENTE RAG
# ------------------------------------------
st.title("🚜 Mecánico Experto: Ditch Witch & Vermeer")
st.caption("Resolución de crisis mecánicas e ingeniería de fluidos en tiempo real.")

# 4. HERRAMIENTAS DEL AGENTE
@tool
def buscar_en_manuales_supabase(query: str) -> str:
    """Busca especificaciones exactas, torques y soluciones en la base de datos de manuales."""
    try:
        query_vector = embeddings.embed_query(query)
        respuesta_db = supabase.rpc(
            "buscar_documentos", 
            {"query_embedding": query_vector, "match_threshold": 0.4, "match_count": 3}
        ).execute()
        
        if not respuesta_db.data:
            return "No se encontraron registros coincidentes en los manuales de la base de datos."
            
        fragmentos = [row["contenido"] for row in respuesta_db.data]
        return "\n\n".join(fragmentos)
    except Exception as e:
        return f"Error al buscar en la base de datos: {e}"

@tool
def listar_manuales_cargados() -> str:
    """Única herramienta para saber qué manuales, marcas o documentos hay en la base de datos."""
    try:
        respuesta = supabase.table("documentos_hdd").select("metadata").execute()
        if not respuesta.data:
            return "Actualmente no tienes ningún manual cargado."
            
        nombres_archivos = set()
        for fila in respuesta.data:
            if "metadata" in fila and "fuente" in fila["metadata"]:
                nombres_archivos.add(fila["metadata"]["fuente"])
                
        if not nombres_archivos:
            return "No se pudo identificar el nombre de los manuales."
            
        lista_manuales = "\n".join([f"- {nombre}" for nombre in nombres_archivos])
        return f"Tengo acceso a los siguientes manuales:\n{lista_manuales}"
    except Exception as e:
        return f"Error al consultar el inventario: {e}"

tools = [buscar_en_manuales_supabase, listar_manuales_cargados]
tool_node = ToolNode(tools)

# 5. ORQUESTACIÓN CON LANGGRAPH
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

INSTRUCCION_MAESTRA = """
Eres un Ingeniero de Soporte Técnico Senior especializado en maquinaria HDD.
Reglas:
1. Responde de forma hiper-estructurada y directa.
2. Mapea el Spanglish de campo al inglés técnico.
3. SIEMPRE basa tu diagnóstico en los fragmentos recuperados.
4. NUNCA inventes manuales ni marcas que no estén en tu base de datos. Si te preguntan qué información tienes, usa SIEMPRE tu herramienta de listar manuales.
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash", 
    temperature=0.1, 
    google_api_key=GOOGLE_KEY,
    system_instruction=INSTRUCCION_MAESTRA
).bind_tools(tools)

def call_model(state: AgentState):
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
# 6. ENTORNO DE CONVERSACIÓN (TEXTO Y VOZ)
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if len(st.session_state.chat_history) == 0:
    with st.chat_message("assistant"):
        st.write("Sistema en línea conectado. Escribe o graba un mensaje de voz indicando la falla del equipo.")

# Mostrar historial
for message in st.session_state.chat_history:
    if isinstance(message, SystemMessage):
        continue
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.write(message.content)

# Inputs de Texto y Audio
user_text = st.chat_input("Ej: perdi fuerza de empuje en mi ditch witch")
user_audio = st.audio_input("🎙️ O graba un mensaje de voz describiendo el problema")

# Lógica para procesar la entrada del usuario (sea texto o voz)
entrada_final = None

if user_audio is not None and not user_text:
    with st.spinner("Escuchando y transcribiendo tu mensaje de voz con Gemini..."):
        # Transcripción del audio directamente con Gemini
        modelo_transcriptor = genai.GenerativeModel('gemini-1.5-flash')
        audio_bytes = user_audio.read()
        
        # Le enviamos el audio binario a Gemini para que lo convierta a texto
        respuesta_audio = modelo_transcriptor.generate_content([
            "Transcribe exactamente el problema técnico que el usuario describe en este audio. Solo devuelve el texto transcrito, sin saludos.",
            {"mime_type": "audio/wav", "data": audio_bytes}
        ])
        
        entrada_final = respuesta_audio.text.strip()
        st.info(f"🗣️ **Transcripción:** {entrada_final}")

elif user_text:
    entrada_final = user_text

# Ejecución del Agente si hay una entrada válida
if entrada_final:
    with st.chat_message("user"):
        if user_text: # Solo dibujamos si fue texto, el de audio ya se dibujó en el info()
            st.write(entrada_final)

    respaldo_historial = list(st.session_state.chat_history)
    
    try:
        with st.chat_message("assistant"):
            with st.spinner("Consultando base de datos vectorial en la nube..."):
                mensajes_envio = respaldo_historial + [HumanMessage(content=entrada_final)]
                inputs = {"messages": mensajes_envio}
                output = agentic_graph.invoke(inputs)
                
                st.session_state.chat_history = output["messages"]
                raw_content = output["messages"][-1].content
                
                final_response = raw_content[0].get("text", "") if isinstance(raw_content, list) else raw_content
                st.write(final_response)
                
    except Exception as e:
        st.error(f"Error en la comunicación con el agente: {e}")
