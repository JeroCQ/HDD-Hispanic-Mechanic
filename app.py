import os
import streamlit as st
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage

# Define tu System Prompt de forma global o en un archivo de configuración separado (.env o config.yaml)
INSTRUCCION_MAESTRA = """
Eres un Ingeniero de Soporte Técnico Senior especializado en maquinaria HDD (Ditch Witch y Vermeer).
Reglas de operación:
1. Responde de forma hiper-estructurada, directa y sin saludos motivacionales.
2. Si el usuario te habla en Spanglish de campo (ej. 'remer', 'drill rod'), mapea el término al inglés técnico.
3. SIEMPRE basa tu diagnóstico en los manuales recuperados. Si la respuesta no está en el contexto, responde: "Dato no disponible en los manuales cargados. Contacte a soporte de fábrica."
4. Cita las especificaciones de torque o presión con absoluta precisión.
5. Termina con preguntas clave que sean necesarias o que te ayuden a responder mejor, especificando porqué esta información es relevante.

"""

def call_model(state: AgentState):
    messages = state["messages"]
    
    # Verificamos si el SystemMessage ya está en el historial para no duplicarlo
    if not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=INSTRUCCION_MAESTRA)] + messages
        
    response = llm.invoke(messages)
    return {"messages": [response]}

# 1. CONFIGURACIÓN VISUAL B2B INDUSTRIAL
st.set_page_config(page_title="Asistente Técnico HDD", layout="centered")
st.title("🚜 Copiloto Técnico HDD: Ditch Witch & Vermeer")
st.caption("Resolución de Crisis en Campo | Diagnóstico de Maquinaria y Fluidos de Perforación")

# 2. VALIDACIÓN DE CREDENCIALES
google_key = os.environ.get("GEMINI_API_KEY") or st.sidebar.text_input("Ingresa tu Gemini API Key", type="password")

if not google_key:
    st.info("Por favor, ingresa tu API Key de Gemini en la barra lateral para activar el asistente de campo.")
    st.stop()

# 3. BASE DE DATOS VECTORIAL (DOCUMENTACIÓN OEM Y FLUIDOS)
@st.cache_resource(show_spinner=True)
def inicializar_base_datos_hdd(api_key):
    try:
        # Ingesta técnica con mapeo semántico (Inglés OEM - Spanglish de campo)
        documentos_tecnicos = [
            "Ditch Witch JT20 / JT32 - Sistema Hidráulico: La pérdida de fuerza de empuje (thrust pressure) acompañada de código de falla hidráulica en bomba auxiliar indica caída de presión en el circuito auxiliar. Solución: Inspeccionar válvula de alivio en el bloque colector izquierdo (left manifold block). Verificar señal en solenoide Y4. Torque de reajuste de la válvula: 35 lb-ft (47 N·m).",
            
            "Vermeer D24x40 S3 / D40x55 - Fluidos en Arena (Running Sand): En terrenos de arena suelta húmeda, la pérdida de retorno de lodo requiere suspender el avance. Ajuste de mezcla por cada 1,000 galones de agua: agregar 35-40 lbs de bentonita de alta producción (Premium Gel) y 1-2 cuartos de polímero líquido (SUSPEND-IT). Viscosidad objetivo en Embudo Marsh: 48-52 segundos antes de rotar.",
            
            "Sistema de Seguridad Strike Alert / ESAS (Ditch Witch & Vermeer): Luz roja intermitente con alarma sonora continua tras clavar estacas indica falla de autocomprobación por alta resistencia a tierra (impedancia > 100 ohmios). Solución: Apagar motor, limpiar terminales de cables para eliminar oxidación, verificar que el aislamiento del cable al chasis no esté agrietado. Presionar 'Test' por 3 segundos para aislamiento manual.",
            
            "Mantenimiento de Barras y Roscas (Drill Rods): El torque de apriete (makeup torque) en barras de Ditch Witch JT20 debe mantenerse estrictamente en los parámetros OEM para evitar estiramiento de roscas. Utilizar grasa para roscas con base de cobre al 40-60%. Nunca iniciar perforación piloto si el indicador de desgaste de mordazas del prensa-barras supera el límite de tolerancia física."
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        docs = text_splitter.create_documents(documentos_tecnicos)
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-001", 
            google_api_key=api_key
        )
        
        vector_store = InMemoryVectorStore.from_documents(docs, embeddings)
        return vector_store.as_retriever(search_kwargs={"k": 2})
    except Exception as e:
        st.error(f"Error al inicializar la base de datos técnica: {e}")
        return None

retriever = inicializar_base_datos_hdd(google_key)
if retriever is None:
    st.stop()

# 4. HERRAMIENTA DE BÚSQUEDA DEL AGENTE
@tool
def buscar_manuales_y_fluidos(query: str) -> str:
    """Busca especificaciones exactas en manuales de servicio Ditch Witch, Vermeer, sistemas de fluidos y códigos Strike Alert."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

tools = [buscar_manuales_y_fluidos]
tool_node = ToolNode(tools)

# 5. ARQUITECTURA DE CONTROL (LangGraph)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Modelo de producción optimizado para llamadas a herramientas y baja latencia
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0.1,  # Temperatura baja para asegurar precisión técnica sin alucinaciones
    google_api_key=google_key
).bind_tools(tools)

def call_model(state: AgentState):
    # System prompt embebido para forzar el rol B2B y traducción cross-lingual interna
    messages = state["messages"]
    system_instruction = (
        "Actúas como un Ingeniero de Soporte Técnico Senior especializado en Perforación Horizontal Dirigida (HDD). "
        "Tu objetivo es resolver problemas mecánicos y de fluidos en campo para operarios que hablan español o Spanglish. "
        "Cuando el usuario use términos como 'remer', 'bomba de lodo', 'estacas' o 'fuerza de empuje', asócialos con la documentación OEM en inglés. "
        "Da respuestas directas, imperativas, estructuradas y cita textualmente las especificaciones de torque, viscosidad o páginas del manual recuperadas."
    )
    
    # Inyectamos la instrucción en el flujo si es el inicio
    if len(messages) == 1:
        messages = [HumanMessage(content=system_instruction)] + messages
        
    response = llm.invoke(messages)
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

# 6. INTERFAZ DE USUARIO EN CAMPO
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="⚙️ Sistema operativo. Copiloto HDD listo. Ingresa el código de error, marca de máquina (Ditch Witch / Vermeer) o las condiciones del suelo para calcular la dosificación de lodos.")
    ]

for message in st.session_state.chat_history:
    # Omitimos mostrar la instrucción del sistema en la UI si existe
    if "Actúas como un Ingeniero de Soporte" in message.content:
        continue
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.write(message.content)

if user_input := st.chat_input("Ej: código Strike Alert parpadeando en rojo / mezcla para arena en Vermeer"):
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analizando telemetría y manuales técnicos OEM..."):
            inputs = {"messages": st.session_state.chat_history}
            output = agentic_graph.invoke(inputs)
            
            st.session_state.chat_history = output["messages"]
            raw_content = output["messages"][-1].content
            
            if isinstance(raw_content, list):
                final_response = raw_content[0].get("text", "")
            else:
                final_response = raw_content
                
            st.write(final_response)
