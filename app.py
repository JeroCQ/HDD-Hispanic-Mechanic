import os
import streamlit as st
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

# EL NOMBRE REAL: ChatGoogleGenerativeAI y GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# 1. CONFIGURACIÓN VISUAL
st.set_page_config(page_title="Mecánico", layout="centered")
st.title("⚡ Mecánico: Ditch Witch & Vermeer")
st.caption("Cuál es el problema? | Especifíca las caracteristicas de la máquina")

# 2. VALIDACIÓN DE CREDENCIALES
google_key = os.environ.get("GEMINI_API_KEY") or st.sidebar.text_input("Ingresa tu Gemini API Key", type="password")

if not google_key:
    st.info("Por favor, ingresa tu API Key de Gemini en la barra lateral para activar al Coach.")
    st.stop()

# 3. BASE DE DATOS VECTORIAL
# 3. BASE DE DATOS VECTORIAL (ACTUALIZADA)
@st.cache_resource(show_spinner=True)
def inicializar_base_datos(google_key):
    # EL MODELO CORRECTO ES: text-embedding-004
    try:
        documentos = [
            # SISTEMAS HIDRÁULICOS Y MECÁNICOS (DITCH WITCH / VERMEER)
            "Falla de Empuje en Ditch Witch (Thrust Pressure): Si la máquina pierde fuerza de empuje de forma súbita y la pantalla reporta un código de falla hidráulica en la bomba auxiliar, indica una caída de presión crítica en el circuito auxiliar. Solución inmediata: Inspeccionar la válvula de alivio situada en el bloque colector izquierdo (left manifold block). Verificar si el solenoide Y4 recibe señal eléctrica de la ECU. El torque de reajuste oficial para esta válvula es de 35 lb-ft (47 N·m).",
            
            "Desgaste de Mordazas y Prensa-Barras: Para evitar el deslizamiento de la columna de perforación durante el avance o el backreaming, el grosor de los dientes de las mordazas del prensa-barras (slip jaws) debe revisarse cada 50 horas de operación. Si el desgaste supera el límite de tolerancia física del 20%, se debe suspender la operación. El torque de apriete (makeup torque) en las roscas de las barras (drill rods) debe respetar estrictamente los límites del fabricante para evitar el estiramiento y fractura del metal.",
            
            # INGENIERÍA DE FLUIDOS Y DISEÑO DE LODOS DE PERFORACIÓN
            "Fluidos para Terrenos de Arena Suelta (Running Sand): Las formaciones de arena húmeda o inestable tienden a colapsar el pozo piloto y causar pérdida de retorno del lodo de perforación. Protocolo de contingencia: Detener el avance mecánico inmediatamente. Por cada 1,000 galones de agua en el tanque de mezcla, dosificar entre 35 y 40 libras de bentonita de alta producción (Premium Gel) para sellar las paredes del hueco, combinado con 1 a 2 cuartos de galón de polímero líquido (SUSPEND-IT) para levantar la arena pesada. La viscosidad medida en el Embudo Marsh debe mantenerse estrictamente en un rango de 48 a 52 segundos antes de reanudar la rotación.",
            
            "Fluidos para Terrenos de Arcilla Reactiva (Sticky Clay): Las arcillas plásticas absorben el agua del lodo, se expanden y se pegan a la cabeza de perforación o al reamer, causando bloqueos por torque alto. Protocolo de dosificación: Mezclar por cada 1,000 galones de agua de 15 a 25 libras de bentonita para control de filtración, añadiendo 1 galón de inhibidor de arcilla (CON-DET o similar) para humectar y evitar el embolamiento (bit balling). Mantener la viscosidad Marsh baja, entre 34 y 38 segundos, para facilitar el flujo de retorno hacia el pozo de entrada.",
            
            # SEGURIDAD CRÍTICA Y SISTEMAS ELECTRÓNICOS DE CAMPO
            "Sistema de Alerta Eléctrica Strike Alert / ESAS: Si la luz roja se enciende de forma intermitente acompañada de una alarma sonora continua tras haber hincado las estacas de tierra en terreno húmedo, el sistema indica una falla de autocomprobación por alta resistencia a tierra (impedancia superior a 100 ohmios). Protocolo de diagnóstico seguro: Apagar el motor de perforación, limpiar los terminales de los cables de prueba conectados a las estacas para remover óxido o lodo seco, y verificar que el aislamiento del cable principal que conecta al chasis no presente grietas ni cortes. Presionar el botón 'Test' durante 3 segundos para realizar el aislamiento manual antes de reanudar operaciones.",
            
            "Localización y Sistemas de Guía (Walkover Tracking): Las interferencias pasivas (estructuras metálicas, concreto reforzado) o activas (líneas de alta tensión enterradas) descalibran la lectura de profundidad y pendiente de la sonda (sonde/beacon) alojada en la cabeza de perforación. Antes de iniciar el cruce, es obligatorio realizar una calibración de fondo (Roll-Ahead Calibration) a una distancia de 10 pies (3 metros) del receptor y verificar la intensidad de la señal en los ejes X e Y para evitar desviaciones del perfil de diseño del pozo."
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        docs = text_splitter.create_documents(documentos)
        
        # MODELO ACTUALIZADO
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2-preview", 
            google_api_key=google_key
        )
        
        vector_store = InMemoryVectorStore.from_documents(docs, embeddings)
        return vector_store.as_retriever(search_kwargs={"k": 2})
    except Exception as e:
        st.error(f"Error al conectar con Gemini: {e}")
        return None

# --- LLAMADA EN EL MAIN ---
if google_key:
    retriever = inicializar_base_datos(google_key)
    if retriever is None:
        st.stop() # Detiene la app si la inicialización falló

# 4. CAPACIDAD DE BÚSQUEDA DEL AGENTE
@tool
def buscar_en_base_de_datos(query: str) -> str:
    """Busca información en la base de datos sobre procedimientos de diagnóstico y soluciones."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

tools = [buscar_en_base_de_datos]
tool_node = ToolNode(tools)

# 5. ARQUITECTURA DE CONTROL (LangGraph)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Aquí también se corrigió el nombre de la clase
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3, google_api_key=google_key).bind_tools(tools)

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

# 6. ENTORNO DE CONVERSACIÓN
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Bienvenido. Sé todo sobre la mecánica de las perforadosras. ¿Qué problema tienes?")]

for message in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.write(message.content)

if user_input := st.chat_input("Ej: ¿Cómo calibro mi máquina?"):
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Pensando ..."):
            inputs = {"messages": st.session_state.chat_history}
            output = agentic_graph.invoke(inputs)
            raw_content = output["messages"][-1].content
            # Verificamos si LangChain nos devolvió una lista compleja o un texto directo
            if isinstance(raw_content, list):
                final_response = raw_content[0].get("text", "")
            else:
                final_response = raw_content
            st.write(final_response)
            st.session_state.chat_history.append(AIMessage(content=final_response))
