import streamlit as st
#from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

st.set_page_config(page_title="LangChain: Chat with PostgreSQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with PostgreSQL DB")

# Load environment variables from .env file
load_dotenv()

# PostgreSQL connection details
postgres_host = os.getenv("POSTGRES_HOST")
postgres_user = os.getenv("POSTGRES_USER")
postgres_password = os.getenv("POSTGRES_PASSWORD")
postgres_db = os.getenv("POSTGRES_DB")

# Groq API Key
api_key = os.getenv("GROQ_API_KEY")

if not all([postgres_host, postgres_user, postgres_password, postgres_db]):
    st.error("Please make sure all PostgreSQL credentials are set in the .env file.")
    st.stop()

if not api_key:
    st.error("Please make sure the Groq API key is set in the .env file.")
    st.stop()

# LLM model
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

@st.cache_resource(ttl="2h")
def configure_db():
    return SQLDatabase(create_engine(
        f"postgresql+psycopg2://{postgres_user}:{postgres_password}@{postgres_host}/{postgres_db}"
    ))

# Configure database
db = configure_db()

# Toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Agent
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[streamlit_callback])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
