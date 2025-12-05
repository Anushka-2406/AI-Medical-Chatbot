import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# -----------------------------------------------------
# 1. LOAD YOUR RAG LLM MODEL HERE
# -----------------------------------------------------
from main import main_chain   # <-- your RAG chain that takes ONLY a text string


# -----------------------------------------------------
# 2. INITIALIZE CHAT HISTORY
# -----------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []


# -----------------------------------------------------
# 3. DISPLAY PREVIOUS CHAT MESSAGES
# -----------------------------------------------------
st.title("📚 RAG Chatbot")

for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -----------------------------------------------------
# 4. CHAT INPUT
# -----------------------------------------------------
user_input = st.chat_input("Ask your question...")

if user_input:
    # Add user message
    st.session_state["history"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # -----------------------------------------------------
    # 5. CALL YOUR RAG MODEL (STRING INPUT)
    # -----------------------------------------------------
    try:
        # DIRECT STRING CALL
        answer = main_chain.invoke(user_input)

        # Convert LangChain Message → string if needed
        if hasattr(answer, "content"):
            answer = answer.content

    except Exception as e:
        answer = f"Error: {e}"

    # -----------------------------------------------------
    # 6. SAVE AND DISPLAY ANSWER
    # -----------------------------------------------------
    st.session_state["history"].append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.write(answer)
