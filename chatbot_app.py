import streamlit as st
from transformers import pipeline

# --- 1. Load the Model ---
# Cache the model so it only downloads and loads once
@st.cache_resource
def load_chatbot_model():
    """Loads the text generation pipeline for the chatbot."""
    # Using the same model you specified: GPT-2
    return pipeline("text-generation", model="gpt2")

chatbot = load_chatbot_model()

# --- 2. Streamlit UI Setup ---
st.set_page_config(page_title="Hugging Face GPT-2 Chatbot", layout="centered")
return pipeline("text-generation", model="distilgpt2")

st.caption("Built with Streamlit and Hugging Face Transformers")

# --- 3. Chat History Setup ---
# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Main Chat Logic ---
# Accept user input
if prompt := st.chat_input("Say something to the AI..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("AI is thinking..."):
            # The core generation function
            response = chatbot(
                prompt,
                max_length=60 + len(prompt.split()), # Adjust max_length dynamically
                do_sample=True,
                top_p=0.95,
                top_k=60,
                num_return_sequences=1,
            )
            ai_response = response[0]['generated_text'].strip()
            
            # Post-process: remove the user's prompt from the generated text
            if ai_response.startswith(prompt):
                 ai_response = ai_response[len(prompt):].strip()
            
            # Ensure a clean response start
            if ai_response.startswith((':', ',', '.', '!', '?')):
                 ai_response = ai_response[1:].strip()
            
            st.markdown(ai_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
