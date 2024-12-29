import ollama
import streamlit as st
import tempfile
import os
 
SYSTEM_PROMPT = """You are an AI diagnostic assistant specialized exclusively in X-ray analysis.
 Your sole purpose is to analyze uploaded X-ray images and answer questions specifically related to X-ray diagnostics.
   If an uploaded image is not an X-ray, respond only with: 'I only know X-rays.' Do not provide any additional analysis, information,
   or engage with non-X-ray images. For any queries or actions unrelated to X-rays, also respond only with: 'I only know X-rays.'"""
text_system_prmot="You are an AI assistant specialized in answering questions exclusively about X-ray diagnostics. You will only respond to queries directly related to X-ray interpretation, procedures, or technology. If the query is unrelated to X-rays, respond only with: 'I only know X-rays.' Do not engage with or answer unrelated questions."
DEFAULT_MODEL = "llava"
 
def ai_response(user_input):
    try:
        response = ollama.chat(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": text_system_prmot},
                {"role": "user", "content": user_input},
            ],
        )
       
        return response['message']['content']
    except Exception as e:
        return f"Error generating AI response: {str(e)}"
 
 
def run_inference(image_path: str):
    try:
        stream = ollama.chat(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "diagnose the image", "images": [image_path]},
            ],
            stream=True,
        )
        diagnosis = ""
        for chunk in stream:
            diagnosis += chunk["message"]["content"]
        return diagnosis
    except Exception as e:
        return f"Error during inference: {str(e)}"
 
st.title("AI Interface")
st.write("Interact with the AI by entering text or uploading an image for diagnosis.")
 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
 
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
 
st.session_state.user_input = st.text_input("Enter your query:", st.session_state.user_input)
 
uploaded_image = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
 
if st.button("Submit"):
    if st.session_state.user_input and not uploaded_image: 
        response = ai_response(st.session_state.user_input)
        st.session_state.chat_history.append({"user": st.session_state.user_input, "ai": response})
        st.session_state.user_input = "" 
    elif uploaded_image and not st.session_state.user_input: 
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
 
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_image.read())
            temp_path = temp_file.name
 
        if os.path.exists(temp_path):
            diagnosis = run_inference(temp_path)
            st.subheader("Diagnosis Result")
            st.write(diagnosis)
        else:
            st.error("Error saving the uploaded image.")
    else:
        st.warning("Please enter a text query or upload an image (not both).")
 
st.subheader("Chat History")
for chat in st.session_state.chat_history:
    st.write(f"**You:** {chat['user']}")
    st.write(f"**AI:** {chat['ai']}")
 
st.sidebar.title("About")
st.sidebar.write("This is an AI-powered interface for interacting with a language model.")
 
