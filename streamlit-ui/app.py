import streamlit as st
import requests

st.title("TubeSage - Ask anything about YouTube videos")


# Initialize messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = []


# Function to fetch data from the API
def fetch_youtube_transcript(url):
    # This odd URL is due to the fact that we are running the Streamlit app in a Docker container
    response = requests.post("http://tubesage:5000/v1/transcribe-video", json={"video_url": url})
    if response.status_code == 200:
        return response.json().get("transcription", "")
    else:
        st.error("Failed to fetch transcript")
        return ""


youtube_url = st.text_input("Enter YouTube URL")

if st.button("Fetch Transcript"):
    if youtube_url:
        transcript = fetch_youtube_transcript(youtube_url)
    else:
        st.warning("Please enter a YouTube URL")

# Display chat messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input and API response
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Make a stateless API call to your backend API with the last user message
    response = requests.post("http://tubesage:5000/v1/invoke", json={"input": prompt, "video_url": youtube_url})
    if response.status_code == 200:
        assistant_response = response.json().get("response", "")
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
    else:
        st.error("Failed to fetch response from the API")
