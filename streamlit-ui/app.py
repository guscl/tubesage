import os
import streamlit as st
import requests

# This odd URL is due to the fact that we are running the Streamlit
TUBESAGE_API = os.environ.get("TUBESAGE_API", "http://tubesage:5000")
TUBESAGE_API_KEY = os.environ.get("TUBESAGE_API_KEY", "tubesage_api_key")

st.title(":mage: TubeSage - Ask anything about YouTube videos")

# Initialize messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = []


# Function to fetch data from the API
def fetch_youtube_transcript(url):
    try:
        response = requests.post(
            f"{TUBESAGE_API}/v1/transcribe-video",
            json={"video_url": url},
            headers={"Authorization": f"ApiKey {TUBESAGE_API_KEY}"},
        )
        response.raise_for_status()
        st.session_state.transcript = response.json().get("transcription", "")
        st.info("Video transcribed, you can start chatting with the assistant now!")
        # restart the chat
        st.session_state.messages = []
    except Exception:
        st.session_state.transcript = ""
        st.error(
            "Failed to fetch transcript, this is problably because there isn't a valid transcription for the video"
        )


youtube_url = st.text_input("Enter YouTube URL")

if st.button("Fetch Transcript"):
    if youtube_url:
        fetch_youtube_transcript(youtube_url)
    else:
        st.warning("Please enter a YouTube URL")

# Display chat messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input and API response
if youtube_url and "transcript" in st.session_state and st.session_state.transcript:
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            response = requests.post(
                f"{TUBESAGE_API}/v1/invoke",
                json={"input": prompt, "video_url": youtube_url},
                headers={"Authorization": f"ApiKey {TUBESAGE_API_KEY}"},
            )
            response.raise_for_status()
            assistant_response = response.json().get("response", "")
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch response from the API: {e}")
