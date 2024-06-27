from flask import Flask, request, jsonify
import openai
from openai import OpenAI
from twilio.twiml.messaging_response import MessagingResponse
import os
from pydub import AudioSegment
import speech_recognition as sr
import requests
from result3 import main
from tempfile import NamedTemporaryFile
from twilio.rest import Client
from requests.auth import HTTPBasicAuth

# Init the Flask App
app = Flask(__name__)

# Initialize the OpenAI API key
# export OPENAI_API_KEY=YOUR API KEY
openai.api_key = os.environ.get("OPENAI_API_KEY")
# For twilio
auth_token = os.environ.get("AUTH_TOKEN")
account_sid = os.environ.get("ACCOUNT_SID")


# Define a function to generate answers using GPT-3
def generate_answer(question):
    client = OpenAI()
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=question + " Please keep the response under 1500 characters",
        max_tokens=1024,
        temperature=0
    )
    

    answer = response.choices[0].text.strip()
    return answer


def transcribe_audio(audio_url):
    # Download the audio file
    audio_file = requests.get(audio_url)
    print("audio url:", audio_url)
    with open("audio.wav", "wb") as f:
        f.write(audio_file.content)
    
    print("here")
    # Convert audio to the correct format if necessary
    audio = AudioSegment.from_file("audio.wav")
    audio.export("audio_converted.wav", format="wav")
    print("here1")

    # Transcribe audio file
    recognizer = sr.Recognizer()
    print("here2")
    audio_file = sr.AudioFile("audio_converted.wav")
    with audio_file as source:
        audio_data = recognizer.record(source)
    text = recognizer.recognize_google(audio_data)
    return text


# Define a route to handle incoming requests
@app.route('/chatgpt', methods=['POST'])
def chatgpt():
    incoming_que = request.values.get('Body', '').lower()
    num_media = int(request.values.get('NumMedia', 0))
    if num_media > 0:
        # If there is media, handle it (assuming it's audio for this example)
        print("User passes in media file")
        media_url = request.values.get('MediaUrl0', '')
        print(media_url)
        #response = requests.get(media_url)
        #response = requests.get(media_url, auth=HTTPBasicAuth(account_sid, auth_token))
        client = Client(account_sid, auth_token)
        response = requests.get(media_url, auth=HTTPBasicAuth(account_sid, auth_token))
        print("Response: ", response)
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(response.content)
            temp_audio_path = temp_audio.name
        val = main(temp_audio_path)
        bot_resp = MessagingResponse()
        msg = bot_resp.message()
        msg.body(val[:1500])
        print("\nBot answer: ", val)
        print("Deleting temp file...")
        os.unlink(temp_audio_path)
        return str(bot_resp)
  
        
    
    print("Question: ", incoming_que)
    # Generate the answer using GPT-3
    answer = generate_answer(incoming_que)
    print("BOT Answer: ", answer)
    bot_resp = MessagingResponse()
    msg = bot_resp.message()
    msg.body(answer[:1500])
    return str(bot_resp)


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5000)

