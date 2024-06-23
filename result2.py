import os
import json
import requests
import time
import sys
#from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
#load_dotenv()
HUME_API_KEY = '3GXYbzLoMI7ykRCiQ8lGkDv85BykVd2jYx12Nhfmvf39hRmN'
BASE_URL = 'https://api.hume.ai/v0'
BATCH_JOBS_URL = f'{BASE_URL}/batch/jobs'

# Initialize OpenAI Client
client = OpenAI()

def start_hume_job(file_path):
    with open(file_path, 'rb') as file:
        files = {'file': file}
        data = {
            'json': json.dumps({
                'models': {
                    'prosody': {
                        'granularity': 'utterance',
                        'identify_speakers': True
                    }
                }
            })
        }
        headers = {'X-Hume-Api-Key': HUME_API_KEY}
        response = requests.post(BATCH_JOBS_URL, files=files, data=data, headers=headers)
    
    if response.status_code == 200:
        return response.json()['job_id']
    else:
        raise Exception(f"Failed to start job: {response.text}")

def get_job_status(job_id):
    headers = {'X-Hume-Api-Key': HUME_API_KEY}
    response = requests.get(f'{BATCH_JOBS_URL}/{job_id}', headers=headers)
    return response.json()['state']['status']

def get_job_predictions(job_id):
    headers = {'X-Hume-Api-Key': HUME_API_KEY}
    response = requests.get(f'{BATCH_JOBS_URL}/{job_id}/predictions', headers=headers)
    return response.json()

def analyze_audio(file_path):
    job_id = start_hume_job(file_path)
    print(f"Started audio analysis job with ID: {job_id}")
    
    while True:
        status = get_job_status(job_id)
        print(f"Audio analysis job status: {status}")
        if status == 'COMPLETED':
            break
        time.sleep(5)
    
    predictions = get_job_predictions(job_id)
    print("Prediction\n", predictions)
    return predictions

def get_top_3_emotions(emotions):
    sorted_emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)
    return sorted_emotions[:3]

def process_audio_data(audio_predictions):
    processed_data = []
    
    audio_chunks = audio_predictions[0]['results']['predictions'][0]['models']['prosody']['grouped_predictions']
    
    for audio_chunk_group in audio_chunks:
        speaker_id = audio_chunk_group['id']
        for audio_chunk in audio_chunk_group['predictions']:
            chunk_data = {
                'speaker_id': speaker_id,
                'transcribed_text': audio_chunk['text'],
                'begin': audio_chunk['time']['begin'],
                'end': audio_chunk['time']['end'],
                'emotions': get_top_3_emotions(audio_chunk['emotions'])
            }
            processed_data.append(chunk_data)
    
    # Sort the processed data by time
    processed_data.sort(key=lambda x: x['begin'])
    
    return processed_data

def write_audio_data_to_file(processed_data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in processed_data:
            f.write(f"Speaker ID: {chunk['speaker_id']}\n")
            f.write(f"Transcribed Text: {chunk['transcribed_text']}\n")
            f.write(f"Time: {chunk['begin']:.2f}s - {chunk['end']:.2f}s\n\n")
            
            f.write("Top 3 Emotions:\n")
            for emotion in chunk['emotions']:
                f.write(f"  - {emotion['name']}: {emotion['score']:.4f}\n")
            f.write("\n" + "-"*50 + "\n\n")

def get_openai_messages(transcript):
    return [{
        "role": "system",
        "content": "You are an executive coach specializing in providing feedback on verbal communication. Your task is to analyze audio transcripts and provide detailed feedback. The feedback should be based on a transcript that includes sentiment analysis of the speaker's voice. Your job is to improve the user's emotional intelligence and communication skills, highlighting patterns and trends the user might have missed. The feedback should be structured to include: 1. A summary of the conversation. 2. Feedback on what was done well. 3. Feedback on areas for improvement. 4. Insights or trends that the user may have missed. Be sure to refer to speakers by proper labels (Speaker 1, Speaker 2, etc.)."
    }, {
        "role": "user",
        "content": f"Analyze the following transcript of a conversation, which includes emotion analysis of the speaker's voice. Provide a summary of the conversation, feedback on what was done well, and feedback on areas for improvement. Do not include raw numbers from the analysis. Here is the transcript: {transcript}. Limit your output to 1500 characters."
    }, {
        "role": "assistant",
        "content": "Ensure that the feedback is constructive and actionable, providing specific examples from the transcript when relevant. Explain what was done well, what areas need improvement, and what insights or trends the speaker may have missed. Focus on actionable feedback that can be easily measured and implemented to improve communication skills."
    }]

def get_feedback(messages):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages)

    return completion.choices[0].message.content

def main(audio_path):
    # Analyze audio and get predictions
    print(f"Analyzing audio file: {audio_path}")
    audio_predictions = analyze_audio(audio_path)

    # Process audio data
    print("Processing audio data...")
    processed_data = process_audio_data(audio_predictions)
    
    # Write processed audio data to a text file
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = f"{base_name}_analysis.txt"
    write_audio_data_to_file(processed_data, output_path)
    print(f"Audio analysis data has been written to {output_path}")

    # Read the audio analysis file
    with open(output_path, 'r', encoding='utf-8') as f:
        audio_transcript = f.read()

    # Send to OpenAI for feedback
    messages = get_openai_messages(audio_transcript)
    feedback = get_feedback(messages)
    print("Feedback from OpenAI:")
    print(feedback)

    # Write the feedback to a file
    feedback_path = f"{base_name}_feedback.txt"
    with open(feedback_path, 'w', encoding='utf-8') as f:
        f.write(feedback)
    print(f"Feedback has been written to {feedback_path}")
    return feedback

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_audio_file>")
        sys.exit(1)
    
    audio_file_path = sys.argv[1]
    if not os.path.exists(audio_file_path):
        print(f"Error: The file {audio_file_path} does not exist.")
        sys.exit(1)
    
    main(audio_file_path)
