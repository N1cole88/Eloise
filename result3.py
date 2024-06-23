import requests
import json
import time
import os
#from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
#load_dotenv()

# Initialize OpenAI Client
client = OpenAI()

# Hume AI API credentials
HUME_API_KEY ='3GXYbzLoMI7ykRCiQ8lGkDv85BykVd2jYx12Nhfmvf39hRmN' 

# API endpoints
BASE_URL = 'https://api.hume.ai/v0'
BATCH_JOBS_URL = f'{BASE_URL}/batch/jobs'

def start_job(file_path):
    """Start a new job for audio analysis"""
    with open(file_path, 'rb') as file:
        files = {'file': file}
        data = {
            'json': json.dumps({
                'models': {
                    'prosody': {
                        'granularity': 'utterance',
                        'identify_speakers': True
                    },
                    'language': {
                        'granularity': 'word',
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
    """Check the status of a job"""
    headers = {'X-Hume-Api-Key': HUME_API_KEY}
    response = requests.get(f'{BATCH_JOBS_URL}/{job_id}', headers=headers)
    return response.json()['state']['status']

def get_job_predictions(job_id):
    """Get the predictions for a completed job"""
    headers = {'X-Hume-Api-Key': HUME_API_KEY}
    response = requests.get(f'{BATCH_JOBS_URL}/{job_id}/predictions', headers=headers)
    return response.json()

def extract_emotions_and_transcript(predictions, top_n=3):
    """Extract the text, top N emotions, and timing for each prediction"""
    result = []
    prosody_predictions = predictions['predictions'][0]['models']['prosody']['grouped_predictions']
    
    for speaker_group in prosody_predictions:
        speaker_id = speaker_group['id']
        for pred in speaker_group['predictions']:
            text = pred['text']
            begin_time = pred['time']['begin']
            end_time = pred['time']['end']
            emotions = sorted(pred['emotions'], key=lambda x: x['score'], reverse=True)[:top_n]
            top_emotions = [(emotion['name'], round(emotion['score'], 4)) for emotion in emotions]
            result.append({
                'speaker_id': speaker_id,
                'text': text,
                'begin': begin_time,
                'end': end_time,
                'top_emotions': top_emotions
            })
    
    return result

def write_to_file(results, output_file):
    """Write the extracted results to a text file"""
    with open(output_file, 'w', encoding='utf-8') as file:
        for result in results:
            file.write(f"Speaker ID: {result['speaker_id']}\n")
            file.write(f"Transcribed Text: {result['text']}\n")
            file.write(f"Time: {result['begin']:.2f}s - {result['end']:.2f}s\n\n")
            file.write("Top 3 Emotions:\n")
            for emotion, score in result['top_emotions']:
                file.write(f"  - {emotion}: {score:.4f}\n")
            file.write("\n" + "-"*50 + "\n\n")

def get_feedback(transcript):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": "You are an executive coach specialized in providing feedback to managers. Your task is to analyze conversations and provide detailed feedback. The feedback should be based on a transcript that includes sentiment analysis of the speaker's voice. Your job is to improve the user's emotional intelligence, highlighting interactions and trends the user might have missed. The feedback should be structured to include: 1. A summary of the conversation. 2. Feedback on what was done well. 3. Feedback on what wasn't done well. 4. Insights or trends that the user may have missed. Be sure to refer to speakers by their Speaker ID."
        }, {
            "role": "user",
            "content": f"Please analyze the following transcript of a conversation, which includes sentiment analysis of the speaker's voice. Provide a summary of the conversation, feedback on what was done well, and feedback on what wasn't done well. Here is the transcript: {transcript}"
        }, {
          "role": "assistant",
          "content": "Based on the provided transcript, here is the feedback: 1) Summary of Feedback, 2) Feedback on What Was Done Well, 3) Feedback on What Wasn't Done Well. 4) Trends or insights that the user may have missed. Please ensure that the feedback is constructive and actionable, providing specific examples from the transcript when relevant."
        }])

    return completion.choices[0].message.content

def main(file_path):
    print("Starting emotion analysis...")
    job_id = start_job(file_path)
    print(f"Job started with ID: {job_id}")

    while True:
        status = get_job_status(job_id)
        print(f"Job status: {status}")
        if status == 'COMPLETED':
            break
        time.sleep(5)

    print("Job completed. Fetching predictions...")
    predictions = get_job_predictions(job_id)[0]["results"]
    print("Predictions: \n", predictions)
    emotions_and_transcript = extract_emotions_and_transcript(predictions)

    # Generate output file name
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = f"{base_name}_analysis.txt"

    write_to_file(emotions_and_transcript, output_file)
    print(f"Analysis written to {output_file}")

    # Read the analysis file for feedback
    with open(output_file, 'r', encoding='utf-8') as f:
        analysis_content = f.read()

    print("Generating feedback...")
    feedback = get_feedback(analysis_content)
    return feedback    
    # Write feedback to file
    feedback_file = f"{base_name}_feedback.txt"
    with open(feedback_file, 'w', encoding='utf-8') as f:
        f.write(feedback)
    print(f"Feedback written to {feedback_file}")


