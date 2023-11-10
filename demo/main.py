from flask import (
    Flask, request, jsonify
)
import pandas as pd
import requests
import json
import io
import re

from src.arpa_to_ipa import arpa_to_ipa
import soundfile as sf

app = Flask(__name__)

def run_force_alignment_and_gop(transcript, wav_path):
    url = "http://14.162.145.55:8888/gop"

    payload = {'transcript': transcript}
    
    files=[
        ('wav',('test.wav',open(wav_path,'rb'),'audio/wav'))
    ]
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    
    return response.json()

def run_scoring(inputs):
    url = "http://14.162.145.55:6868/scoring"

    payload = json.dumps(inputs)

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.json()

def get_alignment(inputs):
    alignments = pd.DataFrame(
        inputs["alignments"], columns=["phone", "start_frame", "duration"])
    
    alignments["word_ids"] = inputs["word_ids"]
    alignments["end_frame"] = alignments["start_frame"] + alignments["duration"]

    return alignments

def get_word_scores(metadata):
    # metadata["raw_word_scores"] = metadata["word_scores"]
    for name, group in metadata.groupby("word_ids"):
        word_score = round(group["word_scores"].mean(), 2)
        for index in group.index:
            metadata.loc[index, "word_scores"] = word_score

    return metadata

def get_ipa_from_arpabet(metadata):
    metadata["ipa"] = metadata.phone.apply(arpa_to_ipa)

    return metadata

def normalize(text):
    text = re.sub(r'[\!@#$%^&*\(\)\\\.\"\,\?\;\:\+\-\_\/\|~`]', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.upper().strip()
    return text

def post_process(metadata, transcript):
    sentence = {
        "text": transcript,
        "score": round(metadata["utterance_scores"].mean(), 2),
        "duration": 0,
        "ipa": "",
        "words": []
    }

    words = transcript.split()
    for word_id, word in enumerate(words):
        phonemes = metadata[metadata.word_ids == word_id]
        tmp_word = {
            "text": word,
            "score": metadata[metadata.word_ids == word_id].word_scores.mean(),
            "arpabet": "",
            "ipa": "",
            "phonemes": [],
            "start_index": 0,
            "end_index": 0,
            "start_time": 0,
            "end_time": 0,
        }
        for index in phonemes.index:
            phone = {
                "arpabet": phonemes["phone"][index],
                "score": round(phonemes["phone_scores"][index], 2),
                "ipa": phonemes["ipa"][index],
                "start_index": 0,
                "end_index": 0,
                "start_time": phonemes["start_frame"][index]*0.01,
                "end_time": phonemes["end_frame"][index]*0.01,
                "sound_most_like": phonemes["phone"][index],
            }

            tmp_word["phonemes"].append(phone)

        tmp_word["start_time"] = tmp_word["phonemes"][0]["start_time"]
        tmp_word["end_time"] = tmp_word["phonemes"][-1]["end_time"]

        sentence["words"].append(tmp_word)

    return {
        "version": "v0.1.0",
        "utterance": [sentence, ]
    }

def run(transcript, wav_path):
    step_1_result = run_force_alignment_and_gop(
        transcript=transcript,
        wav_path=wav_path) 
    alignments = get_alignment(step_1_result.copy())

    step_2_result = run_scoring(inputs=step_1_result.copy()) 
    scores = pd.DataFrame(step_2_result)

    metadata = pd.concat([alignments, scores], axis=1)
    metadata = get_word_scores(metadata)
    metadata = get_ipa_from_arpabet(metadata)

    result = post_process(metadata, transcript)

    return result

@app.route('/pronunciation_scoring', methods=['POST'])
def scoring_endpoint():
    transcript = request.form.get('transcript')
    wavefile = request.files.get('wav')

    transcript = normalize(transcript)
    wav_path = "wav/test.wav"
    waveform, samplerate = sf.read(io.BytesIO(wavefile.read()))
    sf.write(wav_path, waveform, samplerate=samplerate)
    
    result = run(
        transcript=transcript,
        wav_path=wav_path
    )

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=9999)
