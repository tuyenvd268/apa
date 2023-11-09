from flask import Flask, request, jsonify
import soundfile as sf
import numpy as np
import time
import yaml
import base64
import io

from align import Aligner
from gop import GOP

app = Flask(__name__)

def load_config(path):
    config_fh = open(path, "r")
    config_dict = yaml.safe_load(config_fh)
    return config_dict

@app.route('/gop', methods=['POST'])
def gop_endpoint():
    """
        api endpoint
    """
    transcript = request.form.get('transcript')
    wavefile = request.files.get('wav')

    waveform, samplerate = sf.read(io.BytesIO(wavefile.read()))
    sf.write(configs["audio-path"], waveform, samplerate=samplerate)

    alignment = aligner.run(
        wav_path=configs["audio-path"], 
        text=transcript)
    
    gop_features, phones = gop_recipe.run(
        align_path=configs["align-path"], 
        prob_path=configs["prob-path"], 
    )

    processed_alignments, processed_phones, gop_features = post_process(alignment, gop_features, phones)
    
    result = {
        "phonemes": processed_phones,
        "gop_features": gop_features,
        "alignments": processed_alignments
    }
    return jsonify(result)

def post_process(alignment, gop_features, phones):
    processed_alignment, processed_phone = [], []
    processed_gop_features = []

    phones = phones.strip().split()
    for index in range(len(phones)):
        assert phones[index] == alignment[index][0]
        if phones[index] == "SIL":
            continue

        temp_alignment = list(alignment[index])
        temp_alignment[0] = temp_alignment[0].split("_")[0]

        processed_alignment.append(temp_alignment)
        processed_phone.append(phones[index].split("_")[0])
        processed_gop_features.append(gop_features[index])
    
    processed_gop_features = np.stack(processed_gop_features, axis=0)

    assert len(processed_alignment) == len(processed_phone)
    assert len(processed_alignment) == processed_gop_features.shape[0]

    # gop_features = base64.b64encode(gop_features.tobytes()).decode('utf-8')
    processed_alignment=processed_alignment
    processed_phone = processed_phone
    processed_gop_features = processed_gop_features.tolist()

    return processed_alignment, processed_phone, processed_gop_features


if __name__ == "__main__":
    configs = load_config("configs/config.yaml")

    aligner = Aligner(configs=configs)
    gop_recipe = GOP(configs=configs)

    app.run(host="0.0.0.0", debug=False, port=6868)