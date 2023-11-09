from flask import Flask, request, jsonify
import soundfile as sf
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

@app.route('/pronunciation_scoring', methods=['POST'])
def infer_endpoint():
    """
        api endpoint
    """
    transcript = request.form.get('transcript')
    wavefile = request.files.get('wav')

    waveform, samplerate = sf.read(io.BytesIO(wavefile.read()))
    sf.write(configs["audio-path"], waveform, samplerate=samplerate)

    start = time.time()
    aligner.run(
        wav_path=configs["audio-path"], 
        text=transcript)
    end = time.time()

    print("duration: ", end-start)
    
    gop_dict = gop_recipe.run(
        align_path=configs["align-path"], 
        prob_path=configs["prob-path"], 
    )

    return jsonify(gop_dict)

if __name__ == "__main__":
    configs = load_config("config.yaml")

    aligner = Aligner(configs=configs)
    gop_recipe = GOP(configs=configs)

    app.run(host="0.0.0.0", debug=False, port=6868)