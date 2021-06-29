from fairseq.data.encoders.fastbpe import fastBPE
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fairseq.data import Dictionary
import argparse
import numpy as np

import torch

from flask import Flask
from flask import request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

class Model:

    def __init__(self):
        self.vocab = Dictionary()
        self.model = None
        self.bpe = None
        self.label_map = {
            0: 'Enjoyment',
            1: 'Sadness',
            2: 'Disgust',
            3: 'Anger',
            4: 'Fear',
            5: 'Surprise',
            6: 'Other',
        }

    def load_model(self, model_path, bpe_path, dict_path):
        parser = argparse.ArgumentParser()
        parser.add_argument('--bpe-codes',
                            default=bpe_path,
                            required=False,
                            type=str,
                            help='path to fastBPE BPE'
                            )
        args, unknown = parser.parse_known_args()
        self.model = torch.load(model_path, map_location=torch.device('cpu'))
        self.bpe = fastBPE(args)
        self.vocab.add_from_file(dict_path)

    def predict(self, text):
        subwords = '<s> ' + self.bpe.encode(text) + ' </s>'
        encoded_sent = self.vocab.encode_line(subwords, append_eos=True, add_if_not_exist=False).long().tolist()
        ids = pad_sequences([encoded_sent], maxlen=125, dtype="long", value=0, truncating="post", padding="post")
        mask = [[int(token_id > 0) for token_id in ids[0]]]

        inputs = torch.tensor(ids)
        masks = torch.tensor(mask)

        outputs = self.model(inputs,
                             token_type_ids=None,
                             attention_mask=masks)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label = self.label_map[int(np.argmax(logits[0]))]
        return label


model = Model()


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/api', methods=['GET', 'POST'])
def predict():
    data = request.get_json()
    text = data['fname']
    print(text)
    result = model.predict(text)
    response = jsonify(label=result)
    # response.headers.add("Access-Control-Allow-Origin", "http://localhost:8000/api")
    return response


if __name__ == '__main__':
    bpe_path = "./PhoBERT_base_transformers/bpe.codes"
    model_path = './model/model_trained.m'
    dict_path = "./PhoBERT_base_transformers/dict.txt"
    model.load_model(model_path, bpe_path, dict_path)
    app.run(host='localhost', port=8000)
