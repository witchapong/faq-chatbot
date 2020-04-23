from flask import Flask
from flask_restful import Api, Resource, reqparse
import pandas as pd
import numpy as np
import tensorflow_hub as hub
import tensorflow_text
from pythainlp.tokenize import word_tokenize
from gensim.summarization.bm25 import BM25

qa_df = pd.read_csv('src/covid19_qa.csv')

# USE
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
q_embs = []
for question in qa_df['question'].values:
    q_embs.append(model(question).numpy())

q_embs = np.vstack(q_embs)

# bm25
bm25_scorer = BM25([word_tokenize(sent) for sent in qa_df['question'].values])

class QuestionClassifier(Resource):

    parser = reqparse.RequestParser()
    parser.add_argument('question',
        type=str,
        required=True,
        help="Question cannot be empty."
        )
    parser.add_argument('model',
        type=str,
        required=True,
        help="Model type can be either 'use' or 'bm25'."
        )

    @staticmethod
    def get_intent_use(sentence):
        sent_vec = model(sentence).numpy()
        sim_score = sent_vec @ q_embs.T
        return np.argmax(sim_score)

    @staticmethod
    def get_intent_bm25(sentence):
        tokenized_sent = word_tokenize(sentence)
        scores = bm25_scorer.get_scores(tokenized_sent)
        return np.argmax(scores)

    def get(self):
        payload = self.__class__.parser.parse_args()
        if payload['model'] =='use': q_id = self.get_intent_use(payload['question'])
        elif payload['model'] == 'bm25': q_id = self.get_intent_bm25(payload['question'])
        else: return {'message':"Model type can be either 'use' or 'bm25'."}
        
        matched_question = qa_df.iloc[q_id]['question']
        answer = qa_df.iloc[q_id]['answer']
        return {'matched_question':matched_question,'answer':answer}

def create_app():
    app = Flask(__name__)
    app.config['PROPAGATE_EXCEPTIONS'] = True
    app.secret_key = 'mick'
    return app

def create_api(app):
    api = Api(app)
    api.add_resource(QuestionClassifier, '/question_classifier')

app = create_app()
create_api(app)

if __name__ == "__main__":
    app.run(port=5000, debug=True)