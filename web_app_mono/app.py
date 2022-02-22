import flask
import pickle
import pandas as pd
import requests
import joblib

# Use pickle to load in the pre-trained model.
with open(f'model/svc_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        narration = flask.request.form['narration']
        input_variables = pd.DataFrame([[narration]],
                                       columns=['narration'],
                                       dtype=object)
        prediction = model.predict(input_variables)[0]
        return flask.render_template('main.html',
                                     original_input={'narration':narration},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run()