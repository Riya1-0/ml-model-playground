from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', columns=None)

@app.route('/train', methods=['POST'])
def train():
    # Case 1: CSV uploaded
    if 'csv_file' in request.files:
        file = request.files['csv_file']
        df = pd.read_csv(file)
        columns = df.columns.tolist()

        return render_template(
            'index.html',
            columns=columns
        )

    # Case 2: Target selected
    target = request.form.get('target')
    problem_type = request.form.get('problem_type')

    if not target:
        return "Error: Target not selected"

    return render_template(
        'results.html',
        target=target,
        problem_type=problem_type
    )

if __name__ == '__main__':
    app.run(debug=True)
