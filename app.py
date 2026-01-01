from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    file = request.files['csv_file']
    target = request.form['target']
    problem_type = request.form['problem_type']

    # For now, just pass values to results page
    return render_template(
        'results.html',
        filename=file.filename,
        target=target,
        problem_type=problem_type
    )

if __name__ == '__main__':
    app.run(debug=True)
