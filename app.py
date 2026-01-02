from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Metrics
from sklearn.metrics import accuracy_score, r2_score

app = Flask(__name__)

df_global = None   # ✅ FIX 1

@app.route('/')
def home():
    return render_template('index.html', columns=None)


@app.route('/train', methods=['POST'])
def train():
    global df_global   # ✅ FIX 2

    # =========================
    # CASE 1: CSV UPLOAD
    # =========================
    if 'csv_file' in request.files:
        file = request.files['csv_file']
        df_global = pd.read_csv(file)
        columns = df_global.columns.tolist()

        return render_template(
            'index.html',
            columns=columns
        )

    # =========================
    # CASE 2: MODEL TRAINING
    # =========================
    target = request.form.get('target')
    problem_type = request.form.get('problem_type')

    if df_global is None:
        return "Error: No CSV uploaded"

    if not target:
        return "Error: Target not selected"

    df = df_global   # ✅ USE STORED DATAFRAME

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # MODEL SELECTION
    # =========================
    if problem_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor()
        }

    # =========================
    # STEP 3.5 — TRAIN MODELS
    # =========================
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        if problem_type == "classification":
            score = accuracy_score(y_test, predictions)
        else:
            score = r2_score(y_test, predictions)

        results.append({
            "model": name,
            "score": round(score, 4)
        })

    return render_template(
        'results.html',
        results=results,
        target=target,
        problem_type=problem_type
    )


if __name__ == '__main__':
    app.run(debug=True)
