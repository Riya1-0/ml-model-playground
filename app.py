from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import os

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Metrics
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import numpy as np


app = Flask(__name__)

df_global = None   # ✅ FIX 1

@app.route('/')
def home():
    return render_template('index.html', columns=None)


@app.route('/train', methods=['POST'])
def train():
    global df_global  

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
   

    # Encode categorical feature columns
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    feature_names = X.columns.tolist()

    # Handle missing values (NaN)
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    X = pd.DataFrame(X, columns=feature_names)


    y = df[target]
    if problem_type == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y)


    if problem_type == "classification" and len(np.unique(y)) > 20:
        return "Error: Selected target is continuous. Please choose Regression."


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


    best_model_obj = None

    # =========================
    # STEP 3.5 — TRAIN MODELS
    # =========================
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        if problem_type == "classification":
            acc = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='weighted')

            results.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "F1": round(f1, 4),
                "model_obj": model
            })

        else:
            rmse = np.sqrt(mean_squared_error(y_test, predictions))

            results.append({
                "Model": name,
                "RMSE": round(rmse, 4),
                "model_obj": model
            })

    if problem_type == "classification":
        best_row = max(results, key=lambda x: x["Accuracy"])
    else:
        best_row = min(results, key=lambda x: x["RMSE"])

    best_model = best_row["Model"]
    best_model_obj = best_row["model_obj"]

    if problem_type == "classification":
        chart_scores = [row["Accuracy"] for row in results]
    else:
        chart_scores = [row["RMSE"] for row in results]



    # STEP 5.3: Feature Importance

    explanation = "Explainability not available for this model."
    plot_path = None

    if hasattr(best_model_obj, "feature_importances_"):

        importances = best_model_obj.feature_importances_

        # 🔒 SAFETY CHECK
        if len(importances) == len(feature_names):

            feature_importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            top_feature = feature_importance_df.iloc[0]["Feature"]
            explanation = f"The model mainly depends on {top_feature} to make predictions."

            # ===== PLOT =====
            plt.figure(figsize=(6, 4))
            plt.barh(
                feature_importance_df["Feature"],
                feature_importance_df["Importance"]
            )
            plt.xlabel("Importance")
            plt.title("Feature Importance")
            plt.gca().invert_yaxis()

            plot_path = "static/feature_importance.png"
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()


        
    return render_template(
        'results.html',
        results=results,
        explanation=explanation,
        best_model=best_model,
        target=target,
        plot_path=plot_path,
        chart_scores=chart_scores,
        problem_type=problem_type
    )


if __name__ == '__main__':
    app.run(debug=True)
