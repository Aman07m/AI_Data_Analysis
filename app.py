# ---------- IMPORTANT FOR WINDOWS ----------
import matplotlib
matplotlib.use("Agg")

# ---------- IMPORTS ----------
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import os

# ---------- OPTIONAL AI (SUMMARY ONLY) ----------
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
USE_AI = os.getenv("USE_AI", "false").lower() == "true"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq = Groq(api_key=GROQ_API_KEY) if USE_AI and GROQ_API_KEY else None

# ---------- APP ----------
app = Flask(__name__)
CORS(app)

# ---------- COLOR PALETTE ----------
COLORS = [
    "#4F46E5", "#22C55E", "#F97316",
    "#EF4444", "#06B6D4", "#A855F7"
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Colorful AI-Assisted EDA Engine 🚀"})


# ---------- HELPERS ----------
def numeric_cols(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_cols(df):
    return df.select_dtypes(exclude=np.number).columns.tolist()


def has_variance(series):
    return series.nunique() > 1 and series.var() > 0


def rank_numeric(df, cols):
    return sorted(cols, key=lambda c: df[c].var(), reverse=True)


def rank_categorical(df, cols):
    return sorted(cols, key=lambda c: df[c].nunique())


# ---------- AI SUMMARY (SAFE) ----------
def ai_summary(profile, insights):
    if not groq:
        return None

    prompt = f"""
    You are a data analyst.

    Dataset profile:
    {profile}

    Key insights from charts:
    {insights}

    Write a concise narrative summary (5–6 lines) explaining
    the overall story of the dataset in simple language.
    """

    response = groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


# ---------- MAIN API ----------
@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # ---------- LOAD DATA ----------
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Invalid file type"}), 400

        # ---------- COLUMN PROFILING ----------
        num_cols = rank_numeric(df, numeric_cols(df))
        cat_cols = rank_categorical(df, categorical_cols(df))

        graphs = []
        insights = []

        # =====================================================
        # 1️⃣ ONE TOP-10 GRAPH (BEST AVAILABLE)
        # =====================================================
        if cat_cols and 2 <= df[cat_cols[0]].nunique() <= 50:
            graphs.append(("top10_cat", cat_cols[0]))
            insights.append(f"Top categories dominate '{cat_cols[0]}'.")
        elif num_cols and has_variance(df[num_cols[0]]):
            graphs.append(("top10_num", num_cols[0]))
            insights.append(f"Top values highlight extremes in '{num_cols[0]}'.")

        # =====================================================
        # 2️⃣ NUMERIC DISTRIBUTIONS (UP TO 2 COLUMNS)
        # =====================================================
        for col in num_cols:
            if has_variance(df[col]):
                graphs.append(("hist", col))
                insights.append(f"Distribution of '{col}' shows spread and skewness.")
            if len([g for g in graphs if g[0] == "hist"]) == 2:
                break

        # =====================================================
        # 3️⃣ NUMERIC RELATIONSHIP (ONLY IF CORRELATED)
        # =====================================================
        if len(num_cols) >= 2:
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    corr = df[num_cols[i]].corr(df[num_cols[j]])
                    if corr is not None and abs(corr) >= 0.3:
                        graphs.append(("scatter", num_cols[i], num_cols[j]))
                        insights.append(
                            f"'{num_cols[i]}' and '{num_cols[j]}' show correlation (≈ {corr:.2f})."
                        )
                        break
                if any(g[0] == "scatter" for g in graphs):
                    break

        # =====================================================
        # LIMIT TO MAX 6 GRAPHS
        # =====================================================
        graphs = graphs[:6]
        insights = insights[:6]

        if not graphs:
            return jsonify({
                "image": None,
                "insights": ["No meaningful graphs could be generated from this dataset."]
            })

        # =====================================================
        # PLOTTING
        # =====================================================
        fig, axes = plt.subplots(len(graphs), 1, figsize=(12, 4 * len(graphs)))
        if len(graphs) == 1:
            axes = [axes]

        for idx, (ax, g) in enumerate(zip(axes, graphs)):
            color = COLORS[idx % len(COLORS)]

            if g[0] == "top10_cat":
                counts = df[g[1]].value_counts().head(10)
                ax.bar(counts.index.astype(str), counts.values, color=color)
                ax.set_title(f"Top 10 Categories of {g[1]}")
                ax.set_xlabel(g[1])
                ax.set_ylabel("Count")
                ax.tick_params(axis="x", rotation=45)

            elif g[0] == "top10_num":
                top = df[g[1]].sort_values(ascending=False).head(10)
                ax.bar(top.index.astype(str), top.values, color=color)
                ax.set_title(f"Top 10 Highest Values of {g[1]}")
                ax.set_xlabel("Record Index")
                ax.set_ylabel(g[1])

            elif g[0] == "hist":
                ax.hist(df[g[1]], bins=20, color=color)
                ax.set_title(f"Distribution of {g[1]}")
                ax.set_xlabel(g[1])
                ax.set_ylabel("Frequency")

            elif g[0] == "scatter":
                ax.scatter(df[g[1]], df[g[2]], color=color)
                ax.set_title(f"{g[1]} vs {g[2]}")
                ax.set_xlabel(g[1])
                ax.set_ylabel(g[2])

        plt.tight_layout()

        # =====================================================
        # IMAGE CONVERSION
        # =====================================================
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close()

        image_base64 = base64.b64encode(buffer.read()).decode()

        # =====================================================
        # DATASET PROFILE
        # =====================================================
        profile = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "numeric_columns": len(num_cols),
            "categorical_columns": len(cat_cols),
            "missing_values": int(df.isnull().sum().sum()),
            "column_names": df.columns.tolist()
        }

        # =====================================================
        # AI SUMMARY (SAFE FALLBACK)
        # =====================================================
        try:
            summary = ai_summary(profile, insights) if USE_AI else None
        except Exception as e:
            print("AI SUMMARY FAILED:", e)
            summary = None

        if not summary:
            summary = (
                "Automated exploratory data analysis was performed. "
                "The charts highlight dominant categories, key distributions, "
                "and meaningful relationships within the dataset."
            )

        return jsonify({
            "image": image_base64,
            "insights": insights,
            "summary": summary,
            "profile": profile
        })

    except Exception as e:
        print("BACKEND ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run()
