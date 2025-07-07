# Heart-Disease-Risk-Predictor

# ❤️ Heart Disease Risk Predictor

This is a simple **Machine Learning web app** built with **Streamlit** that predicts the risk of heart disease based on user-provided clinical data.

> ⚠️ **Disclaimer:** This tool is for **educational purposes only**. It is **not a medical diagnostic tool**.

---

## 🚀 Features

✅ Predicts risk of heart disease using a **Random Forest Classifier**  
✅ Interactive sliders and dropdowns for inputting clinical details  
✅ Displays model accuracy  
✅ Visualizes data correlations with a heatmap  
✅ Runs locally in your browser using **Streamlit**

---

## 📊 **Dataset**

This project uses the **`heart_cleveland_upload.csv`** file, which is derived from the **Cleveland Heart Disease dataset** available on **[Kaggle](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)**.

---

## 🧩 **Technologies Used**

- **Python**
- **Pandas** & **NumPy**
- **Scikit-learn** (Random Forest Classifier)
- **Seaborn** & **Matplotlib** (for data visualization)
- **Streamlit** (for the web app)

---

## 📁 **Project Structure**

```plaintext
📂 Heart Disease Risk Predictor/
 ├── app.py
 ├── heart_cleveland_upload.csv
 ├── requirements.txt
 └── README.md
```

## ⚙️ **How to Run**
1️⃣ Clone this repo:
- git clone https://github.com/yourusername/heart-disease-risk-predictor.git
- cd heart-disease-risk-predictor

2️⃣ Create a virtual environment (optional but recommended):
- python -m venv venv
- source venv/bin/activate  # Linux/macOS
# OR
- venv\Scripts\activate     # Windows

3️⃣ Install dependencies:
- pip install -r requirements.txt

4️⃣ Run the app:
- streamlit run app.py

5️⃣ Open your browser → http://localhost:8501

## 📌 **How it works**
- Loads the Cleveland Heart Disease dataset (heart_cleveland_upload.csv).

- Trains a Random Forest Classifier on the data.

- Takes user input for clinical features like age, blood pressure, cholesterol, etc.

- Predicts risk as Low Risk or High Risk.

- Displays the prediction probability.

- Optionally shows raw data & correlation heatmap.
