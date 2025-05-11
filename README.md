# 🧠 Thyroid Cancer Recurrence Prediction

An end-to-end machine learning project that predicts the **likelihood of thyroid cancer recurrence** using deep learning and a Streamlit-based web interface. This system helps in early risk assessment to support medical decision-making.

---

## 📌 Project Overview

Thyroid cancer, though often treatable, has the potential for recurrence. This project aims to develop a predictive model that can assess recurrence risks based on patient demographics and clinical data using AI.

---

## 📸 App View 
| Web View                          | Another Features                 |
| --------------------------------- | -------------------------------- | 
![image](https://github.com/user-attachments/assets/61c00f62-f0ec-493f-8b1b-73c25b91add5)|![image](https://github.com/user-attachments/assets/904a4c14-081e-445e-83c1-f79a20463139)|

---

## FlowChart 
![Flowchart](https://github.com/user-attachments/assets/a245bf68-4471-47b9-95dc-2aff69a8f6c5)

---

## 🎯 Goals

- Train a robust machine learning model to predict thyroid cancer recurrence.
- Address class imbalance and optimize prediction threshold.
- Build an interactive web interface for real-time prediction.
- Help medical professionals with supportive decision-making tools.

---

## 📊 Dataset Overview

The dataset contains anonymized patient records, including:

- **Demographics**: Age, Gender  
- **Medical History**: Smoking, Radiotherapy, Physical Examination  
- **Cancer Details**: TNM Staging, Pathology Type, Risk Classification  
- **Target**: Recurrence (`Yes` or `No`)

The dataset is **imbalanced**, with fewer "Yes" (recurrence) cases.

---

## 🧰 Tech Stack

| Component         | Tools / Libraries                        |
|-------------------|-------------------------------------------|
| Language          | Python                                    |
| Data Processing   | Pandas, NumPy                             |
| ML / DL           | Scikit-learn, TensorFlow / Keras          |
| Imbalance Handling| SMOTE (from `imblearn`), Class Weights    |
| Preprocessing     | ColumnTransformer, StandardScaler, OneHot |
| Deployment        | Streamlit                                 |
| Model Storage     | Pickle (`.pkl`), Keras (`.h5`)            |

---

## ⚙️ Machine Learning Pipeline

1. **Data Cleaning & Encoding**
2. **Scaling Numerical Features**
3. **Handling Class Imbalance** using SMOTE
4. **Model Building** with Keras Sequential API
5. **Model Evaluation** (ROC, AUC, Confusion Matrix)
6. **Threshold Optimization** for improved sensitivity
7. **Saving Artifacts** (`model.h5`, `preprocessor.pkl`, `threshold.pkl`)

---

## 🧪 Model Architecture

```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
````

* **Loss Function**: Binary Crossentropy
* **Optimizer**: Adam
* **Output**: Probability of recurrence (thresholded to Yes/No)

---

## 🌐 Streamlit App (Frontend)

The web app includes:

* Sidebar form to enter patient data
* Predict button to estimate recurrence risk
* Result display with probability and interpretation
* Sections explaining the model and thyroid cancer

To run the app:

```bash
streamlit run app.py
```

---

## 📁 Files & Folders

| File                         | Description                 |
| ---------------------------- | --------------------------- |
| `thyroid_cancer_model.ipynb` | Model training notebook     |
| `app.py`                     | Streamlit web application   |
| `thyroid_model.h5`           | Trained Keras model         |
| `preprocessor.pkl`           | Data preprocessing pipeline |
| `threshold.pkl`              | Custom prediction threshold |

---

## 🚀 Future Improvements

* Use SHAP for model explainability
* Train with more diverse datasets
* Improve UI with additional medical recommendations
  
---

## ✅ Conclusion

This project demonstrates how AI can assist in critical healthcare predictions. Deep learning, combined with effective preprocessing and Streamlit deployment, makes it a valuable tool for doctors and researchers.

---

## 👨‍💻 Author
For queries or collaborations, feel free to connect:  
<p align="center">
  <a href="https://www.linkedin.com/in/aritramukherjeeofficial/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
  <a href="https://x.com/AritraMofficial" target="_blank">
    <img src="https://img.shields.io/badge/Twitter-%231DA1F2.svg?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter">
  </a>
  <a href="https://www.instagram.com/aritramukherjee_official/?__pwa=1" target="_blank">
    <img src="https://img.shields.io/badge/Instagram-%23E4405F.svg?style=for-the-badge&logo=instagram&logoColor=white" alt="Instagram">
  </a>
  <a href="https://leetcode.com/u/aritram_official/" target="_blank">
    <img src="https://img.shields.io/badge/LeetCode-%23FFA116.svg?style=for-the-badge&logo=leetcode&logoColor=white" alt="LeetCode">
  </a>
  <a href="https://github.com/AritraOfficial" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-%23181717.svg?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="https://discord.com/channels/@me" target="_blank">
    <img src="https://img.shields.io/badge/Discord-%237289DA.svg?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
  </a>
  <a href="mailto:aritra.work.official@gmail.com" target="_blank">
    <img src="https://img.shields.io/badge/Email-%23D14836.svg?style=for-the-badge&logo=gmail&logoColor=white" alt="Email">
  </a>
</p>

--- 

## 👨‍⚕️ Disclaimer

This tool is for **educational and research purposes only**. It is **not a replacement for professional medical diagnosis**.
