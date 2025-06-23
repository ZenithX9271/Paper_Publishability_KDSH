# 📘 **Paper Publishability & Conference Prediction Tool**  
*A project for Kharagpur Data Science Hackathon (KDSH) – Jan 2025*

---

## **Project Overview**

This repository contains a complete ML pipeline that:

- 🔹 **Classifies** research papers as either *Publishable* or *Non-Publishable*
- 🔹 **Recommends** top-tier conferences (*CVPR, NeurIPS, EMNLP, TMLR, KDD*) for publishable papers

---

## **Tech Stack & ML Techniques**

- **Language:** Python  
- **Libraries:** `scikit-learn`, `xgboost`, `nltk`, `PyMuPDF`, `joblib`  
- **Preprocessing:**  
  - PDF text extraction using **PyMuPDF**
  - **Auto-labeling** from folder structure  
  - **Lemmatization**, **Tokenization**, and **TF-IDF** vectorization  

- **Models Used:**  
  - `RandomForestClassifier` – for binary classification (*publishable or not*)  
  - `XGBClassifier` – for multiclass conference prediction (*cvpr, neurips, etc.*)

---

## **Folder Structure**

```
.
├── preprocessing.py              ← *Complete PDF & NLP pipeline*
├── publishability_model.py       ← *Binary classifier training*
├── conference_model.py           ← *Multiclass classifier training*
├── results.csv                   ← Final output format
├── models/
│   ├── publishability_rf_model.pkl
│   ├── conference_xgb_model.pkl
├── data/
│   └── Reference/
│       ├── Publishable/
│       │   ├── cvpr/
│       │   ├── neurips/
│       ├── Non-Publishable/
```

---

## **Data Format** `(* Required Folder Naming *)`

- All PDFs must be placed inside the `data/Reference/` directory.
- Use the following folder structure:
```
Reference/
├── Publishable/
│   ├── cvpr/
│   ├── neurips/
│   ├── ...
├── Non-Publishable/
```
- Folder names are used to **auto-generate binary and multiclass labels**

---


## **Sample Output Format – `results.csv`**

| Paper ID | Publishable | Conference | Rationale                                                  |
|----------|-------------|------------|------------------------------------------------------------|
| P001     | 1           | cvpr       | Paper aligns with deep learning and computer vision focus. |
| P002     | 0           | na         | na                                                         |

---

