# 🚀 SkillSync AI — Interview Trainer Agent

<p align="center">
  <b>AICTE × IBM Edunet Internship | Problem Statement #22</b>
</p>

<p align="center">
  <a href="https://skillsync-ai-interview-trainer.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/AI-IBM%20Granite-blue" alt="IBM Granite">
  <img src="https://img.shields.io/badge/Status-Deployed-success" alt="Status">
</p>

**SkillSync AI** is an AI-powered interview preparation platform that analyzes resumes, generates personalized interview questions, evaluates answers, and identifies skill gaps.

## ✨ Features

* 📄 **Resume Analysis** – Extract skills, education, and experience from PDF/DOCX.
* 🤖 **AI Question Generation** – Personalized questions based on resume and job role.
* 🔍 **RAG Pipeline** – Context retrieval using FAISS.
* 🎤 **Mock Interviews** – Interactive interview practice with feedback.
* 📊 **Performance Dashboard** – Scores and visual analytics using Plotly.
* 🎯 **Skill Gap Analysis** – Identify missing skills and improvement areas.
* 📂 **Dataset Analysis** – Benchmarking using 2,385 records.
* 🛡️ **Demo Mode** – Explore application workflows without IBM credentials.

## 🛠️ Tech Stack

`Python` · `Streamlit` · `IBM Granite` · `FAISS` · `Sentence Transformers` · `Pandas` · `Scikit-learn` · `Plotly`

## 📸 Application Preview

*Screenshots coming soon.*

## ⚙️ Run Locally

```bash
git clone https://github.com/akshkantiwal6/SkillSync-AI.git
cd SkillSync-AI

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

For live IBM Granite integration, configure `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` as environment variables. Without credentials, the app runs in Demo Mode with predefined responses.

## 📁 Project Structure

```text
SkillSync-AI/
├── app.py
├── config.py
├── requirements.txt
├── data/
├── knowledge_base/
└── modules/
```

## 🎯 Objective

Making interview preparation smarter through resume intelligence, AI-assisted practice, performance evaluation, and personalized skill development.

---

<p align="center">
  <b>Built with ❤️ for AICTE × IBM Edunet Internship</b>
</p>
