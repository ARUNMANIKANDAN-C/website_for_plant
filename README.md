
# 🌿 Plant Disease Classifier API

This is a web-based application built using **FastAPI** that allows users to upload plant leaf images and receive predictions on possible plant diseases using a pre-trained TensorFlow model.

![FastAPI](https://img.shields.io/badge/Built%20With-FastAPI-009688?style=flat-square&logo=fastapi)
![TensorFlow](https://img.shields.io/badge/Powered%20By-TensorFlow-FF6F00?style=flat-square&logo=tensorflow)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🚀 Features

- Upload plant leaf images to detect diseases.
- Predicts from 38 different plant disease classes.
- Web frontend with HTML upload interface.
- Clean API response with class name, confidence, and full raw scores.

---

## 🧠 Model

The model is a TensorFlow/Keras `.h5` file trained on a dataset such as **PlantVillage**. It expects images of shape **(224, 224, 3)**.

> ✅ You must place your model file as `model.h5` in the project root.

---

## 📂 Project Structure

```
project/
├───app
│   ├───static
│   │   ├───img
│   │   └───uploads
│   ├───templates
│   └───__pycache__
└───test
```

---

## ⚙️ Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/plant-disease-classifier.git
   cd plant-disease-classifier
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   uvicorn app.main:app --reload
   ```

5. **Open in browser**
   Visit [http://localhost:8000](http://localhost:8000)

---

## 🧪 API Endpoint

### `/check_image` [POST]

**Description:** Upload an image file and receive prediction.

- **Payload:** Multipart form with an image file
- **Response:**
```json
{
  "result": "Tomato early blight",
  "confidence": "0.94",
  "raw_prediction": [...],
  "predicted_index": 30,
  "predicted_class": "Tomato_early_blight"
}
```

---

## 📋 Example Class Names

Some of the plant disease classes include:

- Apple scab
- Blueberry healthy
- Corn gray leaf spot
- Potato late blight
- Tomato mosaic virus
- Strawberry leaf scorch
- ...and more!

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgments

- [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- [FastAPI](https://fastapi.tiangolo.com/)
- [TensorFlow](https://www.tensorflow.org/)

---
