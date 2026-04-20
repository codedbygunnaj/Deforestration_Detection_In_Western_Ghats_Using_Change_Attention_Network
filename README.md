# 🌍 Deforestation Detection using Siamese Deep Learning

## 🚀 Overview

This project implements an automated **deforestation detection system** for the Western Ghats (India) using **multi-temporal satellite imagery** and deep learning.

The system compares Sentinel-2 images from two different time periods and detects areas where forest loss has occurred using a **Siamese Convolutional Neural Network**.

---

## 🧠 Key Features

* 📡 Satellite Data: Sentinel-2 (multi-spectral imagery)
* 🌳 Ground Truth: Hansen Global Forest Change Dataset
* 🧹 Preprocessing:

  * Cloud masking
  * Water masking
  * Terrain filtering (slope)
* 🧩 Model:

  * Siamese UNet (Change Detection)
  * Feature difference learning
* 📊 Output:

  * Binary deforestation maps
  * Change probability maps

---

## ⚙️ Pipeline

```text
Google Earth Engine → GeoTIFF Tiles → PyTorch Dataset → Siamese Model → Deforestation Map
```

---

## 🗂️ Project Structure

```text
.
├── minor2.ipynb        # Main Colab notebook (end-to-end pipeline)
├── README.md
├── requirements.txt
```

---

## 📦 Dataset

Due to large file sizes, the dataset is **not included** in this repository.

### 📌 How data is generated:

* Tiles are exported from **Google Earth Engine (GEE)**
* Each tile contains:

  * 6 bands (T1 - past)
  * 6 bands (T2 - recent)
  * 1 band (deforestation label)

---

## ▶️ How to Run

### 1️⃣ Open in Google Colab

Upload or open `minor2.ipynb` in Colab.

### 2️⃣ Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3️⃣ Set dataset path

```python
path = "/content/drive/MyDrive/minor2/data/minor_wGhats_tiles"
```

### 4️⃣ Run all cells 🚀

---

## 📊 Sample Results

### 🌿 Input (T1 vs T2)

* Before and after satellite imagery

### 🔴 Output

* White regions indicate deforestation

(Add your screenshots here)

---

## ⚠️ Notes

* Dataset is generated using GEE and stored on Google Drive
* NaN values are handled during preprocessing
* Class imbalance handled via weighted loss

---

## 🧪 Future Improvements

* Add CBAM attention module (Change Attention Network)
* Improve model architecture (FPN decoder)
* Expand dataset coverage
* Deploy as web dashboard

---

## 🧑‍💻 Author

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
