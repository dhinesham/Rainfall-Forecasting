
# 🌧️ Rainfall Forecasting for Irrigation Planning using N-BEATS (PyTorch)

## 📌 Overview

This project uses the **N-BEATS deep learning model** (Neural Basis Expansion Analysis for Time Series) to forecast **7-day rainfall** in Bengaluru, India.
The forecasts can be used by farmers and irrigation planners to optimize water usage and agricultural scheduling.

* **Model**: N-BEATS implemented in PyTorch
* **Dataset**: Daily rainfall records (`blr_rain.csv`)
* **Forecast Horizon**: 7 days
* **Goal**: Provide reliable short-term rainfall predictions for smarter irrigation planning.

---

## 🗂️ Project Structure

```
├── blr_rain.csv               # Dataset (Bengaluru daily rainfall)
├── train_nbeats.py            # Training script
├── predict_nbeats.py          # Prediction + plotting script
├── nbeats_blr_rain.pth        # Trained model (saved weights)
├── scaler_blr.save            # Saved MinMaxScaler
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/rainfall-forecast-nbeats.git
cd rainfall-forecast-nbeats
```

### 2. Create Environment (Anaconda recommended)

```bash
conda create -n rainfall python=3.10
conda activate rainfall
```

### 3. Install Dependencies

```bash
# Install GPU-enabled PyTorch (CUDA 12.1 example)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset

* **Source**: Indian Meteorological Department (IMD) / OpenCity dataset for Bengaluru
* **File**: `blr_rain.csv`
* **Columns**:

  * `_id`: Row ID (dropped)
  * `date`: Daily timestamp
  * `Rain`: Daily rainfall (mm) → **target**
  * `Temp Max`, `Temp Min`: Daily temperature (optional features)

### Example (first rows):

| date       | Rain | Temp Max | Temp Min |
| ---------- | ---- | -------- | -------- |
| 1951-01-01 | 0.0  | 28.3     | 17.5     |
| 1951-01-02 | 2.1  | 29.1     | 18.2     |

---

## 🚀 Usage

### 1. Train Model

```bash
python train_nbeats.py
```

* Trains N-BEATS on last 70+ years of Bengaluru rainfall.
* Saves trained model as `nbeats_blr_rain.pth`.
* Saves fitted scaler as `scaler_blr.save`.

---

### 2. Make Forecast

```bash
python predict_nbeats.py
```

* Loads the trained model.
* Predicts rainfall for the **next 7 days**.
* Plots forecast graph.
* Prints forecast values:

```
🌧 Next 7 days rainfall forecast (mm):
2024-09-02: 3.21 mm
2024-09-03: 1.07 mm
...
```

---

## 📈 Example Output Plot

*(screenshot your Matplotlib output and include here)*

```
7-Day Rainfall Forecast (Bengaluru)
```

Orange line = Predicted rainfall for next 7 days.

---

## 🧠 Model Details

* **Architecture**: N-BEATS (generic architecture, stack of fully-connected blocks)
* **Input Window**: 30 days historical rainfall
* **Forecast Horizon**: 7 days
* **Loss Function**: MSE
* **Optimizer**: Adam (lr=0.001)
* **Training Epochs**: 20 (configurable)

---

## 🔮 Future Improvements

* [ ] Add **temperature features** (`Temp Max`, `Temp Min`) for multivariate forecasting.
* [ ] Include **uncertainty estimation** using Monte Carlo Dropout.
* [ ] Train on multiple cities (transfer learning for Indian regions).
* [ ] Deploy as a **Flask/FastAPI web app** for farmers.

---

## 📚 References

* Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2019). *N-BEATS: Neural basis expansion analysis for interpretable time series forecasting*. ICLR.
* [PyTorch Official Docs](https://pytorch.org/)
* [OpenCity IMD Dataset](https://data.opencity.in/)

---

## 👨‍💻 Author

* **Your Name** – *Machine Learning & AI Enthusiast*
* 🌱 Focused on **AI for Agriculture**
* 📫 Reach me at: [your.email@example.com](mailto:your.email@example.com)

---

👉 Do you want me to also create a **`environment.yml` (Conda style)** file alongside `requirements.txt` so anyone can just do `conda env create -f environment.yml` and get everything (including CUDA) in one go?
