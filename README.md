# Violence Detection using LSTM & YOLO-Pose

> A Deep Learning pipeline to automatically detect fighting and violent behaviors in video streams using Pose Estimation and Long Short-Term Memory (LSTM) neural networks.

## Tech Stack

![Python](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white) ![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)

## 🧠 Pipeline Architecture

<img width="1429" height="1389" alt="diagramma_pipeline" src="https://github.com/user-attachments/assets/0652b110-4e26-4b6a-8104-94e04f3f2169" />

As shown in the diagram, the project is structured in a sequential pipeline:

* **1. Pose Estimation (`1_estrai_keypoints.py`)**: Uses YOLO-Pose to extract human pose landmarks (skeletons) from raw video frames.
* **2. LSTM Training (`2_addestra_lstm_v2_modello_ottimizzato.py`)**: Trains an optimized LSTM network on sequences of 45 frames to capture the temporal dynamics of the fight.
* **3. Inference (`3_inferenza.py`)**: Runs the trained model on new video streams in real-time.

## Requirements

**Python version:** 3.14.7

```bashh
python -m venv env_tesi
source env_tesi/bin/activate   # on Linux/macOS
pip install -r requirements.txt
```
Dataset not included due to size/licensing. See:
- [RWF-2000](https://www.kaggle.com/datasets/vulamnguyen/rwf2000?select=RWF-2000)
- [Real Life Violence Situations](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)
- [Hockey Fight Videos](https://www.kaggle.com/datasets/yassershrief/hockey-fight-vidoes)

## 🚀 How to Run

### 1. Extract keypoints from your dataset
```bash
python 1_estrai_keypoints.py
```
> Make sure to set the correct input/output paths in the script before running.

### 2. Train the LSTM model
```bash
python 2_addestra_lstm_v2_modello_ottimizzato.py
```
> Update the `CARTELLA_INPUT` input folder.

### 3. Test the model on new data
```bash
python 3_inferenza.py [-h] (--video VIDEO | --webcam | --test TEST) [--no-save]
```

## Preview

<img width="1256" height="706" alt="situazione_normale1" src="https://github.com/user-attachments/assets/c07ff2f8-7b59-440e-83da-f8747d08243a" />

<img width="1253" height="701" alt="situazione_normale2" src="https://github.com/user-attachments/assets/6f08de55-8854-40fe-a0eb-cee3b88cc7e0" />

<img width="1250" height="707" alt="rissa1" src="https://github.com/user-attachments/assets/66708add-b042-493b-b1ad-c50735fb74f0" />

> Note: diagram labels are in Italian (original thesis language).


