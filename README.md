# **Capstone Project**

## **Objective**  
This project aims to assist individuals with motor impairments by analyzing their brain's EEG signals to predict intended body movements.  

## **Dataset**  
The dataset has been sourced from [BBCI IV Competition](https://www.bbci.de/competition/iv/).  
- Multiple datasets are available, varying by the number of electrodes used in the EEG skull cap.  
- This repository provides reference data for a **22-channel configuration**.  

![image](https://github.com/user-attachments/assets/174da9c0-db14-4956-bf71-730e0e7a7091)


## **Data Preprocessing**  
1. **File Conversion:**  
   - EEG Lab (MATLAB tool) was used to convert `.gdf` files into `.csv` format for efficient processing.  
   - The columns represent the 22 electrodes placed across the skull, and the rows indicate voltage readings over time.

2. **Labeling:**  
   - Training datasets were labeled based on `EVENT.TYPE` values:  
     - **1:** Left Arm  
     - **2:** Right Arm  
     - **3:** Foot  
     - **4:** Tongue  

3. **Cleaning & Baseline Correction:**  
   - Removed rows with `0` or `NaN` values.  
   - Applied **baseline correction** using EEG Lab.  

4. **Bandpass Filtering:**  
   - Applied a bandpass filter (10–30 Hz) to focus on brain activity related to motor tasks.  

![image](https://github.com/user-attachments/assets/69651634-9c7c-4efc-96e9-c6a1898c70bd)

5. **Smoothing with Savitzky-Golay Filter:**  
   - **Why:** To reduce noise while preserving signal features like peaks and edges.  
   - **Steps:**  
     1. Identified optimal window sizes for critical channels (C3, Cz, C4).  
     2. Selected the window with minimum log dispersion.  
     3. Applied the Savitzky-Golay filter with a polynomial order of 3.  

![image](https://github.com/user-attachments/assets/787f7795-ff33-48ad-bab3-7d04fbf6af0c)

6. **Normalization & Standardization:**  
   - Applied **z-score standardization** and **min-max normalization** for feature scaling.  

![image](https://github.com/user-attachments/assets/1341f45b-e2fb-4d0a-a878-d9e3b8ce38ac)


## **Feature Extraction**  

### **Temporal Features**  
- The preprocessed data was divided into **blocks of 500 rows** based on ground truth labels (1–4).  
- Temporal features were extracted using **LSTM with skip connections**.

### **Spatial Features**  
- Spatial features were extracted using **Residual Connections** from blocks of 500 rows.  
- Irrelevant features with excessive zero values were removed.  

**Architecture Diagram:**  

![cnn_model_architecture](https://github.com/user-attachments/assets/f2d786e3-e0ec-4362-9a7f-75ea88e3a2e7)

---

## **Fusion**  
- Combined **temporal** and **spatial features** into two mixed files:  
  - **Temporal_Mixed**  
  - **Spatial_Mixed**  
- Final fusion was achieved using **Feature Pyramid Fusion**, followed by the removal of irrelevant columns.

**Architecture Diagram:**  

![model_architecture](https://github.com/user-attachments/assets/125ff670-60e6-448b-90d4-f9b49ee5634c)

---

## **Accuracy**  
- Used an **Inception-based CNN** with 5-fold cross-validation for classification.  
- Achieved an accuracy of **80%** on the 22-channel dataset.  
- The model utilizes Inception modules for feature extraction and reports average accuracy and loss.  

![image](https://github.com/user-attachments/assets/f98d0dd1-bd75-4fe0-8275-a57a8f962354)  
![image](https://github.com/user-attachments/assets/6834231b-2900-4598-9772-a2ded60ac9e8)
