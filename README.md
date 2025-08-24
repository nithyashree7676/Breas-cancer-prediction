 Breast Cancer Prediction using Artificial Neural Network (ANN)

 Objective
This project builds an Artificial Neural Network (ANN) to predict whether a breast tumor is *malignant* (cancerous) or *benign* (non-cancerous).  
The goal is to assist in early detection of breast cancer using machine learning techniques.


 Dataset
- *Source*: [Breast Cancer Wisconsin (Diagnostic) Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset) (built into scikit-learn).  
- *Records*: 569 patient samples  
- *Features*: 30 numeric tumor characteristics (mean radius, texture, perimeter, smoothness, etc.)  
- *Target classes*:  
  - 0 = Malignant  
  - 1 = Benign  


 Preprocessing
- Data split into *80% training* and *20% testing*  
- Features normalized using *StandardScaler*  
- Labels encoded as binary (0/1)


 Model Architecture
- *Input Layer*: 30 features  
- *Hidden Layer 1*: Dense(32, ReLU) + BatchNormalization + Dropout(0.3)  
- *Hidden Layer 2*: Dense(16, ReLU)  
- *Output Layer*: Dense(1, Sigmoid)  


 Training
- Optimizer: *Adam*  
- Loss Function: *Binary Crossentropy*  
- Epochs: *50*  
- Batch Size: *32*  
- Early stopping and learning rate reduction applied to avoid overfitting
  

 Results
- *Accuracy*: ~96–97%  
- *ROC-AUC Score*: ~0.99  
- *Confusion Matrix*: Very few false negatives (high sensitivity)  


 Tools & Libraries
- Python  
- NumPy, Pandas  
- Scikit-learn  
- TensorFlow / Keras  
- Matplotlib, Seaborn  


 Sample Results
- *Model Accuracy Curve*  
- *Loss Curve*  
- **ROC Curve (AUC ~
