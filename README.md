# Unsupervised Anomaly Detection using Autoencoders for Predictive Maintenance

This project applies deep learning for unsupervised anomaly detection on machine vibration signals, simulating predictive maintenance in industrial systems. The main method is a 1D autoencoder trained on the NASA IMS Bearing dataset to identify abnormal behavior without labeled data.

## Project Workflow
1. **Data Loading & Preprocessing**
   - NASA IMS Bearing Dataset (Set 1) is used, containing high-frequency vibration data from four bearings.
   - Data is windowed into fixed-size segments (100 samples per window) and normalized to zero mean/unit variance.
2. **Model Architecture**
   - A 1D dense autoencoder is built using TensorFlow/Keras.
   - The encoder compresses input windows; the decoder reconstructs them.
   - The model is trained to minimize mean squared error (MSE) between input and reconstruction.
3. **Anomaly Detection**
   - Reconstruction error is calculated for each window.
   - Windows with error above a threshold (e.g., 40th percentile) are flagged as anomalies.
4. **Evaluation**
   - Stratified sampling ensures balanced evaluation.
   - Metrics: Precision, Recall, F1-score, ROC curve/AUC, Confusion Matrix.
   - Visualizations: Histogram of reconstruction error, anomaly timeline, ROC curve, confusion matrix, t-SNE latent space.
5. **Multimodal LLM Integration**
   - Plots are interpreted using vision-language models (e.g., BLIP-2) for natural language explanations.

## Key Features
- End-to-end workflow: data loading, preprocessing, model training, anomaly scoring, and evaluation.
- Uses deep autoencoder for unsupervised learning of normal patterns.
- Comprehensive evaluation with standard metrics and visualizations.
- Integration with multimodal LLMs for automated plot interpretation.

## How to Run
1. Install requirements:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn tensorflow torch transformers
   ```
2. Download the NASA IMS Bearing dataset and update the ZIP_PATH in the notebook.
3. Open and run `notebook/Unsupervised_Anomalies_detection_Mya Ei Win.ipynb` step by step.

## Folder Structure
- `notebook/Unsupervised_Anomalies_detection_Mya Ei Win.ipynb`: Main notebook with code and analysis
- `Anomaly Detection.pptx.pdf`: Project presentation
- `Anomaly_Detection_Project_Report.pdf`: Detailed report
- `README.md`: Project overview and instructions

## Suggestions for Improvement
- Tune autoencoder architecture and threshold selection
- Experiment with other unsupervised models (e.g., LSTM autoencoders, One-Class SVM)
- Extend MLOps integration for deployment and monitoring
- Apply to other datasets for robustness
