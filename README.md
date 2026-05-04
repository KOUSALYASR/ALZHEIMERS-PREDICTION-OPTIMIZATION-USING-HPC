# Alzheimer’s Prediction Optimization using High-Performance Computing

## Overview
This project presents a High-Performance Computing (HPC)-based deep learning framework for early prediction of Alzheimer’s disease using MRI images. The system leverages parallel processing, distributed computing, and GPU acceleration to improve model performance and computational efficiency.

---

## Objective
- Accurately classify Alzheimer’s disease stages from MRI scans  
- Reduce training time using parallel and distributed computing techniques  
- Build a scalable and efficient prediction system for medical imaging  

---

## Dataset
MRI-based Alzheimer’s dataset  

Four classification categories:
- Non-Demented  
- Very Mild Demented  
- Mild Demented  
- Moderate Demented  

---

## Methodology

### Data Preprocessing
- Image normalization and resizing  
- Parallel preprocessing using Python multiprocessing  
- Efficient handling of large-scale MRI data  

### Model Development
- Convolutional Neural Network (CNN) for classification  
- GPU acceleration using CUDA  
- Model training using PyTorch  

### High-Performance Computing Techniques
- Multiprocessing for parallel data preprocessing  
- MPI (Message Passing Interface) for distributed training  
- Parallel hyperparameter tuning  
- Performance optimization using HPC strategies  

---

## System Architecture
MRI Data → Preprocessing (Parallel) → CNN Model (GPU) → Distributed Training (MPI) → Prediction  

---

## Results
- Achieved approximately **96.9% classification accuracy**  
- Significantly reduced training time using HPC techniques  
- Improved scalability and efficiency for large datasets  
- Consistent performance across all Alzheimer’s stages  

---

## Performance Metrics
- Accuracy: **96.9%**  
- Model evaluated using Precision, Recall, and F1-score  

---

## Technologies Used
- Python  
- PyTorch  
- CUDA  
- MPI (mpi4py)  
- NumPy  
- Pandas  
- Matplotlib  

---

## Key Features
- High-speed training using GPU acceleration  
- Distributed processing for scalability  
- Automated hyperparameter tuning  
- Efficient handling of large medical datasets  

---

## Conclusion
This project demonstrates that integrating High-Performance Computing with deep learning significantly enhances both prediction accuracy and computational efficiency. The system provides a scalable solution for early Alzheimer’s diagnosis with potential for real-world healthcare applications.

---

## Future Work
- Deploy the system on cloud platforms (AWS) with scalable APIs for real-time clinical usage  
- Integrate the model with hospital information systems (HIS/PACS) for seamless workflow integration  
