# Question 4: Reproducible Machine Learning Pipeline with DVC and MLflow

This repository demonstrates a fully reproducible, end-to-end Machine Learning pipeline across heterogeneous environments (Partner A on Windows and Partner B on Linux) using **DVC** with S3-compatible cloud storage (Backblaze B2) and **MLflow** for experiment tracking and model registry management.

---

## 1. Environment Setup

Clone the repository and set up the Conda environment:

```bash
git clone <your-repository-url>
cd aiops-assignment1-q4
conda env create -f environment.yml
conda activate aiops-q4-env
