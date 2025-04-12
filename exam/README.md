# MTE546 Exam

---

This repository contains the solutions for the MTE546 exam. Each question is implemented in a folder. Follow the instructions below to set up the environment and run the code for each question.

---

## Prerequisites

Ensure you have the following installed on your system:
- Python 3.8 or higher
- pip (Python package manager)
- Any additional dependencies listed in `requirements.txt`

## Setup Instructions

1. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Project Structure

```
├── src/
│   ├── q1/                     # Question 1 Python files
│   │
│   ├── q2/                     # Question 2 Python files
│   │
│   ├── q3/                     # Question 3 Python files
│
├── img/                        # Output images
│
├── data/                       # Submission data
│
├── Heart_of_Gold_Improp_drive/ # Provided data
│
├── Nerual_network_data2        # Provided data
│
├── regression data             # Provided data
│
├── requirements.txt            # List of dependencies
└── README.md                   # Project documentation
```

---

## Running the Code

Each question is implemented in a separate script or folder. Below are the instructions to run the code for each question:

### Question 1
```bash
python q1/original_model.py    # Plot original predicitons
python q1/lin_regression.py    # Compute new model
python q1/boot_strap.py        # Compute model with bootstrap method
python q1/compare.py           # Compare original and derived model
```

### Question 2=
```bash
cd question2
python q2/visualize_data.py     # Visualize the provided data
python q2/ekf.py                # EKF implementation
python q2/adaptive_ekf.py       # Adaptive EKF implementation
python q2/run_ekf.py            # Runs the EKFs and generates plots and metrics
```

### Question 3
```bash
cd question3
python q3/seperate.py           # Train seperate NNs and generate plots
python q3/fused.py           # Train fused NNs and generate plots
```
