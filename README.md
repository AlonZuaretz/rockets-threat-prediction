# Predicting The Next Rocket Alert

A brief description of your project, including its purpose, goals, and any unique features or contributions.

## Table of Contents
- [Datasets](#datasets)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Setup](#setup)
- [Usage](#usage)
- [Future Work](#future-work)

## Datasets
  1. **Articles dataset:**
     
  3. **Alerts dataset:**

## Model Architecture

## Results

## Setup
Follow the steps below to set up the project on a Windows system using Python.
### Prerequisites
Before starting, ensure the following are installed:
- Python 3.7 or later (download from [python.org](https://www.python.org/))
- pip (comes with Python installation)
- [Git](https://git-scm.com/)
### Installation
 Open Command Prompt or PowerShell and do the following:
1. **Clone the repository:**
     ```cmd
     git clone https://github.com/username/repository.git
     cd repository
     ```
2. **Create a virtual environment:**
     ```cmd
     python -m venv venv
     venv\Scripts\activate
     ```
 3. **Install Requirements**
    ```cmd
    pip install -r requirements.txt
    ```

## Usage
  1. **Train the model:**
      You may change parameters in the main script.<br>
      In particular, if you want to re-generate embeddings for the articles, set:
      ```python
      read_embedded_articles = False
      ```
      Otherwise, embeddings will be read from a predefined .csv file (for faster performance)
    
      Next, run main from Command Prompt or PowerShell:
      ```cmd
      python main.py
      ```
      Running the main script both trains and tests the model using the data in the "data" directory.<br>
      The trained model and additional train information are saved under the 'results' directory.

  2. **Predict using the trained model**


## Future Work
      
     



