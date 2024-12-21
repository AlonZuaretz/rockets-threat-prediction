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
### Alerts Dataset
The **Threats Dataset** provides the primary temporal input for the model, capturing patterns of rocket alerts across various cities in Israel. The dataset is sourced from publicly available Home Front Command (Pikud Ha'oref) data.

#### Structure
Each entry in the raw dataset includes:
- **Date**: The day the alert occurred.
- **Time**: The specific time of the alert.
- **Day of the Week**: The weekday of the alert.
- **Type of Alert**: Nature of the event (e.g., rocket or missile fire).
- **Location**: City or area affected.

**Example (Raw):**

#### Preprocessing
1. **Timestamp Binning**: Alerts are grouped into 6-hour intervals, creating a single timestamp for combined alerts.
2. **Normalization**: Time-related features are scaled using Min-Max normalization (e.g., hour 12:00 → 0.6667).
3. **Multi-Hot Encoding**: A binary vector represents affected locations in each time interval.
4. **Sequence Generation**: Sequential samples are combined into fixed-length sequences for model input.

**Example (Processed):**




### Articles Dataset
The **Articles Dataset** provides contextual information by embedding war-related news articles. These embeddings enrich the model’s temporal predictions with insights from real-world events.

#### Structure
Each entry in the raw dataset includes:
- **Date**: Publication date of the article.
- **Time**: Time the article was posted.
- **Title**: Headline of the article.
- **Summary**: A brief description of the article content.

**Example (Raw):**

#### Preprocessing
1. **Normalization**: Time-related features are Min-Max normalized similarly to the threats dataset.
2. **Embedding Generation**: Titles and summaries are passed through AlephBERT, producing a 1536-dimensional embedding.
   - 728 dimensions for the title.
   - 728 dimensions for the summary.
3. **Sequence Synchronization**: Articles are aligned with threat sequences to ensure temporal coherence.

**Example (Processed):**


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

  2. **Predict using the trained model:**
      First, make sure you have:<br>
      1. Trained model
      2. parameters in main_eval.py match the parameters used to train the model
      3. Alerts dataset (.csv) in the same format as described here.
      4. Articles dataset (.txt) in the same format as described here.
      Use main_eval.py under evaluate directory.
      ```cmd
      python evaluate/main_eval.py --param1 'trained model.pth' --param2 'alerts path.csv' --param3 'articles directory'
      ```
  
   

## Future Work
      
     



