# ACLNDA
Asymmetric Graph Contrastive Learning Framework for Predicting Non-Coding RNA-Disease Associations in Heterogeneous Graphs
1. Data Processing:
- Run the data processing script to preprocess the input data:
  ```
  python run_process.py
  ```
- This script will process the raw data and generate the necessary files for the subsequent steps.

2. Node Embedding:
- Run the node embedding script to generate node embeddings:
  ```
  python hete/main/=.py
  ```
- This script will use the processed data and generate node embeddings using the specified embedding technique.

3. Prediction:
- Run one of the prediction scripts to make predictions:
  - For LDA prediction:
    ```
    python code/LDA_prediction.py
    ```
  - For MDA prediction:
    ```
    python code/MDA_prediction.py
    ```
  - For LMI prediction:
    ```
    python code/LMI_prediction.py
    ```
- Each prediction script will use the generated node embeddings and make predictions based on the corresponding method.
