# ACLNDA
Asymmetric Graph Contrastive Learning Framework for Predicting Non-Coding RNA-Disease Associations in Heterogeneous Graphs
1. Data Processing:
- Run the data processing script to preprocess the input data:
  ```
  python run_process.py
  ```

2. Node Embedding:
- Run the node embedding script to generate node embeddings:
  ```
  python hete/main.py
  ```

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
