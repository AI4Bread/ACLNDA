# ACLNDA: Asymmetric Graph Contrastive Learning Framework for Predicting Non-Coding RNA-Disease Associations in Heterogeneous Graphs

Non-coding RNAs (ncRNAs), including long non-coding RNAs (lncRNAs) and microRNAs (miRNAs), play crucial roles in gene expression regulation and are significant in disease associations and medical research. Accurate ncRNA-disease association prediction is essential for understanding disease mechanisms and developing treatments. Existing methods often focus on single tasks like lncRNA-disease associations (LDAs), miRNA-disease associations (MDAs), or lncRNA-miRNA interactions (LMIs) and fail to exploit heterogeneous graph characteristics. We propose ACLNDA, an Asymmetric graph Contrastive Learning framework for analyzing heterophilic ncRNA-disease associations. It constructs inter-layer adjacency matrices from original lncRNA, miRNA, and disease associations, and uses a Top-K Intra-Layer Similarity Edges construction approach to form a triple-layer heterogeneous graph. Unlike traditional works, to account for both node attribute features(ncRNA/disease) and node preference features(association), ACLNDA employs an asymmetric yet simple graph contrastive learning framework to maximize one-hop neighborhood context and two-hop similarity, extracting ncRNA-disease features without relying on graph augmentations or homophily assumptions, reducing computational cost while preserving data integrity. Our framework is capable of applying to a universally range of potential LDA, MDA, and LMI association predictions. Further experimental results demonstrate superior performance against other existing state-of-the-art baseline methods, which shows its important potential for providing insights into disease diagnosis and therapeutic target identification.

# Requirements

- dgl
- matplotlib
- networkx
- numpy
- seaborn
- torch
- torch_geometric
- tqdm
- pandas
- sklearn

# RUN ACLNDA
1. Data Processing:
- Run the data processing script to preprocess the input data:
  ```
  python code/process.py
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
