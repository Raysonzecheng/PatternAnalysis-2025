# Siamese Network for ISIC 2020 Melanoma Classification

## Description
This project implements a **Siamese Neural Network** to classify **melanoma vs normal skin lesions** using the ISIC 2020 dataset. The model learns to measure the similarity between pairs of skin images instead of directly classifying them.

##  How It Works
The network uses a shared **ResNet18 encoder** to extract embeddings from two input images.  
A **contrastive loss** is used to minimize the distance between embeddings of the same class and maximize it between different classes.



##  Files
| File | Description |
|------|--------------|
| `modules.py` | Model definition and loss function |
| `dataset.py` | Dataset loader and preprocessing |
| `train.py` | Training and validation logic |
| `predict.py` | Example inference script |
| `README.md` | Documentation |

##  Training Details
- Optimizer: Adam, LR = 1e-4  
- Epochs: 10  
- Batch size: 16  
- Accuracy: ~0.8 on test set  

##  Example Output
