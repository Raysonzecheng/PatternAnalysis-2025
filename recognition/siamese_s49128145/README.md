# Siamese network for Classification of ISIC 2020 Data Set
## Project Introduction
### Project Summary and Aim

The objective of this project is to develop a two-stage classifier based on a Siamese network architecture to perform binary classification on the ISIC 2020 Kaggle Challenge dataset, distinguishing between normal and melanoma skin lesions.

The final model should achieve approximately 0.80 accuracy on the test set, demonstrating strong generalization performance on unseen dermoscopic images.


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
