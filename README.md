# 👶 Affect Detection in Infants using Vision Transformers (ViT)

This project fine-tunes a Vision Transformer (ViT) to classify infant facial expressions into **Positive**, **Neutral**, and **Negative**, using the *City Infant Faces* dataset. It also visualizes model decisions using **SmoothGrad**.

---

## 📂 Project Structure
├── EDA.py # Exploratory Data Analysis <br>
├── data_processing.py # Preprocessing and DataLoaders <br>
├── model_training.py # ViT fine-tuning <br>
├── model_testing.py # Evaluation & misclassification analysis <br>
├── smoothgrad.py # Visualizing with SmoothGrad <br>
├── trained_vit_model.pth # Saved model weights <br>


## 📦 Requirements

- Python 3.9+
- PyTorch
- Hugging Face `transformers`, `datasets`
- OpenCV
- Matplotlib
- tqdm

  ## 🧠 Model Overview

- **Base:** ViT-Base-Patch16-224-in21k  
- **Pretrained on:** AffectNet, FER2013, MMI  
- **Fine-tuned on:** City Infant Faces (Color images)  
- **Classification Labels:** Positive, Neutral, Negative  

**Accuracy Results:**

| Dataset  | Accuracy |
|----------|----------|
| Dev Set  | 92%      |
| Test Set | 84%      |

---

## 🔍 SmoothGrad Insights

**Highlighted facial regions by class:**

- **Positive:** Cheeks & lip corners *(AU6, AU12)*
- **Neutral:** T-zone (forehead, nose, mouth)
- **Negative:** Lip stretch, chin, eye tension *(AU20, AU17, AU6/7)*
