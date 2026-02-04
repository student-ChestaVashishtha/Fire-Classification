# 🚀 Robust Fire Detection using MobileNetV2 & Data-Centric AI

### **Overview**

This project implements a high-efficiency **Convolutional Neural Network (CNN)** for real-time fire detection. Developed as part of my 3rd-year Electronics Engineering studies at **AMU**, the model is optimized for deployment.

The core innovation of this project is not just high accuracy (98.6%), but **model robustness** achieved through targeted **Hard Negative Mining**.

---

## 🔬 Research Focus: Bias Mitigation

During the initial development phase, the model exhibited a significant **feature bias**. It incorrectly classified amber-hued ambient lighting (e.g., yellow streetlights, gold textures) as "Fire" with a **0.62 confidence score**.
<img width="1721" height="806" alt="Screenshot 2026-02-01 182532" src="https://github.com/user-attachments/assets/badce52c-f1cb-4f38-a67a-3d3e6a9003a2" />

### **The Technical Intervention**

To re-calibrate the decision boundary without increasing the model's computational footprint, I implemented a **Data-Centric AI** strategy:

* **Identified Failure Mode:** Over-indexing on high-chroma yellow/orange color histograms.
* **Surgical Fix:** Curated and integrated **35 targeted "Hard Negative" samples** (images of confusing non-fire amber environments).
* **Quantifiable Result:** Misclassification confidence for the same edge cases dropped from **0.62 to 0.06** (a 90% improvement in reliability).
<img width="1567" height="893" alt="Screenshot 2026-02-01 183805" src="https://github.com/user-attachments/assets/11d67ab1-3fc1-4348-9ffd-3a7ff15224e5" />

---

## 🛠️ Technical Stack

* **Architecture:** MobileNetV2 (Transfer Learning from ImageNet)
* **Optimization:** Fine-tuned custom Dense head.
* **Augmentation:** Rotation, Width/Height shifts, and Horizontal flips for generalization.
* **Deployment:** Integrated with **Gradio** and hosted on **Hugging Face Spaces**.

---

## 📊 Dataset Evolution

| Version | Fire Images | Non-Fire Images | Key Observation |
| --- | --- | --- | --- |
| **v1.0 (Baseline)** | 755 | 244 | Failed on amber-hued lighting. |
| **v2.0 (Optimized)** | 755 | **[244 + 35]** | **Eliminated color-bias false positives.** |

---

## 🚀 How to Run

1. **Live Demo:** Access the interactive app here: (https://huggingface.co/spaces/Vashishtha-Chesta-AMU/Fire-Classification)
2. **Local Run:**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
pip install -r requirements.txt
python app.py

```



---

## 👨‍💻 Author

**Chesta Vashishtha** *3rd Year B.Tech (Electronics and Communication Engineering)* *Aligarh Muslim University (AMU)* *CGPA: 8.22/10*

---

