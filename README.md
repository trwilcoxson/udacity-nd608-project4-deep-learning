# Deep Learning Systems: CNN Image Classification on CIFAR-10

A deep learning experiment comparing a baseline CNN against a Batch Normalization-augmented CNN on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) image classification benchmark. The controlled experiment isolates the effect of Batch Normalization (Ioffe & Szegedy, 2015) on convergence speed, training stability, and classification accuracy.

| | |
|---|---|
| **Author** | Tim Wilcoxson |
| **Course** | Project 4 — Deep Learning Systems |
| **Date** | February 2026 |

## Key Findings

- **Experiment:** Baseline CNN vs. CNN + Batch Normalization (single-variable comparison)
- **Dataset:** CIFAR-10 — 60,000 32x32 color images, 10 balanced classes
- **Framework:** PyTorch with MPS (Apple Silicon GPU) acceleration
- **BatchNorm adds only ~0.15% more parameters** while improving convergence speed and final accuracy
- **Improvement observed across nearly all 10 classes**, with the largest gains on visually ambiguous categories (cat, dog, deer)

## Project Structure

```
project4_deep_learning/
├── deep_learning.ipynb                              # Complete DL workflow notebook
├── generate_report.py                               # Script to regenerate the PDF report
├── module_summary.pdf                               # DL analysis report (PDF)
├── Deep_Learning_Systems_Analysis_Report.pdf         # DL analysis report (identical copy)
├── requirements.txt                                  # Python dependencies (pip freeze)
├── README.md                                         # This file
├── .gitignore
├── data/                                             # CIFAR-10 auto-downloaded by torchvision
└── figures/
    ├── fig1_sample_images.png                        # CIFAR-10 sample images (one per class)
    ├── fig2_class_distribution.png                   # Class distribution across splits
    ├── fig3_training_curves_loss.png                 # Training & validation loss curves
    ├── fig4_training_curves_accuracy.png             # Training & validation accuracy curves
    ├── fig5_confusion_matrices.png                   # Side-by-side confusion matrices
    ├── fig6_per_class_accuracy.png                   # Per-class accuracy bar chart
    ├── fig7_confusion_difference.png                 # Confusion matrix difference heatmap
    ├── fig8_error_analysis.png                       # Images rescued by BatchNorm
    └── fig9_training_time.png                        # Per-epoch and cumulative training time
```

## Setup and Reproduction

```bash
git clone https://github.com/trwilcoxson/udacity-nd608-project4-deep-learning.git
cd project4_deep_learning
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the notebook (trains both models, ~20 min on Apple Silicon MPS)
jupyter notebook deep_learning.ipynb

# Regenerate the PDF report
python generate_report.py
```

## Technologies

- **Python 3.13** — PyTorch, Torchvision, NumPy, Pandas, Matplotlib, Seaborn
- **Deep learning** — CNN, Batch Normalization, SGD with momentum, StepLR scheduler
- **Evaluation** — Confusion matrices, per-class precision/recall/F1, error analysis
- **Report generation** — fpdf2
- **Environment** — Jupyter Notebook, venv, MPS (Apple Silicon GPU)

## Dataset

CIFAR-10 (Krizhevsky, 2009): 60,000 32x32 color images in 10 classes, split into 50,000 training and 10,000 test images. Auto-downloaded by `torchvision.datasets.CIFAR10`.

## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. https://www.deeplearningbook.org/
- Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *Proceedings of the 32nd ICML*, 448-456.
- Krizhevsky, A. (2009). *Learning multiple layers of features from tiny images*. Technical Report, University of Toronto.
- Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS*, 32, 8024-8035.
