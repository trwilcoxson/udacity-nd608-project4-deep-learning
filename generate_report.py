"""
Generate the PDF report for the Deep Learning Systems project.

Produces both 'module_summary.pdf' and 'Deep_Learning_Systems_Analysis_Report.pdf'
(identical content) to satisfy rubric criteria that reference each filename.

Usage:
    python generate_report.py
"""

import shutil
from fpdf import FPDF

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_DIR = "."
FIGURES_DIR = f"{PROJECT_DIR}/figures"
OUTPUT_PRIMARY = f"{PROJECT_DIR}/module_summary.pdf"
OUTPUT_COPY = f"{PROJECT_DIR}/Deep_Learning_Systems_Analysis_Report.pdf"

TITLE = "Deep Learning Systems: CNN Image Classification on CIFAR-10"
AUTHOR = "Tim Wilcoxson"
DATE = "February 2026"
COURSE = "Project 4 -- Deep Learning Systems"
DATASET = "CIFAR-10 (Krizhevsky, 2009)"

# Page geometry
PAGE_W = 210  # A4 width in mm
MARGIN = 20
CONTENT_W = PAGE_W - 2 * MARGIN

# Fonts
FONT_BODY = ("Helvetica", "", 11)
FONT_BOLD = ("Helvetica", "B", 11)
FONT_H2 = ("Helvetica", "B", 14)
FONT_H3 = ("Helvetica", "B", 12)
FONT_SMALL = ("Helvetica", "", 9)
FONT_ITALIC = ("Helvetica", "I", 10)

# ---------------------------------------------------------------------------
# Metrics from notebook execution (updated after training)
# ---------------------------------------------------------------------------
BASELINE_ACC = "78.15%"
BATCHNORM_ACC = "81.62%"
BASELINE_PARAMS = "620,362"
BATCHNORM_PARAMS = "621,322"
PARAM_OVERHEAD = "0.15"
ACC_IMPROVEMENT = "+3.47"
BASELINE_TIME = "135.5s (2.3 min)"
BATCHNORM_TIME = "150.8s (2.5 min)"
TIME_OVERHEAD = "+11.3"


# ---------------------------------------------------------------------------
# Report PDF class
# ---------------------------------------------------------------------------
class ReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font(*FONT_SMALL)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, TITLE, align="L")
        self.ln(6)
        self.set_draw_color(180, 180, 180)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font(*FONT_SMALL)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # ---- Helpers ----------------------------------------------------------

    def section_heading(self, number, title):
        self.ln(4)
        self.set_font(*FONT_H2)
        self.set_text_color(30, 60, 120)
        self.cell(0, 10, f"{number}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 60, 120)
        self.line(MARGIN, self.get_y(), MARGIN + CONTENT_W, self.get_y())
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def subsection(self, title):
        self.ln(2)
        self.set_font(*FONT_H3)
        self.set_text_color(50, 80, 140)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
        self.set_text_color(0, 0, 0)

    def body_text(self, text):
        self.set_font(*FONT_BODY)
        self.multi_cell(CONTENT_W, 6, text)
        self.ln(2)

    def bold_text(self, text):
        self.set_font(*FONT_BOLD)
        self.multi_cell(CONTENT_W, 6, text)
        self.ln(1)

    def italic_text(self, text):
        self.set_font(*FONT_ITALIC)
        self.multi_cell(CONTENT_W, 5, text)
        self.ln(1)

    def add_figure(self, path, caption, width=CONTENT_W):
        est_h = width * 0.6 + 15
        if self.get_y() + est_h > 270:
            self.add_page()
        x = (PAGE_W - width) / 2
        self.image(path, x=x, w=width)
        self.ln(2)
        self.set_font(*FONT_ITALIC)
        self.set_text_color(80, 80, 80)
        self.multi_cell(CONTENT_W, 5, caption, align="C")
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def bullet(self, text):
        self.set_font(*FONT_BODY)
        self.cell(6, 6, "-")
        self.multi_cell(CONTENT_W - 6, 6, text)
        self.ln(1)


# ---------------------------------------------------------------------------
# Build the report
# ---------------------------------------------------------------------------
def build_report():
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(MARGIN, MARGIN, MARGIN)

    # =======================================================================
    # TITLE PAGE
    # =======================================================================
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(30, 60, 120)
    pdf.multi_cell(CONTENT_W, 12, TITLE, align="C")
    pdf.ln(10)
    pdf.set_draw_color(30, 60, 120)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(CONTENT_W, 8, AUTHOR, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(CONTENT_W, 8, DATE, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(CONTENT_W, 8, COURSE, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 11)
    pdf.cell(CONTENT_W, 8, f"Dataset: {DATASET}", align="C",
             new_x="LMARGIN", new_y="NEXT")

    # =======================================================================
    # 1. REPORT OVERVIEW
    # =======================================================================
    pdf.add_page()
    pdf.section_heading(1, "Report Overview")
    pdf.body_text(
        "This report presents a deep learning experiment comparing two "
        "convolutional neural network (CNN) architectures on the CIFAR-10 "
        "image classification benchmark. A baseline CNN is compared against "
        "an identical architecture augmented with Batch Normalization "
        "(Ioffe & Szegedy, 2015) to isolate the effect of this single "
        "architectural modification on convergence speed, training stability, "
        "and classification accuracy. Both models are implemented in PyTorch "
        "(Paszke et al., 2019) and trained on Apple Silicon MPS hardware."
    )

    # =======================================================================
    # 2. DATASET AND TASK DESCRIPTION
    # =======================================================================
    pdf.section_heading(2, "Dataset and Task Description")
    pdf.body_text(
        "CIFAR-10 (Krizhevsky, 2009) is a widely-used benchmark dataset "
        "consisting of 60,000 32x32 color images evenly distributed across "
        "10 mutually exclusive classes: airplane, automobile, bird, cat, deer, "
        "dog, frog, horse, ship, and truck. The dataset is split into 50,000 "
        "training images and 10,000 test images, with exactly 6,000 images "
        "per class in the full dataset."
    )
    pdf.body_text(
        "For this experiment, the 50,000 training images were further split "
        "into 45,000 for training and 5,000 for validation using a fixed "
        "random seed for reproducibility. The validation set uses test-time "
        "transforms (normalization only) to provide an unbiased estimate of "
        "generalization performance during training. The test set of 10,000 "
        "images was held out entirely until final evaluation."
    )
    pdf.body_text(
        "The task is multi-class image classification: given a 32x32 RGB "
        "image, predict which of the 10 classes it belongs to. Because the "
        "classes are perfectly balanced, accuracy serves as a valid primary "
        "evaluation metric. Data augmentation (random horizontal flips and "
        "random crops with padding) was applied during training to improve "
        "generalization (Goodfellow et al., 2016)."
    )

    pdf.add_figure(
        f"{FIGURES_DIR}/fig1_sample_images.png",
        "Figure 1. Representative CIFAR-10 images, one per class, showing "
        "the 32x32 resolution and visual diversity of the dataset.",
        width=CONTENT_W - 10,
    )

    pdf.add_figure(
        f"{FIGURES_DIR}/fig2_class_distribution.png",
        "Figure 2. Class distribution across training, validation, and test "
        "splits, confirming balanced representation in all partitions.",
        width=CONTENT_W - 10,
    )

    # =======================================================================
    # 3. MODEL ARCHITECTURE AND DESIGN DECISIONS
    # =======================================================================
    pdf.section_heading(3, "Model Architecture and Design Decisions")

    pdf.subsection("Baseline CNN Architecture")
    pdf.body_text(
        "The baseline model follows a standard CNN pattern with three "
        "convolutional blocks, each consisting of a 3x3 convolution with "
        "same-padding, ReLU activation, and 2x2 max pooling. The channel "
        "dimensions progressively increase from 3 (input RGB) to 32, 64, "
        "and 128, allowing the network to learn increasingly abstract "
        "features at each spatial scale (Goodfellow et al., 2016). After "
        "the convolutional feature extractor, a fully-connected classifier "
        "maps the flattened 128x4x4 feature vector through a 256-unit "
        "hidden layer with ReLU and 50% Dropout (Srivastava et al., 2014) "
        "to the 10-class output."
    )
    pdf.body_text(
        f"This architecture contains {BASELINE_PARAMS} trainable parameters. "
        "The design choices -- three conv blocks with doubling channels, "
        "kernel size 3 with padding 1, and aggressive max pooling -- follow "
        "established practices for small-image classification "
        "(Krizhevsky, 2009). Dropout in the classifier prevents overfitting "
        "to the relatively small training set."
    )

    pdf.subsection("Batch Normalization CNN Architecture")
    pdf.body_text(
        "The experimental model is architecturally identical to the baseline "
        "except for the addition of Batch Normalization layers. BatchNorm2d "
        "is inserted after each convolutional layer (before ReLU), and "
        "BatchNorm1d is inserted after the first fully-connected layer "
        "(before ReLU and Dropout). This follows the placement recommended "
        "by Ioffe & Szegedy (2015) in the original Batch Normalization paper."
    )
    pdf.body_text(
        f"The BatchNorm variant contains {BATCHNORM_PARAMS} parameters -- "
        f"only {PARAM_OVERHEAD}% more than the baseline. This minimal "
        "parameter overhead comes from the learnable affine parameters "
        "(gamma and beta) added per channel, making it an ideal single-"
        "variable experiment: the architectural topology is identical, and "
        "the parameter count difference is negligible."
    )

    # =======================================================================
    # 4. EXPERIMENTAL COMPARISON
    # =======================================================================
    pdf.section_heading(4, "Experimental Comparison")

    pdf.subsection("Experimental Variable")
    pdf.body_text(
        "The single experimental variable is the presence or absence of "
        "Batch Normalization layers. All other hyperparameters are held "
        "constant between the two models: SGD optimizer with momentum 0.9 "
        "and weight decay 5e-4, initial learning rate 0.01 with StepLR "
        "decay (factor 0.1 every 10 epochs), batch size 128, 30 training "
        "epochs, and identical random seeds for weight initialization and "
        "data ordering."
    )

    pdf.subsection("Why Batch Normalization?")
    pdf.body_text(
        "Batch Normalization was proposed by Ioffe & Szegedy (2015) to "
        "address internal covariate shift -- the phenomenon where the "
        "distribution of each layer's inputs changes during training as "
        "the preceding layers' parameters are updated. By normalizing "
        "activations to zero mean and unit variance within each mini-batch, "
        "BatchNorm stabilizes the optimization landscape, enabling higher "
        "learning rates and faster convergence. It also acts as a mild "
        "regularizer by introducing noise through batch statistics "
        "(Goodfellow et al., 2016)."
    )
    pdf.body_text(
        "This makes Batch Normalization an ideal experimental variable: "
        "it has well-documented, observable effects (faster convergence, "
        "smoother loss curves, improved accuracy), it changes only one "
        "aspect of the model (regularization/optimization), and its "
        "parameter overhead is negligible (~0.15%). The comparison is "
        "directly supported by the seminal paper's claims."
    )

    pdf.subsection("Training Protocol")
    pdf.body_text(
        "Both models were trained with identical protocols to ensure a "
        "fair comparison. Before each training run, all random seeds "
        "(Python, NumPy, PyTorch) were reset to the same value (42) to "
        "ensure identical weight initializations and data ordering. "
        "Per-epoch training loss, training accuracy, validation loss, and "
        "validation accuracy were logged for both models, along with "
        "wall-clock training time per epoch."
    )

    pdf.add_figure(
        f"{FIGURES_DIR}/fig3_training_curves_loss.png",
        "Figure 3. Training and validation loss curves for both models. "
        "Dashed lines indicate learning rate decay steps at epochs 10 and 20.",
        width=CONTENT_W - 10,
    )

    pdf.add_figure(
        f"{FIGURES_DIR}/fig4_training_curves_accuracy.png",
        "Figure 4. Training and validation accuracy curves. BatchNorm "
        "converges faster and maintains a consistent accuracy advantage.",
        width=CONTENT_W - 10,
    )

    # =======================================================================
    # 5. RESULTS AND INTERPRETATION
    # =======================================================================
    pdf.section_heading(5, "Results and Interpretation")

    pdf.subsection("Overall Test Accuracy")
    pdf.body_text(
        f"The Baseline CNN achieved a test accuracy of {BASELINE_ACC}, while "
        f"the BatchNorm CNN achieved {BATCHNORM_ACC} -- an improvement of "
        f"{ACC_IMPROVEMENT} percentage points. This improvement is "
        "consistent with the literature on Batch Normalization, which "
        "typically reports accuracy gains of 1-5 percentage points on "
        "CIFAR-10 for comparable architectures (Ioffe & Szegedy, 2015)."
    )

    pdf.subsection("Convergence Speed")
    pdf.body_text(
        "The training curves (Figures 3 and 4) reveal that the BatchNorm "
        "model converges substantially faster than the baseline. In the "
        "first 5 epochs, BatchNorm achieves validation accuracy comparable "
        "to what the baseline requires approximately 10 epochs to reach. "
        "This accelerated convergence is the most prominent benefit of "
        "Batch Normalization, as it reduces the number of epochs needed "
        "to reach a given performance target."
    )

    pdf.subsection("Loss Landscape Smoothing")
    pdf.body_text(
        "The loss curves show that the BatchNorm model exhibits smoother "
        "training dynamics with less oscillation between epochs. This is "
        "consistent with recent theoretical work by Santurkar et al. (2018), "
        "who demonstrated that Batch Normalization's primary benefit may be "
        "smoothing the optimization landscape (making the loss function more "
        "Lipschitz continuous) rather than reducing covariate shift per se."
    )

    pdf.add_figure(
        f"{FIGURES_DIR}/fig5_confusion_matrices.png",
        "Figure 5. Confusion matrices for both models on the test set, "
        "showing prediction distributions across all 10 classes.",
        width=CONTENT_W - 10,
    )

    pdf.add_figure(
        f"{FIGURES_DIR}/fig6_per_class_accuracy.png",
        "Figure 6. Per-class test accuracy comparison. BatchNorm improves "
        "performance on nearly every class, with the largest gains on "
        "visually ambiguous categories.",
        width=CONTENT_W - 10,
    )

    pdf.subsection("Per-Class Analysis")
    pdf.body_text(
        "The per-class accuracy comparison (Figure 6) reveals that BatchNorm "
        "improves accuracy across nearly all classes. The largest gains are "
        "observed on classes that are visually similar and harder to "
        "distinguish: cat, dog, and deer. These classes have high inter-class "
        "confusion (cat/dog, deer/horse) due to similar shapes, textures, "
        "and backgrounds. The normalized activations appear to help the "
        "network learn finer-grained discriminative features for these "
        "challenging categories."
    )

    pdf.add_figure(
        f"{FIGURES_DIR}/fig7_confusion_difference.png",
        "Figure 7. Confusion matrix difference (BatchNorm minus Baseline). "
        "Blue diagonal cells indicate classes where BatchNorm improved; "
        "red off-diagonal cells show shifted error patterns.",
        width=CONTENT_W - 10,
    )

    pdf.add_figure(
        f"{FIGURES_DIR}/fig8_error_analysis.png",
        "Figure 8. Sample images rescued by BatchNorm -- correctly "
        "classified by the BatchNorm model but misclassified by the baseline.",
        width=CONTENT_W - 10,
    )

    pdf.subsection("Training Time")
    pdf.body_text(
        f"The Baseline CNN trained in {BASELINE_TIME}, while the BatchNorm "
        f"CNN required {BATCHNORM_TIME} ({TIME_OVERHEAD}% overhead). "
        "The additional computation from normalization layers adds a small "
        "per-epoch cost. However, because BatchNorm converges faster per "
        "epoch, fewer epochs may be needed to reach a target accuracy, "
        "potentially offering a net training time reduction in practice."
    )

    pdf.add_figure(
        f"{FIGURES_DIR}/fig9_training_time.png",
        "Figure 9. Per-epoch and cumulative training time comparison, "
        "showing the modest computational overhead of Batch Normalization.",
        width=CONTENT_W - 10,
    )

    # =======================================================================
    # 6. LIMITATIONS AND RISKS
    # =======================================================================
    pdf.section_heading(6, "Limitations and Risks")

    pdf.subsection("Dataset Limitations")
    pdf.bullet(
        "Low resolution: CIFAR-10 images are 32x32 pixels, far below the "
        "resolution of modern cameras or real-world deployment scenarios. "
        "The CNN architectures and hyperparameters optimized for this "
        "resolution may not transfer to higher-resolution tasks without "
        "significant modification."
    )
    pdf.bullet(
        "Closed-world assumption: CIFAR-10 contains exactly 10 classes. "
        "Real-world image classification systems must handle open-set "
        "recognition, where inputs may belong to classes not seen during "
        "training (Goodfellow et al., 2016)."
    )
    pdf.bullet(
        "Benchmark saturation: State-of-the-art models achieve 99%+ "
        "accuracy on CIFAR-10 using modern architectures (e.g., Vision "
        "Transformers, large ResNets with extensive augmentation). The "
        "models in this experiment are intentionally simple to clearly "
        "demonstrate the BatchNorm effect, but they do not represent "
        "competitive performance on this benchmark."
    )

    pdf.subsection("Experimental Limitations")
    pdf.bullet(
        "Single seed: Both models were trained with a single random seed. "
        "While this ensures identical initialization for fair comparison, "
        "it does not capture the variance in performance across different "
        "initializations. Running multiple seeds would strengthen the "
        "statistical confidence of the comparison."
    )
    pdf.bullet(
        "Single learning rate: The experiment used a fixed learning rate "
        "of 0.01 for both models. Ioffe & Szegedy (2015) noted that "
        "BatchNorm enables higher learning rates; a comparison across "
        "multiple learning rates would better characterize the interaction "
        "between BatchNorm and optimization dynamics."
    )
    pdf.bullet(
        "MPS hardware: Training was conducted on Apple Silicon MPS, which "
        "has different numerical behavior than CUDA GPUs. Results may differ "
        "slightly on CUDA hardware due to floating-point non-determinism."
    )

    # =======================================================================
    # 7. ETHICAL AND RESPONSIBLE USE
    # =======================================================================
    pdf.section_heading(7, "Ethical and Responsible Use")
    pdf.body_text(
        "While CIFAR-10 is a benign academic benchmark, the image "
        "classification techniques demonstrated here generalize to "
        "applications with significant ethical implications."
    )
    pdf.bullet(
        "Surveillance: CNNs trained on images are foundational to automated "
        "surveillance systems, including facial recognition. Deploying such "
        "systems raises concerns about privacy, consent, and "
        "disproportionate impact on marginalized communities (Buolamwini & "
        "Gebru, 2018). The techniques in this project could contribute to "
        "surveillance applications if applied to different datasets."
    )
    pdf.bullet(
        "Bias in training data: Image classification models inherit biases "
        "present in their training data. CIFAR-10 was curated from web "
        "images and may not represent the full diversity of visual concepts. "
        "Models trained on biased data can systematically underperform on "
        "underrepresented groups or contexts (Mitchell et al., 2019)."
    )
    pdf.bullet(
        "Dual-use risk: The ability to classify images rapidly and "
        "accurately can be used for beneficial applications (medical "
        "imaging, wildlife conservation) or harmful ones (autonomous "
        "weapons, mass content filtering). Responsible deployment requires "
        "considering the downstream application context and implementing "
        "appropriate access controls and monitoring."
    )
    pdf.body_text(
        "Practitioners should evaluate fairness, accountability, and "
        "transparency at every stage of the ML pipeline, from data "
        "collection to deployment (Mitchell et al., 2019)."
    )

    # =======================================================================
    # 8. FUTURE IMPROVEMENTS
    # =======================================================================
    pdf.section_heading(8, "Future Improvements")
    pdf.bullet(
        "Deeper architectures: Replacing the 3-block CNN with a ResNet-style "
        "architecture (He et al., 2016) would enable deeper networks that "
        "benefit even more from Batch Normalization. ResNet's skip "
        "connections combined with BatchNorm have been shown to enable "
        "training of networks with hundreds of layers."
    )
    pdf.bullet(
        "Learning rate tuning: Ioffe & Szegedy (2015) showed that BatchNorm "
        "enables higher learning rates. A systematic comparison of learning "
        "rates (e.g., 0.01, 0.05, 0.1) would reveal whether the BatchNorm "
        "model's advantage increases with more aggressive optimization."
    )
    pdf.bullet(
        "Alternative normalization techniques: Layer Normalization (Ba et "
        "al., 2016), Instance Normalization (Ulyanov et al., 2016), and "
        "Group Normalization (Wu & He, 2018) offer alternatives to Batch "
        "Normalization with different trade-offs, particularly for small "
        "batch sizes or non-image domains."
    )
    pdf.bullet(
        "Transfer learning: Using a pretrained backbone (e.g., ResNet-18 "
        "pretrained on ImageNet) and fine-tuning on CIFAR-10 would likely "
        "achieve significantly higher accuracy while requiring fewer "
        "training epochs (Goodfellow et al., 2016)."
    )
    pdf.bullet(
        "Multi-seed evaluation: Running the experiment across 5-10 random "
        "seeds and reporting confidence intervals would provide stronger "
        "statistical evidence for the observed accuracy improvement."
    )

    # =======================================================================
    # 9. REFERENCES
    # =======================================================================
    pdf.section_heading(9, "References")
    pdf.set_font(*FONT_BODY)

    references = [
        (
            "Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer "
            "Normalization. arXiv preprint arXiv:1607.06450."
        ),
        (
            "Buolamwini, J., & Gebru, T. (2018). Gender Shades: "
            "Intersectional accuracy disparities in commercial gender "
            "classification. Proceedings of the Conference on Fairness, "
            "Accountability and Transparency (FAccT), 77-91."
        ),
        (
            "Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep "
            "Learning. MIT Press. https://www.deeplearningbook.org/"
        ),
        (
            "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual "
            "Learning for Image Recognition. Proceedings of the IEEE "
            "Conference on Computer Vision and Pattern Recognition (CVPR), "
            "770-778."
        ),
        (
            "Ioffe, S., & Szegedy, C. (2015). Batch Normalization: "
            "Accelerating Deep Network Training by Reducing Internal "
            "Covariate Shift. Proceedings of the 32nd International "
            "Conference on Machine Learning (ICML), 448-456."
        ),
        (
            "Krizhevsky, A. (2009). Learning Multiple Layers of Features "
            "from Tiny Images. Technical Report, University of Toronto."
        ),
        (
            "Mitchell, M., et al. (2019). Model Cards for Model Reporting. "
            "Proceedings of the Conference on Fairness, Accountability, and "
            "Transparency (FAccT), 220-229."
        ),
        (
            "Paszke, A., et al. (2019). PyTorch: An Imperative Style, "
            "High-Performance Deep Learning Library. Advances in Neural "
            "Information Processing Systems (NeurIPS), 32, 8024-8035."
        ),
        (
            "Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A. (2018). "
            "How Does Batch Normalization Help Optimization? Advances in "
            "Neural Information Processing Systems (NeurIPS), 31, 2483-2493."
        ),
        (
            "Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., "
            "& Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent "
            "Neural Networks from Overfitting. Journal of Machine Learning "
            "Research, 15(1), 1929-1958."
        ),
        (
            "Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2016). Instance "
            "Normalization: The Missing Ingredient for Fast Stylization. "
            "arXiv preprint arXiv:1607.08022."
        ),
        (
            "Wu, Y., & He, K. (2018). Group Normalization. Proceedings of "
            "the European Conference on Computer Vision (ECCV), 3-19."
        ),
    ]

    for ref in references:
        pdf.multi_cell(CONTENT_W, 5.5, ref)
        pdf.ln(3)

    # =======================================================================
    # OUTPUT
    # =======================================================================
    pdf.output(OUTPUT_PRIMARY)
    shutil.copy2(OUTPUT_PRIMARY, OUTPUT_COPY)
    print(f"Generated: {OUTPUT_PRIMARY}")
    print(f"Copied to: {OUTPUT_COPY}")


if __name__ == "__main__":
    build_report()
