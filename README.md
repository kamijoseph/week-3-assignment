# week three assignment
## tasks:
    - theory
    - classical ml with sklearn (iris species classification using decision trees)
    - deep learning with tensorflow (mnist handwritten digits)
    - nlp with spacy (amazon ner product reviews)

## theory questions and answers

## 1. Short Answer Questions

### **Q1: Primary Differences between TensorFlow and PyTorch**

**Computation Model**
- **TensorFlow:** Uses a *static computation graph* (in TF 1.x) that must be defined before execution. TF 2.x introduced *Eager Execution* for a more dynamic feel.  
- **PyTorch:** Uses a *dynamic computation graph* (define-by-run), created on the fly, which makes debugging and experimentation intuitive.

**Ecosystem**
- **TensorFlow:** Offers a larger ecosystem — TensorBoard (visualization), TensorFlow Lite (mobile), and TensorFlow Serving (deployment).  
- **PyTorch:** Favored for research, highly Pythonic, integrates seamlessly with NumPy and frameworks like Hugging Face Transformers.

**Deployment**
- **TensorFlow:** Stronger production and deployment capabilities.  
- **PyTorch:** Catching up via TorchServe and ONNX export.

**When to Choose**
- **PyTorch:** For research, prototyping, and fast iteration.  
- **TensorFlow:** For scalable production systems and integrated ML pipelines.

---

### **Q2: Use Cases for Jupyter Notebooks in AI Development**

1. **Exploratory Data Analysis (EDA)**  
   Ideal for visualizing datasets, feature engineering, and transforming data interactively using libraries like `matplotlib`, `seaborn`, and `pandas`.

2. **Model Prototyping and Documentation**  
   Allows rapid testing of model architectures with inline visualization, combining code and Markdown for reproducibility and collaboration.

---

### **Q3: How spaCy Enhances NLP Tasks**

- **spaCy** provides advanced features such as:
  - Tokenization, POS tagging, Named Entity Recognition (NER), and dependency parsing.
  - Pretrained models for contextual linguistic understanding.
  - Efficient vectorized operations and word embeddings.

- **Basic Python string operations** (`split()`, `replace()`, etc.) are rule-based and context-unaware, whereas spaCy offers semantic understanding and contextual NLP processing.

---

## 2. Comparative Analysis: Scikit-learn vs TensorFlow

| Feature | **Scikit-learn** | **TensorFlow** |
|----------|------------------|----------------|
| **Target Applications** | Classical ML (regression, classification, clustering, etc.) | Deep Learning (CNNs, RNNs, Transformers, etc.) |
| **Ease of Use for Beginners** | Simple API (`fit()`, `predict()`, `score()`), great for beginners | Requires understanding tensors, gradients, and networks |
| **Community Support** | Mature, stable, widely used in academia and industry | Rapidly evolving, large deep learning community |

**Summary:**  
- **Scikit-learn:** Best for classical ML, small to medium datasets, and quick experimentation.  
- **TensorFlow:** Best for large-scale deep learning and GPU-accelerated model training.

---