# 📚 Table Encoder for TnT

This module is used to pre-train a table encoder that produces dense embeddings for structured tables. It is designed to work with the TnT system for table-aware text-to-SQL generation.

---

## 📦 Installation

Install dependencies via conda:

```bash
conda env create -f environment.yml
```

---

## **🏋️ Training**

### **1. Prepare Training Data**

- Convert your warm-up and training data into `.np` files — each `.np` file should contain a **list of file paths** pointing to your table `.csv` files.
- Place all `.np` files in the same directory.

> 💡 You can use the tools in Table-Encoder/data_preprocess/ to preprocess raw tables from various formats.

### **2. Configure Sentence Transformer**

Create a `.env` file to specify the sentence transformer model (e.g., from HuggingFace):

```bash
# .env
SENTENCE_TRANSFORMER=all-MiniLM-L6-v2
```

### **3. Launch Training**

Run the training script:

```bash
bash Table-Encoder/run.sh --data_path <data_directory>
```

Replace `<data_directory>` with the directory containing your `.np` table data.

### **4. Adjust Hyperparameters (Optional)**

You can modify other training parameters such as:

- Save path
- Batch size
- Model architecture

These can be configured in `Table-Encoder/train.py`.