# DP-GPT: GPT-Driven Gene Text Feature Embedding Fused with Gene Expression Data for Depression Prediction

## Introduction
In recent years, with the improvement of living standards, the prevalence of depression has been steadily increasing, making it a growing public health concern. Currently, the diagnosis of depression primarily relies on self-assessment questionnaires and subjective evaluation by clinicians, both of which are prone to high rates of misdiagnosis and missed diagnosis.  This highlights the need for objective and accurate diagnostic aids. Gene expression data can reveal links between genes and diseases. Studies have shown that gene expression in depression patients differs significantly from healthy individuals, offering potential for early detection. However, existing methods often depend on selecting differentially expressed genes, which may overlook important signals from other genes and are vulnerable to batch effects, limiting model generalization. To address these limitations, we propose DP-GPT: GPT-Driven Gene Text Feature Embedding Fused with Gene Expression Data for Depression Prediction. This method integrates gene expression data with features extracted by GPT. Specifically, gene names and summaries are obtained from the NCBI Gene database, then embedded using GPT to generate feature vectors. These are fused with sample gene expression data and fed into a classifier for prediction. Experiments show that DP-GPT achieves superior performance in depression prediction.

## Requirements
- python==3.8.19
- numpy==1.23.5
- pandas==2.0.3
- ipython==8.12.3
- torch==1.12.1
- scikit-learn==1.3.2


## Data
The original data was sourced from the GEO database, and OpenAI's text embedding API was used to generate text embeddings for each gene. Here we provide a subset of the generated gene embeddings along with collected gene expression values from a portion of the samples.
- `./Data/gene_embedding_ada_test.pickle`: The embeddings were obtained through GPT's text embedding interface based on gene abstracts.
- `./Data/Depression20_normalize_test.csv`: The gene expression dataset contains both depression and healthy control samples.
## Usage
### 1. Data Preprocessing
Firstly, preprocess the input microarray data by normalizing the data. For detailed implementation instructions, please refer to `/DataProcess/preprocess.py`.

### 2. Get Gene_embedding
The corresponding gene embeddings were selected from existing embeddings based on real-world data collected from the GEO database for subsequent prediction tasks.For detailed implementation instructions, please refer to `/DP-GPT/gene_embedding.py`.

### 3. Predicting
Next, we will first perform matrix multiplication between the obtained embeddings and the collected gene expression data, then take the average, and finally use KNN for classification prediction.For detailed implementation instructions, please refer to `/DP-GPT/predict.py`.

### 4. Model Output
The final output of the model includes the predicted class for each sample as well as overall performance evaluation metrics such as Accuracy, Precision, Recall, and F1-score.
## Contact
Please feel free to contact us for any further questions:
- Min Li limin@mail.csu.edu.cn
## References
DP-GPT: GPT-Driven Gene Text Feature Embedding Fused with Gene Expression Data for Depression Prediction
