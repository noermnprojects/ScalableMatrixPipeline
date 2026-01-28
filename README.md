# ⚙️ Scalable Matrix Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Pipeline](https://img.shields.io/badge/Architecture-ETL-orange)
![Status](https://img.shields.io/badge/Performance-High_Throughput-success)

**Scalable-Matrix-Pipeline** is a high-performance **ETL (Extract, Transform, Load)** engine designed to optimize storage and processing of high-dimensional datasets via **Randomized Numerical Linear Algebra (RandNLA)**.

Built for Big Data environments, this pipeline replaces computationally expensive deterministic operations ($\mathcal{O}(mn^2)$) with probabilistic sketching streams ($\mathcal{O}(mn \log k)$), enabling real-time compression and feature extraction on massive image tensors.

---

## 🏗️ Architecture & Data Flow

### 1. Ingestion & Sketching Layer
The pipeline ingests raw dense matrices and applies a **Dimensionality Reduction** pass using Gaussian sketching ($Y = A\Omega$).
* **Streamlined Processing:** Avoids loading the full matrix into memory for singular value decomposition, mimicking streaming algorithms.
* **Adaptive Sampling:** Dynamically adjusts the rank $k$ based on a target error tolerance $\epsilon$ (e.g., $10^{-2}$), ensuring data quality SLAs are met.

### 2. Transformation Engine (NMF & CUR)
The transformation layer cleans and structures the data for downstream analytics:
* **Feature Extraction (NMF):** Decomposes signals into non-negative additive parts ($A \approx WH$), optimizing data for Machine Learning models requiring interpretability.
* **Data Selection (CUR):** Extracts the most significant rows/columns from the raw dataset, preserving sparsity and physical meaning for storage optimization.

### 3. Parallel Channel Processing
The engine treats color images as 3-channel tensors, executing independent processing threads for Red, Green, and Blue channels to maximize throughput and utilize multi-core architectures.

---

## 🚀 Engineering Metrics

* **Throughput:** Log-linear complexity allows processing of 4K+ resolution matrices in sub-second timeframes.
* **Scalability:** Designed to handle "Out-of-Core" datasets where traditional `numpy.linalg.svd` fails due to RAM bottlenecks.
* **Integration:** Outputs standard compressed arrays ready for injection into Data Lakes (Parquet/HDF5) or ML Pipelines.

---

## 📦 Installation & Deployment

```bash
# Clone the repository
git clone [https://github.com/noermnproject/ScalableMatrixPipeline.git](https://github.com/noermnproject/ScalableMatrixPipeline.git)

# Install dependencies
pip install numpy scipy matplotlib pillow
