# Genomic Association Analysis

Transcriptomic and immune microenvironment characterization of imaging-derived recurrence risk in TCGA-LIHC.

## Pipeline

| Step | Script | Input → Output |
|:---:|--------|----------------|
| 1 | `1.prepare_data.R` | TCGA raw counts/TPM + risk xlsx → matched RDS |
| 2 | `2.prepare_deg.R` | matched counts + risk → DEG table |
| 3 | `3.radiogenomic_analysis_ABCDE.R` | matched TPM + risk + DEG → volcano + GO + STRING PPI + CIBERSORT heatmap + boxplots |

Pre-computed matched files and DEG table are included, so you can run step 3 directly:

```r
setwd("path/to/this/directory")
source("3.radiogenomic_analysis_ABCDE.R")
```

## Required R packages

```r
install.packages(c("tidyverse", "ggplot2", "ggpubr", "ggrepel",
    "igraph", "ggraph", "e1071", "DESeq2", "readxl"))
BiocManager::install(c("clusterProfiler", "org.Hs.eg.db"))
```

## Dependencies

| File | Source |
|------|--------|
| `CIBERSORT.R` | Newman et al., 2015, *Nature Methods* |
| `LM22.txt` | CIBERSORT leukocyte gene signature matrix |
| `9606.protein.aliases.v12.0.txt.gz` | STRING v12.0 protein alias mapping |

## Methods

- **Differential expression**: DESeq2 (Wald test, apeglm shrinkage, FDR < 0.10, |log2FC| > 0.5)
- **GO enrichment**: clusterProfiler (hypergeometric test, BH correction)
- **PPI network**: STRING v12.0 (score ≥ 700, REST API) + Maximal Clique Centrality (igraph)
- **Immune deconvolution**: CIBERSORT (ν-SVR, LM22 signature, 100 permutations, QN = FALSE)
- **Group comparison**: Wilcoxon rank-sum test (nominal p-values)
