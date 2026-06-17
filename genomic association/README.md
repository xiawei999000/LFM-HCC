# Genomic Association Analysis

## Scripts

| File | Purpose |
|------|---------|
| `prepare_deg.R` | DESeq2 differential expression: raw counts + risk group → DEG table |
| `radiogenomic_analysis.R` | Downstream analysis: KM survival, volcano plot, GO enrichment, STRING PPI + MCC hub genes, CIBERSORT immune deconvolution |

## Dependencies

| File | Source |
|------|--------|
| `CIBERSORT.R` | Newman et al., 2015, *Nature Methods* |
| `LM22.txt` | CIBERSORT leukocyte gene signature matrix |
| `9606.protein.aliases.v12.0.txt.gz` | STRING v12.0 protein alias mapping |
