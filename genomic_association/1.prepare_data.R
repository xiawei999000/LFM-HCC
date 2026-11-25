# ============================================================================
# prepare_data.R — Match imaging risk scores with TCGA-LIHC RNA-seq data
# Run this ONCE to generate counts_matched.rds, tpm_matched.rds, risk_matched.rds
# ============================================================================
#
# Prerequisites before running this script:
#   1. TCGA-LIHC expression data (see Step 1 below)
#   2. Imaging risk score file (xlsx or csv)
#
# Output:
#   counts_matched.rds  — raw counts matrix (genes × matched patients)
#   tpm_matched.rds     — TPM matrix (genes × matched patients)
#   risk_matched.rds    — matched risk + clinical data
# ============================================================================

library(tidyverse)
library(readxl)
library(readr)

# ============================================================================
# Step 1: Obtain TCGA-LIHC RNA-seq data (choose one method)
# ============================================================================
#
# Method A — Download with TCGAbiolinks (requires internet):
#
#   library(TCGAbiolinks)
#   query <- GDCquery(project = "TCGA-LIHC",
#     data.category = "Transcriptome Profiling",
#     data.type     = "Gene Expression Quantification",
#     workflow.type = "STAR - Counts",
#     sample.type   = "Primary Tumor")
#   GDCdownload(query)
#   exp_data <- GDCprepare(query)
#   counts_raw <- assay(exp_data, "unstranded")        # raw counts
#   tpm_raw    <- assay(exp_data, "tpm_unstrand")      # TPM
#
#   # Filter to primary tumor + deduplicate
#   barcodes <- colnames(counts_raw)
#   is_primary <- substr(barcodes, 14, 15) == "01"
#   counts_raw <- counts_raw[, is_primary]
#   tpm_raw    <- tpm_raw[, is_primary]
#   patient_ids <- substr(colnames(counts_raw), 1, 12)
#   keep <- !duplicated(patient_ids)
#   counts_raw <- counts_raw[, keep]
#   tpm_raw    <- tpm_raw[, keep]
#   colnames(counts_raw) <- colnames(tpm_raw) <- substr(colnames(counts_raw), 1, 12)
#
#   # Annotate with gene symbols
#   gene_info <- rowRanges(exp_data)
#   gene_symbols <- mcols(gene_info)$gene_name
#   rownames(counts_raw) <- rownames(tpm_raw) <- gene_symbols
#
#   saveRDS(counts_raw, "counts_raw.rds")
#   saveRDS(tpm_raw,    "tpm_raw.rds")
#
# Method B — Use pre-downloaded RDS files (if you already have them):
#   Place counts_raw.rds and tpm_raw.rds in the working directory.

# ============================================================================
# Step 2: Load and match data
# ============================================================================

## Uncomment and set your working directory:
setwd("F:\\HCC_LFM\\github\\genomic association")

# Load TCGA expression data
counts_raw <- readRDS("counts_raw.rds")
tpm_raw    <- readRDS("tpm_raw.rds")

cat("Expression data loaded:", ncol(counts_raw), "patients,",
    nrow(counts_raw), "genes\n")

# Load imaging risk scores
risk_df <- read_excel("risk_file.xlsx")

# Standardize patient ID to TCGA 12-character format
risk_df$patient_id <- substr(risk_df$ID, 1, 12)
risk_df <- risk_df[!duplicated(risk_df$patient_id), ]

cat("Risk data:", nrow(risk_df), "patients\n")

# ============================================================================
# Step 3: Match and save
# ============================================================================
common <- intersect(risk_df$patient_id, colnames(counts_raw))
cat("Matched patients:", length(common), "\n")

risk_matched <- risk_df %>%
  filter(patient_id %in% common) %>%
  arrange(patient_id) %>%
  rename(RFS_months = RFS, RFS_event = RFS_status) %>%
  mutate(Risk_group = factor(Risk_group, levels = c("Low", "High")))

counts_matched <- counts_raw[, risk_matched$patient_id, drop = FALSE]
tpm_matched    <- tpm_raw[, risk_matched$patient_id, drop = FALSE]

# Clean column names
colnames(counts_matched) <- risk_matched$patient_id
colnames(tpm_matched)    <- risk_matched$patient_id

# Save
saveRDS(counts_matched, "counts_matched.rds")
saveRDS(tpm_matched,    "tpm_matched.rds")
saveRDS(risk_matched,   "risk_matched.rds")

cat("Saved: counts_matched.rds, tpm_matched.rds, risk_matched.rds\n")
cat(sprintf("  %d patients (High=%d, Low=%d)\n",
            nrow(risk_matched),
            sum(risk_matched$Risk_group == "High"),
            sum(risk_matched$Risk_group == "Low")))
