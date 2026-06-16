# ============================================================================
# prepare_deg.R — DESeq2 differential expression analysis
# Run this ONCE to generate deg_results.csv, or skip if already present.
#
# Input:  counts_matched.rds, risk_matched.rds
# Output: deg_results.csv
# ============================================================================

library(DESeq2)
library(tidyverse)

## Uncomment and set your working directory:
# setwd("path/to/your/data")

# Load data
counts <- readRDS("counts_matched.rds")
risk   <- readRDS("risk_matched.rds")

# Filter low-expression genes
keep <- rowSums(counts >= 10) >= 3
counts <- counts[keep, ]

# DESeq2
dds <- DESeqDataSetFromMatrix(
  countData = round(as.matrix(counts)),
  colData   = risk,
  design    = ~ Risk_group)
dds <- DESeq(dds)

# Extract results
res <- results(dds, contrast = c("Risk_group", "High", "Low"), alpha = 0.10)
coef_name <- resultsNames(dds)[grep("High", resultsNames(dds))]
res_shrunk <- lfcShrink(dds, coef = coef_name, type = "apeglm", quiet = TRUE)

deg_df <- data.frame(
  gene           = rownames(res),
  baseMean       = res$baseMean,
  log2FoldChange = res_shrunk$log2FoldChange,
  lfcSE          = res$lfcSE,
  stat           = res$stat,
  pvalue         = res$pvalue,
  padj           = res$padj,
  stringsAsFactors = FALSE)

# Classify
deg_df$change <- "Stable"
deg_df$change[!is.na(deg_df$padj) & deg_df$padj < 0.10 &
              deg_df$log2FoldChange > 0.5] <- "Up in High-risk"
deg_df$change[!is.na(deg_df$padj) & deg_df$padj < 0.10 &
              deg_df$log2FoldChange < -0.5] <- "Down in High-risk"

cat(sprintf("DEGs: %d up, %d down (padj<0.10, |LFC|>0.5)\n",
            sum(deg_df$change == "Up in High-risk"),
            sum(deg_df$change == "Down in High-risk")))

write.csv(deg_df, "deg_results.csv", row.names = FALSE)
cat("Saved deg_results.csv\n")
