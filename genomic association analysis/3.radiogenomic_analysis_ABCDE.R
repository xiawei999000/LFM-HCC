# ============================================================================
# Radiogenomic Analysis Pipeline
# ============================================================================
# Description: Transcriptomic and immune microenvironment characterization of
#   imaging-derived recurrence risk in TCGA-LIHC.
#
# Input files (place in working directory):
#   - risk_matched.rds        # Matched risk data (data.frame with
#                             #   columns: patient_id, Risk_group (High/Low))
#   - tpm_matched.rds         # TPM expression matrix (genes x patients)
#   - deg_results.csv         # DESeq2 output: gene, log2FoldChange, padj, etc.
#   - LM22.txt                # CIBERSORT signature matrix
#   - CIBERSORT.R             # CIBERSORT R script (Newman et al., 2015)
#   - 9606.protein.aliases.v12.0.txt.gz  # STRING protein alias file
#
# Required R packages (install if missing):
#   tidyverse, ggplot2, ggpubr, ggrepel,
#   clusterProfiler, org.Hs.eg.db, igraph, ggraph, e1071
#
# Output:
#   output/GO_BP_Downregulated.csv
#   output/PPI_top10_hub_genes.csv
#   output/PPI_edges.csv
#   output/CIBERSORT_result.csv
#   output/CIBERSORT_group_comparison.csv
#
# Usage:
#   1. Place all input files in a single directory
#   2. Set that directory as working directory (setwd())
#   3. Source this script
# ============================================================================

library(tidyverse)
library(ggplot2)
library(ggpubr)
library(ggrepel)
library(clusterProfiler)
library(org.Hs.eg.db)
library(igraph)
library(ggraph)
library(e1071)

## Uncomment and set your working directory:
# setwd("path/to/your/data")

# Create output directory
dir.create("output", showWarnings = FALSE)

# Global color scheme
color_high <- "#D43F3A"
color_low  <- "#2E77BB"
color_grey <- "grey80"

# ============================================================================
# Load data
# ============================================================================
risk_df <- readRDS("risk_matched.rds")
tpm     <- readRDS("tpm_matched.rds")
deg_df  <- read.csv("deg_results.csv", stringsAsFactors = FALSE)

cat("Patients:", nrow(risk_df), "\n")
cat("High:", sum(risk_df$Risk_group == "High"),
    "Low:", sum(risk_df$Risk_group == "Low"), "\n")

# ============================================================================
# Panel a: Volcano plot
# ============================================================================
narrative_genes <- c(
  "PLPP2","MUC5B","CACNA1H","CPA6","DUOX2",
  "CKMT2","IGHV1-69","EOMES","CD38",
  "SH2D1A","BTLA","SLAMF6","CD27","CD8A","IL2RB",
  "LAG3","TIGIT","CTLA4","PDCD1")

volcano_df <- deg_df %>%
  filter(!is.na(padj)) %>%
  mutate(neg_log10_padj = -log10(padj),
         label = ifelse(gene %in% narrative_genes, gene, ""))

p_a <- ggplot(volcano_df, aes(x = log2FoldChange, y = neg_log10_padj)) +
  geom_point(aes(color = change), size = 1.2, alpha = 0.75) +
  scale_color_manual(
    values = c("Up in High-risk" = color_high,
               "Down in High-risk" = color_low,
               "Stable" = color_grey),
    labels = c("Down in high-risk", "Stable", "Up in high-risk")) +
  geom_text_repel(aes(label = label), size = 4.5, max.overlaps = 25,
                  box.padding = 0.3, min.segment.length = 0.1, force = 2) +
  geom_hline(yintercept = -log10(0.10), linetype = "dashed",
             color = "grey50", linewidth = 0.4) +
  geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed",
             color = "grey50", linewidth = 0.4) +
  scale_y_continuous(limits = c(0, 7.5), expand = expansion(mult = c(0, 0.05))) +
  labs(x = expression(log[2]~"FC (High-risk vs Low-risk)"),
       y = expression(-log[10]~"adjusted P value")) +
  theme_classic(base_size = 20) +
  theme(legend.position = c(0.02, 0.98), legend.justification = c(0, 1),
        legend.title = element_blank(), legend.text = element_text(size = 15),
        legend.background = element_rect(fill = "white", color = "grey80", linewidth = 0.3),
        axis.title = element_text(size = 21), axis.text = element_text(size = 18))

# ============================================================================
# Panel b: GO Biological Process enrichment (downregulated DEGs)
# Note: upregulated genes (High-risk) returned 0 significant BP terms,
# so only downregulated genes are shown. Both directions were tested.
# ============================================================================
down_genes <- deg_df$gene[deg_df$change == "Down in High-risk"]
up_genes   <- deg_df$gene[deg_df$change == "Up in High-risk"]
universe <- rownames(tpm)

# Upregulated: 0 significant terms (tested, result is empty)
ego_up <- enrichGO(gene = up_genes, universe = universe, OrgDb = org.Hs.eg.db,
                   keyType = "SYMBOL", ont = "BP", pAdjustMethod = "BH",
                   pvalueCutoff = 0.05, qvalueCutoff = 0.2)
cat("GO terms (upregulated):", nrow(as.data.frame(ego_up)), "\n")

# Downregulated: immune-enriched
ego <- enrichGO(gene = down_genes, universe = universe, OrgDb = org.Hs.eg.db,
                keyType = "SYMBOL", ont = "BP", pAdjustMethod = "BH",
                pvalueCutoff = 0.05, qvalueCutoff = 0.2)
go_result <- as.data.frame(ego)
cat("GO terms (downregulated):", nrow(go_result), "\n")
write.csv(go_result, "output/GO_BP_Downregulated.csv", row.names = FALSE)

go_plot <- go_result %>%
  arrange(p.adjust) %>%
  head(8) %>%
  mutate(
    Description = gsub(paste0("adaptive immune response based on somatic ",
      "recombination of immune receptors built from immunoglobulin ",
      "superfamily domains"), "adaptive immune response", Description),
    Description = str_wrap(Description, 20),
    GeneRatio_num = sapply(strsplit(GeneRatio, "/"),
                           function(x) as.numeric(x[1]) / as.numeric(x[2])))

p_b <- ggplot(go_plot, aes(x = GeneRatio_num, y = reorder(Description, Count))) +
  geom_point(aes(size = Count, color = -log10(p.adjust))) +
  scale_color_gradient(low = color_low, high = color_high,
                       name = expression(-log[10]~"adjusted P value")) +
  scale_size_continuous(name = "Gene count", range = c(3, 7)) +
  labs(x = "Gene Ratio", y = "") +
  scale_x_continuous(expand = expansion(mult = c(0.05, 0.35)),
                     breaks = scales::pretty_breaks(n = 3)) +
  theme_classic(base_size = 20) +
  theme(axis.text.y = element_text(size = 15, lineheight = 0.9, hjust = 0),
        axis.text.x = element_text(size = 13), axis.title.x = element_text(size = 19),
        legend.position = c(0.95, 0.20), legend.background = element_blank(),
        legend.text = element_text(size = 14), legend.title = element_text(size = 15),
        legend.key.size = unit(0.45, "cm"), plot.margin = margin(5, 5, 5, 5))

# ============================================================================
# Panel c: STRING PPI network + MCC hub genes
# ============================================================================
cat("\n=== STRING PPI + MCC ===\n")

aliases <- read.table(
  gzfile("9606.protein.aliases.v12.0.txt.gz"),
  header = FALSE, sep = "\t", stringsAsFactors = FALSE,
  quote = "", comment.char = "", fill = TRUE)
colnames(aliases) <- c("string_id", "alias", "source")

alias_map <- aliases %>%
  filter(alias %in% down_genes) %>%
  distinct(string_id, alias)
string_ids <- unique(alias_map$string_id)
cat("Mapped:", length(string_ids), "proteins from",
    length(unique(alias_map$alias)), "genes\n")

id_string <- paste(string_ids, collapse = "%0d")
url <- paste0("https://string-db.org/api/tsv/network?identifiers=",
              id_string, "&species=9606&required_score=700")
ppi_raw <- read.table(url, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
ppi <- ppi_raw %>%
  filter(stringId_A %in% string_ids & stringId_B %in% string_ids)

id2gene <- alias_map %>% distinct(string_id, .keep_all = TRUE)
ppi_edges <- data.frame(
  from  = id2gene$alias[match(ppi$stringId_A, id2gene$string_id)],
  to    = id2gene$alias[match(ppi$stringId_B, id2gene$string_id)],
  score = ppi$score,
  stringsAsFactors = FALSE) %>%
  filter(!is.na(from) & !is.na(to))

g <- simplify(graph_from_data_frame(
  ppi_edges[, c("from", "to")], directed = FALSE))
comps <- components(g)
g <- induced_subgraph(g,
  names(which(comps$membership == which.max(comps$csize))))
cat("Network:", vcount(g), "nodes,", ecount(g), "edges\n")

# Maximal Clique Centrality
all_cliques <- max_cliques(g, min = 3)
mcc_values <- rep(0, vcount(g))
names(mcc_values) <- V(g)$name
for (cl in all_cliques) {
  for (v in cl$name) mcc_values[v] <- max(mcc_values[v], length(cl))
}
top10_mcc <- sort(mcc_values, decreasing = TRUE)[1:min(10, length(mcc_values))]

hub_df <- data.frame(
  rank = 1:length(top10_mcc),
  Gene = names(top10_mcc),
  MCC  = as.integer(top10_mcc))
print(hub_df)
write.csv(hub_df, "output/PPI_top10_hub_genes.csv", row.names = FALSE)
write.csv(ppi_edges, "output/PPI_edges.csv", row.names = FALSE)

# Subgraph of hub genes only
g_hub <- induced_subgraph(g, names(top10_mcc))
lfc_vec <- setNames(deg_df$log2FoldChange, deg_df$gene)
V(g_hub)$mcc    <- mcc_values[V(g_hub)$name]
V(g_hub)$log2FC <- lfc_vec[V(g_hub)$name]

set.seed(42)
p_c <- ggraph(g_hub, layout = "fr") +
  geom_edge_link(alpha = 0.5, color = "grey25", edge_width = 1.0) +
  geom_node_point(aes(size = mcc, fill = log2FC), pch = 21,
                  color = "grey15", alpha = 0.95, stroke = 0.4) +
  scale_fill_gradient2(low = color_low, mid = "white", high = "grey70",
                       midpoint = -1, name = "log2FC") +
  scale_size_continuous(range = c(4, 12), name = "MCC") +
  geom_node_text(aes(label = name), size = 6.0, repel = TRUE,
                 max.overlaps = 15, fontface = "bold", color = "grey10") +
  labs(title = NULL) +
  theme_void() +
  theme(legend.position = "bottom", legend.box = "horizontal",
        legend.text = element_text(size = 10),
        legend.title = element_text(size = 14),
        legend.key.width = unit(1.2, "cm"),
        legend.spacing.x = unit(0.3, "cm"))

# ============================================================================
# Panel d: Immune cell infiltration heatmap (CIBERSORT z-score)
# ============================================================================
cat("\n=== Heatmap ===\n")

# --- Step 1: Run CIBERSORT ---
tpm_input <- as.data.frame(tpm)
tpm_input$Gene <- rownames(tpm_input)
tpm_input <- tpm_input[, c("Gene", setdiff(colnames(tpm_input), "Gene"))]
write.table(tpm_input, "tpm_cibersort_input.txt",
            sep = "\t", row.names = FALSE, quote = FALSE)

source("CIBERSORT.R")
cib_result <- CIBERSORT(sig_matrix = "LM22.txt",
  mixture_file = "tpm_cibersort_input.txt", perm = 100, QN = FALSE)

cell_fractions <- cib_result[, 1:22, drop = FALSE]
common <- intersect(risk_df$patient_id, rownames(cell_fractions))
cell_fractions <- cell_fractions[common, , drop = FALSE]
immune_mat <- as.matrix(cell_fractions[risk_df$patient_id, , drop = FALSE])
cat("CIBERSORT dim:", dim(immune_mat), "(samples x cells)\n")

# --- Step 2: Transpose to cells × samples ---
heat_mat <- t(immune_mat)  # 22 cells × N samples
rownames(heat_mat) <- gsub("\\.", " ", rownames(heat_mat))
cat("Heatmap matrix:", dim(heat_mat), "(cells x samples)\n")
cat("Row names:", paste(head(rownames(heat_mat), 3), collapse=", "), "...\n")
cat("Col names:", paste(head(colnames(heat_mat), 3), collapse=", "), "...\n")

# --- Step 3: Sort columns by risk group, Low-risk first ---
sample_order <- risk_df %>%
  mutate(Risk_group = factor(Risk_group, levels = c("Low", "High"))) %>%
  arrange(Risk_group) %>%
  pull(patient_id)
sample_order <- intersect(sample_order, colnames(heat_mat))

heat_mat <- heat_mat[, sample_order, drop = FALSE]
n_low  <- sum(risk_df$Risk_group[match(sample_order, risk_df$patient_id)] == "Low")
n_high <- sum(risk_df$Risk_group[match(sample_order, risk_df$patient_id)] == "High")
div_x <- n_low + 0.5
cat("Samples: Low=", n_low, "High=", n_high, "\n")

# --- Step 4: LM22 standard order ---
lm22_order <- c(
  "B cells naive", "B cells memory", "Plasma cells",
  "T cells CD8", "T cells CD4 naive", "T cells CD4 memory resting",
  "T cells CD4 memory activated", "T cells follicular helper",
  "T cells regulatory Tregs", "T cells gamma delta",
  "NK cells resting", "NK cells activated",
  "Monocytes", "Macrophages M0", "Macrophages M1", "Macrophages M2",
  "Dendritic cells resting", "Dendritic cells activated",
  "Mast cells resting", "Mast cells activated",
  "Eosinophils", "Neutrophils")
lm22_present <- intersect(lm22_order, rownames(heat_mat))
heat_mat <- heat_mat[lm22_present, , drop = FALSE]

# --- Step 5: Row-wise z-score, handle zero-variance rows ---
heat_mat_z <- t(scale(t(heat_mat)))
# Check for zero-variance rows (NaN from 0/0)
nan_rows <- which(apply(heat_mat_z, 1, function(x) all(is.nan(x))))
if (length(nan_rows) > 0) {
  cat("Zero-variance rows (set to 0):", paste(names(nan_rows), collapse=", "), "\n")
  heat_mat_z[nan_rows, ] <- 0
}
heat_mat_z <- pmin(pmax(heat_mat_z, -2), 2)

# --- Step 6: Build risk annotation ---
anno_labels <- risk_df$Risk_group[match(sample_order, risk_df$patient_id)]
stopifnot(all(!is.na(anno_labels)))

# --- Step 7: Melt ---
heatmap_df <- as.data.frame(heat_mat_z, check.names = FALSE) %>%
  rownames_to_column("Cell_type") %>%
  pivot_longer(-Cell_type, names_to = "Patient", values_to = "Z") %>%
  mutate(Cell_type = factor(Cell_type, levels = rownames(heat_mat_z)),
         Patient   = factor(Patient,   levels = sample_order))

# --- Step 8: Plot ---
anno_df <- data.frame(Patient = factor(sample_order, levels = sample_order),
                      Group = anno_labels)

p_anno <- ggplot(anno_df, aes(x = Patient, y = 1, fill = Group)) +
  geom_tile() +
  geom_vline(xintercept = div_x, color = "white", linewidth = 1.2) +
  scale_fill_manual(name = "Risk group",
                    values = c(High = color_high, Low = color_low)) +
  theme_void() +
  theme(legend.position = "right",
        legend.text = element_text(size = 12), legend.title = element_text(size = 13),
        legend.margin = margin(0, 40, 0, 0),
        plot.margin = margin(15, 5, 0, 0))

p_heat <- ggplot(heatmap_df, aes(x = Patient, y = Cell_type, fill = Z)) +
  geom_tile() +
  geom_vline(xintercept = div_x, color = "white", linewidth = 1.2) +
  scale_fill_gradient2(low = color_low, mid = "white", high = color_high,
                       midpoint = 0, limits = c(-2, 2), name = "Z-score",
                       na.value = "white") +
  labs(x = NULL, y = NULL) +
  theme_minimal(base_size = 13) +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
        axis.text.y = element_text(size = 11, face = "bold"),
        panel.grid = element_blank(),
        legend.position = "right", legend.text = element_text(size = 11),
        legend.title = element_text(size = 12), plot.margin = margin(0, 5, 0, 5))

p_d <- ggarrange(p_anno, p_heat, ncol = 1, heights = c(0.04, 0.96),
                 align = "v", common.legend = FALSE, legend = "right")

# ============================================================================
# Panel e: CIBERSORT immune cell deconvolution boxplots
# ============================================================================
cat("\n=== CIBERSORT boxplots ===\n")

# Reuse CIBERSORT results from Panel d
write.csv(as.data.frame(immune_mat) %>% rownames_to_column("patient_id"),
          "output/CIBERSORT_result.csv", row.names = FALSE)

# Wilcoxon test
comp_results <- lapply(colnames(immune_mat), function(ct) {
  vals <- immune_mat[, ct]
  wt <- wilcox.test(
    vals[risk_df$Risk_group == "High"],
    vals[risk_df$Risk_group == "Low"], exact = FALSE)
  data.frame(Cell_type = ct, p_value = wt$p.value)
})
comp_df <- bind_rows(comp_results) %>%
  arrange(p_value)
write.csv(comp_df, "output/CIBERSORT_group_comparison.csv", row.names = FALSE)
print(comp_df %>% dplyr::select(Cell_type, p_value) %>% head(8))

# Prepare data for boxplot
cell_long <- as.data.frame(immune_mat) %>%
  rownames_to_column("patient_id") %>%
  pivot_longer(-patient_id, names_to = "Cell_type", values_to = "Fraction") %>%
  left_join(risk_df %>% dplyr::select(patient_id, Risk_group),
            by = "patient_id") %>%
  left_join(comp_df, by = "Cell_type")

cell_order <- comp_df$Cell_type
cell_long$Cell_type <- factor(cell_long$Cell_type, levels = cell_order)

star_df <- comp_df %>%
  filter(p_value < 0.05) %>%
  mutate(x = match(Cell_type, cell_order))

p_e <- ggplot(cell_long, aes(x = Cell_type, y = Fraction, fill = Risk_group)) +
  geom_boxplot(outlier.size = 0.3, linewidth = 0.4, width = 0.6) +
  scale_y_continuous(limits = c(-0.01, 0.4),
                     expand = expansion(mult = c(0, 0.05))) +
  geom_text(data = star_df, aes(x = x, y = 0.38, label = "*"),
            inherit.aes = FALSE, size = 9, color = "black") +
  scale_fill_manual(name = "Risk group",
                    values = c(High = color_high, Low = color_low)) +
  labs(x = "", y = "CIBERSORT fraction", title = NULL) +
  theme_classic(base_size = 18) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 16),
        legend.position = c(0.99, 0.95), legend.justification = c(1, 1),
        legend.background = element_blank(),
        legend.text = element_text(size = 15),
        legend.title = element_text(size = 14))

# ============================================================================
# Assemble combined figure
# ============================================================================
# Row 1: Volcano (A) + GO (B)
row1 <- ggarrange(p_a, p_b, ncol = 2, labels = c("A", "B"),
                  font.label = list(size = 24, face = "bold"))
# Row 2: PPI (C) + Heatmap (D)
row2 <- ggarrange(p_c, p_d, ncol = 2, labels = c("C", "D"),
                  font.label = list(size = 24, face = "bold"))
# Row 3: CIBERSORT boxplots (E) full width
p_combined <- ggarrange(row1, row2, p_e, ncol = 1,
  labels = c("", "", "E"),
  font.label = list(size = 24, face = "bold"),
  heights = c(1.15, 0.85, 1))

dir.create("output", showWarnings = FALSE)
pdf("output/Combined_Figure_ABCDE.pdf", width = 16, height = 22)
print(p_combined)
dev.off()
png("output/Combined_Figure_ABCDE.png", width = 16, height = 22, units = "in", res = 300)
print(p_combined)
dev.off()
cat("\nFigure saved to output/Combined_Figure_ABCDE.pdf|png\n")
