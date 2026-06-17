# ============================================================================
# Radiogenomic Analysis Pipeline
# ============================================================================
# Description: Transcriptomic and immune microenvironment characterization of
#   imaging-derived recurrence risk in TCGA-LIHC.
#
# Input files (place in working directory):
#   - risk_matched.rds        # Matched risk + clinical data (data.frame with
#                             #   columns: patient_id, Risk_group (High/Low),
#                             #   RFS_months, RFS_event)
#   - tpm_matched.rds         # TPM expression matrix (genes x patients)
#   - deg_results.csv         # DESeq2 output: gene, log2FoldChange, padj, etc.
#   - LM22.txt                # CIBERSORT signature matrix
#   - CIBERSORT.R             # CIBERSORT R script (Newman et al., 2015)
#   - 9606.protein.aliases.v12.0.txt.gz  # STRING protein alias file
#
# Required R packages (install if missing):
#   tidyverse, ggplot2, ggpubr, ggrepel, survival, survminer,
#   clusterProfiler, org.Hs.eg.db, igraph, ggraph, e1071
#
# Output:
#   output/Combined_Figure.pdf
#   output/Combined_Figure.png
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
library(survival)
library(survminer)
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
# Panel a: Kaplan-Meier survival curve
# ============================================================================
km_fit <- survfit(Surv(RFS_months, RFS_event) ~ Risk_group, data = risk_df)
logrank_p <- 1 - pchisq(survdiff(
  Surv(RFS_months, RFS_event) ~ Risk_group, data = risk_df)$chisq, 1)

km_plot <- ggsurvplot(km_fit, data = risk_df, pval = FALSE,
  risk.table = TRUE, risk.table.height = 0.22, risk.table.y.text = FALSE,
  xlab = "Months", ylab = "Recurrence-free survival",
  palette = c(color_low, color_high),
  legend.title = "Risk Group", legend.labs = c("Low", "High"),
  legend = c(0.85, 0.85),
  ggtheme = theme_classic(base_size = 20),
  break.time.by = 12, surv.plot.height = 0.78,
  tables.theme = theme_cleantable(base_size = 12), risk.table.fontsize = 5)
km_plot$plot <- km_plot$plot +
  annotate("text", x = 0, y = 0.10, label = sprintf("Log-rank p = %.3f", logrank_p),
           size = 7, hjust = 0) +
  theme(axis.title = element_text(size = 21), axis.text = element_text(size = 18))
p_a <- km_plot$plot

# ============================================================================
# Panel b: Volcano plot
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

p_b <- ggplot(volcano_df, aes(x = log2FoldChange, y = neg_log10_padj)) +
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
# Panel c: GO Biological Process enrichment
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

p_c <- ggplot(go_plot, aes(x = GeneRatio_num, y = reorder(Description, Count))) +
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
# Panel d: STRING PPI network + MCC hub genes
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
p_d <- ggraph(g_hub, layout = "fr") +
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
# Panel e: CIBERSORT immune cell deconvolution
# ============================================================================
cat("\n=== CIBERSORT ===\n")

# Prepare TPM input
tpm_input <- as.data.frame(tpm)
tpm_input$Gene <- rownames(tpm_input)
tpm_input <- tpm_input[, c("Gene", setdiff(colnames(tpm_input), "Gene"))]
write.table(tpm_input, "tpm_cibersort_input.txt",
            sep = "\t", row.names = FALSE, quote = FALSE)

source("CIBERSORT.R")
cib_result <- CIBERSORT(
  sig_matrix   = "LM22.txt",
  mixture_file = "tpm_cibersort_input.txt",
  perm = 100, QN = FALSE)

cell_fractions <- cib_result[, 1:22, drop = FALSE]
common <- intersect(risk_df$patient_id, rownames(cell_fractions))
cell_fractions <- cell_fractions[common, , drop = FALSE]
cat("CIBERSORT samples:", nrow(cell_fractions), "\n")

immune_mat <- as.matrix(cell_fractions[risk_df$patient_id, , drop = FALSE])
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
row1 <- ggarrange(p_a, p_b, ncol = 2, labels = c("a", "b"),
                  font.label = list(size = 24, face = "bold"))
row2 <- ggarrange(p_c, p_d, ncol = 2, labels = c("c", "d"),
                  font.label = list(size = 24, face = "bold"))
p_combined <- ggarrange(row1, row2, p_e, ncol = 1,
  labels = c("", "", "e"),
  font.label = list(size = 24, face = "bold"),
  heights = c(1, 1.35, 1))

pdf("output/Combined_Figure.pdf", width = 16, height = 22)
print(p_combined)
dev.off()
png("output/Combined_Figure.png", width = 16, height = 22,
    units = "in", res = 300)
print(p_combined)
dev.off()

cat("\nAnalysis complete. Output saved to output/\n")
