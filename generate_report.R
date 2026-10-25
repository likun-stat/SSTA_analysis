# ============================================================================
# generate_report.R
#
# Gathers test-set scores and October-2026 forecasts from whichever of
# D-EESN / DSNO / FNO-DSTM have finished running, builds a combined
# comparison table and summary figures, and saves them for report.tex to
# embed. Safe to run at any point -- each model's block is guarded by
# file.exists(), so partial results (e.g. while a long FNO-DSTM run is
# still in progress) are handled gracefully; rerun after each model
# finishes to pick up its results.
# ============================================================================

metrics <- c("MSE_model", "CRPS", "twCRPS_u90", "twCRPS_u95", "MAE_extreme", "CRPS_extreme")
cov_levels <- c("50%", "80%", "90%", "95%")

models <- list()

if (file.exists("deesn_h6_comparison.rds")) {
  d <- readRDS("deesn_h6_comparison.rds")
  models[["D-EESN"]] <- list(scores = d$scores[metrics], cov = d$cov, CFG = d$CFG,
                             config_str = sprintf("numLayers=%d, layerOneDimRed=%d, E=%d",
                                                   d$CFG$numLayers, d$CFG$layerOneDimRed, d$E_FINAL))
}
if (file.exists("dsno_h6_comparison.rds")) {
  d <- readRDS("dsno_h6_comparison.rds")
  models[["DSNO (tuned)"]] <- list(scores = d$scores[metrics], cov = d$cov, CFG = d$CFG,
                           config_str = sprintf("p=27, K=%d, D=%d, W=%d, E=500", d$CFG$K, d$CFG$D, d$CFG$W))
}
if (file.exists("chris_exact_h6_comparison.rds")) {
  d <- readRDS("chris_exact_h6_comparison.rds")
  models[["DSNO (Chris-exact)"]] <- list(scores = d$scores[metrics], cov = d$cov, CFG = d$CFG,
                           config_str = sprintf("p=%d, K=%d, D=%d, W=%d, E=500", d$CFG$p, d$CFG$K, d$CFG$D, d$CFG$W))
}
if (file.exists("fno_dstm2d_h6_tuned_comparison.rds")) {
  d <- readRDS("fno_dstm2d_h6_tuned_comparison.rds")
  models[["FNO-DSTM"]] <- list(scores = d$scores_phys[metrics], cov = d$cov_phys, CFG = d$CFG,
                               config_str = sprintf("width=%d, modes_lon=%d, modes_lat=%d, 1 seed x 500 ens (test fit)",
                                                     d$CFG$width, d$CFG$modes_lon, d$CFG$modes_lat))
}

cat(sprintf("Models with completed test-set evaluation: %s\n", paste(names(models), collapse = ", ")))
if (length(models) == 0) { cat("Nothing to report yet.\n"); quit(save = "no") }

# ---- Combined test-set score table ----
score_tab <- data.frame(metric = metrics)
for (nm in names(models)) score_tab[[nm]] <- sapply(metrics, function(mm) models[[nm]]$scores[[mm]])
cat("\n================ Test-set scores (h=6, physical grid) ================\n")
print(score_tab, row.names = FALSE, digits = 4)

cov_tab <- data.frame(level = cov_levels)
for (nm in names(models)) cov_tab[[nm]] <- models[[nm]]$cov[cov_levels]
cat("\n================ Empirical coverage (nominal level in rows) ================\n")
print(cov_tab, row.names = FALSE, digits = 3)

saveRDS(list(score_tab = score_tab, cov_tab = cov_tab,
             config_str = sapply(models, function(x) x$config_str)),
        "report_tables.rds")
cat("\nSaved report_tables.rds\n")

# ---- Figures ----
all_cols <- c("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2")
model_cols <- all_cols[seq_along(models)]

png("report_fig_crps.png", width = 1100, height = 650, res = 130)
par(mar = c(9, 4.5, 3, 1))
bp <- barplot(as.matrix(score_tab[score_tab$metric %in% c("CRPS", "twCRPS_u90", "twCRPS_u95"), -1]),
             beside = TRUE, col = c("#4C72B0", "#DD8452", "#55A868")[1:3],
             names.arg = names(models), las = 2,
             legend.text = c("CRPS", "twCRPS (u90)", "twCRPS (u95)"),
             args.legend = list(x = "topright", bty = "n", cex = 0.85),
             main = "Test-set CRPS and extreme-weighted CRPS (lower = better)", ylab = "score", cex.names = 0.8)
dev.off()
cat("Saved report_fig_crps.png\n")

png("report_fig_coverage.png", width = 1000, height = 650, res = 130)
par(mar = c(4.5, 4.5, 3, 1))
nominal <- c(0.5, 0.8, 0.9, 0.95)
plot(nominal, nominal, type = "l", lty = 2, col = "grey50", xlim = c(0.4, 1), ylim = c(0, 1),
     xlab = "nominal coverage", ylab = "empirical coverage",
     main = "Calibration: empirical vs. nominal interval coverage")
for (i in seq_along(models)) lines(nominal, cov_tab[[names(models)[i]]], type = "b", pch = 19, col = model_cols[i])
legend(0.4, 1.02, c("perfect calibration", names(models)), col = c("grey50", model_cols),
       lty = c(2, rep(1, length(models))), pch = c(NA, rep(19, length(models))), bty = "n", cex = 0.85)
dev.off()
cat("Saved report_fig_coverage.png\n")

# ---- Oct-2026 forecast summary (EOF1, the dominant mode -- ENSO-related) ----
# D-EESN and DSNO forecast directly in EOF-coefficient space (draws_pc).
# FNO-DSTM forecasts directly on the physical grid (draws_ocean_torch_order);
# its EOF1-equivalent is obtained by projecting those grid-space draws back
# onto the EOF1 spatial pattern, load("eof_results.RData")'s Psi_K/col_mean
# (same projection final_plot_oct2026.R's reconstruction inverts), after
# reordering from torch-canonical to EOF/R-native ocean-cell order via
# perm_t2r (as in final_plot_oct2026.R).
fc <- list()
if (file.exists("final_deesn_oct2026.rds")) fc[["D-EESN"]] <- readRDS("final_deesn_oct2026.rds")$draws_pc
if (file.exists("final_dsno_oct2026.rds")) fc[["DSNO (tuned)"]] <- readRDS("final_dsno_oct2026.rds")$draws_pc
if (file.exists("chris_exact_oct2026.rds")) fc[["DSNO (Chris-exact)"]] <- readRDS("chris_exact_oct2026.rds")$draws_pc
if (file.exists("final_fno_dst_oct2026.rds")) {
  d <- readRDS("final_fno_dst_oct2026.rds")
  if (!is.null(d$draws_pc)) {
    fc[["FNO-DSTM"]] <- d$draws_pc
  } else if (!is.null(d$draws_ocean_torch_order)) {
    load("eof_results.RData")  # Psi_K, col_mean
    load("perm_t2r.RData")     # perm_t2r
    phys_r <- d$draws_ocean_torch_order[perm_t2r, ]           # (m, n_ens), R-native order
    fc[["FNO-DSTM"]] <- t(Psi_K) %*% (phys_r - col_mean)       # (K, n_ens), EOF-coefficient space
  }
}
if (length(fc) > 0) {
  cat("\n================ Oct-2026 forecast, EOF1 coefficient ================\n")
  for (nm in names(fc)) {
    v <- fc[[nm]][1, ]
    cat(sprintf("%-10s median=%.1f  mean=%.1f  [5%%,95%%]=[%.1f,%.1f]\n",
                nm, median(v), mean(v), quantile(v, 0.05), quantile(v, 0.95)))
  }
}
cat("\nDone.\n")
