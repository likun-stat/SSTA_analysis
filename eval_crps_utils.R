# ============================================================================
# eval_crps_utils.R
#
# Shared CRPS / tail-weighted CRPS / extreme-conditional evaluation
# utilities, matching the methodology described in
# gru_jump_mixture_comparison.tex ("Evaluation Methodology: Proper Scoring
# for Extremes"). No implementation of these existed anywhere in this
# project's saved .R files prior to this one -- the numbers in that tex
# were produced by code that was never saved to disk. This file is written
# to be the durable, reusable version going forward.
#
# All scoring here operates on the MODEL (log) scale (X = log(Z_obs)), never
# the original/exponentiated scale, for the reason given in the tex: E[exp(Y)]
# is infinite for a genuine Student-t Y regardless of degrees of freedom, so
# naive Monte-Carlo estimates of original-scale point/interval statistics are
# unstable estimates of an infinite population quantity. CRPS's rank-based
# construction has no such pathology on the log scale.
# ============================================================================

# ------------------------------------------------------------------------
# 1. Fair (bias-corrected) ensemble CRPS estimator, O(n log n) via the
#    sorted-vector identity (matches Eq. in gru_jump_mixture_comparison.tex,
#    \S "CRPS"):
#
#    CRPS_hat(x_1:n, y) = (1/n) sum_i |x_i - y|
#                         - (1/(n(n-1))) sum_{k=1}^n (2k-n-1) x_(k)
#
#    The second term is the "fair"/bias-corrected estimate of
#    (1/2) E|X-X'| for X,X' iid ~ F (Ferro et al. 2008): it excludes the
#    n self-pairs (i=j) that a naive plug-in second-moment estimator would
#    include, which is what "fair" means here -- it does NOT refer to
#    tail-weighting (that's a separate concept, \S 2 below).
# ------------------------------------------------------------------------
crps_fair_1 <- function(x, y) {
  n <- length(x)
  if (n < 2L) stop("crps_fair_1 needs at least 2 ensemble draws.")
  xs <- sort(x)
  term1 <- mean(abs(x - y))
  k <- seq_len(n)
  term2 <- sum((2 * k - n - 1) * xs) / (n * (n - 1))
  term1 - term2
}

# Vectorized over a (n_origin x d x n_ens) prediction array and an
# (n_origin x d) truth matrix -- the shape returned by
# direct_hstep_ensemble_gru()/_lstm(). Returns an (n_origin x d) matrix of
# per-cell CRPS values (model/log scale).
crps_fair_array <- function(pred_array, truth) {
  n_origin <- dim(pred_array)[1]; d <- dim(pred_array)[2]
  stopifnot(dim(truth)[1] == n_origin, dim(truth)[2] == d)
  out <- matrix(NA_real_, n_origin, d)
  for (i in seq_len(n_origin)) for (j in seq_len(d)) {
    out[i, j] <- crps_fair_1(pred_array[i, j, ], truth[i, j])
  }
  out
}

# ------------------------------------------------------------------------
# 2. Tail-weighted CRPS (Gneiting & Ranjan 2011): apply the censoring
#    (chaining) function v(x) = max(x, u) to BOTH the ensemble draws and
#    the truth before computing CRPS. This concentrates scoring weight on
#    the upper tail above threshold u, exactly as in
#    gru_jump_mixture_comparison.tex \S "Tail-weighted CRPS".
# ------------------------------------------------------------------------
twcrps_fair_array <- function(pred_array, truth, u) {
  n_origin <- dim(pred_array)[1]; d <- dim(pred_array)[2]
  out <- matrix(NA_real_, n_origin, d)
  for (i in seq_len(n_origin)) for (j in seq_len(d)) {
    x_censored <- pmax(pred_array[i, j, ], u)
    y_censored <- max(truth[i, j], u)
    out[i, j] <- crps_fair_1(x_censored, y_censored)
  }
  out
}

# ------------------------------------------------------------------------
# 3. Pooled empirical thresholds u_90 / u_95 from the TRUE test-set values
#    (model/log scale), shared across every method for a like-for-like
#    comparison -- matches "the pooled empirical 90th/95th percentiles of
#    the true test-set values" in the tex.
# ------------------------------------------------------------------------
pooled_extreme_thresholds <- function(truth, probs = c(0.90, 0.95)) {
  q <- quantile(as.vector(truth), probs = probs, na.rm = TRUE)
  names(q) <- paste0("u", probs * 100)
  q
}

# ------------------------------------------------------------------------
# 4. One-call summary matching Table~\ref{tab:results}'s columns:
#    MSE_model, CRPS, twCRPS_u90, twCRPS_u95, MAE|extreme, CRPS|extreme.
#    pred_mean: (n_origin x d) ensemble-mean point forecast, model scale
#    (this is what MSE_model/MAE|extreme are computed against, matching
#    direct_hstep_ensemble_gru()'s own pred_mean/mspe_model_scale fields).
# ------------------------------------------------------------------------
summarize_extreme_scores <- function(pred_array, truth, pred_mean, u90, u95) {
  crps_mat <- crps_fair_array(pred_array, truth)
  tw90_mat <- twcrps_fair_array(pred_array, truth, u90)
  tw95_mat <- twcrps_fair_array(pred_array, truth, u95)

  extreme_idx <- which(as.vector(truth) > u95)
  mse_model <- mean((pred_mean - truth)^2, na.rm = TRUE)
  mae_extreme <- mean(abs(as.vector(pred_mean) - as.vector(truth))[extreme_idx], na.rm = TRUE)
  crps_extreme <- mean(as.vector(crps_mat)[extreme_idx], na.rm = TRUE)

  list(
    MSE_model = mse_model,
    CRPS = mean(crps_mat, na.rm = TRUE),
    twCRPS_u90 = mean(tw90_mat, na.rm = TRUE),
    twCRPS_u95 = mean(tw95_mat, na.rm = TRUE),
    MAE_extreme = mae_extreme,
    CRPS_extreme = crps_extreme,
    n_extreme = length(extreme_idx),
    n_cells = length(as.vector(truth)),
    crps_mat = crps_mat, tw90_mat = tw90_mat, tw95_mat = tw95_mat
  )
}

# ------------------------------------------------------------------------
# 5. Empirical coverage at a set of central nominal levels, from an ensemble
#    predictive sample -- used for the calibration sections.
# ------------------------------------------------------------------------
empirical_coverage <- function(pred_array, truth, levels = c(0.50, 0.80, 0.90, 0.95)) {
  n_origin <- dim(pred_array)[1]; d <- dim(pred_array)[2]
  out <- setNames(numeric(length(levels)), paste0(levels * 100, "%"))
  for (li in seq_along(levels)) {
    alpha <- (1 - levels[li]) / 2
    covered <- 0L; total <- 0L
    for (i in seq_len(n_origin)) for (j in seq_len(d)) {
      qs <- quantile(pred_array[i, j, ], probs = c(alpha, 1 - alpha), na.rm = TRUE)
      covered <- covered + as.integer(truth[i, j] >= qs[1] && truth[i, j] <= qs[2])
      total <- total + 1L
    }
    out[li] <- covered / total
  }
  out
}

# ------------------------------------------------------------------------
# 6. Self-check: verify crps_fair_1 against the closed-form Gaussian CRPS,
#    CRPS(N(mu,sigma^2), y) = sigma * [ w*(2*Phi(w)-1) + 2*phi(w) - 1/sqrt(pi) ],
#    w = (y-mu)/sigma. Run once at source() time; matches the tex's claim
#    ("verified against the closed-form Gaussian CRPS ... at n=500-5000").
# ------------------------------------------------------------------------
.crps_gaussian_closed_form <- function(mu, sigma, y) {
  w <- (y - mu) / sigma
  sigma * (w * (2 * pnorm(w) - 1) + 2 * dnorm(w) - 1 / sqrt(pi))
}

.verify_crps_fair_1 <- function(n_list = c(500, 2000, 5000), seed = 1) {
  set.seed(seed)
  mu <- 0.7; sigma <- 1.3; y <- 1.9
  closed <- .crps_gaussian_closed_form(mu, sigma, y)
  cat(sprintf("Closed-form Gaussian CRPS: %.5f\n", closed))
  for (n in n_list) {
    draws <- rnorm(n, mu, sigma)
    est <- crps_fair_1(draws, y)
    cat(sprintf("  n=%5d: fair-CRPS estimate = %.5f (diff = %+.5f)\n", n, est, est - closed))
  }
  invisible(closed)
}

if (identical(Sys.getenv("EVAL_CRPS_UTILS_SELFTEST"), "1")) {
  .verify_crps_fair_1()
}
