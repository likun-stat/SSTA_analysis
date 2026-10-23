# ============================================================================
# deesn_eof_h6.R
#
# Patrick McDermott's Deep Ensemble Echo State Network (D-EESN; McDermott &
# Wikle 2019, Environmetrics, "Deep echo state networks with uncertainty
# quantification for spatio-temporal forecasting") applied to the K=27 EOF
# coefficient series, at h=6 lead -- same as every other model in this study.
#
# We use the AUTHORS' OWN public implementation (BD-EESN/Functions/
# masterBRNNFunctions_001.R + ensembleESNCPPFuncts_001.cpp, vendored directly
# into this directory -- originally downloaded and Eigen-compatibility-patched
# by a sibling project's Lorenz-96 D-EESN benchmark run, from
# github.com/schafert/BD-EESN), not a
# reimplementation: this is a genuinely different architecture family (random
# reservoirs + ridge-regression readout, no gradient training at all) from
# every other model here, so matching their exact vendor code is the only way
# to be faithful to the paper.
#
# Architecture summary (see McDermott & Wikle 2019, Sec 3 "Deep/Multi-Layer
# ESN"): each of numLayers-1 "reservoir layers" maps its (curNhOne-dim)
# echo-state expansion down to a dimRedDim-dim EOF bottleneck before feeding
# the next layer -- this hierarchical bottlenecking (not any single reservoir)
# is the paper's headline contribution over a flat/shallow ESN. The final
# layer's reservoir + all bottleneck "feature links" (skip connections from
# every layer) feed one ridge-regression readout to the p=27 EOF-coefficient
# target, DIRECTLY (no log/exp -- unlike the Lorenz-96 benchmark's positive
# lognormal data, EOF coefficients are signed, matching every other
# EOF-coefficient model's own "identity" response-transform convention in
# this study).
#
# Uncertainty quantification: like Dual-Stream NO, D-EESN (non-Bayesian
# variant, which is what we use here -- the fully Bayesian rjMCMC version in
# the same authors' repo is a much heavier, separate model) has NO innovation
# term of its own; all predictive spread comes from an ensemble of E
# independently-drawn RANDOM reservoirs (fixed random W/U weight matrices per
# member, only the ridge readout is "fit"), read off as ensemble disagreement
# -- the authors' own prescribed uncertainty-quantification mechanism (their
# Sec 4), and exactly the "deep ensemble" pattern already used for Dual-
# Stream NO in this study.
# ============================================================================

AUTHOR_CODE_DIR <- "BD-EESN"

.old_wd <- getwd()
setwd(AUTHOR_CODE_DIR)
source(file.path("Functions", "masterBRNNFunctions_001.R"))
setwd(.old_wd)

# Globals referenced inside the authors' function file (same values used by
# this project's own run_mcdermott_deesn_on_X.R Lorenz-96 benchmark).
quadInd <- FALSE          # no quadratic/interaction reservoir features
featureLinkInd <- TRUE    # include every layer's EOF bottleneck as a skip-connected "feature link" into the final readout (the paper's own default)
scaleByHInd <- FALSE      # scale EOF bottleneck features by their OVERALL sd, not per-hidden-unit

load("eof_results.RData")   # A_K, Psi_K, col_mean

# ----------------------------------------------------------------------
# Domain-agnostic ESN constants, fixed across the whole tuning grid --
# the paper's own Lorenz-96 defaults for "how big/sparse/scaled is a
# generic reservoir" (analogous to Dual-Stream NO's W=36-months-fixed
# treatment): not this paper's headline design axes, so not worth
# spending a tuning budget on.
# ----------------------------------------------------------------------
CUR_NH_ONE <- 84L   # first-stage reservoir width (every layer's own internal reservoir, before EOF bottlenecking)
CUR_NH_TWO <- 64L   # final-stage reservoir width (feeds the readout, alongside all layers' feature links)
CUR_RIDGE <- 1e-3   # ridge penalty (generic; the paper's own per-numLayers values were genetic-algorithm-tuned specifically for Lorenz-96 and not portable)
PI_W <- 0.10; PI_U <- 0.10; W_WIDTH <- 0.10; U_WIDTH <- 0.10   # reservoir sparsity / weight scale, both stages (paper's own Lorenz-96 defaults)
M_EMBED <- 3L        # number of embedding lags (paper's own example value)
DIM_RED_ESN <- "EOF"

#' Build the authors' embedding/design-matrix object directly on the EOF
#' coefficient matrix `rawData` (T x p, identity scale -- no log).
#'
#' `train_target_end`: last absolute row usable as a TRAINING target (no
#' leakage past this row). `test_origins`: absolute rows to forecast FROM
#' (the model looks tauEmb*m rows back from each, plus the row itself).
#'
#' curTrainLen is capped to `min(test_origins) - 2L` rather than the
#' maximal leakage-safe value: this guarantees the authors' own
#' "in-between" reservoir warm-up gap (curTrainLen+1):(min(test_origins)-1)
#' always has length >= 1, avoiding R's `a:b` operator silently producing a
#' REVERSED 2-element sequence when a==b+1 (a pure indexing-safety fix, not
#' a change to the model; costs at most one training window).
build_deesn_design <- function(rawData, train_target_end, test_origins, tauEmb, m = M_EMBED) {
  max_train_len_noleak <- train_target_end - tauEmb
  curTrainLen <- min(max_train_len_noleak, min(test_origins) - 2L)
  stopifnot(curTrainLen >= m * tauEmb + 1L)

  yTrain <- rawData[(tauEmb + 1L):(curTrainLen + tauEmb), , drop = FALSE]
  testLen <- length(test_origins)

  inbetween_idx <- (curTrainLen + 1L):(min(test_origins) - 1L)
  stopifnot(length(inbetween_idx) >= 1L, all(diff(c(curTrainLen, inbetween_idx)) == 1L))

  obj <- createEmbedRNNData(
    curTrainLen = curTrainLen, m = m, tauEmb = tauEmb,
    yTrain = yTrain, rawDataInput = rawData, numLocs = ncol(rawData),
    curXTestIndex = test_origins, curTestLen = testLen,
    curInBetweenIndexes = inbetween_idx, curLenInbetween = length(inbetween_idx)
  )
  obj$curTrainLen <- curTrainLen
  obj$curLenInbetween <- length(inbetween_idx)
  obj
}

#' Fit E independent D-EESN reservoirs (numLayers-deep, layerOneDimRed-wide
#' EOF bottleneck at each layer) and return their pooled out-of-sample
#' forecasts as an (n_origin x p x E) array -- same convention as every
#' other model's `arr_coef`/`pred_array` in this study.
fit_deesn_ensemble <- function(des, p, numLayers, layerOneDimRed, E, seed0 = 1L,
                                curNhOne = CUR_NH_ONE, curNhTwo = CUR_NH_TWO, ridge = CUR_RIDGE) {
  setParObjOne <- setParsEESN(ridge, curNhOne, p, M_EMBED)
  setParObjTwo <- setParsEESN(ridge, curNhTwo, layerOneDimRed, 0L)
  numVPars <- curNhTwo + (numLayers - 1L) * layerOneDimRed
  ridgeMat <- diag(ridge, numVPars)
  nColsUOne <- ncol(des$designMatrix)
  nColsUTwo <- layerOneDimRed + 1L
  deltaVec <- rep(0.5, numLayers)   # generic mid-range leak/scale factor per layer (see file header: the paper's own deltaVec values were GA-tuned specifically for Lorenz-96)

  testLen <- nrow(des$designMatrixOutSample)
  pred_array <- array(NA_real_, dim = c(testLen, p, E))

  hMatdimOne <<- curNhOne   # global the vendor code reads inside genResCPP calls

  for (e in seq_len(E)) {
    set.seed(seed0 + e - 1L)
    hMatObj <- multiLayerDeepESNFor(
      curNhOne, des$lenInSampleEmb, des$designMatrix, setParObjOne$sampVecESN,
      W_WIDTH, PI_W, PI_U, U_WIDTH, nColsUOne, deltaVec, setParObjOne$stratValuesXtemp,
      testLen, DIM_RED_ESN, layerOneDimRed, curNhTwo, setParObjTwo$sampVecESN,
      U_WIDTH, PI_W, PI_U, W_WIDTH, nColsUTwo, setParObjTwo$stratValuesXtemp,
      ridgeMat, des$curYInSamp, des$curInSampSD, des$addScaleMat,
      des$designMatrixOutSample, des$inbetweenDesignMatrix, des$curLenInbetween,
      featureLinkInd, numLayers
    )
    pred_array[, , e] <- hMatObj$outSampFor   # already (testLen x p), rescaled to original EOF-coefficient units
  }
  pred_array
}
