# =====================================================================
# Heart Disease Prediction: full pipeline in R (tidymodels)
# Steps: clean -> verify coding -> EDA -> compare models (repeated CV)
#        -> final test evaluation -> threshold tuning -> calibration
#        -> odds ratios -> importance -> error analysis
# =====================================================================

# ---- 0. Packages -----------------------------------------------------
pkgs <- c("tidyverse", "tidymodels", "janitor", "vip", "glmnet",
          "ranger", "xgboost", "probably", "skimr", "corrplot", "broom")
to_install <- setdiff(pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install)
invisible(lapply(pkgs, library, character.only = TRUE))

tidymodels_prefer()
set.seed(123)

# ---- 1. Load and clean ----------------------------------------------
heart_raw <- read_csv("heart.csv") |> clean_names()   # edit the path if needed
glimpse(heart_raw)
cat("Duplicate rows:", sum(duplicated(heart_raw)), "\n")

heart <- heart_raw |>
  distinct() |>                                        # remove duplicates BEFORE splitting
  mutate(
    ca   = if_else(ca == 4, NA_real_, as.numeric(ca)), # 4 is invalid
    thal = if_else(thal == 0, NA_real_, as.numeric(thal)),  # 0 = missing
    sex     = factor(sex, levels = c(0, 1), labels = c("female", "male")),
    cp      = factor(cp, levels = 0:3,
                     labels = c("typical", "atypical", "non_anginal", "asymptomatic")),
    fbs     = factor(fbs, levels = 0:1, labels = c("no", "yes")),
    restecg = factor(restecg, levels = 0:2,
                     labels = c("normal", "st_t_abnormal", "lv_hypertrophy")),
    exang   = factor(exang, levels = 0:1, labels = c("no", "yes")),
    slope   = factor(slope, levels = 0:2, labels = c("up", "flat", "down")),
    thal    = factor(thal, levels = 1:3,
                     labels = c("normal", "fixed_defect", "reversible_defect")),
    # first level = the "event" for tidymodels
    target  = factor(target, levels = c(1, 0),
                     labels = c("disease", "no_disease"))
  )

skim(heart)
colSums(is.na(heart))

# ---- 2. VERIFY target coding (do this before writing any insight) ----
# Clinically: more blocked vessels (ca) and asymptomatic chest pain should go
# with HIGHER disease rates. If you see the opposite, your target is flipped.
heart |> group_by(ca)  |> summarise(n = n(), disease_rate = mean(target == "disease"))
heart |> group_by(cp)  |> summarise(n = n(), disease_rate = mean(target == "disease"))
heart |> group_by(exang) |> summarise(n = n(), disease_rate = mean(target == "disease"))
# If the direction is reversed, flip the labels in the mutate() above:
#   target = factor(target, levels = c(0, 1), labels = c("disease", "no_disease"))
# and say in your write-up how you verified it.

# ---- 3. EDA ----------------------------------------------------------
heart |> count(target) |> mutate(pct = n / sum(n))

# Disease rate by categorical variables
heart |>
  select(target, sex, cp, fbs, restecg, exang, slope, thal) |>
  pivot_longer(-target, names_to = "variable", values_to = "level") |>
  filter(!is.na(level)) |>
  group_by(variable, level) |>
  summarise(rate = mean(target == "disease"), n = n(), .groups = "drop") |>
  ggplot(aes(level, rate, fill = variable)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~variable, scales = "free_x") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Disease rate by category", x = NULL, y = "Disease rate") +
  theme_minimal()

# Numeric variables by target
heart |>
  select(target, age, trestbps, chol, thalach, oldpeak) |>
  pivot_longer(-target) |>
  ggplot(aes(target, value, fill = target)) +
  geom_boxplot(show.legend = FALSE) +
  facet_wrap(~name, scales = "free_y") +
  labs(title = "Numeric predictors by disease status", x = NULL, y = NULL) +
  theme_minimal()

# Correlation among numeric variables
heart |>
  select(age, trestbps, chol, thalach, oldpeak, ca) |>
  drop_na() |> cor() |>
  corrplot(method = "color", type = "upper", addCoef.col = "black", number.cex = 0.7)

# Age bands
heart |>
  mutate(age_band = cut(age, c(0, 44, 54, 64, Inf),
                        labels = c("<45", "45-54", "55-64", "65+"))) |>
  group_by(age_band, sex) |>
  summarise(rate = mean(target == "disease"), n = n(), .groups = "drop")

# ---- 4. Split + resampling ------------------------------------------
split <- initial_split(heart, prop = 0.80, strata = target)
train <- training(split)
test  <- testing(split)

folds <- vfold_cv(train, v = 5, repeats = 5, strata = target)

# ---- 5. Preprocessing recipe (all inside the resampling = no leakage) -
rec <- recipe(target ~ ., data = train) |>
  step_impute_median(all_numeric_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors())

# ---- 6. Model specifications ----------------------------------------
logit_spec <- logistic_reg() |> set_engine("glm")

lasso_spec <- logistic_reg(penalty = tune(), mixture = 1) |> set_engine("glmnet")

rf_spec <- rand_forest(trees = 500, mtry = tune(), min_n = tune()) |>
  set_engine("ranger") |> set_mode("classification")

xgb_spec <- boost_tree(trees = tune(), tree_depth = tune(), learn_rate = tune(),
                       min_n = tune()) |>
  set_engine("xgboost") |> set_mode("classification")

wf_set <- workflow_set(
  preproc = list(rec = rec),
  models  = list(logit = logit_spec, lasso = lasso_spec, rf = rf_spec, xgb = xgb_spec)
)

# ---- 7. Compare models with repeated CV -----------------------------
metrics <- metric_set(roc_auc, accuracy, sens, spec)

wf_res <- wf_set |>
  workflow_map(
    "tune_grid",
    resamples = folds,
    grid      = 15,
    metrics   = metrics,
    control   = control_grid(save_pred = TRUE, save_workflow = TRUE),
    seed      = 123,
    verbose   = TRUE
  )

rank_results(wf_res, rank_metric = "roc_auc", select_best = TRUE) |>
  select(wflow_id, .metric, mean, std_err) |>
  arrange(.metric, desc(mean))

autoplot(wf_res, rank_metric = "roc_auc", metric = "roc_auc", select_best = TRUE) +
  labs(title = "Model comparison (repeated 5-fold CV)") +
  theme_minimal()

# ---- 8. Final model: logistic regression (interpretable) ------------
# If the ranking in step 7 shows another model clearly ahead, swap the spec.
# With ~300 rows, logistic regression is usually tied for best.
final_wf  <- workflow() |> add_recipe(rec) |> add_model(logit_spec)
final_fit <- last_fit(final_wf, split, metrics = metrics)

collect_metrics(final_fit)

test_preds <- collect_predictions(final_fit) |>
  bind_cols(test |> select(-target))

# ROC curve
roc_curve(test_preds, truth = target, .pred_disease) |>
  autoplot() + labs(title = "ROC curve (test set)")

conf_mat(test_preds, truth = target, estimate = .pred_class) |> autoplot("heatmap")

# ---- 9. Threshold tuning (prioritise recall) ------------------------
thr <- threshold_perf(test_preds, truth = target, estimate = .pred_disease,
                      thresholds = seq(0.10, 0.90, by = 0.05),
                      event_level = "first")

thr |>
  filter(.metric %in% c("sensitivity", "specificity", "j_index")) |>
  ggplot(aes(.threshold, .estimate, colour = .metric)) +
  geom_line(linewidth = 1) +
  labs(title = "Sensitivity / specificity vs threshold", x = "Threshold", y = NULL) +
  theme_minimal()

chosen_threshold <- 0.35   # <- choose from the plot; state the trade-off in your write-up
thr |> filter(.threshold == chosen_threshold)

# ---- 10. Calibration -------------------------------------------------
cal_plot_breaks(test_preds, truth = target, estimate = .pred_disease,
                num_breaks = 5, event_level = "first")

# ---- 11. Odds ratios (clinical interpretation) ----------------------
# Numeric predictors were standardised, so their odds ratios are per 1 SD.
fit_parsnip <- extract_fit_parsnip(final_fit)

or_tbl <- tidy(fit_parsnip, exponentiate = TRUE, conf.int = TRUE) |>
  filter(term != "(Intercept)") |>
  arrange(desc(estimate))
or_tbl

or_tbl |>
  ggplot(aes(estimate, reorder(term, estimate))) +
  geom_point() +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  scale_x_log10() +
  labs(title = "Odds ratios (95% CI)", x = "Odds ratio (log scale)", y = NULL) +
  theme_minimal()

# ---- 12. Permutation importance from a random forest ----------------
rf_best <- wf_res |>
  extract_workflow_set_result("rec_rf") |>
  select_best(metric = "roc_auc")

rf_imp_spec <- rand_forest(trees = 500, mtry = rf_best$mtry, min_n = rf_best$min_n) |>
  set_engine("ranger", importance = "permutation") |>
  set_mode("classification")

rf_fit <- workflow() |> add_recipe(rec) |> add_model(rf_imp_spec) |> fit(train)

rf_fit |> extract_fit_parsnip() |>
  vip(num_features = 15) +
  labs(title = "Random forest permutation importance") +
  theme_minimal()

# ---- 13. Error analysis: who does the model miss? -------------------
false_neg <- test_preds |>
  filter(target == "disease", .pred_disease < chosen_threshold)

false_neg |> select(age, sex, cp, thalach, oldpeak, exang, ca, thal, .pred_disease)

# Compare false negatives with correctly caught cases
test_preds |>
  filter(target == "disease") |>
  mutate(caught = .pred_disease >= chosen_threshold) |>
  group_by(caught) |>
  summarise(n = n(), mean_age = mean(age), mean_thalach = mean(thalach),
            mean_oldpeak = mean(oldpeak))

# ---- 14. Save outputs ------------------------------------------------
# dir.create("figures", showWarnings = FALSE)
# ggsave("figures/roc_curve.png", width = 6, height = 4, dpi = 300)
# saveRDS(extract_workflow(final_fit), "heart_model.rds")

sessionInfo()
