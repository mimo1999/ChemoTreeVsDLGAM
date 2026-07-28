#!/bin/bash
prefix="${3:-}"
feature_types=("VMD") #("D" "VD" "VM" "MD") #"standard" "V" "M" "D" "VD" "VM" "MD"
models=("CatBoost") #("Random Forest" "Logistic Regression" "Xgboost" "CatBoost" "Gradient Boosting")
agg_intervals=("24") # "12" "6" "3")


# Common function to run the command safely (no eval). Trailing args are
# optional overrides (oversampling, grid_search, top_k_features,
# selector_type, load_data, run_prefix) so callers that need non-default
# values (e.g. the benchmark10 mode below) don't have to duplicate this
# function — every existing caller that omits them keeps the old behavior.
run_command() {
    local dataset="$1"
    local cohort="$2"
    local model_type="$3"
    local feat_type="$4"
    local feature_selection="$5"
    local agg_interval="$6"
    local action="$7"
    local oversampling="${8:-minority}"
    local grid_search="${9:-1}"
    local top_k_features="${10:-250}"
    local selector_type="${11:-tree}"
    local load_data="${12:-1}"
    local run_prefix="${13:-$prefix}"

    echo ">>> Running: $dataset | $cohort | $model_type | $feat_type | $agg_interval | $action | oversampling=$oversampling grid_search=$grid_search top_k=$top_k_features selector=$selector_type"

    python -m ml_model_training.ml_main \
        --dataset "$dataset" \
        --cohort "$cohort" \
        --model_type "$model_type" \
        --features LAB DEMO \
        --feature_method concatenate \
        --oversampling "$oversampling" \
        --grid_search "$grid_search" \
        --num_folds 5 \
        --feature_selection "$feature_selection" \
        --top_k_features "$top_k_features" \
        --selector_type "$selector_type" \
        --load_data "$load_data" \
        --feat_type "$feat_type" \
        --agg_interval "$agg_interval" \
        --prefix "$run_prefix" \
        --action "$action"
}

# ---------- MIMIC ----------
run_mimic() {
    action="$1"
    echo "Running MIMIC pipeline: $action"
    
    dataset="MIMIC_IV"
    cohorts=("mimic_cohort_aplasia_45_days" "mimic_cohort_NF_30_days")
    feature_selection="1"

    for cohort in "${cohorts[@]}"; do
        for feat_type in "${feature_types[@]}"; do
            for interval in "${agg_intervals[@]}"; do
                if [ "$action" = "prepare" ]; then
                    echo "=========================================="
                    echo "Preparing: $cohort, $feat_type, $interval"
                    run_command "$dataset" "$cohort" "Random Forest" "$feat_type" "$feature_selection" "$interval" "$action"
                else
                    for model in "${models[@]}"; do
                        echo "=========================================="
                        echo "Training: $model, $cohort, $feat_type, $interval"
                        run_command "$dataset" "$cohort" "$model" "$feat_type" "$feature_selection" "$interval" "$action"
                    done
                fi
            done
        done
    done
}

# ---------- UKER ----------
run_uker() {
    action="$1"
    echo "Running UKEr pipeline: $action"
    
    dataset="UKEr"
    cohorts=("uker_cohort_NF_30_days" "uker_cohort_aplasia_45_days")
    feature_selection="0"

    for cohort in "${cohorts[@]}"; do
        for feat_type in "${feature_types[@]}"; do
            for interval in "${agg_intervals[@]}"; do
                if [ "$action" = "prepare" ]; then
                    echo "=========================================="
                    echo "Preparing: $cohort, $feat_type, $interval"
                    run_command "$dataset" "$cohort" "Random Forest" "$feat_type" "$feature_selection" "$interval" "$action"
                else
                    for model in "${models[@]}"; do
                        echo "=========================================="
                        echo "Training: $model, $cohort, $feat_type, $interval"
                        run_command "$dataset" "$cohort" "$model" "$feat_type" "$feature_selection" "$interval" "$action"
                    done
                fi
            done
        done
    done
}

# ---------- BENCHMARK10 ----------
# 10-model comparison run: mimic_cohort_NF_30_days, feat_type=VMD,
# feature_selection=1, top_k=250, selector=tree, grid_search=0.
# oversampling=minority for every model except EBM (oversampling=none, per
# its tuned best-config which assumes balanced sample_weight instead).
# SSTGAM here uses whatever fit_method is currently set in
# config/ml_config_params_best.yaml's SSTGAM.mimic_cohort_NF_30_days block
# (tree = the tuned production config) — running the spline/boost variant
# separately requires toggling that block first, since there is no CLI
# override for a single wrapper's fit_method.
run_benchmark10() {
    action="$1"
    local run_prefix="${2:-benchmark10}"
    echo "Running 10-model benchmark: $action"

    local dataset="MIMIC_IV"
    local cohort="mimic_cohort_NF_30_days"
    local feat_type="VMD"
    local feature_selection="1"
    local agg_interval="24"
    local top_k="250"
    local selector="tree"
    local grid_search="0"

    if [ "$action" = "prepare" ]; then
        echo "=========================================="
        echo "Preparing (oversampling=minority): $cohort, $feat_type"
        run_command "$dataset" "$cohort" "Random Forest" "$feat_type" "$feature_selection" "$agg_interval" "prepare" "minority" "$grid_search" "$top_k" "$selector" 0 "$run_prefix"
        echo "=========================================="
        echo "Preparing (oversampling=none, for EBM): $cohort, $feat_type"
        run_command "$dataset" "$cohort" "Random Forest" "$feat_type" "$feature_selection" "$agg_interval" "prepare" "none" "$grid_search" "$top_k" "$selector" 0 "$run_prefix"
        return
    fi

    local benchmark_models=("GAM" "EBM" "GAMINET" "CatBoost" "Random Forest" "IGANN" "MGAM" "DGAM" "SSTGAM" "HE-EBM")
    for model in "${benchmark_models[@]}"; do
        local oversampling="minority"
        if [ "$model" = "EBM" ]; then
            oversampling="none"
        fi
        echo "=========================================="
        echo "Training: $model | oversampling=$oversampling"
        run_command "$dataset" "$cohort" "$model" "$feat_type" "$feature_selection" "$agg_interval" "train" "$oversampling" "$grid_search" "$top_k" "$selector" 1 "$run_prefix"
    done
}

# ---------- MAIN ----------
if [ "$1" = "mimic" ]; then
    if [ "$2" = "full_run" ]; then
        run_mimic "prepare"
        echo "MIMIC data preparation completed. Starting training..."
        run_mimic "train"
    else
        run_mimic "$2"
    fi
elif [ "$1" = "uker" ]; then
    if [ "$2" = "full_run" ]; then
        run_uker "prepare"
        echo "UKEr data preparation completed. Starting training..."
        run_uker "train"
    else
        run_uker "$2"
    fi
elif [ "$1" = "benchmark10" ]; then
    if [ "$2" = "full_run" ]; then
        run_benchmark10 "prepare" "$3"
        echo "Benchmark10 data preparation completed. Starting training..."
        run_benchmark10 "train" "$3"
    else
        run_benchmark10 "$2" "$3"
    fi
else
    echo "Usage: $0 [mimic|uker|benchmark10] [prepare|train|full_run] [prefix]"
    echo "  mimic prepare: Prepare MIMIC data only"
    echo "  mimic train:   Train MIMIC models only"
    echo "  mimic full_run: Prepare + train MIMIC"
    echo "  uker prepare:  Prepare UKEr data only"
    echo "  uker train:    Train UKEr models only"
    echo "  uker full_run: Prepare + train UKEr"
    echo "  benchmark10 prepare|train|full_run: 10-model NF_30_days/VMD/top250/tree-FS/no-grid-search benchmark"
    echo "  prefix:        Optional prefix for output files (default: empty)"
fi
