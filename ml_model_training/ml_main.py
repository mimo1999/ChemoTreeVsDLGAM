"""
Main entry point for ML model training.
"""

import argparse
import pandas as pd
import numpy as np
import random
from pathlib import Path
from config.constants import PROJECT_ROOT, RANDOM_SEED
from utils.io_utils import build_paths
from ml_model_training.data_loader import DataLoader
from ml_model_training.ml_trainer import MLTrainer
from ml_model_training.ml_evaluator import Result_evaluation
from utils.preprocessing_utils import oversample_minority_with_groups

# Set all random seeds for maximum reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class MLTrainingPipeline:
    """Main pipeline for ML model training."""
    
    def __init__(self, target_cohort: str, saved_data_path: str, model_type: str,
                 feature_combination_method: str, oversampling_method: str,
                 feature_selection_boolen: bool, training_data_type: list,
                 load_training_data_boolen: bool, grid_search_boolen: bool,
                 num_folds: int = 5,
                 feat_type: str = 'standard',
                 agg_interval: int = 24,
                 prefix: str = '',
                 random_seed: int = None):
        
        # Store configuration
        self.target_cohort = target_cohort
        self.saved_data_path = saved_data_path
        self.model_type = model_type
        self.feature_combination_method = feature_combination_method
        self.oversampling_method = oversampling_method
        self.feature_selection_boolen = feature_selection_boolen
        self.training_data_type = training_data_type
        self.load_training_data_boolen = load_training_data_boolen
        self.grid_search_boolen = grid_search_boolen
        self.num_folds = num_folds
        self.feat_type = feat_type
        self.agg_interval = agg_interval
        self.prefix = prefix
        self.random_seed = random_seed
        
        # Build paths
        base = Path(self.saved_data_path)
        self.paths = build_paths(
            base=base,
            cohort=self.target_cohort,
            feature_method=self.feature_combination_method,
            oversampling=self.oversampling_method,
            feature_selection=self.feature_selection_boolen,
            grid_search=self.grid_search_boolen,
            feat_type=self.feat_type,
            agg_interval=self.agg_interval,
            prefix=self.prefix,
            random_seed=self.random_seed
        )
        
        # Initialize components
        self.data_loader = DataLoader(
            target_cohort=self.target_cohort,
            feature_combination_method=self.feature_combination_method,
            training_data_type=self.training_data_type,
            feature_selection_boolen=self.feature_selection_boolen,
            oversampling_method=self.oversampling_method,
            saved_data_path=self.saved_data_path,
            paths=self.paths,
            feat_type=self.feat_type,
            agg_interval=self.agg_interval
        )
        
        self.trainer = MLTrainer(
            model_type=self.model_type,
            feature_combination_method=self.feature_combination_method,
            grid_search=self.grid_search_boolen,
            results_path=str(self.paths.outputs),
            cohort=self.target_cohort
        )
        
        self.evaluator = Result_evaluation()
        
    
    def _print_configuration(self):
        """Print training configuration."""
        print("=" * 60)
        print("MACHINE LEARNING TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"{'Parameter':<30} {'Value':<30}")
        print("-" * 60)
        print(f"{'Target Cohort':<30} {self.target_cohort:<30}")
        print(f"{'Model Type':<30} {self.model_type:<30}")
        print(f"{'Oversampling':<30} {self.oversampling_method:<30}")
        print(f"{'Feature Selection':<30} {self.feature_selection_boolen:<30}")
        print(f"{'Training Data Types':<30} {str(self.training_data_type):<30}")
        print(f"{'Feature Method':<30} {self.feature_combination_method:<30}")
        print(f"{'Grid Search':<30} {self.grid_search_boolen:<30}")
        print(f"{'Number of Folds':<30} {self.num_folds:<30}")
        print(f"{'Load Training Data':<30} {self.load_training_data_boolen:<30}")
        print("=" * 60)
        print(f"{'Temporal resolution':<30} {self.agg_interval:<30}")
        print(f"{'Feature Type':<30} {self.feat_type:<30}")
        print(f"{'Prefix':<30} {self.prefix:<30}")
        print("Starting cross-validation...")
        print()
    
    def prepare_training_data(self):
        """Build feature matrices for all folds."""

        
        for fold in range(self.num_folds):
            print(f"Building feature matrices for fold {fold}...")
            data_per_fold = self.data_loader.load_and_process_data(fold, load_training_data_boolen=False)
            
            if data_per_fold is None:
                print(f'training data not build for fold {fold}!')
                break
        
        print("Feature matrices building completed!")
    
    def train(self):
        """Main training loop for machine learning models."""
        
        self._print_configuration()
        
        for fold in range(self.num_folds):
            print("========================== FOLD {0:2d} =========================".format(fold))
            
            # Load and process data
            data_per_fold = self.data_loader.load_and_process_data(fold, self.load_training_data_boolen)
            
            if data_per_fold is None:
                print(f'Training data not found for fold {fold}!')
                break
            
            X_train, Y_train, X_test, Y_test, X_val, Y_val, subj_ids_train, subj_ids_val = data_per_fold
            
            # Combine train and validation sets for tree-based models
            X_train = pd.concat([X_train, X_val], ignore_index=True)
            Y_train = pd.concat([Y_train, Y_val], ignore_index=True)
            subj_ids_train = pd.concat([subj_ids_train, subj_ids_val], ignore_index=True)
            
            
            # Apply oversampling if needed
            if self.oversampling_method == 'minority':
                print("Oversampling applied. ")
                X_train, Y_train, subj_ids_train = oversample_minority_with_groups(
                    X_train, Y_train, subj_ids_train, seed=RANDOM_SEED
                )
                #print(f'Positive samples in train after oversampling: {Y_train.sum()}')
            
            # Train model and get results
            test_metrics, best_params = self.trainer.train_model(X_train, Y_train, X_test, Y_test, fold, subj_ids_train)
            
            self.evaluator.evaluation_metrics(test_metrics, best_params, fold, self.model_type)
        
        # Save comprehensive results
        self.evaluator.save_results(self.model_type, self.paths.results)


def main():
    """Main entry point for ML training."""
    parser = argparse.ArgumentParser(description="Train ML model")
    
    parser.add_argument('--dataset', type=str, default='MIMIC_IV', help='Dataset name')
    parser.add_argument('--cohort', type=str, default='mimic_cohort_NF_30_days', help='Target cohort')
    parser.add_argument('--model_type', type=str,
                       choices=['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Xgboost', 'CatBoost', 'GAM', 'GAMINET'],
                       default='Random Forest')
    parser.add_argument('--features', nargs='+', default=['LAB', 'DEMO'], 
                       help='List of feature types (e.g., LAB DEMO)')
    parser.add_argument('--feature_method', type=str, choices=['concatenate', 'aggregate'], 
                       default='concatenate')
    parser.add_argument('--oversampling', type=str, default='minority')
    parser.add_argument('--grid_search', type=lambda x: bool(int(x)), default=True)
    parser.add_argument('--feature_selection', type=lambda x: bool(int(x)), default=False)
    parser.add_argument('--load_data', type=lambda x: bool(int(x)), default=False)
    parser.add_argument('--feat_type', type=str,
                       choices=["standard","V", "M", "D", "VMD", "VM", "VD", "MD", "A"],
                       default="standard")
    parser.add_argument('--agg_interval', type=int, choices=[3, 6, 12, 24], default=24,
                    help='Aggregation interval in hours (e.g., 3, 6, 12, 24)') 
    parser.add_argument('--num_folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--prefix', type=str, default='', 
                       help='Prefix for output files')
    parser.add_argument('--action', type=str, default='train', 
                       choices=['prepare', 'train'],
                       help='Action to perform: prepare or train')

    args = parser.parse_args()

    saved_data_path = PROJECT_ROOT / args.dataset / 'saved_data'

    # Initialize training pipeline
    pipeline = MLTrainingPipeline(
        target_cohort=args.cohort,
        saved_data_path=saved_data_path,
        model_type=args.model_type,
        feature_combination_method=args.feature_method,
        oversampling_method=args.oversampling,
        feature_selection_boolen=args.feature_selection,
        training_data_type=args.features,
        load_training_data_boolen=args.load_data,
        grid_search_boolen=args.grid_search,
        num_folds=args.num_folds,
        feat_type=args.feat_type,
        agg_interval = args.agg_interval,
        prefix=args.prefix
    )

    # Execute the requested action
    if args.action == 'prepare':
        pipeline.prepare_training_data()
    elif args.action == 'train':
        pipeline.train()


if __name__ == "__main__":
    main()
