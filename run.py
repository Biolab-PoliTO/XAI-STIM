"""
Command-Line Interface for MER Dataset Preparation and Training

This script provides a simple command-line interface to the complete pipeline:
- Dataset preparation (preprocessing, overlap computation, splitting)
- Dataset organization (into train/val/test folders)
- Model training
- Model inference

Usage:
    python run.py prepare      # Run dataset preparation
    python run.py organize     # Organize into train/val/test
    python run.py verify       # Verify organized dataset
    python run.py stats        # Show dataset statistics

-----------------------------------------------------------------------------------------------
Author: Fabrizio Sciscenti
        Ph.D. Student,
        BIOLAB, Department of Electronics and Telecommunications (DET), Politecnico di Torino
        https://biolab.polito.it/people/fabrizio-sciscenti/
        fabrizio.sciscenti@polito.it

Last Updated: 17/12/2025
-----------------------------------------------------------------------------------------------
"""

import sys
import os
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

import config

def run_prepare():
    """Run dataset preparation pipeline."""
    print("="*80)
    print("RUNNING DATASET PREPARATION")
    print("="*80)
    
    import prepare_dataset as pd
    import pandas as pd_lib
    
    # Check if processed data exists
    if os.path.exists(config.OUTPUT_PATH):
        response = input(f"\nProcessed data already exists at {config.OUTPUT_PATH}\nReload existing? (y/n): ")
        if response.lower() == 'y':
            print("Loading existing data...")
            segments, seg_meta = pd.import_signals_and_meta_from_folder(
                config.OUTPUT_PATH,
                dataset_name_list=config.DATASET_NAMES
            )
        else:
            print("Processing from scratch...")
            segments, seg_meta = None, None
    else:
        segments, seg_meta = None, None
    
    if segments is None:
        # Import raw data
        print("\nImporting raw dataset...")
        X, meta = pd.import_dataset(
            config.BASE_PATH,
            config.DATASET_NAMES,
            config.SAMPLING_FREQUENCIES,
            allowed_electrodes=config.ALLOWED_ELECTRODES
        )
        
        if X is None or meta is None:
            print("Error: Failed to import dataset")
            return False
        
        # Compute overlap
        if os.path.exists(config.META_OVERLAP_PATH):
            print(f"\nLoading overlap values from {config.META_OVERLAP_PATH}")
            meta = pd_lib.read_csv(config.META_OVERLAP_PATH)
        else:
            print("\nComputing overlap values...")
            meta = pd.compute_overlap(
                meta,
                w=config.WINDOW_SIZE,
                eta_m=config.MIN_OVERLAP,
                eta_M=config.MAX_OVERLAP,
                eta_step=config.OVERLAP_STEP,
                verbose=config.VERBOSE
            )
            meta.to_csv(config.META_OVERLAP_PATH, index=False)
            print(f"Saved overlap values to {config.META_OVERLAP_PATH}")
        
        # Extract segments
        print("\nExtracting segments...")
        segments, seg_meta = pd.extract_segments(
            X, meta,
            w=config.WINDOW_SIZE,
            main_output_path=config.OUTPUT_PATH
        )
    
    # Generate splits
    print("\nGenerating test/construction splits...")
    if config.FIXED_TEST_PAIRS is not None and len(config.FIXED_TEST_PAIRS) > 0:
        print(f"Using {len(config.FIXED_TEST_PAIRS)} fixed test patients")
        folds = pd.split_constr_test_fixed(
            segments, seg_meta,
            main_output_path=os.path.join(config.OUTPUT_PATH, 'constr_test_splits_assignments') if config.OUTPUT_PATH else None,
            test_size=config.TEST_SIZE,
            fixed_test_pairs=config.FIXED_TEST_PAIRS
        )
    else:
        print("Generating random balanced splits")
        folds = pd.split_constr_test(
            segments, seg_meta,
            main_output_path=os.path.join(config.OUTPUT_PATH, 'constr_test_splits_assignments') if config.OUTPUT_PATH else None,
            test_size=config.TEST_SIZE
        )
    
    print("\nDataset preparation complete!")
    print("\nGenerated files:")
    print(f"  - test_set_option_*.csv (5 options)")
    print(f"  - construction_folds_*.csv")
    print(f"  - patient_assignments_summary.csv")
    print(f"  - Processed segments in {config.OUTPUT_PATH}")
    
    return True


def run_organize():
    """Run dataset organization."""
    print("="*80)
    print("RUNNING DATASET ORGANIZATION")
    print("="*80)
    
    import organize_dataset as od
    
    # Check if assignment CSV exists
    if not os.path.exists(config.ASSIGNMENT_CSV):
        print(f"\nError: Assignment CSV not found: {config.ASSIGNMENT_CSV}")
        print("\nAvailable test set options:")
        for i in range(1, 6):
            option_file = f"test_set_option_{i}_seed_{config.RANDOM_SEEDS[i-1]}.csv"
            if os.path.exists(option_file):
                print(f"  - {option_file}")
        print("\nPlease select one and rename/copy to data_split.csv")
        return False
    
    # Check if output exists
    if os.path.exists(config.ORGANIZED_PATH):
        response = input(f"\nOrganized data already exists at {config.ORGANIZED_PATH}\nOverwrite? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return False
    
    # Organize
    print(f"\nOrganizing dataset...")
    stats = od.organize_dataset_by_assignment(
        segments_path=config.OUTPUT_PATH,
        assignment_csv=config.ASSIGNMENT_CSV,
        output_path=config.ORGANIZED_PATH,
        copy_files=config.COPY_FILES,
        dataset_names=config.DATASET_NAMES
    )
    
    # Create metadata
    print("\nCreating metadata...")
    metadata = od.create_dataset_metadata(
        config.ORGANIZED_PATH,
        dataset_names=config.DATASET_NAMES
    )
    
    # Verify
    print("\nVerifying organization...")
    verification = od.verify_organization(
        config.ORGANIZED_PATH,
        config.ASSIGNMENT_CSV
    )
    
    print("\n✓ Dataset organization complete!")
    print(f"\nOrganized dataset available at: {config.ORGANIZED_PATH}")
    
    return True


def run_verify():
    """Verify organized dataset."""
    print("="*80)
    print("VERIFYING ORGANIZED DATASET")
    print("="*80)
    
    import organize_dataset as od
    
    if not os.path.exists(config.ORGANIZED_PATH):
        print(f"\nError: Organized dataset not found: {config.ORGANIZED_PATH}")
        print("Please run 'python run.py organize' first.")
        return False
    
    if not os.path.exists(config.ASSIGNMENT_CSV):
        print(f"\nError: Assignment CSV not found: {config.ASSIGNMENT_CSV}")
        return False
    
    verification = od.verify_organization(
        config.ORGANIZED_PATH,
        config.ASSIGNMENT_CSV
    )
    
    return True


def run_stats():
    """Show dataset statistics."""
    print("="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    import numpy as np
    import pandas as pd
    
    # Check what exists
    has_processed = os.path.exists(config.OUTPUT_PATH)
    has_organized = os.path.exists(config.ORGANIZED_PATH)
    
    print(f"\nProcessed data: {'✓' if has_processed else '✗'} ({config.OUTPUT_PATH})")
    print(f"Organized data: {'✓' if has_organized else '✗'} ({config.ORGANIZED_PATH})")
    
    if not has_organized:
        print("\nNo organized dataset found. Run 'python run.py organize' first.")
        return False
    
    # Load metadata
    metadata_file = os.path.join(config.ORGANIZED_PATH, 'dataset_metadata.csv')
    if not os.path.exists(metadata_file):
        print(f"\nMetadata file not found: {metadata_file}")
        return False
    
    meta = pd.read_csv(metadata_file)
    
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"Total segments: {len(meta)}")
    
    if 'dataset' in meta.columns:
        print(f"\nSegments per dataset:")
        for dataset in sorted(meta['dataset'].unique()):
            n = len(meta[meta['dataset'] == dataset])
            pct = 100 * n / len(meta)
            print(f"  {dataset:15s}: {n:6d} ({pct:5.1f}%)")
    
    # Split statistics
    print(f"\n{'='*80}")
    print("SPLIT STATISTICS")
    print(f"{'='*80}")
    
    for split in ['train', 'val', 'test']:
        split_meta = meta[meta['split'] == split]
        print(f"\n{split.upper()}:")
        print(f"  Total segments: {len(split_meta)}")
        
        if 'dataset' in split_meta.columns:
            print(f"  Per dataset:")
            for dataset in sorted(split_meta['dataset'].unique()):
                n = len(split_meta[split_meta['dataset'] == dataset])
                pct = 100 * n / len(split_meta) if len(split_meta) > 0 else 0
                print(f"    {dataset:15s}: {n:6d} ({pct:5.1f}%)")
        
        if 'class' in split_meta.columns:
            print(f"  Class distribution:")
            for cls in sorted(split_meta['class'].unique()):
                n = len(split_meta[split_meta['class'] == cls])
                pct = 100 * n / len(split_meta) if len(split_meta) > 0 else 0
                print(f"    Class {cls}: {n:6d} ({pct:5.1f}%)")
        
        if 'patient' in split_meta.columns:
            n_patients = split_meta['patient'].nunique()
            print(f"  Unique patients: {n_patients}")
    
    return True


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MER Dataset Preparation and Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog= """
                Examples:
                python run.py prepare      # Prepare dataset (preprocess, compute overlap, split)
                python run.py organize     # Organize into train/val/test
                python run.py train        # Train models
                python run.py infer        # Run inference on trained models
                python run.py verify       # Verify organized dataset
                python run.py stats        # Show statistics
                """
    )
    
    parser.add_argument(
        'command',
        choices=['prepare', 'organize', 'train', 'infer', 'verify', 'stats'],
        help='Command to run'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom config file (optional)'
    )
    
    args = parser.parse_args()
    
    # Load custom config if provided
    if args.config:
        print(f"Loading custom config from: {args.config}")
        # Import custom config
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_config", args.config)
        custom_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_config)
        # Update config module
        for key in dir(custom_config):
            if key.isupper():  # Only copy uppercase (config) variables
                setattr(config, key, getattr(custom_config, key))
    
    # Print config summary
    print("\nConfiguration:")
    print(f"  Datasets: {config.DATASET_NAMES}")
    print(f"  Window size: {config.WINDOW_SIZE} samples")
    print(f"  Overlap range: {config.MIN_OVERLAP:.0%} - {config.MAX_OVERLAP:.0%}")
    print(f"  Test size: {config.TEST_SIZE} patients/dataset")
    print()
    
    # Run command
    success = False
    
    if args.command == 'prepare':
        success = run_prepare()
    
    elif args.command == 'organize':
        success = run_organize()
    
    elif args.command == 'verify':
        success = run_verify()
    
    elif args.command == 'stats':
        success = run_stats()
    
    # Exit
    if success:
        print("\nCommand completed successfully!")
        sys.exit(0)
    else:
        print("\nCommand failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
