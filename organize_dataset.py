"""
Dataset Organization for MER STN Classification
===============================================

This script organizes preprocessed MER segments (through prepare_dataset.py) into 
train/val/test splits based on a predefined assignment CSV file.

It reads patient assignments from a CSV file (e.g., data_split.csv with columns patient, dataset, 
assignment being 'train', 'val', or 'test') and copies the corresponding segment files into 
separate folders for each split.

Usage:
------
python organize_dataset.py

Or import and use the functions programmatically:
    from organize_dataset import organize_dataset_by_assignment
    organize_dataset_by_assignment(
        segments_path='path/to/segments',
        assignment_csv='data_split.csv',
        output_path='path/to/organized'
    )

-----------------------------------------------------------------------------------------------
Author: Fabrizio Sciscenti
        Ph.D. Student,
        BIOLAB, Department of Electronics and Telecommunications (DET), Politecnico di Torino
        https://biolab.polito.it/people/fabrizio-sciscenti/
        fabrizio.sciscenti@polito.it

Last Updated: 17/12/2025
-----------------------------------------------------------------------------------------------
"""

import os
import shutil
import numpy as np
import pandas as pd


def load_assignment_csv(csv_path):
    """
    Load and validate assignment CSV file.
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file with columns: patient, dataset, assignment

    Input CSV format:
    -----------------
    patient,    dataset,    assignment
    P001,       AlphaMap,   train
    P002,       NeuroSet,   train
    P004,       NeuroOmega, val
    P011,       AlphaMap,   test
    ...
    
    Returns:
    --------
    df : pd.DataFrame
        Assignment dataframe
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Assignment CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['patient', 'dataset', 'assignment']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")
    
    # Validate assignment values
    valid_assignments = {'train', 'val', 'test'}
    invalid = df[~df['assignment'].isin(valid_assignments)]['assignment'].unique()
    if len(invalid) > 0:
        raise ValueError(f"Invalid assignment values found: {invalid}. Must be one of {valid_assignments}")
    
    print(f"Loaded {len(df)} patient assignments from {csv_path}")
    print(f"  Train: {len(df[df['assignment'] == 'train'])}")
    print(f"  Val: {len(df[df['assignment'] == 'val'])}")
    print(f"  Test: {len(df[df['assignment'] == 'test'])}")
    
    return df


def organize_dataset_by_assignment(segments_path, assignment_csv, output_path, 
                                   copy_files=True, dataset_names=None):
    """
    Organize preprocessed segments into train/val/test folders based on assignments.
    
    Parameters:
    -----------
    segments_path : str
        Path to preprocessed segments (organized by dataset)
        Expected structure: segments_path/dataset_name/patient_id.npy
    assignment_csv : str
        Path to CSV file with patient assignments
    output_path : str
        Output directory for organized dataset
    copy_files : bool
        If True, copy files. If False, create symlinks (faster but requires permissions)
    dataset_names : list of str, optional
        List of dataset names to process. If None, processes all in assignment CSV
    
    Returns:
    --------
    stats : dict
        Statistics about the organization process
    """
    # Load assignments
    assignments_df = load_assignment_csv(assignment_csv)
    
    if dataset_names is None:
        dataset_names = assignments_df['dataset'].unique()
    
    # Initialize statistics
    stats = {
        'train': {'patients': 0, 'segments': 0, 'files_copied': 0},
        'val': {'patients': 0, 'segments': 0, 'files_copied': 0},
        'test': {'patients': 0, 'segments': 0, 'files_copied': 0},
        'errors': []
    }
    
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_path, split), exist_ok=True)
    
    # Process each dataset
    for dataset_name in dataset_names:
        print(f"\nProcessing dataset: {dataset_name}")
        dataset_path = os.path.join(segments_path, dataset_name)
        
        if not os.path.exists(dataset_path):
            print(f"  Warning: Dataset path not found: {dataset_path}")
            stats['errors'].append(f"Dataset path not found: {dataset_path}")
            continue
        
        # Get all patients for this dataset
        dataset_assignments = assignments_df[assignments_df['dataset'] == dataset_name]
        
        for _, row in dataset_assignments.iterrows():
            patient_id = row['patient']
            assignment = row['assignment']
            
            # Destination: output_path/assignment/dataset_name/
            dest_dataset_dir = os.path.join(output_path, assignment, dataset_name)
            os.makedirs(dest_dataset_dir, exist_ok=True)

            try: 
                # Update statistics
                stats[assignment]['patients'] += 1
                
                # Count segments
                npy_file = os.path.join(dataset_path, f"{patient_id}.npy")
                meta_file = os.path.join(dataset_path, f"{patient_id}.csv")

                if os.path.exists(npy_file) and os.path.exists(meta_file):
                    segments = np.load(npy_file, allow_pickle=True)
                    meta = pd.read_csv(meta_file)
                    n_segments = len(segments) if isinstance(segments, np.ndarray) else 1
                    stats[assignment]['segments'] += n_segments
                    stats[assignment]['files_copied'] += 1
                    np.save(os.path.join(dest_dataset_dir, f"{patient_id}.npy"), segments)
                    meta.to_csv(os.path.join(dest_dataset_dir, f"{patient_id}.csv"), index=False)
                    if not copy_files:
                        os.remove(npy_file)
                        os.remove(meta_file)
                
                print(f"  {assignment}: {patient_id} ({dataset_name})")
                
            except Exception as e:
                error_msg = f"Error processing {patient_id} ({dataset_name}): {str(e)}"
                print(f"  Error: {error_msg}")
                stats['errors'].append(error_msg)

        # Remove empty dataset directories
        if not os.listdir(dataset_path):
            os.rmdir(dataset_path)
    
    # Print summary
    print(f"\n{'='*80}")
    print("ORGANIZATION COMPLETE")
    print(f"{'='*80}")
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        print(f"  Patients: {stats[split]['patients']}")
        print(f"  Segments: {stats[split]['segments']}")
        print(f"  Files copied: {stats[split]['files_copied']}")
    
    if stats['errors']:
        print(f"\nErrors encountered: {len(stats['errors'])}")
        for error in stats['errors'][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more")
    
    # Save statistics
    stats_file = os.path.join(output_path, 'organization_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("Dataset Organization Statistics\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Source: {segments_path}\n")
        f.write(f"Assignment CSV: {assignment_csv}\n")
        f.write(f"Output: {output_path}\n\n")
        
        for split in ['train', 'val', 'test']:
            f.write(f"\n{split.upper()}:\n")
            f.write(f"  Patients: {stats[split]['patients']}\n")
            f.write(f"  Segments: {stats[split]['segments']}\n")
            f.write(f"  Files: {stats[split]['files_copied']}\n")
        
        if stats['errors']:
            f.write(f"\n\nErrors ({len(stats['errors'])}):\n")
            for error in stats['errors']:
                f.write(f"  - {error}\n")
    
    print(f"\nStatistics saved to: {stats_file}")
    
    return stats


def create_dataset_metadata(output_path, dataset_names=None):
    """
    Create comprehensive metadata file for the organized dataset.
    
    Scans the organized dataset and creates a CSV with all segments and their properties.
    
    Parameters:
    -----------
    output_path : str
        Path to organized dataset (with train/val/test folders)
    dataset_names : list of str, optional
        Datasets to include in metadata
    
    Returns:
    --------
    metadata_df : pd.DataFrame
        Comprehensive metadata
    """
    all_metadata = []
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(output_path, split)
        
        if not os.path.exists(split_path):
            continue
        
        # Iterate through datasets
        for dataset_name in os.listdir(split_path):
            dataset_path = os.path.join(split_path, dataset_name)
            
            if not os.path.isdir(dataset_path):
                continue
            
            if dataset_names is not None and dataset_name not in dataset_names:
                continue
            
            # Iterate through patients
            for npy_file in os.listdir(dataset_path):
                if not npy_file.endswith('.npy'):
                    continue
                
                # Load patient metadata CSV if it exists
                csv_file = os.path.join(dataset_path, npy_file.replace('.npy', '.csv'))
                patient_id = npy_file.replace('.npy', '')
                npy_file = os.path.join(dataset_path, npy_file)

                if os.path.exists(csv_file):
                    patient_meta = pd.read_csv(csv_file)
                    patient_meta['split'] = split
                    all_metadata.append(patient_meta)
                else:
                    if os.path.exists(npy_file):
                        segments = np.load(npy_file, allow_pickle=True)
                        n_segments = len(segments) if isinstance(segments, np.ndarray) else 1
                        
                        basic_meta = pd.DataFrame([{
                            'patient': patient_id,
                            'dataset': dataset_name,
                            'split': split,
                            'n_segments': n_segments
                        }])
                        all_metadata.append(basic_meta)
    
    if not all_metadata:
        print("Warning: No metadata found")
        return pd.DataFrame()
    
    metadata_df = pd.concat(all_metadata, ignore_index=True)
    
    # Save metadata
    metadata_file = os.path.join(output_path, 'dataset_metadata.csv')
    metadata_df.to_csv(metadata_file, index=False)
    print(f"\nMetadata saved to: {metadata_file}")
    print(f"Total segments: {len(metadata_df)}")
    
    return metadata_df


def verify_organization(output_path, assignment_csv):
    """
    Verify that the organized dataset matches the assignment CSV.
    
    Parameters:
    -----------
    output_path : str
        Path to organized dataset
    assignment_csv : str
        Path to assignment CSV
    
    Returns:
    --------
    verification_report : dict
        Report with any mismatches or issues
    """
    assignments_df = load_assignment_csv(assignment_csv)
    
    report = {
        'missing_patients': [],
        'extra_patients': [],
        'misplaced_patients': [],
        'verified': 0
    }
    
    # Track what we find in the organized dataset
    found_patients = set()
    
    # Scan organized dataset
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(output_path, split)
        
        if not os.path.exists(split_path):
            continue
        
        for dataset_name in os.listdir(split_path):
            dataset_path = os.path.join(split_path, dataset_name)
            
            if not os.path.isdir(dataset_path):
                continue
            
            for patient_id in os.listdir(dataset_path):
                patient_path = os.path.join(dataset_path, patient_id)
                
                if not os.path.isdir(patient_path):
                    continue
                
                found_patients.add((patient_id, dataset_name, split))
                
                # Check if this patient should be in this split
                expected = assignments_df[
                    (assignments_df['patient'] == patient_id) & 
                    (assignments_df['dataset'] == dataset_name)
                ]
                
                if len(expected) == 0:
                    report['extra_patients'].append((patient_id, dataset_name, split))
                elif expected.iloc[0]['assignment'] != split:
                    report['misplaced_patients'].append({
                        'patient': patient_id,
                        'dataset': dataset_name,
                        'found_in': split,
                        'should_be_in': expected.iloc[0]['assignment']
                    })
                else:
                    report['verified'] += 1
    
    # Check for missing patients
    for _, row in assignments_df.iterrows():
        patient_id = row['patient']
        dataset_name = row['dataset']
        assignment = row['assignment']
        
        if (patient_id, dataset_name, assignment) not in found_patients:
            report['missing_patients'].append((patient_id, dataset_name, assignment))
    
    # Print verification report
    print(f"\n{'='*80}")
    print("VERIFICATION REPORT")
    print(f"{'='*80}")
    print(f"Verified: {report['verified']} patients correctly placed")
    
    if report['missing_patients']:
        print(f"\nMissing patients: {len(report['missing_patients'])}")
        for patient, dataset, split in report['missing_patients'][:10]:
            print(f"  - {patient} ({dataset}) should be in {split}")
        if len(report['missing_patients']) > 10:
            print(f"  ... and {len(report['missing_patients']) - 10} more")
    
    if report['extra_patients']:
        print(f"\nExtra patients (not in CSV): {len(report['extra_patients'])}")
        for patient, dataset, split in report['extra_patients'][:10]:
            print(f"  - {patient} ({dataset}) found in {split}")
        if len(report['extra_patients']) > 10:
            print(f"  ... and {len(report['extra_patients']) - 10} more")
    
    if report['misplaced_patients']:
        print(f"\nMisplaced patients: {len(report['misplaced_patients'])}")
        for item in report['misplaced_patients'][:10]:
            print(f"  - {item['patient']} ({item['dataset']}): found in {item['found_in']}, should be in {item['should_be_in']}")
        if len(report['misplaced_patients']) > 10:
            print(f"  ... and {len(report['misplaced_patients']) - 10} more")
    
    if not report['missing_patients'] and not report['extra_patients'] and not report['misplaced_patients']:
        print("\n✓ All patients correctly organized!")
    
    return report


def main():
    """
    Main execution function.
    
    Modify the paths below to match your setup.
    """
    # Configuration
    segments_path = 'dataset_processed_windows'
    assignment_csv = 'data_split.csv'
    output_path = 'dataset'
    dataset_names = ['AlphaMap', 'NeuroSet', 'NeuroOmega']
    
    print("="*80)
    print("MER Dataset Organization")
    print("="*80)
    print(f"Source: {segments_path}")
    print(f"Assignment CSV: {assignment_csv}")
    print(f"Output: {output_path}")
    print("="*80)
    
    # Organize dataset
    stats = organize_dataset_by_assignment(
        segments_path=segments_path,
        assignment_csv=assignment_csv,
        output_path=output_path,
        copy_files=True,  # Set to False for moving files
        dataset_names=dataset_names
    )
    
    # Create comprehensive metadata
    print("\nCreating dataset metadata...")
    metadata = create_dataset_metadata(output_path, dataset_names=dataset_names)
    
    # Verify organization
    print("\nVerifying organization...")
    verification = verify_organization(output_path, assignment_csv)
    
    print("\nDataset organization complete!")


if __name__ == '__main__':
    main()
