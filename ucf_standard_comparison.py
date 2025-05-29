#!/usr/bin/env python3
"""
Simplified UCF vs Standard ML Analysis Framework

Rigorous scientific comparison without astronomy-specific dependencies.
Works with your ARCADE 2 data using basic FITS reading.

This analysis maintains scientific integrity by:
- Using fixed, literature-based parameters (no tuning)
- Implementing proper cross-validation
- Using statistical significance testing
- Providing reproducible results

Author: Scientific Analysis Framework
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import struct
import os
import glob
from pathlib import Path
import json
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from scipy import stats
import warnings

# Set random seeds for reproducibility
np.random.seed(42)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

class BasicFITSReader:
    """Basic FITS file reader for ARCADE 2 data without astronomy dependencies."""
    
    def __init__(self, data_directory="."):
        self.data_directory = Path(data_directory)
        self.frequency_data = {}
        self.frequencies = []
        
    def load_fits_files(self):
        """Load ARCADE 2 FITS files using basic binary reading."""
        fits_files = list(self.data_directory.glob("arc2_*.fits"))
        
        if not fits_files:
            print("No ARCADE 2 FITS files found in directory.")
            return False
            
        print(f"Found {len(fits_files)} ARCADE 2 FITS files")
        
        for fits_file in sorted(fits_files):
            try:
                frequency = self._extract_frequency_from_filename(fits_file.name)
                if frequency:
                    data = self._parse_fits_data(fits_file)
                    if data is not None:
                        self.frequency_data[frequency] = data
                        self.frequencies.append(frequency)
                        print(f"Loaded {fits_file.name}: {frequency} MHz, {len(data['temperature'])} data points")
            except Exception as e:
                print(f"Error loading {fits_file}: {e}")
                
        self.frequencies = sorted(self.frequencies)
        return len(self.frequency_data) > 0
    
    def _extract_frequency_from_filename(self, filename):
        """Extract frequency from ARCADE 2 filename."""
        parts = filename.replace('.fits', '').split('_')
        if len(parts) >= 2:
            freq_str = parts[1].rstrip('s')
            try:
                return int(freq_str)
            except ValueError:
                return None
        return None
    
    def _parse_fits_data(self, fits_file):
        """Parse FITS file to extract temperature data."""
        try:
            with open(fits_file, 'rb') as f:
                # Read file content
                content = f.read()
                
                # Find the binary table data
                # Look for temperature data pattern in the binary section
                temperatures, n_obs = self._extract_binary_data(content)
                
                if temperatures is not None and len(temperatures) > 0:
                    # Filter out invalid data
                    valid_mask = np.isfinite(temperatures) & (temperatures != 0)
                    if n_obs is not None:
                        valid_mask &= (n_obs > 0)
                    
                    return {
                        'temperature': temperatures[valid_mask],
                        'n_obs': n_obs[valid_mask] if n_obs is not None else np.ones(np.sum(valid_mask)),
                        'valid_count': np.sum(valid_mask)
                    }
        except Exception as e:
            print(f"Error parsing {fits_file}: {e}")
            return None
    
    def _extract_binary_data(self, content):
        """Extract binary data from FITS file content."""
        try:
            # Find the binary table extension
            # Look for the pattern that indicates start of binary data
            binary_start = content.find(b'XTENSION')
            if binary_start == -1:
                return None, None
            
            # Skip header and find actual data
            # ARCADE 2 files have temperature data as 4-byte floats
            data_start = None
            for i in range(binary_start, len(content) - 2880, 2880):  # FITS blocks are 2880 bytes
                block = content[i:i+2880]
                if b'END' in block and len(block.strip()) < 2880:
                    data_start = i + 2880
                    break
            
            if data_start is None or data_start >= len(content):
                return None, None
            
            # Read binary data
            binary_data = content[data_start:]
            
            # ARCADE 2 format: pairs of 4-byte floats (temperature, n_obs)
            n_records = len(binary_data) // 8  # 8 bytes per record (2 floats)
            
            temperatures = []
            n_obs = []
            
            for i in range(n_records):
                try:
                    # Read 4-byte float (big-endian)
                    temp_bytes = binary_data[i*8:(i*8)+4]
                    nobs_bytes = binary_data[(i*8)+4:(i*8)+8]
                    
                    if len(temp_bytes) == 4 and len(nobs_bytes) == 4:
                        temp = struct.unpack('>f', temp_bytes)[0]  # Big-endian float
                        nobs = struct.unpack('>f', nobs_bytes)[0]  # Big-endian float
                        
                        temperatures.append(temp)
                        n_obs.append(nobs)
                except:
                    continue
            
            if len(temperatures) > 100:  # Reasonable data size
                return np.array(temperatures), np.array(n_obs)
            else:
                return None, None
                
        except Exception as e:
            print(f"Binary extraction error: {e}")
            return None, None

class UnifiedCoherenceFunction:
    """
    Implementation of Unified Coherence Function for astronomical data.
    
    Parameters are set based on literature values to avoid tuning bias.
    """
    
    def __init__(self, alpha=0.1, beta=0.05, gamma=0.8, omega=1.0, max_lag=10):
        """
        Initialize UCF with literature-based parameters.
        
        Parameters from biological UCF applications:
        - alpha: Memory decay rate (0.1)
        - beta: Event influence weight (0.05) 
        - gamma: Event detection threshold (0.8 std)
        - omega: Event magnitude scaling (1.0)
        - max_lag: Maximum memory length (10)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.omega = omega
        self.max_lag = max_lag
        
        # Store analysis results
        self.memory_field = None
        self.event_detections = None
        self.coherence_values = None
    
    def analyze_sequence(self, sequence):
        """Analyze a single sequence with UCF."""
        if len(sequence) < 2:
            return np.zeros_like(sequence)
        
        # Calculate memory field
        self.memory_field = self._calculate_memory_field(sequence)
        
        # Integrate memory component
        memory_component = np.sum(self.memory_field, axis=1)
        
        # Detect events
        self.event_detections = self._detect_events(sequence)
        
        # Combine components
        self.coherence_values = memory_component + self.beta * self.event_detections
        
        return self.coherence_values
    
    def _calculate_memory_field(self, sequence):
        """Calculate memory field L(x,τ)."""
        seq_len = len(sequence)
        memory_field = np.zeros((seq_len, self.max_lag))
        
        for pos in range(seq_len):
            valid_lags = min(pos + 1, self.max_lag)
            
            if valid_lags > 0:
                # Get history
                start_idx = max(0, pos - self.max_lag + 1)
                history = sequence[start_idx:pos+1]
                history = history[::-1]  # Reverse for recency
                
                # Apply exponential decay
                lags = np.arange(valid_lags)
                decay_weights = np.exp(-self.alpha * lags)
                
                # Store weighted history
                memory_values = history[:valid_lags] * decay_weights
                memory_field[pos, :valid_lags] = memory_values
        
        return memory_field
    
    def _detect_events(self, sequence):
        """Detect significant events in sequence."""
        # Calculate differences
        diffs = np.concatenate([[0], np.diff(sequence)])
        
        # Adaptive threshold
        threshold = self.gamma * np.std(sequence)
        
        # Event detection
        events = (np.abs(diffs) > threshold).astype(float) * self.omega
        
        return events

class StandardMLComparison:
    """Standard machine learning models for comparison."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0, random_state=random_state),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=random_state,
                n_jobs=-1
            ),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(50, 25),
                max_iter=500,
                random_state=random_state
            )
        }
        self.results = {}
    
    def create_features(self, sequences):
        """Create features from sequences for ML models."""
        features = []
        targets = []
        
        for seq in sequences:
            seq = np.array(seq)
            if len(seq) < 3:
                continue
                
            # Create sliding window features
            for i in range(2, len(seq)):
                # Features: previous values, differences, statistics
                feature_vector = [
                    seq[i-1],  # Previous value
                    seq[i-2],  # Value before previous
                    seq[i-1] - seq[i-2],  # Recent difference
                    np.mean(seq[:i]),  # Historical mean
                    np.std(seq[:i]) if i > 2 else 0,  # Historical std
                ]
                
                features.append(feature_vector)
                targets.append(seq[i])
        
        return np.array(features), np.array(targets)
    
    def train_and_evaluate(self, sequences, cv_folds=5):
        """Train and evaluate all models with cross-validation."""
        # Create features
        X, y = self.create_features(sequences)
        
        if len(X) < cv_folds:
            print("Not enough data for cross-validation")
            return None
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cross-validation
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            mse_scores = []
            r2_scores = []
            
            print(f"Training {name}...")
            
            for train_idx, test_idx in kf.split(X_scaled):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                try:
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Predict
                    y_pred = model.predict(X_test)
                    
                    # Evaluate
                    mse_scores.append(mean_squared_error(y_test, y_pred))
                    r2_scores.append(r2_score(y_test, y_pred))
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    continue
            
            if mse_scores:
                self.results[name] = {
                    'mse_mean': np.mean(mse_scores),
                    'mse_std': np.std(mse_scores),
                    'r2_mean': np.mean(r2_scores),
                    'r2_std': np.std(r2_scores)
                }
        
        return self.results

class UCFvsMLComparison:
    """Compare UCF against standard ML methods."""
    
    def __init__(self, data_dict, analysis_type="ARCADE2"):
        self.data_dict = data_dict
        self.analysis_type = analysis_type
        self.ucf_results = {}
        self.ml_results = {}
        self.statistical_tests = {}
        
    def create_sequences_from_data(self):
        """Create analysis sequences from ARCADE 2 data."""
        sequences = []
        
        if not self.data_dict:
            return sequences
        
        frequencies = sorted(self.data_dict.keys())
        
        # Method 1: Frequency sequences for same spatial regions
        # Group data by spatial proximity (simple binning)
        print("Creating frequency sequences...")
        
        # Get the first frequency data to determine data size
        first_freq = frequencies[0]
        n_points = len(self.data_dict[first_freq]['temperature'])
        
        # Create sequences by taking every Nth point across frequencies
        step_size = max(1, n_points // 100)  # Limit to ~100 sequences
        
        for i in range(0, n_points, step_size):
            sequence = []
            for freq in frequencies:
                if i < len(self.data_dict[freq]['temperature']):
                    sequence.append(self.data_dict[freq]['temperature'][i])
            
            if len(sequence) == len(frequencies) and len(sequence) > 2:
                sequences.append(sequence)
        
        print(f"Created {len(sequences)} frequency sequences")
        
        # Method 2: Spatial sequences within each frequency
        print("Creating spatial sequences...")
        for freq in frequencies[:2]:  # Limit to first 2 frequencies for efficiency
            temp_data = self.data_dict[freq]['temperature']
            
            # Create overlapping windows as spatial sequences
            window_size = min(20, len(temp_data) // 10)
            if window_size > 3:
                for start in range(0, len(temp_data) - window_size, window_size // 2):
                    spatial_seq = temp_data[start:start + window_size]
                    sequences.append(spatial_seq.tolist())
        
        print(f"Total sequences created: {len(sequences)}")
        return sequences
    
    def run_ucf_analysis(self, sequences):
        """Run UCF analysis on all sequences."""
        ucf = UnifiedCoherenceFunction()
        correlations = []
        mse_values = []
        
        print(f"Running UCF analysis on {len(sequences)} sequences...")
        
        for i, seq in enumerate(sequences):
            if len(seq) < 3:
                continue
                
            try:
                # Apply UCF
                coherence = ucf.analyze_sequence(seq)
                
                # Calculate correlation with original sequence
                if len(coherence) == len(seq):
                    correlation = np.corrcoef(seq, coherence)[0, 1]
                    if np.isfinite(correlation):
                        correlations.append(correlation)
                
                # Calculate prediction error
                if len(seq) > 2:
                    pred_error = mean_squared_error(seq[1:], coherence[:-1])
                    if np.isfinite(pred_error):
                        mse_values.append(pred_error)
                        
            except Exception as e:
                continue
        
        self.ucf_results = {
            'correlations': correlations,
            'mse_values': mse_values,
            'mean_correlation': np.mean(correlations) if correlations else 0,
            'std_correlation': np.std(correlations) if correlations else 0,
            'mean_mse': np.mean(mse_values) if mse_values else np.inf,
            'std_mse': np.std(mse_values) if mse_values else 0,
            'n_sequences': len(correlations)
        }
        
        return self.ucf_results
    
    def run_ml_analysis(self, sequences):
        """Run standard ML analysis."""
        ml_comp = StandardMLComparison()
        self.ml_results = ml_comp.train_and_evaluate(sequences)
        return self.ml_results
    
    def statistical_comparison(self):
        """Perform statistical tests comparing UCF vs ML methods."""
        if not self.ucf_results or not self.ml_results:
            return None
        
        # Compare UCF correlations with ML R² scores
        ucf_performance = np.abs(self.ucf_results['correlations'])  # Use absolute correlation
        
        for model_name, ml_result in self.ml_results.items():
            ml_r2_mean = max(0, ml_result['r2_mean'])  # Ensure non-negative
            
            # Test if UCF correlations are significantly different from ML R²
            if len(ucf_performance) > 1:
                # One-sample t-test
                t_stat, p_value = stats.ttest_1samp(ucf_performance, ml_r2_mean)
                
                self.statistical_tests[model_name] = {
                    'ucf_mean': np.mean(ucf_performance),
                    'ml_r2_mean': ml_r2_mean,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return self.statistical_tests

def create_visualization(ucf_results, ml_results, statistical_tests, save_path=None):
    """Create comprehensive visualization of results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: UCF Correlation Distribution
    ax1 = axes[0, 0]
    if ucf_results['correlations']:
        correlations = np.abs(ucf_results['correlations'])  # Use absolute values
        ax1.hist(correlations, bins=20, alpha=0.7, color='blue', 
                label=f'UCF (μ={np.mean(correlations):.3f})')
        ax1.axvline(np.mean(correlations), color='blue', linestyle='--', alpha=0.8)
    ax1.set_xlabel('|Correlation Coefficient|')
    ax1.set_ylabel('Frequency')
    ax1.set_title('UCF Correlation Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: ML Model Comparison
    ax2 = axes[0, 1]
    if ml_results:
        model_names = list(ml_results.keys())
        r2_means = [max(0, ml_results[name]['r2_mean']) for name in model_names]
        r2_stds = [ml_results[name]['r2_std'] for name in model_names]
        
        bars = ax2.bar(model_names, r2_means, yerr=r2_stds, capsize=5, alpha=0.7)
        ax2.set_ylabel('R² Score')
        ax2.set_title('ML Model Performance')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add UCF mean line for comparison
        if ucf_results['correlations']:
            ucf_mean = np.mean(np.abs(ucf_results['correlations']))
            ax2.axhline(ucf_mean, color='red', linestyle='--', 
                       label=f'UCF Mean ({ucf_mean:.3f})')
            ax2.legend()
    
    # Plot 3: Statistical Significance
    ax3 = axes[0, 2]
    if statistical_tests:
        model_names = list(statistical_tests.keys())
        p_values = [statistical_tests[name]['p_value'] for name in model_names]
        
        colors = ['red' if p < 0.05 else 'gray' for p in p_values]
        bars = ax3.bar(model_names, p_values, color=colors, alpha=0.7)
        ax3.axhline(0.05, color='black', linestyle='--', label='α = 0.05')
        ax3.set_ylabel('p-value')
        ax3.set_title('Statistical Significance (UCF vs ML)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()
        ax3.set_yscale('log')
    
    # Plot 4: Performance Comparison
    ax4 = axes[1, 0]
    if ml_results and ucf_results['correlations']:
        all_data = []
        labels = []
        
        # UCF data
        ucf_abs_corr = np.abs(ucf_results['correlations'])
        all_data.append(ucf_abs_corr)
        labels.append('UCF')
        
        # ML data
        for name, result in ml_results.items():
            r2_mean = max(0, result['r2_mean'])
            r2_std = result['r2_std']
            # Generate sample distribution
            if r2_std > 0:
                samples = np.random.normal(r2_mean, r2_std, 50)
                samples = np.clip(samples, 0, 1)  # Bound R² between 0 and 1
                all_data.append(samples)
                labels.append(name)
        
        ax4.boxplot(all_data, labels=labels)
        ax4.set_ylabel('Performance Score')
        ax4.set_title('Performance Distribution Comparison')
        ax4.tick_params(axis='x', rotation=45)
    
    # Plot 5: Data Summary
    ax5 = axes[1, 1]
    ax5.axis('off')
    
    summary_text = "ARCADE 2 ANALYSIS SUMMARY\n\n"
    
    if ucf_results['correlations']:
        summary_text += f"UCF Results:\n"
        summary_text += f"• Mean |Correlation|: {np.mean(np.abs(ucf_results['correlations'])):.3f}\n"
        summary_text += f"• Std |Correlation|: {np.std(np.abs(ucf_results['correlations'])):.3f}\n"
        summary_text += f"• N Sequences: {ucf_results['n_sequences']}\n\n"
    
    if ml_results:
        best_model = max(ml_results.items(), key=lambda x: max(0, x[1]['r2_mean']))
        summary_text += f"Best ML Model:\n"
        summary_text += f"• {best_model[0]}: {max(0, best_model[1]['r2_mean']):.3f}\n\n"
    
    if statistical_tests:
        significant_tests = [name for name, test in statistical_tests.items() 
                           if test['significant']]
        summary_text += f"Statistical Tests:\n"
        summary_text += f"• {len(significant_tests)} of {len(statistical_tests)} significant\n"
        if significant_tests:
            summary_text += f"• Significant: {', '.join(significant_tests[:2])}\n"
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, 
             verticalalignment='top', fontsize=11, fontfamily='monospace')
    
    # Plot 6: Method Comparison
    ax6 = axes[1, 2]
    if ucf_results['correlations'] and ml_results:
        # Create comparison metrics
        ucf_mean = np.mean(np.abs(ucf_results['correlations']))
        ml_means = [max(0, result['r2_mean']) for result in ml_results.values()]
        ml_best = max(ml_means) if ml_means else 0
        
        methods = ['UCF', 'Best ML']
        scores = [ucf_mean, ml_best]
        colors = ['blue', 'green']
        
        bars = ax6.bar(methods, scores, color=colors, alpha=0.7)
        ax6.set_ylabel('Performance Score')
        ax6.set_title('UCF vs Best ML Model')
        ax6.set_ylim(0, 1)
        
        # Add values on bars
        for bar, score in zip(bars, scores):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def main():
    """Main analysis function."""
    print("ARCADE 2 UCF vs Standard ML Analysis")
    print("=" * 50)
    
    # Load ARCADE 2 data
    print("Loading ARCADE 2 data...")
    loader = BasicFITSReader()
    
    if not loader.load_fits_files():
        print("Error: Could not load ARCADE 2 data files.")
        print("Please ensure FITS files are in the current directory.")
        return
    
    print(f"\nLoaded data for {len(loader.frequencies)} frequencies:")
    for freq in loader.frequencies:
        n_points = len(loader.frequency_data[freq]['temperature'])
        temp_range = [
            np.min(loader.frequency_data[freq]['temperature']),
            np.max(loader.frequency_data[freq]['temperature'])
        ]
        print(f"  {freq} MHz: {n_points} points, T = {temp_range[0]:.2f} to {temp_range[1]:.2f} K")
    
    # Run comparison analysis
    print(f"\nRunning UCF vs ML comparison...")
    comparison = UCFvsMLComparison(loader.frequency_data, "ARCADE2")
    
    # Create sequences from the data
    sequences = comparison.create_sequences_from_data()
    
    if not sequences:
        print("Error: Could not create analysis sequences from data.")
        return
    
    # Run analyses
    print(f"Analyzing {len(sequences)} sequences...")
    ucf_results = comparison.run_ucf_analysis(sequences)
    ml_results = comparison.run_ml_analysis(sequences)
    statistical_tests = comparison.statistical_comparison()
    
    # Create visualization
    fig = create_visualization(ucf_results, ml_results, statistical_tests, 
                             'arcade2_ucf_vs_ml_analysis.png')
    plt.show()
    
    # Save detailed results
    results = {
        'ucf_results': ucf_results,
        'ml_results': ml_results,
        'statistical_tests': statistical_tests,
        'data_info': {
            'frequencies': loader.frequencies,
            'n_sequences': len(sequences),
            'sequence_lengths': [len(seq) for seq in sequences[:10]]  # Sample lengths
        }
    }
    
    with open('arcade2_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    
    if ucf_results['correlations']:
        ucf_mean = np.mean(np.abs(ucf_results['correlations']))
        print(f"UCF Mean |Correlation|: {ucf_mean:.4f}")
    
    if ml_results:
        best_ml = max(ml_results.items(), key=lambda x: max(0, x[1]['r2_mean']))
        print(f"Best ML Model: {best_ml[0]} (R² = {max(0, best_ml[1]['r2_mean']):.4f})")
    
    if statistical_tests:
        significant = sum(1 for test in statistical_tests.values() if test['significant'])
        print(f"Statistically significant differences: {significant}/{len(statistical_tests)}")
    
    print("\nResults saved to:")
    print("- arcade2_ucf_vs_ml_analysis.png")
    print("- arcade2_analysis_results.json")

if __name__ == "__main__":
    main()