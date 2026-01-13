"""
Clean ECG Denoising Visualizer
==============================

Simple, clean visualizer that creates neat images without overlapping text.
Focuses on clarity and simplicity with minimal text and clean formatting.

Features:
- Clean 3-panel layout (Raw, Noisy, Denoised)
- Minimal text to avoid overlapping
- Clear headings with proper spacing
- Professional appearance without clutter
- High-quality PNG outputs

Author: ECG Signal Processing Research
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for importing the actual denoiser
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ECG-Denoising-Completed'))

try:
    from adaptive_vmd_ecg_denoiser_multi_dataset_validation import AdaptiveVMDECGDenoiserValidator
    REAL_DENOISER_AVAILABLE = True
    print("✅ Real ECG Denoiser algorithm loaded successfully")
except ImportError as e:
    REAL_DENOISER_AVAILABLE = False
    print(f"❌ Real ECG Denoiser not available: {e}")

class CleanECGVisualizer:
    """
    Clean ECG visualizer that creates simple, neat images without text overlap.
    """
    
    def __init__(self, output_dir="ECG-Denoising-Visualizations"):
        """Initialize the clean ECG visualizer."""
        self.output_dir = output_dir
        self.create_output_directories()
        self.setup_clean_plotting_style()
        
        # Initialize the real denoiser
        if REAL_DENOISER_AVAILABLE:
            self.denoiser = AdaptiveVMDECGDenoiserValidator(target_snr_db=50.0)
            print("🔬 Real Adaptive VMD Denoiser initialized")
        else:
            self.denoiser = None
            print("⚠️  Real denoiser not available")
        
        print(f"🎨 Clean ECG Visualizer initialized")
        print(f"📁 Output directory: {self.output_dir}")
    
    def create_output_directories(self):
        """Create clean directory structure."""
        directories = [
            f"{self.output_dir}/Clean_Visualizations",
            f"{self.output_dir}/Clean_Visualizations/MIT-BIH_Results",
            f"{self.output_dir}/Clean_Visualizations/PTB-XL_Results", 
            f"{self.output_dir}/Clean_Visualizations/INCART_Results"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def setup_clean_plotting_style(self):
        """Set up clean, minimal plotting style."""
        plt.style.use('default')
        
        plt.rcParams.update({
            'figure.figsize': (16, 10),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.3,
            'font.size': 11,
            'font.family': 'sans-serif',
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'lines.linewidth': 2.0,
            'legend.fontsize': 10,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.facecolor': 'white'
        })
    
    def create_clean_visualization(self, clean_signal, noisy_signal, denoised_signal,
                                 sampling_rate, dataset_name, record_id, 
                                 performance_metrics, save_path):
        """
        Create clean 3-panel visualization without overlapping text.
        """
        # Create figure with clean layout
        fig, axes = plt.subplots(3, 1, figsize=(16, 10))
        
        # Time vector
        duration = len(clean_signal) / sampling_rate
        time_vector = np.linspace(0, duration, len(clean_signal))
        
        # Clean color scheme
        colors = {
            'clean': '#2E8B57',    # Forest Green
            'noisy': '#DC143C',    # Crimson Red
            'denoised': '#4169E1'  # Royal Blue
        }
        
        # Panel 1: Raw Signal
        axes[0].plot(time_vector, clean_signal, color=colors['clean'], linewidth=2)
        axes[0].set_title('Raw ECG Signal', fontsize=14, fontweight='bold', pad=15)
        axes[0].set_ylabel('Amplitude (mV)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Panel 2: Noisy Signal
        axes[1].plot(time_vector, noisy_signal, color=colors['noisy'], linewidth=2)
        axes[1].set_title(f'Noisy ECG Signal (SNR: {performance_metrics["input_snr_db"]:.1f} dB)', 
                         fontsize=14, fontweight='bold', pad=15)
        axes[1].set_ylabel('Amplitude (mV)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # Panel 3: Denoised Signal
        axes[2].plot(time_vector, denoised_signal, color=colors['denoised'], linewidth=2)
        axes[2].set_title(f'Denoised ECG Signal (SNR: {performance_metrics["output_snr_db"]:.1f} dB)', 
                         fontsize=14, fontweight='bold', pad=15)
        axes[2].set_xlabel('Time (seconds)', fontsize=12)
        axes[2].set_ylabel('Amplitude (mV)', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        # Main title with proper spacing
        fig.suptitle(f'{dataset_name} Dataset - Record {record_id}\nECG Denoising Results', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Adjust layout to prevent overlapping
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        # Save the clean visualization
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Clean visualization saved: {save_path}")
    
    def create_performance_summary(self, results, save_path):
        """Create a clean performance summary visualization."""
        if not results:
            return
        
        # Create clean summary figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Group results by dataset
        datasets = {}
        for result in results:
            dataset_name = result['dataset']
            if dataset_name not in datasets:
                datasets[dataset_name] = []
            datasets[dataset_name].append(result)
        
        # Colors for datasets
        colors = ['#2E8B57', '#DC143C', '#4169E1']
        
        # Plot 1: SNR Performance
        ax1 = axes[0, 0]
        dataset_names = list(datasets.keys())
        output_snrs = [np.mean([r['output_snr'] for r in datasets[name]]) for name in dataset_names]
        
        bars = ax1.bar(range(len(dataset_names)), output_snrs, color=colors[:len(dataset_names)])
        ax1.set_title('Output SNR Performance', fontsize=14, fontweight='bold')
        ax1.set_ylabel('SNR (dB)', fontsize=12)
        ax1.set_xticks(range(len(dataset_names)))
        ax1.set_xticklabels([name.replace(' ', '\n') for name in dataset_names])
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: RMSE Performance
        ax2 = axes[0, 1]
        rmse_values = [np.mean([r['rmse'] for r in datasets[name]]) for name in dataset_names]
        
        bars = ax2.bar(range(len(dataset_names)), rmse_values, color=colors[:len(dataset_names)])
        ax2.set_title('RMSE Performance', fontsize=14, fontweight='bold')
        ax2.set_ylabel('RMSE', fontsize=12)
        ax2.set_xticks(range(len(dataset_names)))
        ax2.set_xticklabels([name.replace(' ', '\n') for name in dataset_names])
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Success Rate
        ax3 = axes[1, 0]
        success_rates = []
        for name in dataset_names:
            dataset_results = datasets[name]
            success_rate = sum(1 for r in dataset_results if r['output_snr'] >= 50) / len(dataset_results) * 100
            success_rates.append(success_rate)
        
        bars = ax3.bar(range(len(dataset_names)), success_rates, color=colors[:len(dataset_names)])
        ax3.set_title('Success Rate (50+ dB)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Success Rate (%)', fontsize=12)
        ax3.set_xticks(range(len(dataset_names)))
        ax3.set_xticklabels([name.replace(' ', '\n') for name in dataset_names])
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.0f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Overall Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate overall stats
        all_snrs = [r['output_snr'] for r in results]
        all_rmses = [r['rmse'] for r in results]
        overall_success = sum(1 for snr in all_snrs if snr >= 50) / len(all_snrs) * 100
        
        stats_text = f"""Overall Performance Summary
        
Total Records: {len(results)}
Average SNR: {np.mean(all_snrs):.2f} dB
Average RMSE: {np.mean(all_rmses):.4f}
Success Rate: {overall_success:.0f}%

Algorithm Status: {'✅ EXCELLENT' if overall_success >= 90 else '👍 GOOD'}"""
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # Main title
        fig.suptitle('ECG Denoising Algorithm Performance Summary', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Save summary
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Performance summary saved: {save_path}")
    
    def process_dataset_clean(self, dataset_name, data_loader_function, record_list):
        """Process dataset with clean visualization output."""
        if not REAL_DENOISER_AVAILABLE:
            print("❌ Real denoiser not available")
            return []
        
        print(f"\n🔬 Processing {dataset_name} Dataset")
        print("-" * 50)
        
        dataset_dir = f"{self.output_dir}/Clean_Visualizations/{dataset_name.replace('-', '_').replace(' ', '_')}_Results"
        os.makedirs(dataset_dir, exist_ok=True)
        
        results = []
        
        for i, record_id in enumerate(record_list):
            print(f"   📊 Processing Record {i+1}/{len(record_list)}: {record_id or 'synthetic'}")
            
            try:
                # Load clean ECG data
                if record_id is not None:
                    clean_ecg, sampling_rate, description = data_loader_function(record_id)
                else:
                    clean_ecg, sampling_rate, description = data_loader_function()
                
                if clean_ecg is None:
                    print(f"      ❌ Failed to load data")
                    continue
                
                # Add noise and apply denoising
                np.random.seed(42 + i)
                input_snr_db = 11.8
                noisy_ecg = self.denoiser.add_awgn_noise(clean_ecg, input_snr_db)
                denoised_ecg = self.denoiser.apply_adaptive_vmd_denoising(noisy_ecg, sampling_rate)
                
                # Compute performance metrics
                performance_metrics = self.denoiser.compute_performance_metrics(
                    clean_ecg, noisy_ecg, denoised_ecg
                )
                
                # Create clean visualization
                record_name = record_id or f"Patient_{i+1}"
                save_path = f"{dataset_dir}/{dataset_name.replace(' ', '_')}_{record_name}_clean.png"
                
                self.create_clean_visualization(
                    clean_ecg, noisy_ecg, denoised_ecg, sampling_rate,
                    dataset_name, record_name, performance_metrics, save_path
                )
                
                # Store results
                result = {
                    'dataset': dataset_name,
                    'record': record_name,
                    'sampling_rate': sampling_rate,
                    'input_snr': performance_metrics['input_snr_db'],
                    'output_snr': performance_metrics['output_snr_db'],
                    'rmse': performance_metrics['rmse'],
                    'correlation': performance_metrics['correlation_coefficient']
                }
                results.append(result)
                
                print(f"      ✅ Output SNR: {performance_metrics['output_snr_db']:.2f} dB")
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
                continue
        
        return results
    
    def run_clean_visualization_suite(self):
        """Run the complete clean visualization suite."""
        print("🎨 CLEAN ECG DENOISING VISUALIZATION SUITE")
        print("="*60)
        print("Creating clean, neat images without overlapping text")
        print("="*60)
        
        if not REAL_DENOISER_AVAILABLE:
            print("❌ Real denoiser not available")
            return
        
        all_results = []
        
        # Check for WFDB availability
        try:
            import wfdb
            wfdb_available = True
            print("✅ WFDB library available")
        except ImportError:
            wfdb_available = False
            print("⚠️  WFDB not available - using synthetic data")
        
        # Process datasets
        if wfdb_available:
            # MIT-BIH dataset
            mitbih_results = self.process_dataset_clean(
                'MIT-BIH', 
                self.denoiser.load_mitbih_arrhythmia_data, 
                ['100', '101', '102']
            )
            all_results.extend(mitbih_results)
            
            # INCART dataset
            incart_results = self.process_dataset_clean(
                'INCART', 
                self.denoiser.load_incart_annotated_data, 
                ['I01', 'I02', 'I03']
            )
            all_results.extend(incart_results)
        
        # PTB-XL dataset
        ptbxl_results = self.process_dataset_clean(
            'PTB-XL', 
            self.denoiser.load_ptbxl_clinical_data, 
            [None, None, None]
        )
        all_results.extend(ptbxl_results)
        
        # Create performance summary
        if all_results:
            summary_path = f"{self.output_dir}/Clean_Visualizations/performance_summary_clean.png"
            self.create_performance_summary(all_results, summary_path)
        
        print(f"\n🎉 CLEAN VISUALIZATION SUITE COMPLETE!")
        print(f"📁 Clean PNG files saved in: {self.output_dir}/Clean_Visualizations")
        print(f"📊 Total visualizations: {len(all_results) + 1}")
        print("="*60)
        
        return all_results


def main():
    """Main function to run the clean ECG visualization system."""
    print("🎨 CLEAN ECG DENOISING VISUALIZER")
    print("Simple, neat images without text overlap")
    print("="*60)
    
    # Initialize clean visualizer
    visualizer = CleanECGVisualizer()
    
    # Run clean visualization suite
    results = visualizer.run_clean_visualization_suite()
    
    if results:
        print(f"\n🎉 CLEAN VISUALIZATION COMPLETE!")
        print(f"📊 Average performance: {np.mean([r['output_snr'] for r in results]):.2f} dB SNR")
        print("All images are clean and neat without overlapping text!")


if __name__ == "__main__":
    main()