#!/usr/bin/env python3

import subprocess
import time
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import argparse
import threading
import signal
import sys
from collections import defaultdict, deque

class NVLinkMonitor:
    def __init__(self, duration=300, update_interval=1):
        self.duration = duration
        self.update_interval = update_interval
        self.monitoring = True
        self.data = defaultdict(lambda: deque(maxlen=300))  # Keep last 5 minutes
        self.timestamps = deque(maxlen=300)
        
    def get_nvlink_throughput(self):
        """Get current NVLink throughput using nvidia-smi"""
        try:
            # Get throughput data
            result = subprocess.run(
                ['nvidia-smi', 'nvlink', '-gt', 'd'],
                capture_output=True, text=True, check=True
            )
            
            throughput_data = {}
            current_gpu = None
            
            for line in result.stdout.split('\n'):
                line = line.strip()
                if line.startswith('GPU'):
                    current_gpu = int(line.split(':')[0].split()[-1])
                    throughput_data[current_gpu] = {}
                elif 'Link' in line and 'Data' in line:
                    parts = line.split()
                    link_num = int(parts[1].rstrip(':'))
                    direction = parts[3].rstrip(':')  # Tx or Rx
                    value_kib = int(parts[4])
                    
                    if current_gpu is not None:
                        key = f"GPU{current_gpu}_Link{link_num}_{direction}"
                        throughput_data[key] = value_kib
                        
            return throughput_data
            
        except subprocess.CalledProcessError as e:
            print(f"Error getting NVLink data: {e}")
            return {}
    
    def calculate_bandwidth(self, current_data, previous_data, time_delta):
        """Calculate bandwidth from cumulative counters"""
        bandwidth = {}
        for key in current_data:
            if key in previous_data and time_delta > 0:
                # Convert KiB/s to MB/s (1 KiB = 1024 bytes)
                diff_kib = current_data[key] - previous_data[key]
                bandwidth_mbs = (diff_kib * 1024) / (time_delta * 1024 * 1024)
                bandwidth[key] = max(0, bandwidth_mbs)  # Ensure non-negative
        return bandwidth
    
    def monitor_bandwidth(self):
        """Monitor NVLink bandwidth continuously"""
        print("🚀 Starting NVLink bandwidth monitoring...")
        print(f"📊 Duration: {self.duration} seconds")
        print("📈 Monitoring GPU-to-GPU transfers...")
        
        previous_data = None
        previous_time = None
        
        try:
            while self.monitoring:
                current_time = time.time()
                current_data = self.get_nvlink_throughput()
                
                if previous_data is not None and previous_time is not None:
                    time_delta = current_time - previous_time
                    bandwidth = self.calculate_bandwidth(current_data, previous_data, time_delta)
                    
                    # Store data for plotting
                    self.timestamps.append(datetime.now())
                    
                    # Aggregate GPU-to-GPU bandwidth (sum of active links)
                    gpu_totals = defaultdict(lambda: {'tx': 0, 'rx': 0})
                    
                    for key, bw in bandwidth.items():
                        if bw > 0.1:  # Only show significant bandwidth
                            parts = key.split('_')
                            gpu = parts[0]  # GPU0, GPU1, etc.
                            direction = parts[2].lower()  # tx or rx
                            gpu_totals[gpu][direction] += bw
                    
                    # Store total bandwidth per GPU
                    for gpu, totals in gpu_totals.items():
                        self.data[f"{gpu}_TX"].append(totals['tx'])
                        self.data[f"{gpu}_RX"].append(totals['rx'])
                    
                    # Print current bandwidth
                    if any(totals['tx'] > 1 or totals['rx'] > 1 for totals in gpu_totals.values()):
                        print(f"\n⚡ {datetime.now().strftime('%H:%M:%S')} - Active NVLink Traffic:")
                        for gpu, totals in gpu_totals.items():
                            if totals['tx'] > 1 or totals['rx'] > 1:
                                print(f"  {gpu}: TX={totals['tx']:.1f} MB/s, RX={totals['rx']:.1f} MB/s")
                
                previous_data = current_data
                previous_time = current_time
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"❌ Error during monitoring: {e}")
        
        self.monitoring = False
    
    def create_plot(self):
        """Create real-time plot of NVLink bandwidth"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('NVLink Bandwidth Monitor - Real-time GPU Communication', fontsize=14)
        
        def animate(frame):
            if not self.data or not self.timestamps:
                return
            
            # Clear axes
            ax1.clear()
            ax2.clear()
            
            # Plot TX bandwidth
            ax1.set_title('NVLink TX Bandwidth (GPU → Others)')
            ax1.set_ylabel('Bandwidth (MB/s)')
            
            # Plot RX bandwidth  
            ax2.set_title('NVLink RX Bandwidth (Others → GPU)')
            ax2.set_ylabel('Bandwidth (MB/s)')
            ax2.set_xlabel('Time')
            
            colors = ['red', 'blue', 'green', 'orange']
            
            for i, gpu_id in enumerate(['GPU0', 'GPU1', 'GPU2', 'GPU3']):
                tx_key = f"{gpu_id}_TX"
                rx_key = f"{gpu_id}_RX"
                
                if tx_key in self.data and len(self.data[tx_key]) > 0:
                    times = list(self.timestamps)[-len(self.data[tx_key]):]
                    
                    # TX plot
                    ax1.plot(times, list(self.data[tx_key]), 
                            label=gpu_id, color=colors[i], linewidth=2)
                    
                    # RX plot
                    if rx_key in self.data:
                        ax2.plot(times, list(self.data[rx_key]), 
                                label=gpu_id, color=colors[i], linewidth=2)
            
            # Formatting
            for ax in [ax1, ax2]:
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(bottom=0)
                
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            plt.tight_layout()
        
        ani = animation.FuncAnimation(fig, animate, interval=1000, cache_frame_data=False)
        plt.show()
        return ani

def main():
    parser = argparse.ArgumentParser(description='NVLink Bandwidth Monitor')
    parser.add_argument('--duration', type=int, default=300, 
                       help='Monitoring duration in seconds (default: 300)')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Update interval in seconds (default: 1.0)')
    parser.add_argument('--plot', action='store_true',
                       help='Show real-time plot')
    parser.add_argument('--no-plot', action='store_true',
                       help='Text-only monitoring')
    
    args = parser.parse_args()
    
    monitor = NVLinkMonitor(duration=args.duration, update_interval=args.interval)
    
    # Start monitoring in background thread
    monitor_thread = threading.Thread(target=monitor.monitor_bandwidth)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print('\n🛑 Stopping monitor...')
        monitor.monitoring = False
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    if args.plot and not args.no_plot:
        try:
            print("📊 Starting real-time plot... (Close plot window to stop)")
            ani = monitor.create_plot()
        except ImportError:
            print("❌ matplotlib not available, using text-only mode")
            monitor_thread.join()
    else:
        print("📊 Text-only monitoring mode")
        monitor_thread.join()

if __name__ == "__main__":
    main() 