"""
Realistic Web Server Resource Usage Simulator
Generates realistic CPU, RAM, and bandwidth usage patterns for web servers.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import math
from typing import Dict, List, Tuple, Optional

class ResourceSimulator:
    """Generates realistic web server resource usage patterns."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the resource simulator with optional random seed."""
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        
        # Base resource usage patterns by server type
        self.server_profiles = {
            'web-frontend': {
                'base_cpu': 25,
                'peak_cpu': 85,
                'base_ram_gb': 4,
                'peak_ram_gb': 12,
                'base_bandwidth_mb': 50,
                'peak_bandwidth_mb': 500,
                'business_hours_factor': 2.5,
                'weekend_factor': 0.6,
                'seasonal_amplitude': 0.3
            },
            'api-backend': {
                'base_cpu': 30,
                'peak_cpu': 90,
                'base_ram_gb': 8,
                'peak_ram_gb': 24,
                'base_bandwidth_mb': 100,
                'peak_bandwidth_mb': 800,
                'business_hours_factor': 3.0,
                'weekend_factor': 0.4,
                'seasonal_amplitude': 0.4
            },
            'database': {
                'base_cpu': 15,
                'peak_cpu': 75,
                'base_ram_gb': 16,
                'peak_ram_gb': 64,
                'base_bandwidth_mb': 20,
                'peak_bandwidth_mb': 200,
                'business_hours_factor': 2.0,
                'weekend_factor': 0.8,
                'seasonal_amplitude': 0.2
            },
            'cache-server': {
                'base_cpu': 10,
                'peak_cpu': 60,
                'base_ram_gb': 8,
                'peak_ram_gb': 32,
                'base_bandwidth_mb': 200,
                'peak_bandwidth_mb': 1000,
                'business_hours_factor': 2.2,
                'weekend_factor': 0.7,
                'seasonal_amplitude': 0.25
            }
        }
    
    def generate_business_hours_factor(self, timestamp: datetime) -> float:
        """Generate business hours activity factor (0.3 to 3.0)."""
        hour = timestamp.hour
        weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
        
        # Base factor for different hours
        if 9 <= hour <= 17:  # Business hours
            base_factor = 1.0
        elif 7 <= hour <= 9 or 17 <= hour <= 21:  # Peak adjacent hours
            base_factor = 0.8
        elif 21 <= hour <= 23 or 6 <= hour <= 7:  # Evening/early morning
            base_factor = 0.6
        else:  # Night hours
            base_factor = 0.3
        
        # Weekend adjustment
        if weekday >= 5:  # Weekend
            base_factor *= 0.6
        
        # Add some randomness
        noise = np.random.normal(0, 0.1)
        factor = base_factor + noise
        
        return max(0.2, min(3.0, factor))
    
    def generate_seasonal_factor(self, timestamp: datetime) -> float:
        """Generate seasonal variation factor based on day of year."""
        day_of_year = timestamp.timetuple().tm_yday
        
        # Peak activity around holidays and end of year
        seasonal_cycle = 1.0 + 0.2 * math.sin(2 * math.pi * day_of_year / 365)
        
        # Holiday spikes
        holiday_periods = [
            (355, 365),  # End of year
            (120, 130),  # Spring period
            (240, 250),  # Late summer
        ]
        
        holiday_factor = 1.0
        for start, end in holiday_periods:
            if start <= day_of_year <= end:
                holiday_factor = 1.3
                break
        
        return seasonal_cycle * holiday_factor
    
    def generate_traffic_spikes(self, base_value: float, timestamp: datetime) -> float:
        """Generate occasional traffic spikes."""
        # 5% chance of a spike during business hours
        # 1% chance during off hours
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        if 9 <= hour <= 17 and weekday < 5:
            spike_probability = 0.05
        else:
            spike_probability = 0.01
        
        if np.random.random() < spike_probability:
            spike_multiplier = np.random.uniform(1.5, 3.0)
            return base_value * spike_multiplier
        
        return base_value
    
    def generate_memory_leak_factor(self, timestamp: datetime, server_id: str) -> float:
        """Simulate gradual memory usage increase (memory leaks)."""
        # Different servers have different leak patterns
        leak_rates = {
            'web-frontend': 0.001,
            'api-backend': 0.002,
            'database': 0.0005,
            'cache-server': 0.0015
        }
        
        rate = leak_rates.get(server_id, 0.001)
        hours_since_midnight = timestamp.hour + timestamp.minute / 60.0
        
        # Memory resets at night (server restart simulation)
        if 2 <= timestamp.hour <= 4:
            return 1.0
        
        # Gradual increase throughout the day
        leak_factor = 1.0 + rate * hours_since_midnight
        return min(leak_factor, 1.5)  # Cap at 50% increase
    
    def generate_cpu_usage(self, server_id: str, timestamp: datetime, 
                          previous_cpu: Optional[float] = None) -> float:
        """Generate realistic CPU usage percentage."""
        profile = self.server_profiles[server_id]
        
        # Base CPU with business hours factor
        business_factor = self.generate_business_hours_factor(timestamp)
        seasonal_factor = self.generate_seasonal_factor(timestamp)
        
        base_cpu = profile['base_cpu'] * business_factor * seasonal_factor
        
        # Add spikes
        cpu_with_spikes = self.generate_traffic_spikes(base_cpu, timestamp)
        
        # Add some correlation with previous value (smoothing)
        if previous_cpu is not None:
            correlation_factor = 0.7
            cpu_with_spikes = correlation_factor * previous_cpu + (1 - correlation_factor) * cpu_with_spikes
        
        # Add noise
        noise = np.random.normal(0, 5)
        final_cpu = cpu_with_spikes + noise
        
        # Ensure realistic bounds
        return max(1, min(100, final_cpu))
    
    def generate_ram_usage(self, server_id: str, timestamp: datetime, 
                          cpu_usage: float, previous_ram: Optional[float] = None) -> Tuple[float, float]:
        """Generate realistic RAM usage in MB. Returns (usage_mb, total_mb)."""
        profile = self.server_profiles[server_id]
        
        # Base RAM correlated with CPU usage
        cpu_correlation = 0.7 + (cpu_usage / 100) * 0.3
        base_ram_gb = profile['base_ram_gb'] * cpu_correlation
        
        # Apply memory leak factor
        leak_factor = self.generate_memory_leak_factor(timestamp, server_id)
        ram_with_leak = base_ram_gb * leak_factor
        
        # Business hours adjustment
        business_factor = self.generate_business_hours_factor(timestamp)
        ram_adjusted = ram_with_leak * (0.8 + 0.4 * business_factor)
        
        # Add some correlation with previous value
        if previous_ram is not None:
            correlation_factor = 0.8
            ram_adjusted = correlation_factor * (previous_ram / 1024) + (1 - correlation_factor) * ram_adjusted
        
        # Add noise
        noise = np.random.normal(0, 0.5)
        final_ram_gb = max(profile['base_ram_gb'] * 0.5, ram_adjusted + noise)
        
        # Determine total RAM (allocated capacity)
        total_ram_gb = max(profile['peak_ram_gb'], final_ram_gb * 1.2)
        
        return final_ram_gb * 1024, total_ram_gb * 1024  # Convert to MB
    
    def generate_bandwidth_usage(self, server_id: str, timestamp: datetime, 
                                cpu_usage: float) -> Tuple[float, float]:
        """Generate realistic bandwidth usage in MB. Returns (inbound, outbound)."""
        profile = self.server_profiles[server_id]
        
        # Base bandwidth correlated with CPU usage
        cpu_factor = 0.5 + (cpu_usage / 100) * 0.5
        base_bandwidth = profile['base_bandwidth_mb'] * cpu_factor
        
        # Business hours factor
        business_factor = self.generate_business_hours_factor(timestamp)
        bandwidth_adjusted = base_bandwidth * business_factor
        
        # Add spikes
        bandwidth_with_spikes = self.generate_traffic_spikes(bandwidth_adjusted, timestamp)
        
        # Seasonal adjustment
        seasonal_factor = self.generate_seasonal_factor(timestamp)
        final_bandwidth = bandwidth_with_spikes * seasonal_factor
        
        # Add noise
        noise = np.random.normal(0, final_bandwidth * 0.1)
        inbound_mb = max(0, final_bandwidth + noise)
        
        # Outbound is typically different ratio depending on server type
        outbound_ratios = {
            'web-frontend': 0.8,  # Serves content
            'api-backend': 0.6,   # API responses
            'database': 0.3,      # Mostly receives queries
            'cache-server': 1.2   # Serves cached content
        }
        
        outbound_ratio = outbound_ratios.get(server_id, 0.7)
        outbound_mb = inbound_mb * outbound_ratio
        
        # CDN factor for web servers (reduces outbound traffic)
        if 'web' in server_id:
            cdn_factor = np.random.uniform(0.6, 0.8)
            outbound_mb *= cdn_factor
        
        return max(0, inbound_mb), max(0, outbound_mb)
    
    def generate_server_data(self, server_id: str, start_date: datetime, 
                           end_date: datetime, interval_minutes: int = 5) -> pd.DataFrame:
        """Generate complete resource data for a server over a date range."""
        timestamps = []
        current_time = start_date
        
        while current_time < end_date:
            timestamps.append(current_time)
            current_time += timedelta(minutes=interval_minutes)
        
        data = []
        previous_cpu = None
        previous_ram = None
        
        for timestamp in timestamps:
            # Generate CPU usage
            cpu_usage = self.generate_cpu_usage(server_id, timestamp, previous_cpu)
            
            # Generate RAM usage
            ram_usage_mb, ram_total_mb = self.generate_ram_usage(
                server_id, timestamp, cpu_usage, previous_ram
            )
            
            # Generate bandwidth usage
            bandwidth_in_mb, bandwidth_out_mb = self.generate_bandwidth_usage(
                server_id, timestamp, cpu_usage
            )
            
            data.append({
                'server_id': server_id,
                'timestamp': timestamp,
                'cpu_usage_percent': round(cpu_usage, 2),
                'ram_usage_mb': round(ram_usage_mb, 2),
                'ram_total_mb': round(ram_total_mb, 2),
                'bandwidth_in_mb': round(bandwidth_in_mb, 2),
                'bandwidth_out_mb': round(bandwidth_out_mb, 2)
            })
            
            # Update previous values for correlation
            previous_cpu = cpu_usage
            previous_ram = ram_usage_mb
        
        return pd.DataFrame(data)
    
    def generate_all_servers_data(self, start_date: datetime, end_date: datetime,
                                interval_minutes: int = 5) -> pd.DataFrame:
        """Generate data for all configured servers."""
        all_data = []
        
        for server_id in self.server_profiles.keys():
            print(f"Generating data for {server_id}...")
            server_data = self.generate_server_data(server_id, start_date, end_date, interval_minutes)
            all_data.append(server_data)
        
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data.sort_values(['timestamp', 'server_id']).reset_index(drop=True)

if __name__ == "__main__":
    # Test the simulator
    simulator = ResourceSimulator(seed=42)
    
    # Generate 7 days of data
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 8)
    
    print("Generating sample data...")
    data = simulator.generate_all_servers_data(start_date, end_date)
    
    print(f"Generated {len(data)} data points")
    print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"Servers: {data['server_id'].unique()}")
    
    # Show sample statistics
    for server in data['server_id'].unique():
        server_data = data[data['server_id'] == server]
        print(f"\n{server} statistics:")
        print(f"  CPU: {server_data['cpu_usage_percent'].mean():.1f}% ± {server_data['cpu_usage_percent'].std():.1f}%")
        print(f"  RAM: {server_data['ram_usage_mb'].mean()/1024:.1f}GB ± {server_data['ram_usage_mb'].std()/1024:.1f}GB")
        print(f"  Bandwidth Out: {server_data['bandwidth_out_mb'].mean():.1f}MB/5min ± {server_data['bandwidth_out_mb'].std():.1f}MB/5min")