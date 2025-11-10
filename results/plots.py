import pandas as pd
import matplotlib.pyplot as plt

# Load your performance data
df = pd.read_csv("bench_clean.csv")  # or whichever CSV you saved
df['peak_rss_gb'] = df['peak_rss_bytes'] / (1024**3)

# --- Wall-clock time vs. parallel files ---
plt.figure(figsize=(8,5))
for tiles, group in df.groupby('num_tiles'):
    plt.plot(group['max_parallel_files'], group['wall_time_sec'], marker='o', label=f'{tiles} tiles')
plt.xlabel('Max Parallel Files')
plt.ylabel('Wall Clock Time (s)')
plt.title('Effect of Parallelism on Wall-Clock Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('wallclock_vs_parallel.png', dpi=300)
plt.show()

# --- Peak RSS vs. parallel files ---
plt.figure(figsize=(8,5))
for tiles, group in df.groupby('num_tiles'):
    plt.plot(group['max_parallel_files'], group['peak_rss_gb'], marker='o', label=f'{tiles} tiles')
plt.xlabel('Max Parallel Files')
plt.ylabel('Peak RSS (GB)')
plt.title('Memory Usage vs. Parallelism')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('rss_vs_parallel.png', dpi=300)
plt.show()