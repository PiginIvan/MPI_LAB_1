import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python plot_mpi_results.py results.csv")
    sys.exit(1)

fn = sys.argv[1]
df = pd.read_csv(fn, header=None, names=['np','total_tosses','time_sec','pi'], encoding='utf-8-sig')
df = df.sort_values('np')

df['np'] = df['np'].astype(int)

if 1 in df['np'].values:
    T1 = float(df[df['np'] == 1]['time_sec'].iloc[0])
else:
    smallest_p = df['np'].min()
    T1 = float(df[df['np'] == smallest_p]['time_sec'].iloc[0])
    print(f"Warning: p=1 not present, using p={smallest_p} as baseline for speedup.")

df['speedup'] = T1 / df['time_sec']
df['efficiency'] = df['speedup'] / df['np']

plt.figure()
plt.plot(df['np'], df['time_sec'], marker='o')
plt.xscale('log', base=2)
plt.xlabel('Number of processes (p)')
plt.ylabel('Time (s)')
plt.title('Execution time vs processes')
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.savefig('time_vs_p.png', dpi=150)

plt.figure()
plt.plot(df['np'], df['speedup'], marker='o')
plt.xscale('log', base=2)
plt.xlabel('Number of processes (p)')
plt.ylabel('Speedup S_p')
plt.title('Speedup vs processes')
plt.grid(True, which='both', ls='--', alpha=0.4)
x = df['np']
plt.plot(x, x, linestyle='--', label='Ideal speedup')
plt.legend()
plt.savefig('speedup_vs_p.png', dpi=150)

plt.figure()
plt.plot(df['np'], df['efficiency'], marker='o')
plt.xscale('log', base=2)
plt.xlabel('Number of processes (p)')
plt.ylabel('Efficiency E_p')
plt.title('Efficiency vs processes')
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.savefig('efficiency_vs_p.png', dpi=150)

print("Plots saved: time_vs_p.png, speedup_vs_p.png, efficiency_vs_p.png")