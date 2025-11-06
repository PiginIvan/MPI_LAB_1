import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python graphs.py results.csv")
    sys.exit(1)

fn = sys.argv[1]

cols = ['N','P','T_par','T_seq','Speedup','Efficiency']
df = pd.read_csv(fn, header=None, names=cols, dtype=str)

df['N'] = pd.to_numeric(df['N'], errors='coerce').astype('Int64')
df['P'] = pd.to_numeric(df['P'], errors='coerce').astype('Int64')
df['T_par'] = pd.to_numeric(df['T_par'], errors='coerce').astype(float)
df['T_seq'] = pd.to_numeric(df['T_seq'], errors='coerce').astype(float)
df['Speedup'] = pd.to_numeric(df['Speedup'], errors='coerce').astype(float)
df['Efficiency'] = pd.to_numeric(df['Efficiency'], errors='coerce').astype(float)

before = len(df)
df = df.dropna(subset=['N','P','T_par'])
after = len(df)
if before != after:
    print(f"Warning: dropped {before-after} invalid rows")

agg = df.groupby(['N','P'], as_index=False).median()

Ns = sorted(agg['N'].unique())
Ps = sorted(agg['P'].unique())

print("Found matrix sizes N:", Ns)
print("Found process counts P:", Ps)

plt.figure(figsize=(8,6))
for p in Ps:
    sub = agg[agg['P']==p].sort_values('N')
    if sub.empty: continue
    plt.plot(sub['N'], sub['T_par'], marker='o', label=f'P={p}')
plt.xlabel('Matrix size N')
plt.ylabel('Parallel time T_par (s)')
plt.title('T_par vs N (lines = different P)')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('time_vs_N.png', dpi=150)
print("Saved: time_vs_N.png")

plt.figure(figsize=(8,6))
markers = ['o','s','^','d','v','p','*','h','x','+']
for i,n in enumerate(Ns):
    sub = agg[agg['N']==n].sort_values('P')
    if sub.empty: continue
    plt.plot(sub['P'], sub['Speedup'], marker=markers[i%len(markers)], linestyle='-', label=f'N={n}')
x = np.array(Ps)
plt.plot(x, x, '--', color='gray', label='ideal S=P')
plt.xscale('log', base=2)
plt.xlabel('Processes P (log2 scale)')
plt.ylabel('Speedup S(P)')
plt.title('Speedup vs P for different N')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('speedup_vs_P_per_N.png', dpi=150)
print("Saved: speedup_vs_P_per_N.png")

plt.figure(figsize=(8,6))
for i,p in enumerate(Ps):
    sub = agg[agg['P']==p].sort_values('N')
    if sub.empty: continue
    plt.plot(sub['N'], sub['Speedup'], marker=markers[i%len(markers)], linestyle='-', label=f'P={p}')
plt.xlabel('Matrix size N')
plt.ylabel('Speedup S')
plt.title('Speedup vs N for different P')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('speedup_vs_N_per_P.png', dpi=150)
print("Saved: speedup_vs_N_per_P.png")

plt.figure(figsize=(8,6))
for i,n in enumerate(Ns):
    sub = agg[agg['N']==n].sort_values('P')
    if sub.empty: continue
    plt.plot(sub['P'], sub['Efficiency'], marker=markers[i%len(markers)], linestyle='-', label=f'N={n}')
plt.xscale('log', base=2)
plt.xlabel('Processes P (log2 scale)')
plt.ylabel('Efficiency E(P)')
plt.title('Efficiency vs P for different N')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('efficiency_vs_P.png', dpi=150)
print("Saved: efficiency_vs_P.png")

print("\nSummary (N, P, T_par, Speedup, Efficiency):")
print(agg[['N','P','T_par','Speedup','Efficiency']].to_string(index=False))

print("\nDone.")
