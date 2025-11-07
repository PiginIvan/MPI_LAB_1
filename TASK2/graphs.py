import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python plot_matvec.py results.csv")
    sys.exit(1)

fn = sys.argv[1]
df = pd.read_csv(fn, header=None, names=['algo','np','n','time_sec'])
df['np'] = df['np'].astype(int)
df['n'] = df['n'].astype(int)

# For each matrix size n, plot time vs p, speedup and efficiency for each algo
sizes = sorted(df['n'].unique())
algos = df['algo'].unique()

# Общие графики со всеми данными
print("Creating combined plots with all data...")

# График времени выполнения (все данные)
plt.figure(figsize=(12, 8))
for idx, row in df.iterrows():
    label = f"{row['algo']}, n={row['n']}, p={row['np']}"
    plt.plot(row['np'], row['time_sec'], 'o', label=label, alpha=0.7)
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Number of processes (p)')
plt.ylabel('Time (s)')
plt.title('Execution time (all data)')
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("time_all_data.png", dpi=150, bbox_inches='tight')
plt.close()

# График времени выполнения с группировкой по алгоритмам
plt.figure(figsize=(12, 8))
for a in algos:
    algo_data = df[df['algo'] == a]
    for n in algo_data['n'].unique():
        n_data = algo_data[algo_data['n'] == n].sort_values('np')
        plt.plot(n_data['np'], n_data['time_sec'], marker='o', 
                label=f"{a}, n={n}", linewidth=2, markersize=6)
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Number of processes (p)')
plt.ylabel('Time (s)')
plt.title('Execution time by algorithm and matrix size')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig("time_by_algorithm.png", dpi=150, bbox_inches='tight')
plt.close()

# График ускорения (speedup) для всех данных
plt.figure(figsize=(12, 8))
for a in algos:
    for n in sizes:
        s = df[(df['algo'] == a) & (df['n'] == n)].sort_values('np')
        if s.empty: 
            continue
        
        # Находим базовое время для p=1
        baseline = s[s['np'] == 1]
        if baseline.empty:
            # Если нет данных для p=1, используем минимальное доступное p
            min_p = s['np'].min()
            baseline = s[s['np'] == min_p]
            if baseline.empty:
                continue
            T1 = float(baseline['time_sec'].iloc[0])
            # Экстраполируем для идеального случая (необязательно)
            print(f"Warning: No p=1 data for {a}, n={n}, using p={min_p} as baseline")
        else:
            T1 = float(baseline['time_sec'].iloc[0])
        
        speedup = T1 / s['time_sec']
        plt.plot(s['np'], speedup, marker='o', linewidth=2, 
                label=f"{a}, n={n}", markersize=6)

# Идеальное ускорение
x_ideal = np.unique(df['np'])
plt.plot(x_ideal, x_ideal, 'k--', label='ideal', linewidth=2)

plt.xscale('log', base=2)
plt.xlabel('Number of processes (p)')
plt.ylabel('Speedup')
plt.title('Speedup (all algorithms and sizes)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig("speedup_all_data.png", dpi=150, bbox_inches='tight')
plt.close()

# График эффективности (efficiency) для всех данных
plt.figure(figsize=(12, 8))
for a in algos:
    for n in sizes:
        s = df[(df['algo'] == a) & (df['n'] == n)].sort_values('np')
        if s.empty: 
            continue
        
        # Находим базовое время для p=1
        baseline = s[s['np'] == 1]
        if baseline.empty:
            # Если нет данных для p=1, используем минимальное доступное p
            min_p = s['np'].min()
            baseline = s[s['np'] == min_p]
            if baseline.empty:
                continue
            T1 = float(baseline['time_sec'].iloc[0])
        else:
            T1 = float(baseline['time_sec'].iloc[0])
        
        speedup = T1 / s['time_sec']
        efficiency = speedup / s['np']
        plt.plot(s['np'], efficiency, marker='o', linewidth=2, 
                label=f"{a}, n={n}", markersize=6)

# Идеальная эффективность
x_eff = np.unique(df['np'])
plt.plot(x_eff, np.ones_like(x_eff), 'k--', label='ideal', linewidth=2)

plt.xscale('log', base=2)
plt.xlabel('Number of processes (p)')
plt.ylabel('Efficiency')
plt.title('Efficiency (all algorithms and sizes)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig("efficiency_all_data.png", dpi=150, bbox_inches='tight')
plt.close()

# Обычные графики (по размерам матриц) - оригинальная функциональность
print("Creating individual size plots...")
for n in sizes:
    sub = df[df['n'] == n].copy()
    
    # График времени выполнения
    plt.figure()
    for a in algos:
        s = sub[sub['algo'] == a].sort_values('np')
        if s.empty: continue
        plt.plot(s['np'], s['time_sec'], marker='o', label=a, linewidth=2, markersize=6)
    plt.xscale('log', base=2)
    plt.xlabel('Number of processes (p)')
    plt.ylabel('Time (s)')
    plt.title(f'Execution time (n={n})')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.savefig(f"time_n{n}.png", dpi=150)
    plt.close()

    # График ускорения
    plt.figure()
    for a in algos:
        s = sub[sub['algo'] == a].sort_values('np')
        if s.empty: continue
        if 1 in s['np'].values:
            T1 = float(s[s['np'] == 1]['time_sec'].iloc[0])
        else:
            T1 = float(s['time_sec'].iloc[0])  # use smallest p as baseline
        speedup = T1 / s['time_sec']
        plt.plot(s['np'], speedup, marker='o', label=a, linewidth=2, markersize=6)
    x = np.unique(sub['np'])
    plt.plot(x, x, '--', label='ideal', linewidth=2)
    plt.xscale('log', base=2)
    plt.xlabel('Number of processes (p)')
    plt.ylabel('Speedup')
    plt.title(f'Speedup (n={n})')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.savefig(f"speedup_n{n}.png", dpi=150)
    plt.close()

    # График эффективности
    plt.figure()
    for a in algos:
        s = sub[sub['algo'] == a].sort_values('np')
        if s.empty: continue
        if 1 in s['np'].values:
            T1 = float(s[s['np'] == 1]['time_sec'].iloc[0])
        else:
            T1 = float(s['time_sec'].iloc[0])
        speedup = T1 / s['time_sec']
        efficiency = speedup / s['np']
        plt.plot(s['np'], efficiency, marker='o', label=a, linewidth=2, markersize=6)
    plt.xscale('log', base=2)
    plt.xlabel('Number of processes (p)')
    plt.ylabel('Efficiency')
    plt.title(f'Efficiency (n={n})')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.savefig(f"efficiency_n{n}.png", dpi=150)
    plt.close()

print("All plots created successfully!")
print("Combined plots: time_all_data.png, time_by_algorithm.png, speedup_all_data.png, efficiency_all_data.png")
print("Individual plots: time_nX.png, speedup_nX.png, efficiency_nX.png for each size X")
print("Sizes found:", sizes)