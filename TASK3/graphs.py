import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('results.csv', header=None, names=['method', 'processes', 'matrix_size', 'execution_time'])

speedup_data = []

for matrix_size in df['matrix_size'].unique():
    time_1proc = df[(df['processes'] == 1) & (df['matrix_size'] == matrix_size)]['execution_time'].values[0]
    
    for processes in df['processes'].unique():
        current_data = df[(df['processes'] == processes) & (df['matrix_size'] == matrix_size)]
        if len(current_data) > 0:
            current_time = current_data['execution_time'].values[0]
            
            if processes == 1:
                speedup = 1.0
            else:
                speedup = time_1proc / current_time
            
            speedup_data.append({
                'matrix_size': matrix_size,
                'processes': processes,
                'speedup': speedup,
                'execution_time': current_time
            })

speedup_df = pd.DataFrame(speedup_data)

plt.figure(figsize=(12, 8))

colors = ['blue', 'red', 'green', 'orange', 'purple']
matrix_sizes = sorted(speedup_df['matrix_size'].unique())

for i, matrix_size in enumerate(matrix_sizes):
    subset = speedup_df[speedup_df['matrix_size'] == matrix_size]
    plt.plot(subset['processes'], subset['speedup'], 
             marker='o', linewidth=2, markersize=8, 
             color=colors[i % len(colors)],
             label=f'N={matrix_size}')

ideal_procs = np.array([1, 4, 9, 16, 25])
plt.plot(ideal_procs, ideal_procs, 'k--', linewidth=1, label='Идеальное ускорение')

plt.xlabel('Количество процессов', fontsize=12)
plt.ylabel('Ускорение', fontsize=12)
plt.title('Ускорение алгоритма Кэннона в зависимости от числа процессов', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks([1, 4, 9, 16, 25])
plt.tight_layout()
plt.savefig('speedup_vs_processes.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 8))

processes_list = sorted(speedup_df['processes'].unique())
for processes in processes_list:
    subset = speedup_df[speedup_df['processes'] == processes]
    plt.plot(subset['matrix_size'], subset['execution_time'], 
             marker='s', linewidth=2, markersize=6,
             label=f'p={processes}')

plt.xlabel('Размер матрицы (N)', fontsize=12)
plt.ylabel('Время выполнения (сек)', fontsize=12)
plt.title('Время выполнения алгоритма Кэннона', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('time_vs_matrix_size.png', dpi=300, bbox_inches='tight')
plt.show()

speedup_df['efficiency'] = speedup_df['speedup'] / speedup_df['processes']

plt.figure(figsize=(12, 8))

for i, matrix_size in enumerate(matrix_sizes):
    subset = speedup_df[speedup_df['matrix_size'] == matrix_size]
    plt.plot(subset['processes'], subset['efficiency'], 
             marker='^', linewidth=2, markersize=8,
             color=colors[i % len(colors)],
             label=f'N={matrix_size}')

plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Идеальная эффективность')
plt.xlabel('Количество процессов', fontsize=12)
plt.ylabel('Эффективность', fontsize=12)
plt.title('Эффективность алгоритма Кэннона', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(0, 1.1)
plt.xticks([1, 4, 9, 16, 25])
plt.tight_layout()
plt.savefig('efficiency.png', dpi=300, bbox_inches='tight')
plt.show()

pivot_speedup = speedup_df.pivot(index='processes', columns='matrix_size', values='speedup')

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_speedup, annot=True, fmt='.2f', cmap='RdYlGn', 
            cbar_kws={'label': 'Ускорение'})
plt.title('Тепловая карта ускорения алгоритма Кэннона', fontsize=14)
plt.xlabel('Размер матрицы')
plt.ylabel('Количество процессов')
plt.tight_layout()
plt.savefig('speedup_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

for matrix_size in matrix_sizes:
    subset = speedup_df[speedup_df['matrix_size'] == matrix_size]
    best_config = subset.loc[subset['speedup'].idxmax()]
    
    print(f"\nРазмер матрицы {matrix_size}:")
    print(f"  Лучшее ускорение: {best_config['speedup']:.2f}x (при {best_config['processes']} процессах)")
    print(f"  Эффективность: {best_config['efficiency']:.3f}")
    print(f"  Время выполнения: {best_config['execution_time']:.3f} сек")

speedup_df.to_csv('results_with_analysis.csv', index=False)