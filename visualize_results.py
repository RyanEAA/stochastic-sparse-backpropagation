from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def metric(df, mean, std, ylabel, title, path):
    fig,ax=plt.subplots(figsize=(9,6)); dense=df[df.model=='dense']; variants=df[df.model!='dense']
    for name,g in variants.groupby('model'):
        g=g.sort_values('keep_ratio'); ax.errorbar(g.keep_ratio,g[mean],yerr=g[std],marker='o',capsize=3,label=name)
    if not dense.empty: ax.axhline(dense.iloc[0][mean],linestyle='--',label='dense')
    ax.set(xlabel='Keep ratio',ylabel=ylabel,title=title); ax.grid(True,alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--summary',type=Path,required=True); p.add_argument('--output-dir',type=Path,default=None); a=p.parse_args()
    df=pd.read_csv(a.summary); out=a.output_dir or a.summary.parent/'plots'; out.mkdir(parents=True,exist_ok=True)
    for ds,g in df.groupby('dataset'):
        d=out/ds; d.mkdir(parents=True,exist_ok=True)
        metric(g,'val_accuracy_mean','val_accuracy_std','Validation accuracy',f'{ds}: accuracy vs keep ratio',d/'accuracy_vs_keep_ratio.png')
        metric(g,'backward_ms_mean','backward_ms_std','Backward time (ms)',f'{ds}: backward time vs keep ratio',d/'backward_time_vs_keep_ratio.png')
        metric(g,'memory_mb_mean','memory_mb_std','Memory (MB)',f'{ds}: memory vs keep ratio',d/'memory_vs_keep_ratio.png')
        metric(g,'epoch_time_mean','epoch_time_std','Epoch time (s)',f'{ds}: epoch time vs keep ratio',d/'epoch_time_vs_keep_ratio.png')
    print(f'Plots saved under {out}')
if __name__=='__main__': main()
