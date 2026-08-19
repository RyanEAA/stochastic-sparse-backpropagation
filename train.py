"""Run one dataset/model experiment and save raw measurements only."""
import argparse, csv, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from data.loaders import build_loaders, normalize_dataset_name, SUPPORTED_DATASETS
from models import build_model, AVAILABLE_MODELS
from training.runtime import set_seed, get_device, synchronize, memory_bytes

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train(); loss_sum=correct=total=0; batches=[]; start=time.perf_counter()
    for bi,(x,y) in enumerate(loader):
        x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True)
        out=model(x); loss=criterion(out,y)
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        synchronize(device); t0=time.perf_counter(); loss.backward(); synchronize(device)
        bw=time.perf_counter()-t0; mem=memory_bytes(device); optimizer.step()
        n=y.size(0); loss_sum += loss.item()*n; c=(out.argmax(1)==y).sum().item(); correct += c; total += n
        batches.append(dict(epoch=epoch,batch=bi,backward_time_s=bw,memory_bytes=mem,batch_accuracy=c/n))
    synchronize(device)
    return loss_sum/total, correct/total, time.perf_counter()-start, batches

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval(); loss_sum=correct=total=0
    for x,y in loader:
        x,y=x.to(device),y.to(device); out=model(x); loss=criterion(out,y); n=y.size(0)
        loss_sum += loss.item()*n; correct += (out.argmax(1)==y).sum().item(); total += n
    return loss_sum/total, correct/total

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, help=f"One of: {', '.join(SUPPORTED_DATASETS)}")
    p.add_argument('--model', required=True, choices=AVAILABLE_MODELS)
    p.add_argument('--keep-ratio', type=float, default=1.0)
    p.add_argument('--epochs', type=int, default=3); p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--subset', type=int, default=0); p.add_argument('--data-dir', default='data')
    p.add_argument('--num-workers', type=int, default=0); p.add_argument('--device', default='auto', choices=['auto','cpu','cuda','mps'])
    p.add_argument('--output-dir', type=Path, required=True)
    a=p.parse_args(); dataset=normalize_dataset_name(a.dataset)
    if a.model != 'dense' and not 0 < a.keep_ratio <= 1: raise ValueError('keep_ratio must be in (0, 1].')
    set_seed(a.seed); device=get_device(a.device)
    train_loader,val_loader=build_loaders(dataset,a.batch_size,a.seed,a.subset,a.data_dir,a.num_workers)
    model=build_model(dataset,a.model,a.keep_ratio).to(device); opt=optim.Adam(model.parameters(),lr=a.lr); criterion=nn.CrossEntropyLoss()
    print(f"dataset={dataset} model={a.model} keep_ratio={a.keep_ratio} seed={a.seed} device={device}")
    erows=[]; brows=[]
    for epoch in range(1,a.epochs+1):
        tl,ta,et,b=train_epoch(model,train_loader,opt,criterion,device,epoch); vl,va=evaluate(model,val_loader,criterion,device)
        print(f"Epoch {epoch}: train_loss={tl:.4f} train_acc={ta:.4f} val_loss={vl:.4f} val_acc={va:.4f} time={et:.2f}s")
        base=dict(dataset=dataset,model=a.model,keep_ratio=(1.0 if a.model=='dense' else a.keep_ratio),seed=a.seed)
        erows.append({**base,'epoch':epoch,'train_loss':tl,'train_accuracy':ta,'val_loss':vl,'val_accuracy':va,'epoch_time_s':et})
        for row in b: brows.append({**base,**row})
    write_csv(a.output_dir/'epochs.csv',erows,['dataset','model','keep_ratio','seed','epoch','train_loss','train_accuracy','val_loss','val_accuracy','epoch_time_s'])
    write_csv(a.output_dir/'batches.csv',brows,['dataset','model','keep_ratio','seed','epoch','batch','backward_time_s','memory_bytes','batch_accuracy'])
if __name__=='__main__': main()
