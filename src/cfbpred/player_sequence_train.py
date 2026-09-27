"""Chronological experiment trainer; auxiliary labels never enter a forward."""
from __future__ import annotations

import time
import numpy as np
import torch

from cfbpred.config import N_MARGIN_BINS,N_TOTAL_BINS
from cfbpred.model import margin_bin,total_bin,smoothed_targets,soft_ce,win_probability,expected_margin
from cfbpred.player_sequence_net import masked_stat_loss
from cfbpred.train import game_batch


def predict(model,d,indices,device,batch_size=128):
    model.eval();margins=[];totals=[]
    with torch.no_grad():
        for start in range(0,len(indices),batch_size):
            j=torch.tensor(indices[start:start+batch_size])
            _,pm,pt=model.pregame(game_batch(d,j,device))
            margins.append(pm.softmax(-1).cpu());totals.append(pt.softmax(-1).cpu())
    return torch.cat(margins),torch.cat(totals)


def fit(model,d,device,train_indices,val_indices,fixed_epochs=None,max_epochs=200,patience=20,
        micro_batch_size=128,aux_weight=1.,log_path=None):
    train_indices,val_indices=np.asarray(train_indices),np.asarray(val_indices)
    if (d["G_season"][train_indices]>=2026).any() or not d["G_labeled"][train_indices].all():
        raise ValueError("Invalid training labels")
    if fixed_epochs is None and (not len(val_indices) or d["G_season"][train_indices].max()>=d["G_season"][val_indices].min()):
        raise ValueError("Selection must follow training chronologically")
    if (d["G_season"][val_indices]>=2026).any():raise ValueError("2026 selection prohibited")
    fi=d["G_fbs_inv"]
    fbs,fcs=train_indices[fi[train_indices]],train_indices[~fi[train_indices]]
    tm=smoothed_targets(margin_bin(d["G_margin"]),N_MARGIN_BINS)
    tt=smoothed_targets(total_bin(d["G_total"]),N_TOTAL_BINS)
    truth=d["G_margin"].numpy()[val_indices];wins=(truth>0).astype(float)
    opt=torch.optim.AdamW(model.pre.parameters(),lr=getattr(model.pre,"learning_rate",.002),
                          weight_decay=getattr(model.pre,"weight_decay",.03))
    best,best_state,since,best_epoch=1e9,None,0,0
    history=[]
    for ep in range(fixed_epochs if fixed_epochs is not None else max_epochs):
        start_time=time.monotonic();model.train()
        sub=np.random.choice(fcs,int(len(fcs)*.25),replace=False) if len(fcs) else np.empty(0,np.int64)
        perm=np.random.permutation(np.r_[fbs,sub]);aux_sum=0.;game_sum=0.
        for start in range(0,len(perm),2048):
            full=perm[start:start+2048];opt.zero_grad()
            for offset in range(0,len(full),micro_batch_size):
                ids=full[offset:offset+micro_batch_size];j=torch.tensor(ids)
                _,pm,pt=model.pregame(game_batch(d,j,device))
                game_loss=soft_ce(pm,tm[j].to(device))+.5*soft_ce(pt,tt[j].to(device))
                loss=game_loss
                head_weight=getattr(model.pre,"roster_head_weight",0.0)
                if head_weight>0:
                    roster_loss=soft_ce(model.pre.roster_margin_logits,tm[j].to(device))
                    loss=loss+head_weight*roster_loss
                    aux_sum+=float(roster_loss.detach().cpu())*len(ids)
                if model.pre.auxiliary:
                    target,mask=d["player_store"].labels(ids,device)
                    auxiliary=masked_stat_loss(model.pre.aux_predictions,target,mask)
                    loss=loss+aux_weight*auxiliary
                    aux_sum+=float(auxiliary.detach().cpu())*len(ids)
                game_sum+=float(game_loss.detach().cpu())*len(ids)
                (loss*(len(ids)/len(full))).backward()
            opt.step()
        record={"epoch":ep+1,"game_loss":game_sum/len(perm),"aux_loss":aux_sum/len(perm)}
        if fixed_epochs is None:
            pm,_=predict(model,d,val_indices,device,min(128,micro_batch_size))
            wp=win_probability(pm.clamp(min=1e-30).log()).numpy().clip(1e-6,1-1e-6)
            em=expected_margin(pm.clamp(min=1e-30).log()).numpy()
            score=float(-np.mean(wins*np.log(wp)+(1-wins)*np.log(1-wp)))
            record.update(validation_wp_logloss=score,validation_margin_mae=float(np.mean(abs(em-truth))))
            if score<best-1e-4:
                best,since,best_epoch=score,0,ep+1
                best_state={k:v.detach().cpu().clone() for k,v in model.pre.state_dict().items()}
            else:since+=1
        record["seconds"]=time.monotonic()-start_time;history.append(record)
        if log_path is not None:
            from cfbpred.pregame_experiments import save_json
            save_json(log_path,history)
        if ep%5==0:print(f" epoch={ep+1} seconds={record['seconds']:.1f} game_loss={record['game_loss']:.4f} aux={record['aux_loss']:.4f} best={best:.5f}",flush=True)
        if fixed_epochs is None and since>=patience:break
    if fixed_epochs is None:model.pre.load_state_dict(best_state)
    model.pre.selected_epochs=fixed_epochs if fixed_epochs is not None else best_epoch
    model.pre.selection_score=best
    return history
