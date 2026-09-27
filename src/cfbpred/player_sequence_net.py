"""Opt-in factorial player experiments inside the same pregame DNN."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from cfbpred.player_inputs import STAT_COLS
from cfbpred.player_net import PlayerPregameTower, PlayerSetEncoder
from cfbpred.player_sequence_inputs import EVENT_COLS, ROLE_COLS

VARIANTS = {"control":(False,False,False), "context":(True,False,False),
            "contribution":(False,True,False), "auxiliary":(False,False,True),
            "combined":(True,True,True)}


def safe_lookup(embedding, indices):
    """Bound embedding/gather rows for the known Apple GPU gather issue."""
    flat=indices.reshape(-1)
    if not len(flat):
        return embedding.weight.new_zeros((*indices.shape,embedding.embedding_dim))
    return torch.cat([embedding(flat[i:i+8192]) for i in range(0,len(flat),8192)]).reshape(*indices.shape,-1)


class SequenceEncoder(nn.Module):
    def __init__(self,n_teams):
        super().__init__()
        self.program=nn.Embedding(n_teams,4,padding_idx=0)
        nn.init.normal_(self.program.weight,std=.05)
        with torch.no_grad():self.program.weight[0].zero_()
        self.event=nn.Sequential(nn.Linear(len(EVENT_COLS)+8,16),nn.GELU())
        self.temporal=nn.Sequential(nn.Linear(20,12),nn.GELU())
        self.attention=nn.Linear(12,1)
        self.output=nn.Linear(24,12)

    def forward(self,batch,shape):
        values=batch["sequence_values"]
        out=values.new_zeros((*shape,12))
        if not batch["sequence_slots"].numel():return out
        encoded=self.event(torch.cat([values,safe_lookup(self.program,batch["sequence_team"]),
                                     safe_lookup(self.program,batch["sequence_opponent"])],-1))
        refs=batch["sequence_refs"]
        # Bound advanced-index row count as well as embedding row count.
        flat=refs.reshape(-1)
        h=torch.cat([encoded[flat[i:i+8192]] for i in range(0,len(flat),8192)]).reshape(*refs.shape,-1)
        h=self.temporal(torch.cat([h,batch["sequence_temporal"]],-1))
        mask=batch["sequence_mask"]
        weights=self.attention(h).squeeze(-1).masked_fill(~mask,-1e4).softmax(-1)*mask
        weights=weights/weights.sum(-1,keepdim=True).clamp(min=1e-8)
        summary=self.output(torch.cat([(h*weights[...,None]).sum(-2),h[:,0]],-1))
        return out.reshape(-1,12).index_copy(0,batch["sequence_slots"],summary).reshape(*shape,12)


class RichPlayerEncoder(PlayerSetEncoder):
    def __init__(self,schema,n_teams,context,contribution):
        super().__init__(len(schema["static_cols"]),len(schema["history_cols"]),len(schema["positions"]))
        self.context,self.contribution=context,contribution
        if context:self.sequence=SequenceEncoder(n_teams)
        if contribution:
            self.role=nn.Sequential(nn.Linear(12+len(ROLE_COLS),16),nn.GELU(),nn.Linear(16,1))
            self.contribution_group=nn.Linear(12,8,bias=False)

    def forward_batch(self,batch):
        scout,history,position,mask=(batch[k] for k in ["player_scout","player_history","player_position","player_mask"])
        h=self.history(history).flatten(-2)
        if "player_slots" in batch:
            slots=batch["player_slots"]
            packed_position=torch.cat([position.reshape(-1)[slots[i:i+8192]] for i in range(0,len(slots),8192)]) if len(slots) else position.new_empty(0)
            packed=self.player(torch.cat([scout,h,safe_lookup(self.position,packed_position)],-1))
            x=packed.new_zeros((mask.numel(),12)).index_copy(0,slots,packed).reshape(*mask.shape,12)
        else:
            x=self.player(torch.cat([scout,h,safe_lookup(self.position,position)],-1))
        if self.context:x=x+self.sequence(batch,mask.shape)
        logits=None
        if self.contribution:logits=self.role(torch.cat([x,batch["player_role"]],-1)).squeeze(-1)
        groups=[]
        for group in range(self.n_positions):
            valid=mask & position.eq(group)
            count=valid.sum(-1,keepdim=True)
            mean=(x*valid[...,None]).sum(-2)/count.clamp(min=1)
            maximum=x.masked_fill(~valid[...,None],-1e4).amax(-2)
            maximum=torch.where(count.gt(0),maximum,0.)
            summary=self.group(torch.cat([mean,maximum,torch.log1p(count.float())/4],-1))
            if logits is not None:
                weights=logits.masked_fill(~valid,-1e4).softmax(-1)*valid
                weights=weights/weights.sum(-1,keepdim=True).clamp(min=1e-8)
                summary=summary+self.contribution_group((x*weights[...,None]).sum(-2))
            groups.append(summary*count.gt(0))
        return self.team(torch.cat(groups,-1))*mask.any(-1,keepdim=True),x


class RichPlayerTower(PlayerPregameTower):
    def __init__(self,n_teams,n_confs,scalar_cols,schema,context,contribution,auxiliary,**kwargs):
        super().__init__(n_teams,n_confs,scalar_cols,schema,**kwargs)
        self.encoder=RichPlayerEncoder(schema,n_teams,context,contribution)
        self.auxiliary=auxiliary
        if auxiliary:self.stat_head=nn.Sequential(nn.Linear(12+48+1,32),nn.GELU(),nn.Linear(32,len(STAT_COLS)))
        self.aux_predictions=None

    def forward_batch(self,batch):
        for key in ["player_scout","player_history","player_position","player_mask"]:
            if key not in batch:raise ValueError(f"Player model requires {key}")
        roster,x=self.encoder.forward_batch(batch)
        self.aux_predictions=None
        if self.auxiliary and self.training:
            own=torch.stack([self.core.team_emb(batch["h_idx"]),self.core.team_emb(batch["a_idx"])],1)
            gp=torch.stack([batch["h_gp01"],batch["a_gp01"]],1)[...,None]
            context=torch.cat([own,own.flip(1),gp],-1)[...,None,:].expand(*x.shape[:-1],49)
            self.aux_predictions=self.stat_head(torch.cat([x,context],-1))
        scalars=torch.cat([batch["scalars"],roster[:,0],roster[:,1]],-1)
        return self.core(scalars,batch["h_idx"],batch["a_idx"],batch["h_conf"],batch["a_conf"],
                         batch["h_hist"],batch["h_opp"],batch["a_hist"],batch["a_opp"],
                         batch.get("h_gp01"),batch.get("a_gp01"))


def masked_stat_loss(predictions,targets,mask):
    """Equal weight per observed stat per game; unknowns have zero gradient."""
    errors=F.smooth_l1_loss(predictions,targets,reduction="none",beta=.1)
    count=mask.sum(dim=(1,2))
    mean=(errors*mask).sum(dim=(1,2))/count.clamp(min=1)
    known=count>0
    per_game=(mean*known).sum(-1)/known.sum(-1).clamp(min=1)
    return per_game.mean()


def make(d,variant):
    from cfbpred.train import make_model,s2d_indices
    context,contribution,auxiliary=VARIANTS[variant]
    model=make_model(d,s2d_gate=True,delta=True)
    model.pre=RichPlayerTower(len(d["vocab"]["team"])+1,len(d["vocab"]["conf"])+1,d["scalar_cols"],
                              d["player_store"].manifest,context,contribution,auxiliary,
                              s2d_idx_h=s2d_indices(d["scalar_cols"],"h"),s2d_idx_a=s2d_indices(d["scalar_cols"],"a"))
    return model
