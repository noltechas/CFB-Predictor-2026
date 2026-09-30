"""Full-career attention and explicit game-specific player context on TabM."""
from __future__ import annotations

import torch
from torch import nn

from cfbpred.architecture_research_net import make as make_tabm
from cfbpred.player_sequence_net import RichPlayerEncoder,safe_lookup
from cfbpred.player_sequence_inputs import ROLE_COLS


class CareerEncoder(nn.Module):
    """Linear-cost attention over ALL prior events, with no padding or cap."""
    def __init__(self,n_teams,event_dim):
        super().__init__()
        self.program=nn.Embedding(n_teams,4,padding_idx=0)
        nn.init.normal_(self.program.weight,std=.05)
        with torch.no_grad():self.program.weight[0].zero_()
        self.event=nn.Sequential(nn.Linear(event_dim+8,16),nn.GELU())
        self.temporal=nn.Sequential(nn.Linear(20,12),nn.GELU())
        self.attention=nn.Linear(12,1)
        self.output=nn.Linear(24,12)

    def forward(self,batch):
        n=len(batch["career_counts"])
        values=batch["career_values"]
        if not batch["career_refs"].numel():return values.new_zeros((n,12))
        encoded=self.event(torch.cat([values,safe_lookup(self.program,batch["career_team"]),
                                     safe_lookup(self.program,batch["career_opponent"])],-1))
        refs=batch["career_refs"];owner=batch["career_owner"]
        h=self.temporal(torch.cat([encoded[refs],batch["career_temporal"]],-1))
        logits=self.attention(h).squeeze(-1)
        maximum=logits.new_full((n,),-torch.inf)
        maximum.scatter_reduce_(0,owner,logits.detach(),reduce="amax",include_self=True)
        weights=(logits-maximum[owner]).exp()
        denominator=weights.new_zeros(n).index_add(0,owner,weights)
        summed=h.new_zeros((n,12)).index_add(0,owner,h*(weights/denominator[owner].clamp(min=1e-12))[:,None])
        latest=h.new_zeros((n,12)).index_add(0,owner,h*batch["career_is_latest"][:,None])
        return self.output(torch.cat([summed,latest],-1))*(batch["career_counts"]>0)[:,None]


class FullRosterEncoder(RichPlayerEncoder):
    # When True, a player reported out leaves the unit's mean, maximum and count as
    # well as its role-weighted term. The deployed full_context variant pooled mean
    # and maximum over every listed player, so an "out" flag reached only one of the
    # three unit statistics.
    pool_available = False

    def __init__(self,schema,n_teams,event_dim,context_dim=0,roles=False):
        super().__init__(schema,n_teams,context=False,contribution=False)
        self.career=CareerEncoder(n_teams,event_dim)
        self.context_dim=context_dim;self.roles=roles
        if context_dim:self.context=nn.Sequential(nn.Linear(context_dim,12),nn.GELU(),nn.Linear(12,12))
        if roles:
            self.role=nn.Sequential(nn.Linear(12+len(ROLE_COLS)+context_dim,24),nn.GELU(),nn.Linear(24,1))
            self.weighted_group=nn.Linear(12,8,bias=False)

    def forward_batch(self,batch):
        scout,history,position,mask=(batch[k] for k in ["player_scout","player_history","player_position","player_mask"])
        slots=batch["player_slots"];pos=position.reshape(-1)[slots]
        h=self.history(history).flatten(-2)
        packed=self.player(torch.cat([scout,h,safe_lookup(self.position,pos)],-1))+self.career(batch)
        context=batch.get("player_context")
        if self.context_dim:
            if context is None or context.shape!=(len(slots),self.context_dim):raise ValueError("Missing player timeline features")
            packed=packed+self.context(context)
        x=packed.new_zeros((mask.numel(),12)).index_copy(0,slots,packed).reshape(*mask.shape,12)
        logits=None
        if self.roles:
            role=batch["player_role"].reshape(-1,len(ROLE_COLS))[slots]
            fields=[packed,role]+([context] if self.context_dim else [])
            scores=self.role(torch.cat(fields,-1)).squeeze(-1)
            logits=scores.new_full((mask.numel(),),-1e4).index_copy(0,slots,scores).reshape(mask.shape)
        available=mask & ~batch.get("player_confirmed_out",torch.zeros_like(mask))
        groups=[]
        for group in range(self.n_positions):
            valid=mask & position.eq(group)
            pooled=(valid & available) if self.pool_available else valid
            count=pooled.sum(-1,keepdim=True)
            mean=(x*pooled[...,None]).sum(-2)/count.clamp(min=1)
            maximum=x.masked_fill(~pooled[...,None],-1e4).amax(-2)
            maximum=torch.where(count.gt(0),maximum,0.)
            summary=self.group(torch.cat([mean,maximum,torch.log1p(count.float())/4],-1))
            if logits is not None:
                active=valid & available
                weights=logits.masked_fill(~active,-1e4).softmax(-1)*active
                weights=weights/weights.sum(-1,keepdim=True).clamp(min=1e-8)
                summary=summary+self.weighted_group((x*weights[...,None]).sum(-2))
            groups.append(summary*count.gt(0))
        return self.team(torch.cat(groups,-1))*mask.any(-1,keepdim=True),x


ROSTER_HEAD_WEIGHT = 0.5   # roster-only margin loss, relative to the game loss
# full_context            : deployed through v5 (no head; out-flag reaches the role term only)
# full_context_rosterhead : + roster-only margin head
# full_context_v2         : + roster-only margin head, and mean/max/count pooled over available players
# full_context_v3         : v2 + per-side availability inputs straight into the core
# full_context_gates      : v2 + one season-to-date gate per stat family
# full_context_extras     : v2 network on the re-admitted venue/schedule/head-to-head scalars
# full_context_v4         : v2 network on cross-season talent percentiles (features_raw.TalentPercentiles)
#                           in place of the raw 247 composite, its missing flag and sos_talent
# full_context_v4_r4/_r8  : v4, loss weighted toward recent seasons (half-life 4 / 8 seasons)
TALENT_FIX_VARIANTS = {"full_context_v4", "full_context_v4_r4", "full_context_v4_r8"}
# full_context_v5a        : v2 + offseason roster movement (portal in/out, draft departures; mvmt_)
# full_context_v5b        : v2 + in-season head-coach change flags (cchg_)
# full_context_v5         : v2 + both
# full_context_fcs100     : v2 trained on every FCS-vs-FCS game each epoch (v2 samples 25%)
MOVEMENT_VARIANTS = {"full_context_v5a", "full_context_v5b", "full_context_v5", "full_context_fcs100"}
HEAD_VARIANTS = {"full_context_rosterhead", "full_context_v2", "full_context_v3",
                 "full_context_gates", "full_context_extras", *TALENT_FIX_VARIANTS, *MOVEMENT_VARIANTS}
POOL_AVAILABLE_VARIANTS = {"full_context_v2", "full_context_v3", "full_context_gates", "full_context_extras",
                           *TALENT_FIX_VARIANTS, *MOVEMENT_VARIANTS}
FAMILY_GATE_VARIANTS = {"full_context_gates"}
# EXCLUDED_SCALARS entries a variant re-admits (train.SCALAR_EXTRAS while loading its data);
# a "-<column>" entry drops that exact column instead
TALENT_FIX_SCALARS = ("tpct", "-h_talent_composite", "-a_talent_composite", "-h_has_talent",
                      "-a_has_talent", "-h_sos_talent", "-a_sos_talent")
SCALAR_EXTRAS_BY_VARIANT = {"full_context_extras": ("venue_grass", "venue_capacity", "kick_local_hour",
                                                    "body_clock_shift", "h2h_"),
                            **{v: TALENT_FIX_SCALARS for v in TALENT_FIX_VARIANTS},
                            "full_context_v5a": ("mvmt_",), "full_context_v5b": ("cchg_",),
                            "full_context_v5": ("mvmt_", "cchg_")}
# share of non-FBS-involved (FCS-vs-FCS) training games sampled into each epoch (default 0.25)
FCS_SHARE_BY_VARIANT = {"full_context_fcs100": 1.0}
# <variant>_nu: the same variant without h/a_upper_share. CFBD historical rosters carry calendar
# years or final-career classes in `year`, so the share reads 1.0 for every team through 2015 and
# falls to 0.47 by 2026 - an era marker, not experience (the player store already drops class).
NO_UPPER = ("-h_upper_share", "-a_upper_share")
for _v in ("full_context_v2", "full_context_v5a", "full_context_v5b", "full_context_v5", "full_context_fcs100"):
    SCALAR_EXTRAS_BY_VARIANT[_v + "_nu"] = SCALAR_EXTRAS_BY_VARIANT.get(_v, ()) + NO_UPPER
    if _v in FCS_SHARE_BY_VARIANT:
        FCS_SHARE_BY_VARIANT[_v + "_nu"] = FCS_SHARE_BY_VARIANT[_v]
NU_VARIANTS = {_v + "_nu" for _v in ("full_context_v2", "full_context_v5a", "full_context_v5b",
                                      "full_context_v5", "full_context_fcs100")}
# data arm + every FCS-vs-FCS game (the rule's combination arm), with and without upper_share
for _v in ("full_context_v5a", "full_context_v5b", "full_context_v5"):
    for _nu in ("", "_nu"):
        _name = _v + "_fcs" + _nu
        SCALAR_EXTRAS_BY_VARIANT[_name] = SCALAR_EXTRAS_BY_VARIANT[_v] + (NO_UPPER if _nu else ())
        FCS_SHARE_BY_VARIANT[_name] = 1.0
        NU_VARIANTS.add(_name)
HEAD_VARIANTS |= NU_VARIANTS
POOL_AVAILABLE_VARIANTS |= NU_VARIANTS
# Training games are weighted 0.5 ** ((latest training season - season) / half_life), rescaled to
# mean 1. Seasons are not exchangeable: FBS scoring fell from ~57-58 points (2014-20) to ~52 (2025)
# and the gap between conference tiers keeps widening, while embeddings are static over 2001-2025.
RECENCY_HALF_LIFE_BY_VARIANT = {"full_context_v4_r4": 4.0, "full_context_v4_r8": 8.0}
CONTEXT_VARIANTS = {"full_context", *HEAD_VARIANTS}


def make(data,variant="full_history"):
    if variant not in {"full_history","full_history_roles",*CONTEXT_VARIANTS}:raise ValueError(variant)
    model=make_tabm(data,"tabm16",absence=variant=="full_context_v3")
    store=data["player_store"]
    context_dim=len(store.context_cols) if variant in CONTEXT_VARIANTS else 0
    model.pre.encoder=FullRosterEncoder(store.manifest,len(data["vocab"]["team"])+1,
                                       len(store.full_manifest["event_cols"]),context_dim,variant!="full_history")
    if variant in HEAD_VARIANTS:model.pre.attach_roster_head(ROSTER_HEAD_WEIGHT)
    if variant in POOL_AVAILABLE_VARIANTS:model.pre.encoder.pool_available=True
    if variant in FAMILY_GATE_VARIANTS:model.pre.core.enable_family_gates(data["scalar_cols"])
    return model
