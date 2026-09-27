import math

import torch

from cfbpred.architecture_research_net import ABSENCE_NAMES, absence_features
from cfbpred.player_sequence_inputs import ROLE_COLS
from cfbpred.player_timeline import CONTEXT_COLS


def _batch(home_status, away_known=False):
    """One game, three home players with recent passing 30/10/0 attempts."""
    mask = torch.tensor([[[True, True, True], [True, True, False]]])
    role = torch.zeros(1, 2, 3, len(ROLE_COLS))
    att = ROLE_COLS.index("recent3_att_value")
    for p, attempts in enumerate((30, 10, 0)):
        role[0, 0, p, att] = math.log1p(attempts) / 5          # stored as log counts
    slots = mask.reshape(-1).nonzero().squeeze(-1)
    context = torch.zeros(len(slots), len(CONTEXT_COLS))
    for p, status in enumerate(home_status):
        if status:
            context[p, CONTEXT_COLS.index("report_known")] = 1
            context[p, CONTEXT_COLS.index(status)] = 1
    if away_known:
        context[3, CONTEXT_COLS.index("report_known")] = 1
    return dict(player_mask=mask, player_role=role, player_slots=slots, player_context=context)


def test_starter_out_takes_his_share_of_recent_passing():
    f = absence_features(_batch(("out", "available", None)), CONTEXT_COLS)[0]
    i = {n: k for k, n in enumerate(ABSENCE_NAMES)}
    assert f[0, i["report_known"]] == 1
    assert torch.isclose(f[0, i["out_share_att"]], torch.tensor(30 / 40), atol=1e-4)
    assert torch.isclose(f[0, i["players_out_log"]], torch.tensor(math.log1p(1) / 3), atol=1e-6)


def test_backup_questionable_is_unsure_not_out():
    f = absence_features(_batch(("available", "questionable", None)), CONTEXT_COLS)[0]
    i = {n: k for k, n in enumerate(ABSENCE_NAMES)}
    assert f[0, i["out_share_att"]] == 0
    assert torch.isclose(f[0, i["unsure_share_att"]], torch.tensor(10 / 40), atol=1e-4)


def test_no_report_is_distinguishable_from_reported_healthy():
    unreported = absence_features(_batch((None, None, None)), CONTEXT_COLS)[0]
    healthy = absence_features(_batch(("available", "available", "available")), CONTEXT_COLS)[0]
    i = {n: k for k, n in enumerate(ABSENCE_NAMES)}
    assert unreported[0, i["report_known"]] == 0 and healthy[0, i["report_known"]] == 1
    # every share is zero in both; only the report flag tells them apart
    assert torch.equal(unreported[0, 1:], healthy[0, 1:])


def test_sides_are_independent():
    f = absence_features(_batch(("out", None, None), away_known=True), CONTEXT_COLS)[0]
    i = {n: k for k, n in enumerate(ABSENCE_NAMES)}
    assert f[1, i["report_known"]] == 1 and f[1, i["out_share_att"]] == 0
