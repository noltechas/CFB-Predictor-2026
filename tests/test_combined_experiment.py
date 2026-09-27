import pandas as pd

from cfbpred.combined_experiment import worth_confirming


def frame():
    return pd.DataFrame(dict(game_id=range(12),season=[2023]*4+[2024]*4+[2025]*4,
        week_c=[1,2,3,4]*3,margin=[7.,-7.,7.,-7.]*3,total=[50.]*12,
        em=[6.,-6.,6.,-6.]*3,et=[50.]*12,mm=[6.,-6.,6.,-6.]*3,
        wp=[.7,.3,.7,.3]*3,crps=[1.]*12,spread_close=[-7.,7.,-7.,7.]*3,fbs_fbs=[True]*12))


def test_tiny_but_repeated_improvement_qualifies_for_more_seeds():
    control=frame();candidate=control.copy()
    candidate.em += .001*candidate.margin.apply(lambda x:1 if x>0 else -1)
    assert worth_confirming(control,candidate)


def test_lower_margin_error_does_not_hide_worse_probability_predictions():
    control=frame();candidate=control.copy()
    candidate.em=candidate.margin
    candidate.wp=1-candidate.wp
    assert not worth_confirming(control,candidate)
