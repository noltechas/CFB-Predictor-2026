from cfbpred.conference_availability import team_name


def test_conference_context_distinguishes_miami_programs():
    teams={'Miami','Miami (OH)','UAlbany','San José State','Central Connecticut'}
    assert team_name('Miami',teams,'MAC')=='Miami (OH)'
    assert team_name('Miami',teams,'ACC')=='Miami'
    assert team_name('vs. Albany',teams)=='UAlbany'
    assert team_name('at San Jose State',teams)=='San José State'
    assert team_name('vs. Central Connecticut State',teams)=='Central Connecticut'
