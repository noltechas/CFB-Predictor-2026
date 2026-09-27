"""Versioned, conservative person resolution. Provider IDs are not person IDs.

Corrections are season AND program scoped. Unverified gaps remain separate
quarantined episodes, never asserted merges. Names are never merge keys.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import pandas as pd

OVERRIDES = Path(__file__).resolve().parents[2] / "data/player_identity_overrides.json"


def norm_id(values):
    return values.astype("string").str.replace(r"\.0$", "", regex=True)


def norm_name(value):
    return re.sub(r"[^a-z]", "", str(value).lower())


def last_name_core(value):
    return norm_name(re.sub(r"\b(?:jr|sr|ii|iii|iv|v)\b\.?", "", str(value), flags=re.I))


def resolve_rosters(roster, rules=None, recruits=None):
    """Return every source roster row with its auditable season-scoped identity."""
    rules = json.loads(OVERRIDES.read_text())["rules"] if rules is None else rules
    r = roster.copy().reset_index(drop=True)
    r["athlete_id"] = norm_id(r.athlete_id)
    if r.athlete_id.isna().any():
        raise ValueError("Roster contains missing provider IDs")
    y = r[["athlete_id", "season"]].drop_duplicates().sort_values(["athlete_id", "season"])
    y["episode"] = y.groupby("athlete_id").season.diff().ge(4).groupby(y.athlete_id).cumsum()
    y["episode_start"] = y.groupby(["athlete_id", "episode"]).season.transform("min")
    y["identity_uncertain"] = y.episode.gt(0) | (y.season-y.episode_start).ge(9)
    r = r.merge(y, on=["athlete_id", "season"], validate="many_to_one")
    r["person_id"] = "cfbd:" + r.athlete_id
    m = r.identity_uncertain
    r.loc[m, "person_id"] += ":unverified:" + r.loc[m, "episode_start"].astype(str)
    r["identity_method"] = "provider_continuity"
    r.loc[m, "identity_method"] = "quarantined_episode"
    if recruits is not None:
        q = recruits[recruits.recruitType.eq("HighSchool")].copy()
        q["recruit_id"] = norm_id(q.id)
        q["athlete_id"] = norm_id(q.athleteId)
        # A consistent, explicit high-school record can bridge missing roster
        # seasons. Requiring its class to precede the ORIGINAL college episode
        # prevents a newly recycled ID from inheriting the older person's past.
        single = q.dropna(subset=["athlete_id"])
        single = single[single.groupby("athlete_id").athlete_id.transform("size").eq(1)].set_index("athlete_id")
        first = r.groupby("athlete_id").season.transform("min")
        recruit_year = r.athlete_id.map(single.year)
        recruit_id = r.athlete_id.map(single.recruit_id)
        explicit = pd.Series([str(qid) in set(map(str,ids)) if pd.api.types.is_list_like(ids) else False
                              for qid,ids in zip(recruit_id,r.recruitIds)],index=r.index)
        name_ok = pd.Series([last_name_core(last) in norm_name(name) and norm_name(first)[:1]==norm_name(name)[:1]
                             for first,last,name in zip(r.firstName,r.lastName,r.athlete_id.map(single.name))],index=r.index)
        bridge = (r.identity_uncertain & explicit & name_ok & recruit_year.between(first-4,first+1)
                  & (r.season-recruit_year).between(0,8))
        r.loc[bridge,"identity_uncertain"] = False
        r.loc[bridge,"person_id"] = "cfbd:" + r.loc[bridge,"athlete_id"]
        r.loc[bridge,"identity_method"] = "explicit_recruit_continuity"
        # Incompatible high-school class dates flag silent collisions even
        # when the reused ID has no four-year gap (e.g. Tyler Brown).
        restart = recruit_year.ge(first+3) & r.season.ge(recruit_year) & explicit & name_ok
        r.loc[restart,"identity_uncertain"] = True
        r.loc[restart,"person_id"] = ("cfbd:"+r.loc[restart,"athlete_id"]+":class_conflict:"
                                        +recruit_year[restart].astype(int).astype(str))
        r.loc[restart,"identity_method"] = "incompatible_recruit_timeline"
    r["recruit_override"] = None
    r["college_start"] = pd.NA
    r["identity_evidence"] = None
    for rule in rules:
        m = (r.athlete_id.eq(rule["athlete_id"]) & r.season.between(rule["from_season"],rule.get("to_season",9999))
             & r.team.isin(rule["teams"]))
        r.loc[m, "person_id"] = rule["person_id"]
        r.loc[m, "identity_uncertain"] = False
        r.loc[m, "identity_method"] = "verified_override"
        r.loc[m, "college_start"] = rule["college_start"]
        r.loc[m, "position"] = rule["position"]
        r.loc[m, "identity_evidence"] = rule["evidence"]
        for ix in r.index[m]:
            r.at[ix, "recruit_override"] = rule["recruit_ids"]
    # Two different programs in the same season do not prove simultaneous
    # membership; annual feeds cannot determine the transfer date.
    multi = r.groupby(["season", "person_id"]).team.transform("nunique").gt(1)
    r.loc[multi, "identity_uncertain"] = True
    r.loc[multi, "identity_method"] = "ambiguous_membership"
    return r


def canonical_rosters(resolved):
    """Verified aliases count once. Prefer the native canonical provider row."""
    r = resolved.copy()
    r["native"] = r.person_id.eq("cfbd:" + r.athlete_id)
    return r.sort_values("native", ascending=False).drop_duplicates(
        ["season", "team", "person_id"]).drop(columns="native").reset_index(drop=True)


def resolve_history(history, identities, rules=None):
    """Only exact season/program/provider joins; unknown rows remain quarantined."""
    h = history.copy(); h["athlete_id"] = norm_id(h.athlete_id)
    keys = ["season", "team", "athlete_id"]
    identity = identities[keys + ["person_id", "identity_uncertain"]].drop_duplicates()
    if identity.duplicated(keys).any():
        raise ValueError("Ambiguous identity crosswalk")
    h = h.merge(identity, on=keys, how="left", validate="many_to_one")
    rules = json.loads(OVERRIDES.read_text())["rules"] if rules is None else rules
    for rule in rules:
        # Official biographies also establish history membership in a missing
        # annual roster year. These bounded overrides do not invent roster
        # membership for future targets.
        m = (h.athlete_id.eq(rule["athlete_id"]) & h.season.between(rule["from_season"],rule.get("to_season",9999))
             & h.team.isin(rule["teams"]))
        h.loc[m,"person_id"] = rule["person_id"]
        h.loc[m,"identity_uncertain"] = False
    h["history_usable"] = h.person_id.notna() & h.identity_uncertain.eq(False)
    return h


def _name_agrees(first, last, recruit_name) -> bool:
    """The last-name-core guard used for every recruit link: last name present in the
    recruit's name and first initials equal. A guard, not positive identity evidence."""
    core = last_name_core(last)
    name = norm_name(recruit_name)
    return bool(core) and core in name and norm_name(first)[:1] == name[:1]


def safe_recruit_links(roster, recruits):
    """Explicit links, scoped to the correct person; reject name conflicts.

    A reviewed override can supersede a conflicting provider athlete ID.
    Otherwise a mismatched provider ID is not accepted merely by name.
    """
    from cfbpred.player_catalog import link_recruits
    r, links, conflicts = link_recruits(roster, recruits)
    # A provider-ID conflict is a recruit the roster's own recruitIds names, whose
    # recruit record carries a different athleteId. Two independent pointers (that
    # explicit roster entry and the player's name) can still outvote one provider
    # field, but only when the competing athleteId does not belong to a roster row
    # that itself matches the recruit's name — then the record may be someone else's.
    conflicts = conflicts.copy()
    conflicts["reject_reason"] = "provider_athlete_id_conflict"
    by_id = r.dropna(subset=["athlete_id"]).groupby("athlete_id")
    rescued = []
    for ix, row in conflicts.iterrows():
        ros = r.loc[r.roster_row.eq(row.roster_row)].iloc[0]
        if not _name_agrees(ros.firstName, ros.lastName, row["name"]):
            continue
        claimants = by_id.get_group(row.recruit_athlete_id) if row.recruit_athlete_id in by_id.groups else r.iloc[0:0]
        if any(_name_agrees(o.firstName, o.lastName, row["name"]) for o in claimants.itertuples()):
            continue
        rescued.append(ix)
    if rescued:
        moved = conflicts.loc[rescued].drop(columns="reject_reason").copy()
        moved["match_method"] = "roster_recruit_id_name_verified"
        links = pd.concat([links, moved], ignore_index=True)
        conflicts = conflicts.drop(index=rescued)
    q = recruits.copy(); q["recruit_id"] = norm_id(q.id)
    q = q.rename(columns={"year": "recruit_year"})
    overrides = []
    for row in r.itertuples():
        if isinstance(row.recruit_override, list):
            links = links[links.roster_row.ne(row.roster_row)]
            for _, recruit in q[q.recruit_id.isin(row.recruit_override)].iterrows():
                overrides.append({**recruit.to_dict(), "roster_row": row.roster_row,
                                  "season": row.season, "match_method": "verified_override"})
    if overrides:
        links = pd.concat([links, pd.DataFrame(overrides)], ignore_index=True)
    check = r.set_index("roster_row")
    links["reject_reason"] = ""
    links.loc[links.roster_row.map(check.identity_uncertain), "reject_reason"] = "identity_uncertain"
    for ix, row in links.iterrows():
        ros = check.loc[row.roster_row]
        if row.recruit_year > ros.season:
            links.at[ix, "reject_reason"] = "future_recruit"
        elif pd.notna(ros.college_start) and row.recruit_year < int(ros.college_start)-2:
            links.at[ix, "reject_reason"] = "wrong_identity_episode"
        elif row.match_method != "verified_override":
            # Last-name agreement is a guard, not positive identity evidence.
            last = last_name_core(ros.lastName)
            if last and (last not in norm_name(row["name"]) or norm_name(ros.firstName)[:1]!=norm_name(row["name"])[:1]):
                links.at[ix, "reject_reason"] = "name_conflict"
    rejected = links[links.reject_reason.ne("")].copy()
    return r, links[links.reject_reason.eq("")].copy(), pd.concat([conflicts, rejected], ignore_index=True)
