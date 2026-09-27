# Publishing Chas’s rankings

The app’s **Chas** tab shows one shared list to everyone. It reads the latest two weekly snapshots from `public.editorial_rankings`. The empty state remains until you publish your first list. No sample rankings are published.

Edit `data/editorial_rankings.json`: set the season and football week (0 means preseason), then put teams in your preferred order, starting with #1. There is no Top 25 restriction. Each team can be a school name, a numeric team ID, or an object with `school` / `team_id` and an optional `note`. A title and introduction are optional.

For example, the following **format example** assigns Ohio State #1 and Georgia #2; replace it with your actual choices:

```json
"teams": [
  { "school": "Ohio State", "note": "Your explanation goes here." },
  "Georgia"
]
```

From the project root, load the same database settings used by the other publishing commands:

```sh
set -a
source .env
set +a

# Look up a school name or ID.
python3 src/cfbpred/editorial_rankings.py --list-teams "Ohio"

# Validate every team and review the full ordered snapshot without writing.
python3 src/cfbpred/editorial_rankings.py --file data/editorial_rankings.json

# Publish the reviewed list for everyone.
python3 src/cfbpred/editorial_rankings.py --file data/editorial_rankings.json --publish
```

Publishing atomically replaces the selected week; earlier weeks stay available. Empty lists, unknown teams, duplicate teams, and invalid metadata are rejected. App users have SELECT access only, with RLS enabled. Database credentials stay in the backend environment. The app caches snapshots for 15 minutes; pull to refresh loads an update immediately. Rank movement compares only consecutive weeks from the same season and author; search preserves published positions.

The migration is `supabase/migrations/20260908204517_editorial_rankings.sql`. This data is independent of `fieldcast_rankings` and is never a model input.
