# Mobile OTA evidence — pregame "why this number" section

Published to the Expo EAS `production` channel on 2026-09-22. No native build and
no App Store / Play Store submission was performed: the change is JavaScript only,
adds no dependency and no native module, so installed apps pick it up on next launch.

- Update group: `ec393975-2be2-4879-8fed-9570523df92c`
- Runtime: `exposdk:57.0.0` (unchanged, so the installed-app audience is unaffected)
- iOS update ID: `01a0c994-4d81-7e05-9b5a-ded084573e8e`
- Android update ID: `01a0c994-4d81-770b-aca1-662fd5dc66f9`
- Message: `Pregame Why this number section: every model input, in win-probability points`
- Previous group on this channel, for rollback: `1573ed24-16ce-4bb0-9900-c1d01a6f2be8`

## What it ships

A "Why this number" card on the game screen, directly under Game info, shown only when
an attribution has been published for that game. Each factor is stated as a positive
percentage with the benefiting team's mark beside it, never a signed number, and any
factor expands to every field behind it with the value and the FBS average it is
measured against. A separate panel ranks individual players inside their own position
unit. The card renders nothing when no attribution exists, so games outside the
published slate are unaffected.

Contributions come from `public.game_explanations`, written by
`cfbpred.explain_game` and `cfbpred.publish_explanations`, which is regenerated for
the upcoming FBS slate as step 10 of `refresh_week.sh` and pruned to the active model
version. The table is read with the same publishable key and read-only policy the rest
of the app uses.

## Verification

Production manifest verification used the Expo update URL from `mobile/app.json` with
`expo-protocol-version: 1`, `expo-platform`, `expo-runtime-version: exposdk:57.0.0` and
`expo-channel-name: production`. Both iOS and Android returned HTTP 200 and the
expected `expo-update-id` response header. The channel pointer was confirmed to serve
update group `ec393975-2be2-4879-8fed-9570523df92c`. Note that `u.expo.dev` rejects a
default Python client fingerprint with a Cloudflare 1010; verify with `curl`.

The mobile suite passed 106 tests, TypeScript type checking passed, and story coverage
reported every exported component has a story. The card was reviewed in Storybook on
web across its favoured, reversed and near-even states before publication, and
anonymous reads of `game_explanations` were confirmed to return all 71 published games.

`EXPO_PUBLIC_STORYBOOK` was explicitly set to `false` for the publish, so the bundle
registers the app root rather than the design library. The EAS `production` environment
holds no variables; the Supabase endpoint and publishable key are compiled into
`mobile/src/config.ts`, so no secret is injected at update time.

The mobile working tree carried 143 modified and 85 untracked files that predate this
work, so the update is recorded against commit `f2fe01e` with a dirty-tree marker.
