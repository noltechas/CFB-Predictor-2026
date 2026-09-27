# Mobile OTA evidence — historical reconstruction disclosures

The tested mobile disclosure and accuracy-filtering update was published to the
Expo EAS `production` channel on 2026-09-22. No native build or App Store /
Play Store submission was performed.

- Update group: `1573ed24-16ce-4bb0-9900-c1d01a6f2be8`
- Runtime: `exposdk:57.0.0`
- iOS update ID: `01a0c72f-7ec1-762d-995b-bff124f1cc11`
- Android update ID: `01a0c72f-7ec1-7d94-af41-79772098faae`
- Message: `Label retrospective historical predictions and exclude reconstructed rows from held-out accuracy`

Production manifest verification used the Expo update URL from `mobile/app.json`
with `expo-protocol-version: 1`, `expo-platform`, and
`expo-runtime-version: exposdk:57.0.0`. Both iOS and Android returned HTTP 200
and the expected `expo-update-id` response header. The production channel and
SDK runtime were unchanged, preserving installed-app audience compatibility.

The mobile test suite passed 106 tests and TypeScript type checking passed before
publication. Ranking season/week selectors remain feed-driven: they request the
available snapshot calendar and do not hardcode a current-season-only list, so
all published seasons can appear once present in the database.
