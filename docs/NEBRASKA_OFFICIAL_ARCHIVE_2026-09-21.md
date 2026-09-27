# Nebraska official archive audit — 2026-09-21

This batch preserves 81 raw HTML responses from the official Huskers.com archive pagination crawl (`raw_batch_002`). The copied responses are SHA-256 hashed in `source_inventory.json`; each entry retains its public URL, retrieval time, JSON-LD metadata, and status-term scan.

The batch is complete for the exposed 2026 archive pagination result, but it exposes no older official article URLs. The pinned schedule contains 313 Nebraska fixtures (304 historical, 9 future, seasons 2001–2026). `game_coverage.json` maps every fixture and distinguishes `source_enumerated_2026_only` from `not_searched_public_official_archive`; no report is marked verified and `facts.json` is empty. No player health or availability is inferred from missing articles, game notes, box scores, or unlisted names.

The 2026 HTML pages are retained as retrospective official evidence. They are not promoted to model facts because this batch did not establish a game-specific player status section with both publisher publication and revision timestamps before each kickoff. PDF metadata, where present, is not publication proof.

Repeatable route: replay the official archive pagination query, archive each response, record URL/retrieval/hash and JSON-LD publication/revision fields, normalize the full Nebraska schedule from the pinned snapshot, join article game/opponent/date, and promote only explicit status headings whose latest publisher timestamp precedes kickoff. Preserve unmatched, incomplete, and pre-2001 links with an out-of-scope marker rather than dropping them.

## Public API discovery and bulk enumeration

The official Nuxt bundle imports a public schedule client whose live routes resolve under `/website-api`. Verified requests:

- `GET /website-api/sports?filter[slug]=football` → football sport id `5`.
- `GET /website-api/schedules?filter[sport.id]=5&per_page=200&page=1` → 146 football schedules, from 1890 through 2035/2033 entries.
- `GET /website-api/schedule-events?filter[schedule_id]=<id>&per_page=200&page=1` for every returned football schedule → 1,486 event records, with pagination metadata recorded per response.
- `GET /website-api/articles/<article_id>` for linked pre/post-event article IDs. The public batch returned 283 article records and then 139 HTTP 429 responses; all successful responses and rate-limit evidence are recorded in `api_route_trace.json` and `raw_api/articles/`.

`historical_game_map.json` matches all 313 pinned model fixtures to official API schedule events by season, opponent, and date. Linked articles are retrospective official source material; no availability fact is promoted without explicit status text plus publisher publication/revision proof before kickoff. The API enumeration materially extends the exposed archive through 1890, including seasons outside the current model scope; these are retained in the raw API inventory for future use.

The sole initial map miss was 2015 BYU. The official event is present as `Brigham Young` (event 4126, schedule 231); the pinned schedule uses `BYU`. It is now matched with an explicit alias marker, not a guessed opponent.

The article endpoint `/website-api/articles/<id>` returned metadata-only objects (for example, no `contentBlocks` or body field in article 2560). The public article page client documents the content route `/website-api/articles/by/slug/{slug}?include=author,categories,contentBlocks,image,links,scheduleEvent`, `article_retry_queue.json` records pending IDs and requires Retry-After/backoff before any future retry. Existing article responses are labeled `metadata_only`, and no facts are derived from them.

`article_request_receipts.json` is the complete cached request receipt ledger for the article metadata pass. No requests were made after the recorded rate-limit boundary; `enumerate_public_api.py` implements the same first-429 stop behavior for reproducibility.

## Content-route cooldown probe

After cooldown, one probe of the discovered content route for pre-event article 2552 returned HTTP 200 with `content-type: application/json` and no `Retry-After` header. A bounded follow-up fetched 24 additional distinct pre-event articles from 2019–2020 at one-second spacing, all HTTP 200; a later bounded continuation brought the total to 37. Their actual `content_blocks` are archived under `raw_api/content/`; `content_inventory.json` records body quotations, publisher `published_at`, missing `modified_at`, and hashes. `game_note_inventory.json` maps 29 linked official game-note PDFs to schedule events. These are retrospective-only because revision timestamps are absent; no player facts were promoted.

## Final bounded content/PDF batch

The public article content route was used only for distinct pre-event article IDs already exposed by the official schedule records. `raw_api/content/` now contains 37 successful full-body JSON responses (including the cooldown probe), mapped to 36 of the 313 current-model fixtures. The metadata pass had 139 HTTP 429 responses; 19 distinct pre-event article IDs remain in `remaining_content_queue.json` because their slugs/metadata were unavailable at the rate-limit boundary. No host rotation or retry was attempted.

The content bodies exposed 29 official game-note PDF URLs. `pdf_batch_receipts.json` records one bounded 25-request HTTPS batch: 25 HTTP 200 responses, with response headers, byte counts, final URLs, local paths, and SHA-256 hashes. Four PDF URLs remain pending in `pdf_inventory.json`; the older HTTP forms were retried over HTTPS only because the HTTP hostname did not resolve. These PDFs and game notes are retrospective evidence only: the linked article records publish times but no publisher revision timestamp, so they produce zero eligible availability facts. Existing 2024 PDF archives were checked for URL overlap before downloading; none of these 29 URLs overlapped.

`coverage_summary.json` reports 313/313 official event matches, 36 model games with a cached pregame article body, 19 model games with a cached game-note PDF, 37 cached content articles, 19 pending content IDs, and 25/29 cached PDFs. `facts.json` remains empty and `quarantine.json` remains the proper destination for any future status text lacking pre-kickoff revision proof. `enumerate_public_api.py` plus the receipt ledgers provide the repeatable route; pending queues can resume only after an appropriate Retry-After cooldown.
