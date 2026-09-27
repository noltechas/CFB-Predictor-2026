# CFB database migration — completed September 14, 2026

## Completed and verified

The user explicitly approved the migration. CFB Production organization `vcdtzebkicecbnkjxueb` is now on Pro with Spend Cap enabled. Its sole project, `ijhekfbtyahlhsfstcuj`, runs standard PostgreSQL 17.6 with heap tables. The approved subscription is $25/month plus applicable tax. The old organization remains Free, and the original OrioleDB project is retained for rollback.

All 16 application table row counts matched the source immediately after restore. All 86 Week 2 FBS-involved games have finals; the real mobile data layer returns 131 modeled games in total, selects Week 2, and loads 710 teams. The mobile web preview visibly renders the final scores. All 1,376 original Storage objects were migrated: 1,373 match their original hashes; three logos were refreshed from their original public CDN sources. Model weights match exactly.

The complete scorer is deployed as active version 2 with JWT verification enabled. Both model files are embedded in the deployed JavaScript, initialized lazily, and make no runtime Storage requests. The new multipart deploy endpoint rejected even the signed-in owner; Supabase's documented JSON create/update endpoints succeeded. `supabase/build_scorer_bundle.cjs` reproduces the deployed bundle from the canonical sources. Its output was verified byte-for-byte against the artifact submitted for deployment. No temporary signed import is used by the live deployment; its unused staging object was removed.

A real write cycle initially hit the Edge compute ceiling. Final backlog processing is now limited to 16 games per invocation, retaining all live games, with a 120-snapshot inference budget. Tests verify backlog advancement and retention of live games. Real write requests now return HTTP 200. The private one-minute scheduler raises an error on any non-200 response; automatic runs at 03:42, 03:43, and 03:44 UTC succeeded. Curve backlog continues draining automatically.

The source scorer is paused. `.env` and the existing score/nightly publication jobs now point to the new database; both launch agents were resumed successfully. The web configuration is switched. Original configuration and the database archive are retained privately under `data/audit_backups/20260913_orioledb/`. The original API remains quota-restricted; recovery should restore the backup into a serving project or clear that restriction before reverting an app endpoint.

The production Expo update group is `79db0a43-a4f2-4a11-874c-3400b940637d`, runtime `exposdk:57.0.0`. Actual production manifest requests confirm iOS update `01a09dff-2d9b-775b-a741-0262d16d92f0` and Android update `01a09dff-2d9b-76da-a125-4e81f3490148` are served. Installed phones apply the update after download and restart.

Validation: 91 mobile tests and TypeScript checking passed; three scorer work-selection tests, the embedded-model network-free check, Deno checking, and bundle reproducibility passed. All public tables retain RLS and foreign-key checks found no missing venues. The four original migration records are preserved, followed by the new private scheduler/access-hardening migration. Final Supabase advisors show no warnings or errors; remaining informational notices include the deliberately server-only logits table, index usage, and connection allocation.

Evidence: `logs/score_repair/migrated-mobile-check.json`, `migration-update-delivery.json`, `latest-scheduled-runs.json`, `mobile-migration-tests.log`, and the private backup directory's row counts, scorer responses, deployment metadata, and final advisor report.

## Why the upgrade failed

CFB Predictor (`ocswpxthayxplgfboqvj`) runs PostgreSQL 17.9 with OrioleDB, on the preview release channel. Supabase rejects paid subscriptions while this project remains in the organization. The other project, Online Social Deduction, is standard PostgreSQL and paused; it is not the preview blocker.

The original organization remains Free. Its attempted upgrade was rejected for the preview-engine restriction. The separate replacement organization was subsequently upgraded successfully.

## Backup and inventory

`data/audit_backups/20260913_orioledb/database.dump` is a read-only, consistent pg_dump archive of public, auth, storage, and migration-history schemas. Size: 83,687,159 bytes. SHA-256: `e6289b0c9789f506a2a651507eb9b878ad0e7594f71933bd3366cd79ff0200c8`.

The entire archive was successfully decoded by pg_restore. The archive was subsequently restored successfully to the new production database and its application row counts verified. Inventory and access policies are saved beside it. All 16 public tables have RLS enabled. There are no Supabase Auth users to migrate. Storage contains 1,376 object metadata records; the database dump does not contain their file contents. Original model files are available locally with matching hashes. Logo objects require copying or regeneration from the original CDN sources if the restricted Storage API cannot serve them.

## Approved migration approach

1. Create a separate organization and standard PostgreSQL project in us-east-1 for CFB production. Apply the already approved single Pro subscription only after checking the estimate remains $25/month plus tax. Keep the original organization Free.
2. Restore public schema, all public data, views, grants, and RLS into standard heap tables. Use pg_restore's `--no-table-access-method` during restoration; the custom archive retains engine metadata. Preserve the managed auth/storage schemas supplied by the new project; restore only application-specific policies/configuration and required migration history. Recreate supported extensions, excluding OrioleDB.
3. Recreate Storage buckets and copy/regenerate the logos. Deploy the bundled-model scorer with JWT verification enabled. Configure its server-side credentials and one-minute scheduler without exposing secrets to mobile clients.
4. Compare table counts, final scores, prediction coverage, rankings, RLS, curves, and model output against the original snapshot. Verify real REST requests and scorer execution before changing the app endpoint.
5. Briefly stop local publication jobs at cutover, synchronize any data changed since the snapshot, switch their database configuration, and resume them. Publish the new endpoint and publishable key through the production Expo update and update the web configuration. Check actual production manifests and API responses.
6. Preserve the original project and archive for rollback. Do not delete its database. Keep the total approved paid subscription at $25/month plus applicable tax, with spend cap enabled.

## Approval-review history

Automatic approval review rejected clicking the dashboard ORIOLEDB service-version control because it might change the production engine and cause downtime. The user subsequently approved the specific migration, which is now complete. A temporary maintenance-handler deployment was also rejected; it was never deployed. Only the complete tested scorer was deployed successfully.

References: [Supabase migration guide](https://supabase.com/docs/guides/platform/migrating-within-supabase/backup-restore), [OrioleDB overview](https://supabase.com/docs/guides/database/orioledb).
