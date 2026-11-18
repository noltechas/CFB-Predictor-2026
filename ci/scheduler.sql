-- GitHub Actions on the clock, started from Supabase.
--
-- GitHub's own `schedule:` trigger is best effort: on 2026-11-18 the 30-minute
-- scores job ran 3 times in 18 hours, the 10-minute pregame check 3 times, and
-- the nightly started 8.5 hours late. pg_cron here is punctual, so it starts
-- each workflow through the REST API (workflow_dispatch) instead.
--
-- The token lives in Supabase Vault as `github_actions_token` (a fine-grained
-- token: these two repositories only, Actions read and write). Nothing secret is
-- in this file; store or rotate the token with
-- ~/IdeaProjects/CBB-Predictor-2026/scripts/store_github_token.py.
--
-- Dispatches carry inputs.scheduled = "true" so the football workflows can still
-- honour the CFB_ACTIONS_ENABLED switch; a hand-started run never sends it.
--
-- Pause everything:  select cron.unschedule(jobname) from cron.job where jobname like 'gh-%';
-- Re-apply this file to restore the jobs.

create schema if not exists ops;
revoke all on schema ops from public, anon, authenticated;

create table if not exists ops.dispatch_log (
  id       bigserial primary key,
  at       timestamptz not null default now(),
  repo     text not null,
  workflow text not null,
  status   int,            -- 204 = GitHub accepted the dispatch
  detail   text            -- the response body (or error) when it did not
);
revoke all on ops.dispatch_log from public, anon, authenticated;
create index if not exists dispatch_log_at on ops.dispatch_log (at desc);

create or replace function ops.dispatch(
  p_repo     text,
  p_workflow text,
  p_inputs   jsonb default '{"scheduled": "true"}',
  p_ref      text  default 'main'
) returns int
language plpgsql security definer
set search_path = pg_catalog, extensions
as $$
declare
  token text;
  resp  record;
begin
  select decrypted_secret into token
    from vault.decrypted_secrets where name = 'github_actions_token' limit 1;
  if token is null then
    insert into ops.dispatch_log (repo, workflow, status, detail)
      values (p_repo, p_workflow, null, 'no github_actions_token in vault');
    return null;
  end if;

  perform extensions.http_set_curlopt('CURLOPT_TIMEOUT', '20');
  select status, content into resp from extensions.http((
    'POST',
    format('https://api.github.com/repos/%s/actions/workflows/%s/dispatches', p_repo, p_workflow),
    array[
      extensions.http_header('Authorization', 'Bearer ' || token),
      extensions.http_header('Accept', 'application/vnd.github+json'),
      extensions.http_header('X-GitHub-Api-Version', '2022-11-28'),
      -- GitHub refuses requests without a User-Agent
      extensions.http_header('User-Agent', 'fieldcast-scheduler')
    ],
    'application/json',
    jsonb_build_object('ref', p_ref, 'inputs', coalesce(p_inputs, '{}'::jsonb))::text
  )::extensions.http_request);

  insert into ops.dispatch_log (repo, workflow, status, detail)
    values (p_repo, p_workflow, resp.status,
            case when resp.status = 204 then null else left(resp.content, 500) end);
  return resp.status;
exception when others then
  insert into ops.dispatch_log (repo, workflow, status, detail)
    values (p_repo, p_workflow, null, left(sqlerrm, 500));
  return null;
end $$;

revoke all on function ops.dispatch(text, text, jsonb, text) from public, anon, authenticated;

-- The schedules, in UTC, the same moments the workflows' cron lines used.
select cron.unschedule(jobname) from cron.job
 where jobname in ('gh-cfb-scores', 'gh-cfb-pregame', 'gh-cfb-nightly', 'gh-cbb-nightly', 'gh-dispatch-log-trim');

select cron.schedule('gh-cfb-scores',  '7,37 * * * *',
  $$select ops.dispatch('noltechas/CFB-Predictor-2026', 'cfb-scores.yml')$$);
select cron.schedule('gh-cfb-pregame', '*/10 * * * *',
  $$select ops.dispatch('noltechas/CFB-Predictor-2026', 'cfb-pregame.yml')$$);
select cron.schedule('gh-cfb-nightly', '30 10 * * *',
  $$select ops.dispatch('noltechas/CFB-Predictor-2026', 'cfb-nightly.yml')$$);
select cron.schedule('gh-cbb-nightly', '15 10,18 * * *',
  $$select ops.dispatch('noltechas/CBB-Predictor-2026', 'nightly.yml')$$);
select cron.schedule('gh-dispatch-log-trim', '0 4 * * *',
  $$delete from ops.dispatch_log where at < now() - interval '30 days'$$);
