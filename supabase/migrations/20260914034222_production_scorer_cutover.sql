-- Production cutover: the edge_service_key Vault secret is provisioned separately.
create extension if not exists http with schema extensions;
create extension if not exists pg_cron with schema pg_catalog;
create schema if not exists cfb_private;
revoke all on schema cfb_private from public, anon, authenticated;
CREATE OR REPLACE FUNCTION cfb_private.invoke_live_scorer()
 RETURNS integer
 LANGUAGE plpgsql
 SET search_path TO ''
AS $function$
declare response extensions.http_response;
begin
 perform extensions.http_set_curlopt('CURLOPT_TIMEOUT','50');
 select * into response from extensions.http(('POST','https://ijhekfbtyahlhsfstcuj.supabase.co/functions/v1/live-scorer',array[extensions.http_header('Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='edge_service_key'))],'application/json','{}')::extensions.http_request);
 if response.status<>200 then raise exception 'live-scorer HTTP %: %',response.status,left(response.content,300);end if;
 return response.status;
end
$function$
;
revoke all on function cfb_private.invoke_live_scorer() from public, anon, authenticated;
alter function public.live_curve_state(bigint[]) set search_path = '';
alter view public.v_model_vs_vegas set (security_invoker = true);
-- Both original policies allowed exactly the same roles and SELECT predicate.
drop policy if exists anon_read_live_predictions on public.live_predictions;
do $$ begin
 if not exists(select 1 from cron.job where jobname='live-scorer')
    and exists(select 1 from vault.secrets where name='edge_service_key') then
  perform cron.schedule('live-scorer','* * * * *','select cfb_private.invoke_live_scorer();');
 end if;
end $$;
