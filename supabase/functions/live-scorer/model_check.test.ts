Deno.test('deployment model check loads frozen files without making network requests', async () => {
  const oldServe = Deno.serve;
  const oldFetch = globalThis.fetch;
  const oldUrl = Deno.env.get('SUPABASE_URL');
  const oldKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
  let handler: ((req: Request) => Promise<Response>) | undefined;
  Deno.env.set('SUPABASE_URL', 'https://test.supabase.co');
  Deno.env.set('SUPABASE_SERVICE_ROLE_KEY', 'test-placeholder');
  Object.defineProperty(Deno, 'serve', { configurable: true, writable: true,
    value: (fn: typeof handler) => { handler = fn; } });
  globalThis.fetch = () => { throw new Error('Model check must not access the network'); };
  try {
    await import('./index.ts');
    if (!handler) throw new Error('No handler registered');
    const response = await handler(new Request('https://test.supabase.co/?check=model'));
    const body = await response.json();
    if (response.status !== 200 || body.model !== 'bundled' || body.storageRequests !== 0)
      throw new Error(JSON.stringify(body));
  } finally {
    globalThis.fetch = oldFetch;
    Object.defineProperty(Deno, 'serve', { configurable: true, writable: true, value: oldServe });
    if (oldUrl == null) Deno.env.delete('SUPABASE_URL'); else Deno.env.set('SUPABASE_URL',oldUrl);
    if (oldKey == null) Deno.env.delete('SUPABASE_SERVICE_ROLE_KEY'); else Deno.env.set('SUPABASE_SERVICE_ROLE_KEY',oldKey);
  }
});
