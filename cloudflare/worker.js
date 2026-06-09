export default {
  async fetch(request, env) {
    const cors = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: cors });
    }

    const url = new URL(request.url);
    const json = (body, status = 200) =>
      new Response(JSON.stringify(body), {
        status,
        headers: { ...cors, 'Content-Type': 'application/json' },
      });

    // POST /vote — store a single slot vote (keyed by trail:date:slot)
    if (request.method === 'POST' && url.pathname === '/vote') {
      let vote;
      try { vote = await request.json(); } catch { return json({ error: 'invalid json' }, 400); }
      const { date, trail, slot } = vote;
      if (!date || !trail || !slot || vote.vote === undefined) {
        return json({ error: 'missing fields: date, trail, slot, vote' }, 400);
      }
      const key = `${trail}:${date}:${slot}`;
      await env.VOTES.put(key, JSON.stringify(vote));
      return json({ ok: true });
    }

    // GET /votes — return all stored votes as array
    if (request.method === 'GET' && url.pathname === '/votes') {
      const list = await env.VOTES.list();
      const votes = await Promise.all(
        list.keys.map(async ({ name }) => {
          const val = await env.VOTES.get(name);
          return val ? JSON.parse(val) : null;
        })
      );
      return json(votes.filter(Boolean));
    }

    return json({ error: 'not found' }, 404);
  },
};
