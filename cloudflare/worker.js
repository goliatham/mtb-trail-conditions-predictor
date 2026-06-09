const WATERMARK_KEY = '__last_retrained_at__';
const GH_REPO = 'goliatham/mtb-trail-conditions-predictor';
const GH_WORKFLOW = 'daily-predict.yml';

export default {
  async scheduled(event, env, ctx) {
    const resp = await fetch(
      `https://api.github.com/repos/${GH_REPO}/actions/workflows/${GH_WORKFLOW}/dispatches`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${env.GITHUB_TOKEN}`,
          'Accept': 'application/vnd.github+json',
          'X-GitHub-Api-Version': '2022-11-28',
          'User-Agent': 'mtcp-votes-worker',
        },
        body: JSON.stringify({ ref: 'main' }),
      }
    );
    if (!resp.ok) {
      const body = await resp.text();
      console.error(`GitHub dispatch failed ${resp.status}: ${body}`);
    }
  },

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

    // GET /votes — return votes since last retrain (or all if no watermark)
    // ?all=1 returns everything regardless of watermark
    if (request.method === 'GET' && url.pathname === '/votes') {
      const watermark = await env.VOTES.get(WATERMARK_KEY);
      const since = url.searchParams.get('all') ? null : watermark;
      const list = await env.VOTES.list();
      const votes = (await Promise.all(
        list.keys
          .filter(({ name }) => name !== WATERMARK_KEY)
          .map(async ({ name }) => {
            const val = await env.VOTES.get(name);
            return val ? JSON.parse(val) : null;
          })
      )).filter(Boolean);

      const filtered = since
        ? votes.filter(v => v.voted_at && v.voted_at > since)
        : votes;

      return json(filtered);
    }

    // POST /retrained — update watermark after a successful retrain
    if (request.method === 'POST' && url.pathname === '/retrained') {
      const ts = new Date().toISOString();
      await env.VOTES.put(WATERMARK_KEY, ts);
      return json({ ok: true, watermark: ts });
    }

    // GET /status — show vote count + watermark (useful for threshold check)
    if (request.method === 'GET' && url.pathname === '/status') {
      const watermark = await env.VOTES.get(WATERMARK_KEY);
      const list = await env.VOTES.list();
      const votes = (await Promise.all(
        list.keys
          .filter(({ name }) => name !== WATERMARK_KEY)
          .map(async ({ name }) => {
            const val = await env.VOTES.get(name);
            return val ? JSON.parse(val) : null;
          })
      )).filter(Boolean);
      const newVotes = watermark
        ? votes.filter(v => v.voted_at && v.voted_at > watermark).length
        : votes.length;
      return json({ total: votes.length, since_retrain: newVotes, last_retrained_at: watermark });
    }

    return json({ error: 'not found' }, 404);
  },
};
