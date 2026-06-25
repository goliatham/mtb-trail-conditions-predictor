const WATERMARK_KEY = '__last_retrained_at__';
const LOG_PREDICT_KEY = '__last_predict__';
const LOG_RETRAIN_KEY = '__last_retrain_check__';
const GH_REPO = 'goliatham/mtb-trail-conditions-predictor';
const GH_WORKFLOW = 'daily-predict.yml';

const isVoteKey = (name) => !name.startsWith('__');

async function countVotes(env, sinceWatermark) {
  const list = await env.VOTES.list();
  const votes = (await Promise.all(
    list.keys
      .filter(({ name }) => isVoteKey(name))
      .map(async ({ name }) => {
        const val = await env.VOTES.get(name);
        return val ? JSON.parse(val) : null;
      })
  )).filter(Boolean);
  const since_retrain = sinceWatermark
    ? votes.filter(v => v.voted_at && v.voted_at > sinceWatermark).length
    : votes.length;
  return { total: votes.length, since_retrain, votes };
}

export default {
  async scheduled(event, env, ctx) {
    const log = { ts: new Date().toISOString(), cron: event.cron };

    if (event.cron === '0 13 * * 1') {
      // Sunday retrain — only if new votes exist since last retrain
      // Reads KV directly rather than calling /status to avoid self-invocation issues
      const watermark = await env.VOTES.get(WATERMARK_KEY);
      const { total, since_retrain } = await countVotes(env, watermark);
      log.action = 'retrain_check';
      log.votes_total = total;
      log.votes_since_retrain = since_retrain;

      if (since_retrain === 0) {
        log.outcome = 'skipped';
        log.reason = 'no new votes since last retrain';
        console.log(`Sunday retrain skipped: 0 new votes`);
        await env.VOTES.put(LOG_RETRAIN_KEY, JSON.stringify(log));
        return;
      }

      const resp = await fetch(
        `https://api.github.com/repos/${GH_REPO}/actions/workflows/retrain.yml/dispatches`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${env.GITHUB_TOKEN}`,
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
            'User-Agent': 'mtcp-votes-worker',
          },
          body: JSON.stringify({ ref: 'main', inputs: { reason: 'weekly auto-retrain' } }),
        }
      );
      log.gh_status = resp.status;
      if (!resp.ok) {
        const body = await resp.text();
        log.outcome = 'dispatch_failed';
        log.error = body;
        console.error(`Retrain dispatch failed ${resp.status}: ${body}`);
      } else {
        log.outcome = 'dispatched';
        console.log(`Sunday retrain: ${since_retrain} new votes, dispatched`);
      }
      await env.VOTES.put(LOG_RETRAIN_KEY, JSON.stringify(log));
      return;
    }

    // Daily predict
    log.action = 'daily_predict';
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
    log.gh_status = resp.status;
    if (!resp.ok) {
      const body = await resp.text();
      log.outcome = 'failed';
      log.error = body;
      console.error(`GitHub dispatch failed ${resp.status}: ${body}`);
    } else {
      log.outcome = 'ok';
    }
    await env.VOTES.put(LOG_PREDICT_KEY, JSON.stringify(log));
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

    // GET /votes — return all votes (?all=1) or only new since last retrain
    if (request.method === 'GET' && url.pathname === '/votes') {
      const watermark = await env.VOTES.get(WATERMARK_KEY);
      const since = url.searchParams.get('all') ? null : watermark;
      const { votes } = await countVotes(env, null);
      const filtered = since
        ? votes.filter(v => v.voted_at && v.voted_at > since)
        : votes;
      return json(filtered);
    }

    // POST /retrained — update watermark and trim KV to 500 most recent votes (~2 months buffer)
    if (request.method === 'POST' && url.pathname === '/retrained') {
      const ts = new Date().toISOString();
      await env.VOTES.put(WATERMARK_KEY, ts);

      const list = await env.VOTES.list();
      const voteKeys = list.keys
        .filter(({ name }) => isVoteKey(name))
        .map(({ name }) => name)
        .sort((a, b) => (b.split(':')[1] || '').localeCompare(a.split(':')[1] || '')); // newest first
      const toDelete = voteKeys.slice(500);
      await Promise.all(toDelete.map(key => env.VOTES.delete(key)));

      return json({ ok: true, watermark: ts, total: voteKeys.length, deleted: toDelete.length });
    }

    // GET /status — vote count + watermark
    if (request.method === 'GET' && url.pathname === '/status') {
      const watermark = await env.VOTES.get(WATERMARK_KEY);
      const { total, since_retrain } = await countVotes(env, watermark);
      return json({ total, since_retrain, last_retrained_at: watermark });
    }

    // GET /debug — last outcome for each cron type (persisted separately to KV)
    if (request.method === 'GET' && url.pathname === '/debug') {
      const [predict, retrain] = await Promise.all([
        env.VOTES.get(LOG_PREDICT_KEY),
        env.VOTES.get(LOG_RETRAIN_KEY),
      ]);
      return json({
        last_predict:       predict ? JSON.parse(predict) : null,
        last_retrain_check: retrain ? JSON.parse(retrain) : null,
      });
    }

    return json({ error: 'not found' }, 404);
  },
};
