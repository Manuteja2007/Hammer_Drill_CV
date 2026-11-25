const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const { test } = require('node:test');
const { execFileSync } = require('node:child_process');
const renderedHTML = execFileSync('py', ['-3', '-c', 'from app import app; from flask import render_template; import sys; ctx=app.test_request_context(); ctx.push(); sys.stdout.buffer.write(render_template("index.html").encode("utf-8")); ctx.pop()'], { env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' } }).toString('utf8');

function harness(reviewJobs = []) {
    const html = renderedHTML;
    const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
    assert.equal(ids.length, new Set(ids).size, 'HTML IDs must be unique');
    const nodes = new Map();
    for (const id of ids) {
        const classes = new Set();
        nodes.set(id, { id, textContent: '', className: '', style: {}, value: '', files: [],
            disabled: false, listeners: {}, children: [], classList: {
                add: value => classes.add(value), remove: value => classes.delete(value),
                toggle: (value, enabled) => enabled ? classes.add(value) : classes.delete(value),
                contains: value => classes.has(value),
            }, removeAttribute(name) { delete this[name]; },
            addEventListener(name, fn) { this.listeners[name] = fn; },
            replaceChildren() { this.children = []; }, append(value) { this.children.push(value); },
        });
    }
    const events = {}, calls = [], sources = [];
    const context = { console, Blob, navigator: { sendBeacon() {} },
        setTimeout: (fn, delay) => { if (delay === 500) queueMicrotask(fn); return 1; }, clearTimeout() {},
        FormData: class { append() {} },
        EventSource: class {
            constructor(url) { this.url = url; sources.push(this); }
            close() { this.closed = true; }
        },
        window: { addEventListener(name, fn) { events[name] = fn; } },
        document: { getElementById(id) {
            assert.ok(nodes.has(id), `Missing HTML element: ${id}`);
            return nodes.get(id);
        }, querySelectorAll: () => [...nodes.values()].filter(node => node.id.startsWith('view-')),
        createElement: () => ({ textContent: '', className: '' }),
        addEventListener(name, fn) { events[name] = fn; } },
        fetch: async (url, options = {}) => {
            calls.push({ url, body: options.body && typeof options.body === 'string' ? JSON.parse(options.body) : null });
            const data = url === '/review_video' ? reviewJobs.shift() : url.startsWith('/start_stream/') ? { feed_url: '/video_feed_file?token=test' } : url.startsWith('/session_summary/') ? {
                elapsed: 12, total_hits: 3, total_misses: 0, best_streak: 3,
                max_velocity: 1, avg_velocity: 0.2, avg_accel: 0.5,
                timeline_text: '3 hits', timeline: [{ type: 'HIT' }, { type: 'HIT' }, { type: 'HIT' }],
            } : { message: 'Upload successful', status: 'success' };
            return { ok: true, json: async () => data };
        },
    };
    vm.createContext(context);
    vm.runInContext(fs.readFileSync('static/app.js', 'utf8') + '\nthis.app = App;', context);
    events.DOMContentLoaded();
    return { nodes, calls, sources, events, context, app: context.app };
}

function stats(overrides = {}) {
    return { time: 4, duration: 20, play_state: 'play', last_event: '--', timeline_text: 'No events yet', ...overrides };
}

test('live session displays analytics and waits for stop before fetching final results', async () => {
    const h = harness();
    await h.app.start('live');
    assert.equal(h.nodes.get('btn-live-pause').disabled, true);
    await h.app.toggleLiveSession();
    h.sources[0].onmessage({ data: JSON.stringify(stats({ hit_count: 3 })) });
    assert.equal(h.nodes.get('live-hit_count').textContent, 3);
    await h.app.toggleLivePause();
    assert.equal(h.calls.at(-1).body.action, 'pause');
    await h.app.stopSession('live');
    assert.equal(h.calls.at(-2).body.action, 'stop');
    assert.equal(h.calls.at(-1).url, '/session_summary/live');
    assert.equal(h.nodes.get('res-duration').textContent, '00:00:12');
    assert.equal(h.nodes.get('res-timeline-visual').children[0].textContent, '3 hits');
    assert.ok(h.nodes.get('view-results').classList.contains('active'));
});

test('upload filenames remain literal text and keyboard seeking sends seek before play', async () => {
    const h = harness();
    await h.app.start('upload_prompt');
    h.nodes.get('video-upload-input').files = [{ name: '<img src=x onerror=alert(1)>.mp4', size: 12 }];
    await h.app.handleUpload();
    assert.match(h.nodes.get('upload-status').textContent, /<img src=x/);
    assert.equal(h.nodes.get('upload-status').innerHTML, undefined);
    h.sources[0].onmessage({ data: JSON.stringify(stats()) });
    h.nodes.get('video-scrubber').listeners.input({ target: { value: 50 } });
    assert.equal(h.nodes.get('video-timer-current').textContent, '00:10');
    await h.nodes.get('video-scrubber').listeners.change({ target: { value: 50 } });
    assert.equal(h.calls.at(-2).body.action, 'seek');
    assert.equal(h.calls.at(-2).body.time, 10);
    assert.equal(h.calls.at(-1).body.action, 'play');
});

test('home releases both modes before clearing feeds', async () => {
    const h = harness();
    await h.app.start('live');
    await h.app.toggleLiveSession();
    await h.app.goHome();
    assert.deepEqual(h.calls.filter(call => call.url === '/video_control').map(call => call.body.action), ['stop', 'stop']);
    assert.equal(h.nodes.get('live-feed').src, undefined);
    assert.equal(h.sources[0].closed, true);
    assert.ok(h.nodes.get('view-landing').classList.contains('active'));
});

test('request errors stay on the current view and show the backend message', async () => {
    const h = harness();
    await h.app.start('live');
    await h.app.toggleLiveSession();
    h.context.fetch = async () => ({ ok: false, status: 409, json: async () => ({ error: 'Camera busy' }) });
    await h.app.stopSession('live');
    assert.equal(h.nodes.get('toast').textContent, 'Camera busy');
    assert.ok(h.nodes.get('view-live').classList.contains('active'));
});

test('oversized uploads are rejected before sending a request', async () => {
    const h = harness();
    await h.app.start('upload_prompt');
    h.nodes.get('video-upload-input').files = [{ name: 'large.mp4', size: 501 * 1024 * 1024 }];
    await h.app.handleUpload();
    assert.equal(h.calls.length, 0);
    assert.match(h.nodes.get('toast').textContent, /500 MB/);
});

test('hand-contact rejections display separately from accepted hits', async () => {
    const h = harness();
    await h.app.start('live');
    await h.app.toggleLiveSession();
    h.sources[0].onmessage({ data: JSON.stringify(stats({ hit_count: 3, rejected_count: 2,
        hand_status: 'Hand contact - tap ignored', hand_check_ms: 9.5, last_event: 'REJECTED' })) });
    assert.equal(h.nodes.get('live-hit_count').textContent, 3);
    assert.equal(h.nodes.get('live-rejected_count').textContent, 2);
    assert.equal(h.nodes.get('live-hand_status').textContent, 'Hand contact - tap ignored');
    assert.equal(h.nodes.get('live-hand_check_ms').textContent, 9.5);
    assert.match(h.nodes.get('live-last_event_badge').className, /hm-badge-rejected/);
});

test('session target follows counted taps and session label stays literal', async () => {
    const h = harness();
    h.nodes.get('session-label').value = '<session> morning set';
    h.nodes.get('goal-taps').value = '5';
    await h.app.start('live');
    assert.equal(h.nodes.get('live-session-name').textContent, '<session> morning set');
    await h.app.toggleLiveSession();
    h.sources[0].onmessage({ data: JSON.stringify(stats({ hit_count: 3, state: 'Tracking' })) });
    assert.equal(h.nodes.get('live-goal-label').textContent, '3 of 5 taps');
    assert.equal(h.nodes.get('live-goal-progress').value, 60);
    h.sources[0].onmessage({ data: JSON.stringify(stats({ hit_count: 7, state: 'Tracking' })) });
    assert.equal(h.nodes.get('live-goal-progress').value, 100);
    assert.match(h.nodes.get('live-goal-label').textContent, /target reached/);
});

test('startup failures show the backend error without opening a video stream', async () => {
    const h = harness();
    await h.app.start('live');
    h.context.fetch = async () => ({ ok: false, status: 409, json: async () => ({ error: 'Camera already in use' }) });
    await h.app.toggleLiveSession();
    assert.equal(h.nodes.get('toast').textContent, 'Camera already in use');
    assert.equal(h.nodes.get('live-playback-label').textContent, 'Could not start');
    assert.equal(h.nodes.get('live-feed').src, undefined);
    assert.equal(h.sources.length, 0);
});

test('downloaded report includes finalized numbers and session notes', async () => {
    const h = harness();
    let exported = null;
    h.context.URL = { createObjectURL(blob) { exported = blob; return 'blob:test'; }, revokeObjectURL() {} };
    h.context.document.body = { append() {} };
    const create = h.context.document.createElement;
    h.context.document.createElement = name => name === 'a' ? { click() {}, remove() {} } : create(name);
    await h.app.start('live');
    await h.app.toggleLiveSession();
    await h.app.stopSession('live');
    h.nodes.get('session-notes').value = 'Keep the ball in frame.';
    await h.app.downloadReport();
    const report = JSON.parse(await exported.text());
    assert.equal(report.total_hits, 3);
    assert.equal(report.session_notes, 'Keep the ball in frame.');
    assert.match(report.interpretation, /estimates/);
});

test('invalid tap targets are explained before a session starts', async () => {
    const h = harness();
    h.nodes.get('goal-taps').value = '2.5';
    await h.app.start('live');
    assert.match(h.nodes.get('toast').textContent, /whole-number tap target/);
    assert.equal(h.calls.length, 0);
});


test('finish recording reviews every frame independently of preview position', async () => {
    const result = { elapsed: 10, total_hits: 2, best_streak: 2, total_frames: 300,
        analysis_scope: 'entire_recording', timeline: [], timeline_text: '2 hits' };
    const h = harness([{status: 'running', processed_frames: 150, total_frames: 300},
        {status: 'complete', result}]);
    await h.app.start('video');
    h.nodes.get('video-upload-input').files = [{ name: 'test.mp4', size: 100 }];
    await h.app.handleUpload();
    h.sources[0].onmessage({data: JSON.stringify(stats({time: 4, duration: 10}))});
    await h.app.seekVideo(20);
    await h.app.stopSession('video');
    assert.equal(h.nodes.get('res-duration').textContent, '00:00:10');
    assert.equal(h.nodes.get('res-total-hits').textContent, 2);
    assert.equal(h.nodes.get('review-progress-bar').value, 50);
    assert.equal(h.nodes.get('review-progress').hidden, true);
    assert.ok(h.nodes.get('view-results').classList.contains('active'));
    assert.equal(h.calls.filter(call => call.url === '/review_video').length, 2);
    assert.equal(h.calls.filter(call => call.url === '/session_summary/video').length, 0);
});

test('failed full video review keeps the workspace available for retry', async () => {
    const h = harness([{status: 'error', error: 'Video decoding failed'}]);
    await h.app.start('video');
    h.nodes.get('video-upload-input').files = [{name: 'test.mp4', size: 100}];
    await h.app.handleUpload();
    await h.app.stopSession('video');
    assert.ok(h.nodes.get('view-video').classList.contains('active'));
    assert.equal(h.nodes.get('btn-video-finish').disabled, false);
    assert.equal(h.nodes.get('toast').textContent, 'Video decoding failed');
});
