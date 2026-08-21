import { useEffect, useSyncExternalStore } from 'react';

// The mark is a 12x12 attention matrix, not an asset. Column 0 is the attention sink,
// the diagonal is causal/local attention and the last column is recency — the three
// canonical head types, which together draw an N.
const RESOLUTION = 12;
const SPREAD = 0.9;
const RAMP_TOKENS = ['--logo-0', '--logo-1', '--logo-2'];

const gaussian = (d, s) => Math.exp(-(d * d) / (2 * s * s));
const clamp01 = (value) => Math.min(1, Math.max(0, value));

const attention = (i, j, n, s) => Math.max(
  gaussian(j, s * 0.85),
  gaussian(j - (n - 1), s * 0.85),
  gaussian(i - j, s),
);

function hexToRgb(hex) {
  const v = parseInt(hex.trim().slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

// The palette is whatever the active theme says it is; the mark never owns a
// colour. Cached per theme: the mark repaints on a frame clock now, and reading
// three custom properties off the root is a style recalc every time. The key
// carries the system preference too, so an OS switch with no data-theme set
// still invalidates it.
let rampCache = { key: null, ramp: null };

function readRamp() {
  const key = `${document.documentElement.dataset.theme ?? 'system'}:${
    window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'}`;
  if (rampCache.key === key) return rampCache.ramp;
  const styles = getComputedStyle(document.documentElement);
  const ramp = RAMP_TOKENS.map((token) => hexToRgb(styles.getPropertyValue(token)));
  rampCache = { key, ramp };
  return ramp;
}

function sampleRamp(ramp, t) {
  const x = t * (ramp.length - 1);
  const lo = ramp[Math.floor(x)];
  const hi = ramp[Math.ceil(x)];
  const f = x - Math.floor(x);
  return `rgb(${lo.map((c, k) => Math.round(c + (hi[k] - c) * f)).join(',')})`;
}

export function paintMark(canvas, size, scale, time = null) {
  // Cells never fall below ~3px: the grid coarsens instead of the drawing shrinking.
  const n = Math.max(5, Math.min(RESOLUTION, Math.floor(size / 3)));
  const spread = SPREAD * (n / RESOLUTION);
  const ramp = readRamp();

  canvas.width = Math.round(size * scale);
  canvas.height = Math.round(size * scale);

  const ctx = canvas.getContext('2d');
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  ctx.beginPath();
  ctx.roundRect(0, 0, size, size, size * 0.16);
  ctx.clip();

  const cell = size / n;
  let hotCell = null;
  if (time !== null) {
    const phase = (time * 1.1) % 6;
    const path = Math.min(2.999, phase < 3 ? phase : 6 - phase);
    const segment = Math.floor(path);
    const position = (path - segment) * (n - 1);
    hotCell = segment === 0
      ? [n - 1 - position, 0]
      : segment === 1
        ? [position, position]
        : [n - 1 - position, n - 1];
  }

  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const pulse = hotCell
        ? 0.75 * gaussian(Math.hypot(i - hotCell[0], j - hotCell[1]), 1.1)
        : 0;
      ctx.fillStyle = sampleRamp(ramp, clamp01(attention(i, j, n, spread) + pulse));
      ctx.fillRect(Math.floor(j * cell), Math.floor(i * cell), Math.ceil(cell), Math.ceil(cell));
    }
  }
}

const subscribeTheme = (onChange) => {
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  return () => observer.disconnect();
};

export const useTheme = () => useSyncExternalStore(
  subscribeTheme,
  () => document.documentElement.dataset.theme,
);

// At 10 fps the pulse stays legible without continuously re-encoding unnecessary
// frames — and it only runs where something is reading the icon back.
export function useAtlasFavicon() {
  const theme = useTheme();

  useEffect(() => {
    const canvas = document.createElement('canvas');
    const link = document.querySelector('link[rel="icon"]');
    link.type = 'image/png';

    const startedAt = performance.now();
    const paintFrame = (time) => {
      paintMark(canvas, 32, 1, time);
      link.href = canvas.toDataURL('image/png');
    };

    // A phone has no tab strip. The icon shows up in the tab switcher, the
    // history and the bookmarks, and every one of those is a snapshot taken
    // once — nothing there re-reads the link. Ten PNG encodes a second off a
    // battery for a surface that does not exist: paint one still frame and go.
    const canBeSeen = window.matchMedia?.('(hover: hover) and (pointer: fine)').matches ?? true;
    const stillness = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    if (!canBeSeen || stillness) {
      paintFrame(null);
      return;
    }

    let timer = null;
    const stop = () => { window.clearInterval(timer); timer = null; };
    // Nor does a backgrounded tab: setInterval keeps firing there, throttled
    // but firing, and the frames it encodes are never shown.
    const run = () => {
      if (timer) return;
      timer = window.setInterval(() => paintFrame((performance.now() - startedAt) / 1000), 100);
    };
    const onVisibility = () => (document.hidden ? stop() : run());

    paintFrame(0);
    run();
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      stop();
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [theme]);
}
