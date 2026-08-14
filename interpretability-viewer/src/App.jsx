import { createContext, useContext, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { AtlasMark } from './AtlasMark';
import { useAtlasFavicon } from './atlas-mark';
import './App.css';
import { FACT_KEYS, lookupWiki } from './wiki';

const EMPTY_OBJ = {};
const BASE_URL = import.meta.env.BASE_URL ?? '/';

// Overlay display preference (heatmap composited over the original), shared by all views.
const OverlayContext = createContext({ enabled: false, opacity: 0.8 });
const useOverlay = () => useContext(OverlayContext);

// Aiming the context card at a subject is available anywhere a name is shown.
// There is one reference surface on the page and this is how anything else
// points at it.
const WikiContext = createContext({ open: () => {} });
const useWiki = () => useContext(WikiContext);

const VS_KEYS = ['mode', 'model', 'dataset', 'classId', 'imageId', 'methods'];

function readStateFromUrl() {
  const params = new URLSearchParams(window.location.search);
  return Object.fromEntries(VS_KEYS.map((k) => [k, params.get(k)]).filter(([, v]) => v != null));
}

function writeStateToUrl(vs) {
  const params = new URLSearchParams();
  for (const k of VS_KEYS) if (vs[k] != null) params.set(k, vs[k]);
  const query = params.toString();
  window.history.replaceState(null, '', query ? `?${query}` : window.location.pathname);
}

function initialTheme() {
  const saved = localStorage.getItem('theme');
  if (saved === 'light' || saved === 'dark') return saved;
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

// Apply once at module load — before first render, no flash, no effect.
document.documentElement.dataset.theme = initialTheme();

function ThemeToggle() {
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme);
  const toggle = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('theme', next);
    setTheme(next);
  };
  return (
    <button
      type="button" className="theme-toggle" onClick={toggle}
      title="Toggle theme" aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
    >{theme === 'dark' ? '☀' : '☾'}</button>
  );
}

const METHOD_CATEGORIES = {
  gradient: {
    label: 'Gradient',
    methods: [
      'GuidedGradCam', 'GradientShap', 'Saliency', 'IntegratedGradients',
      'LayerGradCam', 'InputXGradient', 'DeepLift', 'DeepLiftShap', 'GuidedBackprop',
      'Deconvolution', 'LayerIntegratedGradients',
    ],
  },
  perturbation: {
    label: 'Perturbation',
    methods: [
      'CB-RISE', 'RISE', 'Occlusion', 'KernelShap', 'Lime', 'FeaturePermutation',
      'FeatureAblation', 'ShapleyValueSampling',
    ],
  },
};

function normalize(v) {
  return String(v ?? '').toLowerCase().normalize('NFKD').replace(/[\u0300-\u036f]/g, '').trim();
}

function compareMixedIds(a, b) {
  const an = Number(a), bn = Number(b);
  return Number.isFinite(an) && Number.isFinite(bn) ? an - bn : String(a).localeCompare(String(b));
}

function categorizeMethod(name) {
  const norm = normalize(name);
  for (const [cat, { methods }] of Object.entries(METHOD_CATEGORIES)) {
    if (methods.some((m) => norm.startsWith(normalize(m)))) return cat;
  }
  return 'other';
}

// The checked methods, in the order the sidebar lists them, minus the ones
// this particular image has no output for.
function resolveMethodEntries(methods, outputs = {}) {
  return (methods ?? []).filter((m) => outputs?.[m]).map((m) => [m, outputs[m]]);
}

// Heavy image binaries live on Hugging Face datasets; JSON metadata stays local (in git).
// Set VITE_ASSET_SOURCE=local to serve everything from public/.
const HF_DATASET = 'https://huggingface.co/datasets/Matgc04';
// imagenet-pico is a *private* HF dataset. By default it's served locally (public/).
// Only when VITE_WORKER_URL is set does it get proxied through the Cloudflare Worker
// (which injects the HF token server-side).
const WORKER_ORIGIN = import.meta.env.VITE_WORKER_URL;
const HF_ROUTES = import.meta.env.VITE_ASSET_SOURCE === 'local' ? [] : [
  { test: /^imagenet-pico-ai\/val\//, base: `${HF_DATASET}/neuralatlas-imagenet-pico-ai/resolve/main/`, strip: /^imagenet-pico-ai\// },
  { test: /^outputs\/images\//, base: `${HF_DATASET}/neuralatlas-attributions/resolve/main/`, strip: /^outputs\// },
  ...(WORKER_ORIGIN ? [{ test: /^imagenet-pico\//, base: `${WORKER_ORIGIN}/hf/`, strip: /^imagenet-pico\// }] : []),
];

function resolveAssetUrl(path) {
  if (!path) return null;
  const value = String(path);
  if (/^(?:[a-z]+:)?\/\//i.test(value) || value.startsWith('data:')) return value;
  const rel = value.replace(/^\/+/, '');
  for (const route of HF_ROUTES) {
    if (route.test.test(rel)) return `${route.base}${rel.replace(route.strip, '')}`;
  }
  return `${BASE_URL}${rel}`;
}

async function fetchJson(path, options) {
  const response = await fetch(resolveAssetUrl(path), options);
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`);
  }
  return response.json();
}

function buildLegacyOutputStructure(manifest, runPayloads) {
  const structure = { models: {} };

  for (const model of manifest?.models ?? []) {
    const datasets = manifest?.datasets_by_model?.[model] ?? [];
    const modelNode = structure.models[model] ??= { datasets: {} };

    for (const dataset of datasets) {
      const runKey = `${model}::${dataset}`;
      const run = runPayloads[runKey];
      const datasetNode = modelNode.datasets[dataset] ??= { classes: {} };

      if (run?.summary?.metrics) {
        datasetNode.metrics = run.summary.metrics;
      }

      for (const image of run?.images?.images ?? []) {
        const classId = String(image.class_id);
        const imageId = String(image.image_id);
        const classNode = datasetNode.classes[classId] ??= { images: {} };
        classNode.images[imageId] = {
          outputs: image.outputs ?? {},
          prediction: image.prediction ?? null,
          interpretability_metrics: image.interpretability_metrics ?? {},
          original_url: image.original_url ?? null,
        };
      }
    }
  }

  for (const modelNode of Object.values(structure.models)) {
    const datasets = modelNode?.datasets ?? EMPTY_OBJ;
    const firstDatasetWithMetrics = Object.values(datasets).find((datasetNode) => datasetNode?.metrics);
    if (firstDatasetWithMetrics?.metrics) {
      modelNode.metrics = firstDatasetWithMetrics.metrics;
    }
  }

  return structure;
}

function buildImageRecords(outputStructure, imgCache, lblCache) {
  const records = [];
  for (const [model, { datasets = {} }] of Object.entries(outputStructure?.models ?? {})) {
    for (const [dataset, { classes = {} }] of Object.entries(datasets)) {
      const imgLookup = imgCache?.[dataset] ?? {};
      const lblLookup = lblCache?.[dataset] ?? {};
      for (const [classId, { images = {} }] of Object.entries(classes)) {
        const filenames = imgLookup[classId] ?? [];
        const classLabel = lblLookup[classId] ?? classId;
        for (const [imageId, {
          outputs = {}, prediction = null, original_url: originalUrl = null,
          interpretability_metrics: interpretabilityMetrics = {},
        } = {}] of Object.entries(images)) {
          const filename = filenames[imageId] ?? null;
          records.push({
            model, dataset, classId, classLabel, imageId, filename,
            originalUrl: originalUrl ?? (filename ? `${dataset}/val/${classId}/${filename}` : null),
            outputs,
            prediction,
            interpretabilityMetrics,
          });
        }
      }
    }
  }
  return records.sort((a, b) =>
    a.dataset.localeCompare(b.dataset) || compareMixedIds(a.classId, b.classId) ||
    compareMixedIds(a.imageId, b.imageId) || a.model.localeCompare(b.model)
  );
}

function getClassCompareMatrix(records, { dataset, classId }) {
  if (!dataset || !classId) return { models: [], rows: [] };

  const scoped = records.filter((r) => r.dataset === dataset && r.classId === classId);
  const models = [...new Set(scoped.map((r) => r.model))].sort();

  const byImage = new Map();
  for (const r of scoped) {
    if (!byImage.has(r.imageId)) byImage.set(r.imageId, new Map());
    byImage.get(r.imageId).set(r.model, r);
  }

  const rows = [...byImage.keys()].sort(compareMixedIds).map((imageId) => {
    const rowMap = byImage.get(imageId);
    const exemplar = rowMap.values().next().value;
    return {
      imageId,
      filename: exemplar?.filename ?? null,
      classId,
      classLabel: exemplar?.classLabel ?? classId,
      cells: models.map((model) => ({ model, record: rowMap.get(model) ?? null })),
    };
  });

  return { models, rows };
}

function getModelMetrics(outputStructure) {
  const models = outputStructure?.models ?? EMPTY_OBJ;
  const byModel = {};
  const byModelAndDataset = {};

  for (const [modelName, modelNode] of Object.entries(models)) {
    const datasets = modelNode?.datasets ?? EMPTY_OBJ;
    byModelAndDataset[modelName] = {};

    if (modelNode?.metrics) {
      byModel[modelName] = modelNode.metrics;
    }

    for (const [datasetName, datasetNode] of Object.entries(datasets)) {
      if (datasetNode?.metrics) {
        byModelAndDataset[modelName][datasetName] = datasetNode.metrics;
      }
    }
  }

  return { byModel, byModelAndDataset };
}

function formatMetricPercent(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
}

/* ── Reusable UI Components ─────────────────────────────────── */

const JET_LUT = (() => {
  function pw(t, stops) {
    for (let i = 0; i < stops.length - 1; i++) {
      const [x0, y0] = stops[i], [x1, y1] = stops[i + 1];
      if (t <= x1) return y0 + (y1 - y0) * (t - x0) / (x1 - x0);
    }
    return stops.at(-1)[1];
  }
  return Array.from({ length: 256 }, (_, i) => {
    const t = i / 255;
    return [
      pw(t, [[0, 0], [0.35, 0], [0.66, 1], [0.89, 1], [1, 0.5]]),
      pw(t, [[0, 0], [0.125, 0], [0.375, 1], [0.64, 1], [0.91, 0], [1, 0]]),
      pw(t, [[0, 0.5], [0.11, 1], [0.34, 1], [0.65, 0], [1, 0]]),
    ].map(v => Math.round(Math.max(0, Math.min(1, v)) * 255));
  });
})();

function applyJet(img, canvas, overlay) {
  if (!canvas || !img.naturalWidth || !img.naturalHeight) return;
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const d = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const px = d.data;
  for (let i = 0; i < px.length; i += 4) {
    const gray = px[i];
    const [r, g, b] = JET_LUT[gray];
    // Overlay: amplify low attributions via gamma so diffuse methods are visible.
    const alpha = overlay ? Math.round(Math.pow(gray / 255, 0.5) * 255) : 255;
    px[i] = r; px[i + 1] = g; px[i + 2] = b; px[i + 3] = alpha;
  }
  ctx.putImageData(d, 0, 0);
}

function JetCanvas({ src, className, alt, overlay = false, opacity }) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const overlayRef = useRef(overlay);
  overlayRef.current = overlay;
  useEffect(() => {
    if (!src) return undefined;
    const img = new Image();
    let cancelled = false;

    // Needed so the canvas can read pixels for the jet colormap when the heatmap is
    // served cross-origin (Hugging Face). HF reflects the request Origin in its CORS
    // header, so an anonymous request is allowed and the canvas stays untainted.
    img.crossOrigin = 'anonymous';
    img.decoding = 'async';
    img.onload = () => {
      if (!cancelled) {
        imageRef.current = img;
        applyJet(img, canvasRef.current, overlayRef.current);
      }
    };
    img.src = src;

    return () => {
      cancelled = true;
      imageRef.current = null;
    };
  }, [src]);

  useLayoutEffect(() => {
    if (imageRef.current) applyJet(imageRef.current, canvasRef.current, overlay);
  }, [overlay]);

  if (!src) return null;
  return (
    <canvas
      ref={canvasRef} className={className} role="img" aria-label={alt}
      style={opacity == null ? undefined : { opacity }}
    />
  );
}

// Heatmap, optionally composited over the model-view crop of the original.
function Attribution({ src, originalSrc, alt, className }) {
  const { enabled, opacity } = useOverlay();
  if (!originalSrc) return <JetCanvas className={className} src={src} alt={alt} />;
  return (
    <div className={`${className} overlay-stack`}>
      <img className="overlay-stack__base" src={originalSrc} alt="" loading="lazy" />
      <JetCanvas className="overlay-stack__heat" src={src} alt={alt} overlay={enabled} opacity={enabled ? opacity : 1} />
    </div>
  );
}

// Original shown cropped to the model's view (Resize 256 -> CenterCrop 224);
// click toggles to the full untouched image inside the same box.
function OriginalImage({ src, alt, className }) {
  const [expanded, setExpanded] = useState(false);
  if (!src) return null;
  // The zoom is a transform, which paints outside the element's own box —
  // it needs a frame to be clipped by, the same way .overlay-stack does.
  return (
    <div
      className={`${className} original-frame`}
      role="button" tabIndex={0} aria-pressed={expanded}
      onClick={() => setExpanded((v) => !v)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setExpanded((v) => !v); }
      }}
      title={expanded ? 'Full image — click to crop to model view' : 'Cropped to model view (224) — click for full image'}
    >
      <img className={`original-crop${expanded ? ' is-expanded' : ''}`} src={src} alt={alt} loading="lazy" />
    </div>
  );
}

function formatMetricBadgeValue(value) {
  return Math.abs(value) >= 1000
    ? new Intl.NumberFormat('en-US', { notation: 'compact', maximumSignificantDigits: 3 }).format(value)
    : value.toFixed(2);
}

function MetricBadges({ metrics }) {
  const definitions = {
    mif: 'Most Important First AUC',
    lif: 'Least Important First AUC',
    morph: 'Morphological faithfulness AUC',
    segment: 'Segment-wise deletion AUC',
    infidelity: 'Infidelity (lower is better)',
  };
  const items = Object.entries(definitions)
    .map(([name, title]) => ({ name, title, rawValue: metrics?.[name] }))
    .filter(({ rawValue }) => rawValue != null && rawValue !== '' && Number.isFinite(Number(rawValue)))
    .map(({ name, title, rawValue }) => ({ name, title, value: Number(rawValue) }));

  if (!items.length) return null;
  return (
    <dl className="metric-badges" aria-label="Interpretability metrics">
      {items.map(({ name, title, value }) => (
        <div key={name} className="metric-badge" title={title}>
          <dt>{name}</dt>
          <dd>{formatMetricBadgeValue(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function MiniImage({ caption, src, alt, missingText = 'Not available', variant = 'attribution', originalSrc, metrics, wikiKind }) {
  const resolvedSrc = resolveAssetUrl(src);
  // Name first, then the picture. In a wall of twenty heatmaps you scan for a
  // method by name and then look at what it produced, so the label leads.
  return (
    <figure className="mini-image">
      <figcaption>
        <span className="mini-image__name">{caption}</span>
        {wikiKind && <InfoDot kind={wikiKind} id={caption} label={caption} />}
      </figcaption>
      {!resolvedSrc
        ? <div className="mini-image__missing">{missingText}</div>
        : variant === 'original'
          ? <OriginalImage className="mini-image__asset mini-image__asset--original" src={resolvedSrc} alt={alt} />
          : <Attribution className="mini-image__asset mini-image__asset--attribution" src={resolvedSrc} originalSrc={resolveAssetUrl(originalSrc)} alt={alt} />}
      {resolvedSrc && variant !== 'original' && <MetricBadges metrics={metrics} />}
    </figure>
  );
}

// A switch, not a button: overlay is a state the maps are in, and a switch
// shows which state that is without having to read the label for a tense.
function OverlayControl({ enabled, opacity, onToggle, onOpacity }) {
  return (
    <div className="overlay-control">
      <button
        type="button" className={`switch${enabled ? ' is-on' : ''}`}
        onClick={onToggle} role="switch" aria-checked={enabled}
      >
        <span className="switch__track"><span className="switch__knob" /></span>
        <span className="switch__label">Overlay heatmap</span>
      </button>
      {enabled && (
        <label className="overlay-control__opacity">
          Opacity
          <input
            type="range" min="0" max="1" step="0.05" value={opacity}
            onChange={(e) => onOpacity(Number(e.target.value))}
          />
        </label>
      )}
    </div>
  );
}

// Horizontal, and it rides with the images instead of sitting in the sidebar:
// the scale is a property of the maps being read, not of the selection.
function ColorbarLegend() {
  return (
    <div className="colorbar-legend">
      <span className="colorbar-legend__title">Attribution</span>
      <span className="colorbar-legend__tick" aria-hidden="true">0</span>
      <span className="colorbar-legend__bar" role="img" aria-label="Attribution color scale from 0 to 1" />
      <span className="colorbar-legend__tick" aria-hidden="true">1</span>
      <span className="colorbar-legend__note">normalized per image</span>
    </div>
  );
}

// Render settings sit above the maps they change, sticky under the top bar.
function RenderBar({ overlay, opacity, onToggle, onOpacity }) {
  const ref = useRef(null);

  // The compare header sticks flush under this bar, so its offset *is* this
  // bar's height. Published from a measurement rather than declared as a
  // constant: the bar is content-sized and wraps at narrow widths, and a
  // number guessed in the stylesheet leaves a transparent seam that the maps
  // scroll through.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const publish = () => {
      const { height } = el.getBoundingClientRect();
      document.documentElement.style.setProperty('--renderbar-h', `${height}px`);
    };
    publish();
    const ro = new ResizeObserver(publish);
    ro.observe(el);
    return () => {
      ro.disconnect();
      document.documentElement.style.removeProperty('--renderbar-h');
    };
  }, []);

  return (
    <div className="render-bar" ref={ref}>
      <OverlayControl enabled={overlay} opacity={opacity} onToggle={onToggle} onOpacity={onOpacity} />
      <ColorbarLegend />
    </div>
  );
}

/* ── Reference layer ────────────────────────────────────────────
   Every model, dataset, method and metric has a glossary entry. The dot
   shows the one-liner on hover, and on click aims the context card at the
   top of the page at that entry. On touch, where there is no hover, the
   first tap stands in for it. */

function InfoDot({ kind, id, label }) {
  const entry = lookupWiki(kind, id);
  const { open } = useWiki();
  const rootRef = useRef(null);
  const anchorRef = useRef(null);
  const tipRef = useRef(null);
  const pointer = useRef(null);
  const [anchor, setAnchor] = useState(null);
  const [pos, setPos] = useState(null);
  // Which input is driving, read off the event rather than off a media query:
  // a laptop with a touchscreen is both, and what counts is the one in hand.
  const [byTouch, setByTouch] = useState(false);

  const show = () => {
    const r = anchorRef.current?.getBoundingClientRect();
    if (r) setAnchor(r);
  };
  const hide = () => { setAnchor(null); setPos(null); };

  // Placed after the tip exists, because deciding whether it fits below the dot
  // needs its measured height. useLayoutEffect, so the first painted frame is
  // already in the right place.
  useLayoutEffect(() => {
    if (!anchor || !tipRef.current) return;
    const { width, height } = tipRef.current.getBoundingClientRect();
    const below = anchor.bottom + 8;
    setPos({
      left: Math.max(8, Math.min(anchor.left, window.innerWidth - width - 8)),
      // Flip above the dot rather than run off the bottom of the window.
      top: below + height > window.innerHeight - 8
        ? Math.max(8, anchor.top - 8 - height)
        : below,
    });
  }, [anchor]);

  // A tip a finger opened has no hover to end it: the next touch anywhere else
  // closes it, and so does the page moving under it — the position is measured
  // once, in viewport coordinates, so a scroll would strand it mid-air.
  useEffect(() => {
    if (!anchor || !byTouch) return;
    const away = (e) => { if (!rootRef.current?.contains(e.target)) hide(); };
    document.addEventListener('pointerdown', away);
    window.addEventListener('scroll', hide, { passive: true, capture: true });
    return () => {
      document.removeEventListener('pointerdown', away);
      window.removeEventListener('scroll', hide, { capture: true });
    };
  }, [anchor, byTouch]);

  if (!entry) return null;

  // Touch has no hover, so the tap that would have hovered is the tap that
  // opens: the one-liner never gets read. On a finger the first tap shows it
  // and the second acts on it; a mouse or a keyboard still acts on the first.
  const act = (e) => {
    e.stopPropagation();
    const finger = pointer.current === 'touch' || pointer.current === 'pen';
    pointer.current = null;
    if (finger && !anchor) {
      setByTouch(true);
      show();
      return;
    }
    hide();
    setByTouch(false);
    open(kind, entry.id);
  };

  return (
    <span
      ref={rootRef} className="info-dot"
      /* Filtered by pointer type rather than left as onMouseEnter: touch fires
         a compatibility mouseenter before the click, which would open the tip
         and make the first tap look like the second. */
      onPointerEnter={(e) => { if (e.pointerType === 'mouse') { setByTouch(false); show(); } }}
      onPointerLeave={(e) => { if (e.pointerType === 'mouse') hide(); }}
    >
      <button
        ref={anchorRef} type="button" className="info-dot__btn"
        aria-label={`About ${label ?? entry.title}`}
        onFocus={show} onBlur={hide}
        onPointerDown={(e) => { pointer.current = e.pointerType; }}
        onClick={act}
      >i</button>
      {/* Portalled to the body. The tip is positioned in viewport coordinates,
          and any transformed ancestor — every image card carries one from its
          entry animation — would otherwise become the origin those coordinates
          resolve against and throw the tip across the page. */}
      {anchor && createPortal(
        <span
          ref={tipRef} className="info-tip" role="tooltip"
          style={{ left: pos?.left ?? -9999, top: pos?.top ?? -9999 }}
        >
          <span className="info-tip__title">{entry.title}</span>
          {entry.tags?.length > 0 && <span className="info-tip__tags">{entry.tags.join(' · ')}</span>}
          <span className="info-tip__summary">{entry.summary}</span>
          <span className="info-tip__cta">{byTouch ? 'Tap again to open it at the top' : 'Click to open it at the top'}</span>
        </span>,
        document.body,
      )}
    </span>
  );
}

const factOf = (entry, label) => entry.facts?.find(([k]) => k === label)?.[1] ?? '—';

// Segmentation variants (for example "Lime (SLIC)" and "Lime (KMeans)")
// share one glossary entry. Keep the real method id for selection, but show
// only one chip in the reference card.
function uniqueWikiMethods(methodOptions, activeMethod) {
  const grouped = new Map();
  for (const methodId of methodOptions) {
    const entry = lookupWiki('method', methodId);
    const key = entry?.id?.toLowerCase() ?? String(methodId).toLowerCase();
    const existing = grouped.get(key);
    if (!existing) {
      grouped.set(key, {
        id: methodId,
        label: entry?.title ?? methodId,
      });
    } else if (methodId === activeMethod) {
      // Keep the active segmentation as the target if it is not the first
      // variant in the selected list.
      existing.id = methodId;
    }
  }
  return [...grouped.values()];
}

/* The subject of the screen, stated above the maps: what dataset, what model,
   what method. It replaces the info dots that used to hang off the dataset and
   model crumbs — the answer is now on the page instead of one hover away, and
   the dots stay only where there is no room for a card (the method captions).
   Full-bleed across the content column, like the render strip: it captions
   everything under it, so it is as wide as what it captions. "Read more" grows
   the card downwards rather than opening the side panel — the maps below stay
   in place, they only move down. */

const CONTEXT_TABS = [
  { kind: 'dataset', label: 'Dataset' },
  { kind: 'model', label: 'Model' },
  { kind: 'method', label: 'Method' },
];

function ContextCard({
  dataset, model, method, methodOptions = [], onMethod,
  tab, onTab, expanded, onExpand, hiddenKinds,
}) {
  const ids = { dataset, model, method };
  const wikiMethodOptions = uniqueWikiMethods(methodOptions, method);

  // A tab exists only when its subject is both selectable here and documented —
  // "Across Models" has no single model, and a dataset with no entry would open
  // an empty card.
  const tabs = CONTEXT_TABS.filter((t) =>
    !hiddenKinds?.includes(t.kind) && lookupWiki(t.kind, ids[t.kind])
  );
  if (!tabs.length) return null;

  const active = tabs.find((t) => t.kind === tab) ?? tabs[0];
  const entry = lookupWiki(active.kind, ids[active.kind]);

  return (
    <section className="context-card" aria-label="Current selection reference">
      <nav className="context-card__tabs" role="tablist" aria-label="Reference subject">
        {tabs.map((t) => (
          <button
            key={t.kind} type="button" role="tab"
            aria-selected={t.kind === active.kind}
            className={`context-card__tab${t.kind === active.kind ? ' is-on' : ''}`}
            onClick={() => onTab(t.kind)}
          >{t.label}</button>
        ))}
      </nav>

      <div className="context-card__body">
        {/* Which method the card is reading. Only when more than one is
            checked — with a single method the tab label already says it. */}
        {active.kind === 'method' && wikiMethodOptions.length > 1 && (
          <div className="context-card__picker">
            {wikiMethodOptions.map(({ id, label }) => (
              <button
                key={label} type="button"
                className={`wiki-chip${id === method ? ' is-on' : ''}`}
                onClick={() => onMethod(id)}
              >{label}</button>
            ))}
          </div>
        )}

        {/* Identity on the left, the sentence in the middle, the control on
            the right — one line across the full width instead of a stack in a
            narrow column. */}
        <div className="context-card__lede">
          <div className="context-card__ident">
            <h2 className="context-card__title">{entry.title}</h2>
            <p className="context-card__tags">
              {entry.tags?.map((t) => <span key={t} className="wiki-tag">{t}</span>)}
            </p>
          </div>
          <p className="context-card__summary">{entry.summary}</p>
          <button
            type="button" className="context-card__more"
            aria-expanded={expanded}
            onClick={() => onExpand(!expanded)}
          >
            {expanded ? 'Show less' : 'Read more'}
            <svg className="context-card__chev" viewBox="0 0 10 6" aria-hidden="true">
              <path d="M1 1.5 5 5 9 1.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </div>

        {expanded && (
          <div className="context-card__detail">
            <div className="context-card__slot">
              <span className="context-card__slot-label">What makes it different</span>
              <p className="context-card__slot-text">{entry.differs}</p>
            </div>

            {/* Same fixed dials as the reference panel, laid out across the
                width instead of down a rail. */}
            <dl className="context-card__facts">
              {FACT_KEYS[active.kind].map((label) => (
                <div key={label} className="context-card__fact">
                  <dt>{label}</dt>
                  <dd>{factOf(entry, label)}</dd>
                </div>
              ))}
            </dl>

            <p className="context-card__links">
              {entry.links?.length
                ? entry.links.map((l) => (
                  <a key={l.href} href={l.href} target="_blank" rel="noreferrer noopener">{l.label}</a>
                ))
                : <span className="context-card__nolink">No external source</span>}
            </p>
          </div>
        )}
      </div>
    </section>
  );
}

/* ── Method filter ──────────────────────────────────────────────
   Checkboxes, not a single-choice dropdown: the useful question is
   almost always "these four, without that one". */

// `indeterminate` is a DOM property with no HTML attribute, so it can only be
// set imperatively.
function TriCheckbox({ checked, indeterminate, ...rest }) {
  const ref = useRef(null);
  useEffect(() => { if (ref.current) ref.current.indeterminate = Boolean(indeterminate); }, [indeterminate]);
  return <input ref={ref} type="checkbox" checked={checked} {...rest} />;
}

function MethodFilter({ groups, selected, onChange, disabled, onCollapse }) {
  const [query, setQuery] = useState('');
  const all = useMemo(() => groups.flatMap((g) => g.methods), [groups]);
  const chosen = useMemo(() => new Set(selected), [selected]);

  const q = normalize(query);
  const visible = groups
    .map((g) => ({ ...g, methods: g.methods.filter((m) => !q || normalize(m).includes(q)) }))
    .filter((g) => g.methods.length);

  const setFrom = (next) => onChange(all.filter((m) => next.has(m)));
  const toggle = (m) => {
    const next = new Set(chosen);
    if (next.has(m)) next.delete(m); else next.add(m);
    setFrom(next);
  };
  const toggleGroup = (methods, on) => {
    const next = new Set(chosen);
    for (const m of methods) if (on) next.add(m); else next.delete(m);
    setFrom(next);
  };

  return (
    <div className="method-filter">
      <div className="method-filter__head">
        <span className="tiny-form__label">Methods</span>
        <span className="method-filter__count">{selected.length}/{all.length}</span>
        {onCollapse && (
          <button type="button" className="panel-collapse-btn" onClick={onCollapse} title="Collapse panel">&#8592;</button>
        )}
      </div>

      {disabled || all.length === 0 ? (
        <p className="status-message">Choose a dataset to list its methods.</p>
      ) : (
        <>
          <input
            className="combo__input" value={query} placeholder="Filter methods"
            onChange={(e) => setQuery(e.target.value)}
          />
          <div className="method-filter__actions">
            <button type="button" onClick={() => onChange(all)}>All</button>
            <button type="button" onClick={() => onChange([])}>None</button>
          </div>
          <div className="method-filter__list">
            {visible.length === 0 && <p className="combo__empty">No matches — try a shorter query.</p>}
            {visible.map((group) => {
              const allOn = group.methods.every((m) => chosen.has(m));
              return (
                <div key={group.key} className="method-filter__group">
                  {/* One checkbox for the whole family. Partly-checked shows
                      the dash, so the group's state is readable at a glance
                      without counting its children. */}
                  <label className="method-filter__group-head">
                    <TriCheckbox
                      checked={allOn}
                      indeterminate={!allOn && group.methods.some((m) => chosen.has(m))}
                      onChange={() => toggleGroup(group.methods, !allOn)}
                      aria-label={`All ${group.label} methods`}
                    />
                    <span className="method-filter__group-label">{group.label}</span>
                  </label>
                  {/* The info dot sits outside the <label>, or clicking it
                      would toggle the checkbox on the way through. */}
                  {group.methods.map((m) => (
                    <div key={m} className="method-check">
                      <label className="method-check__label">
                        <input type="checkbox" checked={chosen.has(m)} onChange={() => toggle(m)} />
                        <span className="method-check__name">{m}</span>
                      </label>
                      <InfoDot kind="method" id={m} label={m} />
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

function MethodFigures({ methods, outputs, imageId, originalSrc, interpretabilityMetrics }) {
  const entries = resolveMethodEntries(methods, outputs);
  if (!entries.length) return <MiniImage caption="Method" missingText="No method checked in the sidebar." />;
  return entries.map(([name, url]) => (
    <MiniImage
      key={name} caption={name} src={url} originalSrc={originalSrc} wikiKind="method"
      alt={`${name} for image ${imageId}`} metrics={interpretabilityMetrics?.[name]}
    />
  ));
}

function PredictionBadge({ prediction, classId, labels }) {
  if (!prediction) return null;
  const predId = prediction.predicted_class_id;
  const predLabel = labels?.[predId] ?? `Class ${predId}`;
  const isCorrect = String(predId) === String(classId);
  const confidencePct = prediction.confidence == null
    ? null
    : `${(Number(prediction.confidence) * 100).toFixed(1)}%`;
  // A tinted pill, because a bare line of text floated loose under the image
  // with nothing tying it to the picture. It states a fact and is never a
  // button: no fill saturated enough to read as one, and no hover state.
  return (
    <p
      className={`prediction prediction--${isCorrect ? 'correct' : 'incorrect'}`}
      title={`Predicted class ${predId} \u2014 ${isCorrect ? 'matches' : 'does not match'} class ${classId}`}
    >
      <span
        className="prediction__icon" role="img"
        aria-label={isCorrect ? 'Correct prediction' : 'Incorrect prediction'}
      >
        {isCorrect ? '\u2713' : '\u2717'}
      </span>
      <span className="prediction__label">{predLabel}</span>
      {confidencePct && <span className="prediction__confidence">{confidencePct}</span>}
    </p>
  );
}

function EmptyState({ title, description }) {
  return (
    <div className="empty-state-shell">
      <div className="empty-state">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

// Loading and failure are inline sentences that name the next action —
// no spinner, no skeleton, no toast.
function AppStatus({ children }) {
  return (
    <div className="app-status">
      <p className="status-message" role="status" aria-live="polite">{children}</p>
    </div>
  );
}

// A label and a hairline. Replaces the heading-plus-blurb block that used to
// sit above every section and say nothing the reader didn't already know.
function SectionRule({ label, children }) {
  return (
    <div className="section-rule">
      <span className="section-rule__label">{label}</span>
      <span className="section-rule__line" aria-hidden="true" />
      {children}
    </div>
  );
}

// Stats read left to right on hairlines, not as six floating cards.
function ModelStatsRail({ model, dataset, stats }) {
  if (!model || !dataset || !stats) return null;

  const items = [
    { label: 'Samples', value: stats.total },
    { label: 'Correct', value: stats.correct },
    { label: 'Accuracy', value: formatMetricPercent(stats.accuracy) },
    { label: 'Precision', value: formatMetricPercent(stats.macroPrecision) },
    { label: 'Recall', value: formatMetricPercent(stats.macroRecall) },
    { label: 'F1', value: formatMetricPercent(stats.macroF1) },
  ];

  return (
    <section className="stat-rail" aria-label="Selected model metrics">
      <div className="stat-rail__run">
        <span className="stat-rail__label">Run</span>
        <strong className="stat-rail__run-name">
          {model}<span> / {dataset}</span>
        </strong>
      </div>
      <dl className="stat-rail__stats">
        {items.map((item) => (
          <div key={item.label} className="stat-rail__stat">
            <dt className="stat-rail__label">{item.label}</dt>
            <dd className="stat-rail__value">{item.value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function ModeSwitcher({ value, onChange }) {
  const items = [
    { value: 'single', label: 'Single' },
    { value: 'model_grid', label: 'By Model' },
    { value: 'class_compare', label: 'Across Models' },
  ];
  return (
    <div className="mode-switcher" role="tablist" aria-label="Viewer mode">
      {items.map((item) => (
        <button
          key={item.value} type="button" role="tab"
          aria-selected={value === item.value}
          className={`mode-switcher__tab${value === item.value ? ' mode-switcher__tab--active' : ''}`}
          onClick={() => onChange(item.value)}
        >{item.label}</button>
      ))}
    </div>
  );
}

// Mode is navigation, not a filter — it belongs in the chrome with the
// wordmark and the live readout, not stacked on top of the dropdowns.
function TopBar({ mode, onModeChange, readout, wikiOn, onToggleWiki }) {
  return (
    <header className="topbar">
      <span className="topbar__brand">
        <AtlasMark size={22} />
        <span className="topbar__name">NeuralAtlas</span>
      </span>
      <ModeSwitcher value={mode} onChange={onModeChange} />
      <p className="topbar__readout" role="status" aria-live="polite">{readout}</p>
      <button
        type="button" className={`topbar__wiki${wikiOn ? ' is-on' : ''}`}
        onClick={onToggleWiki} aria-pressed={wikiOn}
        title={wikiOn ? 'Hide reference' : 'Show reference'}
        aria-label={wikiOn ? 'Hide reference' : 'Show reference'}
      >
        {/* An open book, not the word: it sits next to the theme glyph and
            reads as a way out of the page at a glance. */}
        <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M10 5.6C8.6 4.3 6.6 3.8 3.9 4a1 1 0 0 0-.9 1v8.7a1 1 0 0 0 1 1c2.6-.2 4.5.3 6 1.5 1.5-1.2 3.4-1.7 6-1.5a1 1 0 0 0 1-1V5a1 1 0 0 0-.9-1c-2.7-.2-4.7.3-6.1 1.6Z" />
          <path d="M10 5.6v10.6" />
        </svg>
      </button>
      <ThemeToggle />
    </header>
  );
}

/* One step of the selection path. Closed it is a label and its current value —
   what is on screen, stated in one line. Open it is the same searchable list
   that used to live in the rail, just brought to where the answer is read. */
function CrumbSelect({ label, value, items, onSelect, placeholder, disabled }) {
  const inputId = useId();
  const list = (items ?? []).map((i) => typeof i === 'string' ? { value: i, label: i } : i);
  const selectedLabel = value == null ? '' : list.find((i) => i.value === value)?.label ?? String(value);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const rootRef = useRef(null);
  const inputRef = useRef(null);

  const q = normalize(query);
  const filtered = q ? list.filter((i) => !i.isHeader && normalize(i.label).includes(q)) : list;

  const commit = (v) => { onSelect(v); setOpen(false); setQuery(''); };

  // Outside-click rather than blur: the trigger is a button, and blur-to-close
  // would fight the click that is trying to toggle it shut.
  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => { if (!rootRef.current?.contains(e.target)) setOpen(false); };
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  useEffect(() => { if (open) inputRef.current?.focus(); }, [open]);

  return (
    <div className={`crumb${disabled ? ' is-disabled' : ''}`} ref={rootRef}>
      <button
        type="button" id={inputId}
        className={`crumb__trigger${open ? ' is-open' : ''}`}
        disabled={disabled}
        aria-haspopup="listbox" aria-expanded={open}
        onClick={() => { setOpen((o) => !o); setQuery(''); }}
      >
        <span className="crumb__label">{label}</span>
        <span className="crumb__line">
          <span className="crumb__value">{selectedLabel || placeholder}</span>
          {/* The caret is the only thing saying "this opens". Without it the
              crumb reads as a printed caption. */}
          <svg className="crumb__caret" viewBox="0 0 10 6" aria-hidden="true">
            <path d="M1 1.5 5 5 9 1.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
      </button>
      {open && !disabled && (
        <div className="crumb__pop">
          <input
            ref={inputRef} className="combo__input" value={query}
            placeholder={placeholder} aria-label={`Search ${label}`}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') { const f = filtered.find((i) => !i.isHeader); if (f) commit(f.value); }
            }}
          />
          <div className="crumb__list" role="listbox">
            {filtered.length === 0 ? (
              <div className="combo__empty">No matches — try a shorter query.</div>
            ) : filtered.slice(0, 250).map((item) =>
              item.isHeader ? (
                <div key={item.value} className="combo__group-header">{item.label}</div>
              ) : (
                <button
                  type="button" key={item.value} role="option"
                  aria-selected={item.value === value}
                  className={`combo__option${item.value === value ? ' is-on' : ''}`}
                  onClick={() => commit(item.value)}
                >{item.label}</button>
              )
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* The selection — model, dataset, class, image — in the chrome, where it reads
   as the caption for the whole screen instead of four dropdowns in a rail. */
function SelectionBar({ children }) {
  return <div className="selection-bar">{children}</div>;
}

/* ── View Components ────────────────────────────────────────── */

function SingleImageGallery({ imageData, labels }) {
  if (!imageData?.original) {
    return <EmptyState title="No image selected yet" description="Choose a model, dataset, class, and image to inspect." />;
  }

  const outputs = Object.entries(imageData.outputs ?? {});
  return (
    <section className="image-gallery">
      <SectionRule label={`Original + ${outputs.length} attribution${outputs.length === 1 ? '' : 's'}`} />
      <div className="gallery-grid">
        {/* The verdict belongs to the photograph, not to the section: it is a
            fact about this image, so it hangs under this image. */}
        <div className="image-card" style={{ '--delay': '80ms' }}>
          <div className="image-card__header"><h3>Image</h3></div>
          <OriginalImage className="image-card__image image-card__image--original" src={resolveAssetUrl(imageData.original)} alt="Original selection" />
          <PredictionBadge prediction={imageData.prediction} classId={imageData.classId} labels={labels} />
        </div>
        {outputs.map(([method, url]) => (
          <div key={method} className="image-card">
            <div className="image-card__header">
              <h3>{method}</h3>
              <InfoDot kind="method" id={method} label={method} />
            </div>
            <Attribution className="image-card__image image-card__image--attribution" src={resolveAssetUrl(url)} originalSrc={resolveAssetUrl(imageData.original)} alt={`${method} explanation`} />
            <MetricBadges metrics={imageData.interpretabilityMetrics?.[method]} />
          </div>
        ))}
      </div>
    </section>
  );
}

function ModelGridView({ records, methods, ready, labels }) {
  if (!ready) return <EmptyState title="Choose model and dataset" description="Select a model and dataset to browse many images at once." />;
  if (!records.length) return <EmptyState title="No images match your filters" description="Try a different class selection to widen the result set." />;

  return (
    <section className="model-grid" aria-label="Model image grid">
      {records.map((record) => (
        <article key={`${record.model}__${record.dataset}__${record.classId}__${record.imageId}`} className="model-grid-card">
          <header className="model-grid-card__meta">
            <h3>Class {record.classId} - {record.classLabel}</h3>
            <p>Image {record.imageId}{record.filename ? ` - ${record.filename}` : ''}</p>
            <PredictionBadge prediction={record.prediction} classId={record.classId} labels={labels} />
          </header>
          <div className="model-grid-card__images">
            <MiniImage caption="Original" src={record.originalUrl} alt={`Original ${record.imageId}`} missingText="Original unavailable" variant="original" />
            <MethodFigures
              methods={methods} outputs={record.outputs} imageId={record.imageId}
              originalSrc={record.originalUrl} interpretabilityMetrics={record.interpretabilityMetrics}
            />
          </div>
        </article>
      ))}
    </section>
  );
}

function ClassCompareView({ matrix, methods, ready, labels }) {
  if (!ready) return <EmptyState title="Choose dataset and class" description="Pick a dataset and class to align the same image IDs across models." />;
  if (!matrix.rows.length) return <EmptyState title="No aligned rows found" description="No images are available for this class across the selected models." />;

  const style = { '--compare-columns': matrix.models.length + 1 };
  return (
    <section className="compare-matrix" aria-label="Class comparison matrix">
      <div className="compare-header" style={style}>
        <div className="compare-header__cell compare-header__cell--original">Original</div>
        {matrix.models.map((m) => (
          <div key={m} className="compare-header__cell">{m}</div>
        ))}
      </div>
      {matrix.rows.map((row) => {
        const origUrl = row.cells.find((c) => c.record?.originalUrl)?.record?.originalUrl ?? null;
        return (
          <div key={row.imageId} className="compare-row" style={style}>
            <article className="compare-cell compare-cell--original">
              <MiniImage caption={`Image ${row.imageId}`} src={origUrl} alt={`Original image ${row.imageId}`} missingText="Original unavailable" variant="original" />
            </article>
            {row.cells.map((cell) => (
              <article key={`${row.imageId}__${cell.model}`} className="compare-cell" data-model={cell.model}>
                <PredictionBadge prediction={cell.record?.prediction} classId={row.classId} labels={labels} />
                <MethodFigures
                  methods={methods} outputs={cell.record?.outputs ?? {}} imageId={row.imageId}
                  originalSrc={cell.record?.originalUrl}
                  interpretabilityMetrics={cell.record?.interpretabilityMetrics}
                />
              </article>
            ))}
          </div>
        );
      })}
    </section>
  );
}

function resolveSelection(options, currentValue) {
  if (!options.length) return null;
  if (currentValue && options.some((o) => (o.value ?? o) === currentValue)) return currentValue;
  return options[0].value ?? options[0];
}

/* ── Main Form ──────────────────────────────────────────────── */

function ModelForm({ outputStructure }) {
  const modelsStruct = outputStructure?.models ?? EMPTY_OBJ;

  const [vs, setVs] = useState(() => ({
    mode: 'single', model: null, dataset: null, classId: null, imageId: null, methods: null,
    ...readStateFromUrl(),
  }));
  const [panelCollapsed, setPanelCollapsed] = useState(false);
  const [overlay, setOverlay] = useState(false);
  const [overlayOpacity, setOverlayOpacity] = useState(0.8);
  const [imgCache, setImgCache] = useState({});
  const [lblCache, setLblCache] = useState({});
  const [dsStatus, setDsStatus] = useState({});

  const patch = (update) => { const next = { ...vs, ...update }; setVs(next); writeStateToUrl(next); };

  const modelOptions = useMemo(() => Object.keys(modelsStruct).sort(), [modelsStruct]);

  const effectiveModel = resolveSelection(modelOptions, vs.model);

  const datasetOptions = useMemo(() => {
    if (!effectiveModel) return [];
    return Object.keys(modelsStruct[effectiveModel]?.datasets ?? {}).sort(compareMixedIds);
  }, [effectiveModel, modelsStruct]);

  const effectiveDataset = resolveSelection(datasetOptions, vs.dataset);

  // Load dataset metadata (images structure + labels)
  useEffect(() => {
    if (!effectiveDataset) return;
    const controller = new AbortController();
    const { signal } = controller;
    const ds = effectiveDataset;
    const updStatus = (fields) =>
      !signal.aborted && setDsStatus((p) => ({ ...p, [ds]: { ...p[ds], ...fields } }));

    if (!imgCache[ds]) {
      updStatus({ imagesLoading: true, imagesError: null });
      fetchJson(`${ds}/${ds}_structure.json`, { signal })
        .then((data) => { if (!signal.aborted) setImgCache((p) => ({ ...p, [ds]: data })); })
        .catch((e) => { if (e.name !== 'AbortError') updStatus({ imagesError: 'Failed to load.' }); })
        .finally(() => updStatus({ imagesLoading: false }));
    }

    if (!lblCache[ds]) {
      updStatus({ labelsLoading: true, labelsError: null });
      fetchJson('imagenet-mini/imagenet-1k-id2label.json', { signal })
        .then((data) => { if (!signal.aborted) setLblCache((p) => ({ ...p, [ds]: data })); })
        .catch((e) => {
          if (e.name !== 'AbortError') {
            setLblCache((p) => ({ ...p, [ds]: {} }));
            updStatus({ labelsError: 'Failed to load.' });
          }
        })
        .finally(() => updStatus({ labelsLoading: false }));
    }

    return () => controller.abort();
  }, [effectiveDataset, imgCache, lblCache]);

  const imageRecords = useMemo(
    () => buildImageRecords(outputStructure, imgCache, lblCache),
    [outputStructure, imgCache, lblCache]
  );

  const modelMetrics = useMemo(
    () => getModelMetrics(outputStructure),
    [outputStructure]
  );

  const classOptions = useMemo(() => {
    if (!effectiveDataset || !effectiveModel) return [];
    const labels = lblCache[effectiveDataset] ?? {};
    const classIds = Object.keys(modelsStruct[effectiveModel]?.datasets?.[effectiveDataset]?.classes ?? {});
    return classIds.sort(compareMixedIds).map((id) => ({ value: id, label: `${id} - ${labels[id] ?? id}` }));
  }, [effectiveDataset, effectiveModel, lblCache, modelsStruct]);

  const effectiveClassId = resolveSelection(classOptions, vs.classId);

  const imageOptions = useMemo(() => {
    if (!effectiveModel || !effectiveDataset || !effectiveClassId) return [];
    const images = modelsStruct[effectiveModel]?.datasets?.[effectiveDataset]?.classes?.[effectiveClassId]?.images ?? {};
    const filenames = imgCache[effectiveDataset]?.[effectiveClassId] ?? [];
    return Object.keys(images).sort(compareMixedIds).map((id) => ({
      value: id, label: filenames[id] ? `${id} - ${filenames[id]}` : id,
    }));
  }, [effectiveModel, effectiveDataset, effectiveClassId, modelsStruct, imgCache]);

  const effectiveImageId = vs.mode === 'single' ? resolveSelection(imageOptions, vs.imageId) : null;

  const methodGroups = useMemo(() => {
    if (!effectiveDataset) return [];
    const methods = new Set();
    for (const r of imageRecords) if (r.dataset === effectiveDataset) Object.keys(r.outputs).forEach((m) => methods.add(m));
    const sorted = [...methods].sort();

    const grouped = {};
    for (const m of sorted) { const c = categorizeMethod(m); (grouped[c] ??= []).push(m); }
    return [...Object.keys(METHOD_CATEGORIES), 'other']
      .filter((c) => grouped[c]?.length)
      .map((c) => ({ key: c, label: METHOD_CATEGORIES[c]?.label ?? 'Other', methods: grouped[c] }));
  }, [imageRecords, effectiveDataset]);

  const availableMethods = useMemo(() => methodGroups.flatMap((g) => g.methods), [methodGroups]);

  // Absent from the URL means "all of them", so a link keeps working when the
  // dataset gains a method. An explicit list is intersected with what exists.
  const selectedMethods = useMemo(() => {
    if (vs.methods == null) return availableMethods;
    const wanted = new Set(vs.methods.split(',').filter(Boolean));
    return availableMethods.filter((m) => wanted.has(m));
  }, [vs.methods, availableMethods]);

  // The context card is the only reference surface on the page, so its state
  // lives here: which subject it reads, whether the full entry is open, and
  // whether it is on screen at all (the book in the top bar).
  const [contextTab, setContextTab] = useState('model');
  const [contextOpen, setContextOpen] = useState(false);
  const [contextShown, setContextShown] = useState(true);
  const [pickedMethod, setPickedMethod] = useState(null);
  // A method named from the sidebar may not be checked, so it is enough that
  // it be documented — otherwise fall back to the first one on screen.
  const contextMethod = pickedMethod && lookupWiki('method', pickedMethod)
    ? pickedMethod
    : selectedMethods[0] ?? null;

  // Anything that names a subject — an info dot on a method caption, a
  // checkbox in the rail — aims the card at it and opens the full entry.
  const wikiApi = useMemo(() => ({
    open: (kind, id) => {
      setContextShown(true);
      setContextOpen(true);
      if (kind) setContextTab(kind);
      if (kind === 'method' && id) setPickedMethod(id);
    },
  }), []);

  // Without a nudge the card would keep reading whatever was last opened while
  // the selection moved on underneath it, so it follows the last thing touched.
  // Checking a whole family at once is left alone — twenty added methods name
  // no single subject.
  const lastSelection = useRef({ model: null, dataset: null, methods: [] });
  useEffect(() => {
    const prev = lastSelection.current;
    lastSelection.current = { model: effectiveModel, dataset: effectiveDataset, methods: selectedMethods };
    if (prev.model == null && prev.dataset == null) return; // first pass, nothing was touched
    if (effectiveModel !== prev.model) setContextTab('model');
    else if (effectiveDataset !== prev.dataset) { setContextTab('dataset'); setPickedMethod(null); }
    else {
      const added = selectedMethods.filter((m) => !prev.methods.includes(m));
      if (added.length === 1) { setContextTab('method'); setPickedMethod(added[0]); }
    }
  }, [effectiveModel, effectiveDataset, selectedMethods]);

  const setSelectedMethods = (list) => {
    const next = availableMethods.filter((m) => list.includes(m));
    patch({ methods: next.length === availableMethods.length ? null : next.join(',') });
  };

  const singleImageData = useMemo(() => {
    if (vs.mode !== 'single' || !effectiveModel || !effectiveDataset || !effectiveClassId || !effectiveImageId) return null;
    const r = imageRecords.find((i) =>
      i.model === effectiveModel && i.dataset === effectiveDataset && i.classId === effectiveClassId && i.imageId === effectiveImageId
    );
    if (!r) return null;
    return {
      original: r.originalUrl,
      outputs: Object.fromEntries(resolveMethodEntries(selectedMethods, r.outputs)),
      interpretabilityMetrics: r.interpretabilityMetrics,
      prediction: r.prediction,
      classId: r.classId,
    };
  }, [imageRecords, vs.mode, effectiveModel, effectiveDataset, effectiveClassId, effectiveImageId, selectedMethods]);

  const modelGridRecords = useMemo(() => {
    if (!effectiveModel || !effectiveDataset) return [];
    return imageRecords.filter((r) =>
      r.model === effectiveModel && r.dataset === effectiveDataset &&
      (!effectiveClassId || r.classId === effectiveClassId)
    );
  }, [imageRecords, effectiveModel, effectiveDataset, effectiveClassId]);

  const classCompareMatrix = useMemo(
    () => getClassCompareMatrix(imageRecords, { dataset: effectiveDataset, classId: effectiveClassId }),
    [imageRecords, effectiveDataset, effectiveClassId]
  );

  const selectedModelStats = effectiveModel && effectiveDataset
    ? modelMetrics.byModelAndDataset[effectiveModel]?.[effectiveDataset] ?? null
    : null;

  const hasContent =
    (vs.mode === 'single' && Boolean(singleImageData?.original)) ||
    (vs.mode === 'model_grid' && modelGridRecords.length > 0) ||
    (vs.mode === 'class_compare' && classCompareMatrix.rows.length > 0);

  const dsInfo = effectiveDataset ? dsStatus[effectiveDataset] ?? {} : {};
  const isLoading = dsInfo.imagesLoading || dsInfo.labelsLoading;

  const handleModeChange = (mode) => {
    const next = { ...vs, mode };
    if (mode !== 'single') next.imageId = null;
    setVs(next);
    writeStateToUrl(next);
  };

  const methodLabel = (() => {
    if (!availableMethods.length) return 'no methods';
    if (!selectedMethods.length) return 'no method checked';
    if (selectedMethods.length === 1) return selectedMethods[0];
    if (selectedMethods.length === availableMethods.length) return `all ${availableMethods.length} methods`;
    return `${selectedMethods.length} of ${availableMethods.length} methods`;
  })();

  // The readout sits in the top bar next to the mode tabs, so it never has to
  // name the mode — only what the current selection resolves to.
  const summaryText = (() => {
    if (vs.mode === 'single') {
      if (!singleImageData?.original) return 'Complete the selection to begin';
      return `1 image · ${Object.keys(singleImageData.outputs ?? {}).length} attributions`;
    }
    if (vs.mode === 'model_grid') {
      if (!effectiveModel || !effectiveDataset) return 'Select model and dataset';
      const cls = effectiveClassId ? `class ${effectiveClassId}` : 'all classes';
      return `${modelGridRecords.length} images · ${cls} · ${methodLabel}`;
    }
    if (!effectiveDataset || !effectiveClassId) return 'Select dataset and class';
    return `${classCompareMatrix.rows.length} rows · ${classCompareMatrix.models.length} models · ${methodLabel}`;
  })();

  return (
    <OverlayContext.Provider value={{ enabled: overlay, opacity: overlayOpacity }}>
    <WikiContext.Provider value={wikiApi}>
    <TopBar
      mode={vs.mode} onModeChange={handleModeChange} readout={summaryText}
      wikiOn={contextShown} onToggleWiki={() => setContextShown((v) => !v)}
    />
    <SelectionBar>
      {vs.mode !== 'class_compare' && (
        <CrumbSelect
          label="Model" value={effectiveModel} items={modelOptions}
          onSelect={(v) => patch({ model: v, classId: null, imageId: null })}
          placeholder="Search model" disabled={!modelOptions.length}
        />
      )}

      <CrumbSelect
        label="Dataset" value={effectiveDataset}
        items={datasetOptions}
        onSelect={(v) => patch({ dataset: v })}
        placeholder="Search dataset"
        disabled={!datasetOptions.length}
      />

      <CrumbSelect
        label="Class" value={effectiveClassId} items={classOptions}
        onSelect={(v) => patch({ classId: v, imageId: null })}
        placeholder={isLoading ? 'Loading class metadata...' : 'Search class'}
        disabled={!effectiveDataset || !classOptions.length}
      />

      {vs.mode === 'single' && (
        <CrumbSelect
          label="Image" value={effectiveImageId} items={imageOptions}
          onSelect={(v) => patch({ imageId: v })}
          placeholder="Search image id / filename"
          disabled={!effectiveClassId}
        />
      )}
    </SelectionBar>
    {(dsInfo.imagesError || dsInfo.labelsError) && (
      <div className="selection-status">
        <p className="status-message" role="status">Some dataset metadata failed to load.</p>
      </div>
    )}
    <div className={`viewer-layout${panelCollapsed ? ' viewer-layout--collapsed' : ''}`}>
      <aside className={`controls-panel${panelCollapsed ? ' controls-panel--collapsed' : ''}`}>
        {panelCollapsed ? (
          <button className="panel-toggle" onClick={() => setPanelCollapsed(false)} title="Expand controls">
            <span className="panel-toggle__icon">&#8594;</span>
            <span className="panel-toggle__label">Methods</span>
          </button>
        ) : (
          <MethodFilter
            groups={methodGroups} selected={selectedMethods}
            onChange={setSelectedMethods} disabled={!effectiveDataset}
            onCollapse={hasContent ? () => setPanelCollapsed(true) : null}
          />
        )}
      </aside>

      <main className="viewer-content">
        {/* The card names what is on screen; the render strip sets how it is
            drawn. Reading order follows: subject first, then controls, then
            the maps they act on. */}
        {contextShown && (
          <ContextCard
            dataset={effectiveDataset} model={effectiveModel} method={contextMethod}
            methodOptions={selectedMethods} onMethod={setPickedMethod}
            tab={contextTab} onTab={setContextTab}
            expanded={contextOpen} onExpand={setContextOpen}
            hiddenKinds={vs.mode === 'class_compare' ? ['model'] : undefined}
          />
        )}

        <RenderBar
          overlay={overlay} opacity={overlayOpacity}
          onToggle={() => setOverlay((v) => !v)} onOpacity={setOverlayOpacity}
        />

        {vs.mode === 'model_grid' && (
          <ModelStatsRail model={effectiveModel} dataset={effectiveDataset} stats={selectedModelStats} />
        )}

        {vs.mode === 'single' && <SingleImageGallery imageData={singleImageData} labels={lblCache[effectiveDataset]} />}
        {vs.mode === 'model_grid' && (
          <ModelGridView records={modelGridRecords} methods={selectedMethods} ready={Boolean(effectiveModel && effectiveDataset)} labels={lblCache[effectiveDataset]} />
        )}
        {vs.mode === 'class_compare' && (
          <ClassCompareView matrix={classCompareMatrix} methods={selectedMethods}
            ready={Boolean(effectiveDataset && effectiveClassId)} labels={lblCache[effectiveDataset]} />
        )}
      </main>
    </div>
    </WikiContext.Provider>
    </OverlayContext.Provider>
  );
}

async function loadOutputStructure(signal) {
  const manifest = await fetchJson('outputs/manifest.json', { signal });

  const entries = Object.entries(manifest?.runs ?? {}).flatMap(([model, datasets]) =>
    Object.entries(datasets ?? {}).map(([dataset, paths]) => ({ model, dataset, paths }))
  );

  const runPayloads = Object.fromEntries(
    await Promise.all(
      entries.map(async ({ model, dataset, paths }) => {
        const [images, summary] = await Promise.all([
          fetchJson(paths.images, { signal }),
          fetchJson(paths.summary, { signal }),
        ]);
        return [`${model}::${dataset}`, { images, summary }];
      })
    )
  );

  return buildLegacyOutputStructure(manifest, runPayloads);
}

function App() {
  const [outputStructure, setOutputStructure] = useState(null);
  const [error, setError] = useState(null);

  useAtlasFavicon();

  useEffect(() => {
  const controller = new AbortController();

  loadOutputStructure(controller.signal)
    .then(setOutputStructure)
    .catch((e) => {
      if (e.name !== 'AbortError') setError(e);
    });

  return () => controller.abort();
}, []);

  if (error) return <AppStatus>Could not read outputs/manifest.json. Check that the run outputs are published, then reload.</AppStatus>;
  if (!outputStructure) return <AppStatus>Reading run manifest and attribution metadata.</AppStatus>;

  return <div className="app-shell"><ModelForm outputStructure={outputStructure} /></div>;
}

export default App;
