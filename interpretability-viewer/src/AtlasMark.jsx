import { useEffect, useRef } from 'react';
import { paintMark, useTheme } from './atlas-mark';

export function AtlasMark({ size = 32 }) {
  const ref = useRef(null);
  const theme = useTheme();

  // The wordmark carries the animation on a phone, where the favicon it used
  // to live in is never on screen. Thirty frames a second is more than a
  // seven-cell grid can show, and requestAnimationFrame stops on its own when
  // the tab goes to the background.
  useEffect(() => {
    const canvas = ref.current;
    const scale = window.devicePixelRatio || 1;
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
      paintMark(canvas, size, scale);
      return;
    }

    const startedAt = performance.now();
    let painted = 0;
    let frame = requestAnimationFrame(function tick(now) {
      frame = requestAnimationFrame(tick);
      if (now - painted < 1000 / 30) return;
      painted = now;
      paintMark(canvas, size, scale, (now - startedAt) / 1000);
    });
    return () => cancelAnimationFrame(frame);
  }, [size, theme]);

  return (
    <canvas
      ref={ref} className="atlas-mark" style={{ width: size, height: size }}
      role="img" aria-label="NeuralAtlas"
    />
  );
}
