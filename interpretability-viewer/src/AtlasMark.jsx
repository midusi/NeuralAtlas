import { useEffect, useRef } from 'react';
import { paintMark, useTheme } from './atlas-mark';

export function AtlasMark({ size = 32 }) {
  const ref = useRef(null);
  const theme = useTheme();

  useEffect(() => {
    paintMark(ref.current, size, window.devicePixelRatio || 1);
  }, [size, theme]);

  return (
    <canvas
      ref={ref} className="atlas-mark" style={{ width: size, height: size }}
      role="img" aria-label="NeuralAtlas"
    />
  );
}
