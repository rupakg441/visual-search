import { Crop, RotateCcw, Trash2 } from 'lucide-react';
import { useRef, useState, type PointerEvent } from 'react';

interface ImageCropperProps {
  imageUrl: string;
  onApplyCrop: (croppedBlob: Blob) => void;
  onResetCrop: () => void;
  onClearImage: () => void;
  disabled?: boolean;
}

interface CropRect {
  x: number; // percentage 0..100
  y: number;
  w: number;
  h: number;
}

const DEFAULT_CROP: CropRect = { x: 10, y: 10, w: 80, h: 80 };

export function ImageCropper({
  imageUrl,
  onApplyCrop,
  onResetCrop,
  onClearImage,
  disabled,
}: ImageCropperProps) {
  const imgRef = useRef<HTMLImageElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const [cropping, setCropping] = useState(false);
  const [rect, setRect] = useState<CropRect>(DEFAULT_CROP);
  const dragStart = useRef<{ px: number; py: number; rect: CropRect } | null>(null);

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (!cropping) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    dragStart.current = { px: e.clientX, py: e.clientY, rect };
  };

  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (!dragStart.current || !stageRef.current) return;
    const bounds = stageRef.current.getBoundingClientRect();
    const dxPct = ((e.clientX - dragStart.current.px) / bounds.width) * 100;
    const dyPct = ((e.clientY - dragStart.current.py) / bounds.height) * 100;

    const nextX = Math.max(0, Math.min(100 - dragStart.current.rect.w, dragStart.current.rect.x + dxPct));
    const nextY = Math.max(0, Math.min(100 - dragStart.current.rect.h, dragStart.current.rect.y + dyPct));
    setRect({ ...dragStart.current.rect, x: nextX, y: nextY });
  };

  const onPointerUp = () => {
    dragStart.current = null;
  };

  const applyCrop = () => {
    const img = imgRef.current;
    if (!img || !img.naturalWidth) return;

    const sx = Math.round((rect.x / 100) * img.naturalWidth);
    const sy = Math.round((rect.y / 100) * img.naturalHeight);
    const sw = Math.max(16, Math.round((rect.w / 100) * img.naturalWidth));
    const sh = Math.max(16, Math.round((rect.h / 100) * img.naturalHeight));

    const canvas = document.createElement('canvas');
    canvas.width = sw;
    canvas.height = sh;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
    canvas.toBlob(
      (blob) => {
        if (blob) {
          onApplyCrop(blob);
          setCropping(false);
        }
      },
      'image/jpeg',
      0.92,
    );
  };

  return (
    <div>
      <div className="crop-stage" ref={stageRef}>
        <img ref={imgRef} src={imageUrl} alt="Query jewelry preview" />
        {cropping && (
          <div
            className="crop-box"
            role="region"
            aria-label="Drag crop area"
            style={{
              left: `${rect.x}%`,
              top: `${rect.y}%`,
              width: `${rect.w}%`,
              height: `${rect.h}%`,
            }}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
          />
        )}
      </div>

      {cropping && (
        <div style={{ marginTop: '0.6rem', fontSize: '0.8rem', color: '#5c5c57' }}>
          <label htmlFor="crop-size-slider" style={{ display: 'block', marginBottom: '0.25rem' }}>
            Crop box size: {rect.w}%
          </label>
          <input
            id="crop-size-slider"
            type="range"
            min={30}
            max={95}
            value={rect.w}
            onChange={(e) => {
              const size = Number(e.target.value);
              setRect((prev) => ({
                x: Math.min(prev.x, 100 - size),
                y: Math.min(prev.y, 100 - size),
                w: size,
                h: size,
              }));
            }}
            style={{ width: '100%' }}
          />
        </div>
      )}

      <div className="upload-actions" style={{ marginTop: '0.75rem' }}>
        {!cropping ? (
          <button
            type="button"
            className="btn"
            disabled={disabled}
            onClick={() => {
              setRect(DEFAULT_CROP);
              setCropping(true);
            }}
          >
            <Crop size={15} />
            Crop Query
          </button>
        ) : (
          <button type="button" className="btn btn-primary" disabled={disabled} onClick={applyCrop}>
            <Crop size={15} />
            Apply & Search
          </button>
        )}

        {cropping ? (
          <button
            type="button"
            className="btn"
            onClick={() => {
              setCropping(false);
              onResetCrop();
            }}
          >
            <RotateCcw size={15} />
            Cancel
          </button>
        ) : (
          <button type="button" className="btn" disabled={disabled} onClick={onClearImage}>
            <Trash2 size={15} />
            Change Image
          </button>
        )}
      </div>
    </div>
  );
}
