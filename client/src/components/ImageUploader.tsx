import { Camera, ImagePlus, Upload } from 'lucide-react';
import { useRef, useState, type DragEvent } from 'react';
import { resolveImageUrl, type CatalogItem } from '../api/visualSearch';

interface ImageUploaderProps {
  onSelectFile: (blob: Blob, filename: string) => void;
  sampleItems?: CatalogItem[];
  disabled?: boolean;
}

export function ImageUploader({ onSelectFile, sampleItems = [], disabled }: ImageUploaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (!file) return;
    onSelectFile(file, file.name);
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    if (disabled) return;
    handleFiles(e.dataTransfer.files);
  };

  const loadSample = async (item: CatalogItem) => {
    const primary = resolveImageUrl(item);
    let res = await fetch(primary).catch(() => null);
    if (!res || !res.ok) {
      res = await fetch(item.imageUrl);
    }
    const blob = await res.blob();
    onSelectFile(blob, `${item.productId}.webp`);
  };

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload jewelry image by clicking or dragging and dropping"
        className={`dropzone ${dragging ? 'dragging' : ''}`}
        onClick={() => !disabled && fileInputRef.current?.click()}
        onKeyDown={(e) => {
          if ((e.key === 'Enter' || e.key === ' ') && !disabled) {
            e.preventDefault();
            fileInputRef.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <ImagePlus size={34} color="#134023" style={{ marginBottom: '0.5rem' }} />
        <div style={{ fontWeight: 600, fontSize: '0.98rem', color: '#134023' }}>
          Drop a ring or jewelry photo here
        </div>
        <div style={{ fontSize: '0.82rem', color: '#5c5c57', marginTop: '0.25rem' }}>
          JPG, PNG, or WebP up to 10 MB
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp,image/gif"
        className="sr-only"
        aria-label="Choose image file"
        onChange={(e) => {
          handleFiles(e.target.files);
          e.target.value = '';
        }}
      />

      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="sr-only"
        aria-label="Capture photo with camera"
        onChange={(e) => {
          handleFiles(e.target.files);
          e.target.value = '';
        }}
      />

      <div className="upload-actions">
        <button
          type="button"
          className="btn btn-primary"
          disabled={disabled}
          onClick={() => fileInputRef.current?.click()}
        >
          <Upload size={16} />
          File Picker
        </button>
        <button
          type="button"
          className="btn"
          disabled={disabled}
          onClick={() => cameraInputRef.current?.click()}
        >
          <Camera size={16} />
          Camera
        </button>
      </div>

      {sampleItems.length > 0 && (
        <div style={{ marginTop: '1.1rem' }}>
          <div style={{ fontSize: '0.78rem', fontWeight: 600, color: '#5c5c57', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Quick Test From Local Catalog
          </div>
          <div className="sample-strip">
            {sampleItems.slice(0, 4).map((item) => (
              <button
                key={`${item.productId}-${item.imageUrl}`}
                type="button"
                className="sample-thumb"
                onClick={() => void loadSample(item)}
                title={`Search using ${item.title ?? item.productId}`}
              >
                <img
                  src={resolveImageUrl(item)}
                  alt={item.title ?? item.productId}
                  loading="lazy"
                  onError={(e) => {
                    if (e.currentTarget.src !== item.imageUrl) {
                      e.currentTarget.src = item.imageUrl;
                    }
                  }}
                />
                <div style={{ marginTop: '0.2rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {item.productId}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
