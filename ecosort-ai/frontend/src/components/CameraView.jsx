import React, { useRef, useState, useEffect } from 'react';
import { Camera, Upload, RefreshCw, Video, VideoOff, Zap, AlertCircle } from 'lucide-react';
import { detectWaste } from '../api/client.js';

export default function CameraView({ language, onDetectionsUpdated }) {
  const [imagePreview, setImagePreview] = useState(null);
  const [isLiveCamera, setIsLiveCamera] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [autoScan, setAutoScan] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [currentDetections, setCurrentDetections] = useState([]);

  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const autoScanTimerRef = useRef(null);

  const t = {
    en: {
      title: "Vision Feed (Live Camera / Upload)",
      startCamera: "Start Live Camera",
      stopCamera: "Stop Camera",
      captureNow: "Scan Current Frame",
      autoScanOn: "Auto-Scan: ON (3s)",
      autoScanOff: "Auto-Scan: OFF",
      uploadBtn: "Upload Photograph",
      simulateBtn: "Simulate Capture",
      scanning: "Analyzing Waste with YOLO11n...",
      instruction: "Start the live camera or upload a photo of the waste item.",
      cameraPermissionError: "Camera permission denied or camera not found. Please allow access or upload a photo."
    },
    ta: {
      title: "காட்சி ஊட்டம் (நேரலை கேமரா / பதிவேற்றம்)",
      startCamera: "நேரலை கேமராவைத் தொடங்கு",
      stopCamera: "கேமராவை நிறுத்து",
      captureNow: "தற்போதைய சட்டகத்தை ஆய்வு செய்",
      autoScanOn: "தானியங்கி ஆய்வு: ஆன் (3 நொடி)",
      autoScanOff: "தானியங்கி ஆய்வு: ஆஃப்",
      uploadBtn: "புகைப்படத்தைப் பதிவேற்றுக",
      simulateBtn: "உருவகப்படுத்துதல்",
      scanning: "YOLO11n மூலம் கழிவுகளை ஆய்வு செய்கிறது...",
      instruction: "நேரலை கேமராவை இயக்கவும் அல்லது கழிவுப் பொருளின் புகைப்படத்தைப் பதிவேற்றவும்.",
      cameraPermissionError: "கேமரா அணுகல் மறுக்கப்பட்டது அல்லது கிடைக்கவில்லை. அனுமதியை வழங்கவும் அல்லது புகைப்படத்தைப் பதிவேற்றவும்."
    }
  }[language] || {
    title: "Vision Feed (Live Camera / Upload)",
    startCamera: "Start Live Camera",
    stopCamera: "Stop Camera",
    captureNow: "Scan Current Frame",
    autoScanOn: "Auto-Scan: ON (3s)",
    autoScanOff: "Auto-Scan: OFF",
    uploadBtn: "Upload Photograph",
    simulateBtn: "Simulate Capture",
    scanning: "Analyzing Waste with YOLO11n...",
    instruction: "Start the live camera or upload a photo of the waste item.",
    cameraPermissionError: "Camera permission denied or camera not found. Please allow access or upload a photo."
  };

  // Clean up media stream on unmount
  useEffect(() => {
    return () => {
      stopCameraStream();
      if (autoScanTimerRef.current) {
        clearInterval(autoScanTimerRef.current);
      }
    };
  }, []);

  const stopCameraStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsLiveCamera(false);
    setAutoScan(false);
    if (autoScanTimerRef.current) {
      clearInterval(autoScanTimerRef.current);
      autoScanTimerRef.current = null;
    }
  };

  const startCameraStream = async () => {
    setCameraError(null);
    setImagePreview(null);
    setCurrentDetections([]);

    try {
      const constraints = {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'environment'
        },
        audio: false
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsLiveCamera(true);
    } catch (err) {
      console.error('Failed to access webcam:', err);
      setCameraError(t.cameraPermissionError);
      setIsLiveCamera(false);
    }
  };

  const handleCaptureFrame = () => {
    if (!videoRef.current || !isLiveCamera) return;

    const video = videoRef.current;
    const canvas = canvasRef.current || document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      if (blob) {
        triggerDetection(blob);
      }
    }, 'image/jpeg', 0.9);
  };

  const toggleAutoScan = () => {
    if (autoScan) {
      setAutoScan(false);
      if (autoScanTimerRef.current) {
        clearInterval(autoScanTimerRef.current);
        autoScanTimerRef.current = null;
      }
    } else {
      setAutoScan(true);
      // Run once immediately then interval
      handleCaptureFrame();
      autoScanTimerRef.current = setInterval(() => {
        handleCaptureFrame();
      }, 3000);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Stop live camera if running
    if (isLiveCamera) {
      stopCameraStream();
    }

    const reader = new FileReader();
    reader.onload = () => {
      setImagePreview(reader.result);
      triggerDetection(file);
    };
    reader.readAsDataURL(file);
  };

  const triggerDetection = async (fileBlob) => {
    setIsScanning(true);
    try {
      const formData = new FormData();
      formData.append(
        'file',
        fileBlob || new Blob(['sample-image'], { type: 'image/jpeg' }),
        'sample.jpg'
      );
      formData.append('language', language);

      const data = await detectWaste(formData);
      const items = Array.isArray(data)
        ? data
        : (data?.detected_items || []);

      setCurrentDetections(items);
      if (onDetectionsUpdated) {
        onDetectionsUpdated(items);
      }
    } catch (err) {
      console.warn('Backend unavailable, using edge mock fallback:', err);
      const mockItems = [
        {
          class_id: 'plastic_pet_bottle',
          detected_label: 'plastic_bottle',
          item_name: 'Plastic PET Bottle',
          primary_stream: 'Dry waste',
          bin_color: 'Blue',
          confidence: 0.94,
          bounding_box: { x_min: 0.25, y_min: 0.18, x_max: 0.72, y_max: 0.82 },
          disposal_guidance: 'Clean, crush, and place into the Blue Recyclable Bin.'
        }
      ];
      setCurrentDetections(mockItems);
      if (onDetectionsUpdated) {
        onDetectionsUpdated(mockItems);
      }
    } finally {
      setIsScanning(false);
    }
  };

  const handleSimulateCapture = () => {
    if (isLiveCamera) {
      stopCameraStream();
    }
    setImagePreview('https://images.unsplash.com/photo-1605600659908-0ef719419d41?w=800&auto=format&fit=crop&q=60');
    triggerDetection(null);
  };

  const getBoundingBoxStyles = (box) => {
    if (!box) return null;
    const xMin = box.x_min !== undefined ? box.x_min : (Array.isArray(box) ? box[0] : 0);
    const yMin = box.y_min !== undefined ? box.y_min : (Array.isArray(box) ? box[1] : 0);
    const xMax = box.x_max !== undefined ? box.x_max : (Array.isArray(box) ? box[2] : 1);
    const yMax = box.y_max !== undefined ? box.y_max : (Array.isArray(box) ? box[3] : 1);

    const top = Math.max(0, Math.min(100, yMin * 100));
    const left = Math.max(0, Math.min(100, xMin * 100));
    const width = Math.max(2, Math.min(100 - left, (xMax - xMin) * 100));
    const height = Math.max(2, Math.min(100 - top, (yMax - yMin) * 100));

    return { top: `${top}%`, left: `${left}%`, width: `${width}%`, height: `${height}%` };
  };

  const getBoxColorClasses = (binColor) => {
    switch (binColor?.toLowerCase()) {
      case 'green':
        return { border: 'border-emerald-400 bg-emerald-500/15', badge: 'bg-emerald-500 text-slate-950' };
      case 'red':
        return { border: 'border-rose-400 bg-rose-500/15', badge: 'bg-rose-500 text-white' };
      case 'black':
      case 'grey':
      case 'gray':
      case 'deposition':
        return { border: 'border-amber-400 bg-amber-500/15', badge: 'bg-amber-500 text-slate-950' };
      case 'blue':
      default:
        return { border: 'border-blue-400 bg-blue-500/15', badge: 'bg-blue-500 text-white' };
    }
  };

  return (
    <div className="bg-slate-800/60 border border-slate-700/80 rounded-2xl p-5 flex flex-col shadow-xl">
      {/* Hidden canvas for video frame extraction */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Header with status badges */}
      <div className="flex items-center justify-between mb-3">
        <h2 className="font-semibold text-slate-200 text-sm flex items-center gap-2">
          {isLiveCamera ? (
            <Video className="w-4 h-4 text-emerald-400 animate-pulse" />
          ) : (
            <Camera className="w-4 h-4 text-emerald-400" />
          )}
          {t.title}
        </h2>
        <div className="flex items-center gap-2">
          {isLiveCamera && (
            <span className="text-[10px] font-bold bg-rose-500/20 text-rose-300 border border-rose-500/40 px-2 py-0.5 rounded-full flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-rose-500 animate-ping" />
              LIVE
            </span>
          )}
          <span className="text-[11px] font-mono bg-slate-700/60 px-2 py-0.5 rounded text-slate-300">
            YOLO11n-edge
          </span>
        </div>
      </div>

      {/* Vision Viewport: Video or Image Preview */}
      <div className="relative aspect-video w-full rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center group">
        {/* Live Video Element */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover ${isLiveCamera ? 'block' : 'hidden'}`}
        />

        {/* Uploaded or Simulated Image */}
        {!isLiveCamera && imagePreview && (
          <img src={imagePreview} alt="Waste preview" className="w-full h-full object-contain" />
        )}

        {/* Empty Placeholder */}
        {!isLiveCamera && !imagePreview && (
          <div className="text-center p-6 text-slate-500 flex flex-col items-center gap-2">
            <Camera className="w-12 h-12 stroke-[1.5] text-slate-600" />
            <p className="text-xs max-w-xs">{t.instruction}</p>
          </div>
        )}

        {/* Dynamic Bounding Boxes */}
        {(isLiveCamera || imagePreview) &&
          currentDetections.map((item, idx) => {
            const boxStyle = getBoundingBoxStyles(item.bounding_box);
            if (!boxStyle) return null;
            const colorClasses = getBoxColorClasses(item.bin_color);
            const label = item.item_name || item.detected_label || item.label || 'Item';
            const conf = Math.round((item.confidence || 0.85) * 100);

            return (
              <div
                key={idx}
                className={`absolute border-2 rounded pointer-events-none transition-all ${colorClasses.border}`}
                style={boxStyle}
              >
                <span className={`absolute -top-6 left-0 text-[10px] font-bold px-1.5 py-0.5 rounded shadow ${colorClasses.badge}`}>
                  {label.replace(/_/g, ' ')} ({conf}%)
                </span>
              </div>
            );
          })}

        {/* Scanning Spinner Overlay */}
        {isScanning && (
          <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-xs flex flex-col items-center justify-center text-emerald-400 gap-2 z-10">
            <RefreshCw className="w-8 h-8 animate-spin text-emerald-400" />
            <span className="text-xs font-medium">{t.scanning}</span>
          </div>
        )}
      </div>

      {/* Camera Error Message */}
      {cameraError && (
        <div className="mt-3 p-3 rounded-xl bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
          <span>{cameraError}</span>
        </div>
      )}

      {/* Control Action Toolbar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-4">
        {/* 1. Live Camera Toggle */}
        {!isLiveCamera ? (
          <button
            onClick={startCameraStream}
            className="flex items-center justify-center gap-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold py-2.5 px-3 rounded-xl transition shadow-lg shadow-emerald-600/20 cursor-pointer"
          >
            <Video className="w-4 h-4" />
            <span>{t.startCamera}</span>
          </button>
        ) : (
          <button
            onClick={stopCameraStream}
            className="flex items-center justify-center gap-1.5 bg-rose-600 hover:bg-rose-500 text-white text-xs font-semibold py-2.5 px-3 rounded-xl transition shadow-lg shadow-rose-600/20 cursor-pointer"
          >
            <VideoOff className="w-4 h-4" />
            <span>{t.stopCamera}</span>
          </button>
        )}

        {/* 2. Capture & Scan (Active when live) */}
        {isLiveCamera ? (
          <button
            onClick={handleCaptureFrame}
            disabled={isScanning}
            className="flex items-center justify-center gap-1.5 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold py-2.5 px-3 rounded-xl transition shadow-lg shadow-cyan-600/20 disabled:opacity-50 cursor-pointer"
          >
            <Camera className="w-4 h-4" />
            <span>{t.captureNow}</span>
          </button>
        ) : (
          <button
            onClick={handleSimulateCapture}
            className="flex items-center justify-center gap-1.5 bg-slate-700/80 hover:bg-slate-700 text-slate-200 text-xs font-medium py-2.5 px-3 rounded-xl transition border border-slate-600/50 cursor-pointer"
          >
            <Camera className="w-4 h-4 text-slate-300" />
            <span>{t.simulateBtn}</span>
          </button>
        )}

        {/* 3. Auto-Scan Toggle (When live) */}
        {isLiveCamera ? (
          <button
            onClick={toggleAutoScan}
            className={`flex items-center justify-center gap-1.5 text-xs font-semibold py-2.5 px-3 rounded-xl transition border cursor-pointer ${
              autoScan
                ? 'bg-amber-500 text-slate-950 border-amber-400 shadow-lg shadow-amber-500/20 animate-pulse'
                : 'bg-slate-700/80 hover:bg-slate-700 text-slate-200 border-slate-600/50'
            }`}
          >
            <Zap className="w-4 h-4" />
            <span>{autoScan ? t.autoScanOn : t.autoScanOff}</span>
          </button>
        ) : null}

        {/* 4. Upload Photograph Button */}
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileUpload}
          accept="image/*"
          className="hidden"
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          className={`flex items-center justify-center gap-1.5 bg-slate-700/80 hover:bg-slate-700 text-slate-200 text-xs font-medium py-2.5 px-3 rounded-xl transition border border-slate-600/50 cursor-pointer ${
            isLiveCamera ? 'col-span-1' : 'col-span-2'
          }`}
        >
          <Upload className="w-4 h-4 text-slate-300" />
          <span>{t.uploadBtn}</span>
        </button>
      </div>
    </div>
  );
}
