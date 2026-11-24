import React from 'react';
import { Trash2, AlertTriangle, CheckCircle2, Sparkles } from 'lucide-react';

export default function DetectionCard({ detections = [], language = 'en' }) {
  // Normalize detections to always guarantee an array
  const items = Array.isArray(detections)
    ? detections
    : (detections?.detected_items || []);

  const t = {
    en: {
      title: "Detection & Disposal Directives",
      noDetections: "No waste objects analyzed yet. Scan or upload an image to view disposal instructions.",
      binLabel: "Designated Bin",
      confidence: "Confidence",
      guidance: "Disposal Guidance"
    },
    ta: {
      title: "கண்டறிதல் & அகற்றும் வழிகாட்டல்",
      noDetections: "இன்னும் கழிவுப் பொருட்கள் ஆய்வு செய்யப்படவில்லை. வழிமுறைகளைக் காண படத்தை ஸ்கேன் செய்யவும்.",
      binLabel: "குறிக்கப்பட்ட தொட்டி",
      confidence: "நம்பகத்தன்மை",
      guidance: "அகற்றும் வழிகாட்டுதல்"
    }
  }[language] || {
    title: "Detection & Disposal Directives",
    noDetections: "No waste objects analyzed yet. Scan or upload an image to view disposal instructions.",
    binLabel: "Designated Bin",
    confidence: "Confidence",
    guidance: "Disposal Guidance"
  };

  const getBinStyle = (color) => {
    switch (color?.toLowerCase()) {
      case 'green':
        return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';
      case 'blue':
        return 'bg-blue-500/20 text-blue-300 border-blue-500/40';
      case 'red':
        return 'bg-rose-500/20 text-rose-300 border-rose-500/40';
      case 'black':
      case 'grey':
      case 'gray':
      case 'deposition':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/40';
      default:
        return 'bg-slate-700/40 text-slate-300 border-slate-600/40';
    }
  };

  return (
    <div className="bg-slate-800/60 border border-slate-700/80 rounded-2xl p-5 shadow-xl flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h2 className="font-semibold text-slate-200 text-sm flex items-center gap-2">
          <Trash2 className="w-4 h-4 text-cyan-400" />
          {t.title}
        </h2>
        {items.length > 0 && (
          <span className="text-xs bg-emerald-950 text-emerald-400 border border-emerald-800/60 px-2.5 py-0.5 rounded-full font-medium flex items-center gap-1">
            <CheckCircle2 className="w-3.5 h-3.5" />
            {items.length} Object(s)
          </span>
        )}
      </div>

      {items.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center text-center p-8 text-slate-500">
          <Sparkles className="w-10 h-10 text-slate-600 mb-2 stroke-[1.5]" />
          <p className="text-xs max-w-sm">{t.noDetections}</p>
        </div>
      ) : (
        <div className="space-y-3">
          {items.map((item, idx) => {
            const label = item.item_name || item.detected_label || item.label || 'Waste Item';
            const category = item.primary_stream || item.category || 'Segregated Stream';
            const conf = Math.round((item.confidence || 0.85) * 100);
            const binColor = item.bin_color || 'Blue';
            const guidance = item.disposal_guidance ||
              item.segregation_instructions ||
              item.disposal_summary?.[language] ||
              item.disposal_summary?.en ||
              'Deposit in the specified bin according to statutory municipal guidelines.';

            return (
              <div
                key={idx}
                className="bg-slate-900/80 border border-slate-700/60 rounded-xl p-4 transition-all hover:border-slate-600"
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-bold text-white capitalize flex items-center gap-2">
                      {label.replace(/_/g, ' ')}
                      <span className="text-[11px] font-mono font-normal text-slate-400 bg-slate-800 px-2 py-0.5 rounded">
                        {conf}% {t.confidence}
                      </span>
                    </h3>
                    <p className="text-xs text-slate-400 mt-0.5 capitalize">
                      {category.replace(/_/g, ' ')}
                    </p>
                  </div>

                  <div
                    className={`text-xs px-3 py-1.5 rounded-lg border font-bold flex items-center gap-1.5 shadow-sm ${getBinStyle(
                      binColor
                    )}`}
                  >
                    <span className="w-2.5 h-2.5 rounded-full bg-current" />
                    <span>{binColor} Bin</span>
                  </div>
                </div>

                <div className="mt-3 pt-3 border-t border-slate-800 text-xs">
                  <span className="text-slate-400 font-medium block mb-1">
                    {t.guidance}:
                  </span>
                  <p className="text-slate-200 leading-relaxed bg-slate-950/60 p-2.5 rounded-lg border border-slate-800">
                    {guidance}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
