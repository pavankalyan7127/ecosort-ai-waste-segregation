import React, { useEffect, useState } from 'react';
import { getCategories } from '../api/client.js';
import { BookOpen, Search, CheckCircle2, ArrowRight } from 'lucide-react';

export default function WasteGuide({ language }) {
  const [categories, setCategories] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedBin, setSelectedBin] = useState('all');

  useEffect(() => {
    getCategories(language)
      .then((data) => setCategories(data))
      .catch((err) => console.error('Error fetching categories:', err));
  }, [language]);

  const t = {
    en: {
      title: "Waste Segregation Directory & Guide",
      subtitle: "Official rules and bin classifications for clean, source-level sorting.",
      searchPlaceholder: "Search waste item (e.g., bottle, battery, peel)...",
      allBins: "All Categories",
      examples: "Common Items",
      instructions: "Disposal Instructions"
    },
    ta: {
      title: "கழிவுப் பிரிப்பு வழிகாட்டி",
      subtitle: "சரியான மற்றும் முறையான கழிவு அகற்றலுக்கான வழிகாட்டுதல்கள்.",
      searchPlaceholder: "பொருளைத் தேடுங்கள் (எ.கா. பாட்டில், பேட்டரி, பழத்தோல்)...",
      allBins: "அனைத்து வகைகள்",
      examples: "பொதுவான பொருட்கள்",
      instructions: "அகற்றும் வழிமுறைகள்"
    }
  }[language];

  const filteredCategories = categories.filter((cat) => {
    const matchesBin = selectedBin === 'all' || cat.bin_color.toLowerCase().includes(selectedBin);
    const matchesSearch =
      !searchTerm ||
      cat.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      cat.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (cat.examples && cat.examples.some((ex) => ex.toLowerCase().includes(searchTerm.toLowerCase())));
    return matchesBin && matchesSearch;
  });

  return (
    <div className="space-y-6">
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-xl font-bold text-white flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-emerald-400" />
              {t.title}
            </h1>
            <p className="text-xs text-slate-400 mt-1">{t.subtitle}</p>
          </div>

          <div className="flex items-center gap-2">
            <div className="relative">
              <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder={t.searchPlaceholder}
                className="bg-slate-950 border border-slate-700 rounded-xl pl-9 pr-4 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500 w-64"
              />
            </div>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 mt-4 pt-4 border-t border-slate-800/80">
          {['all', 'green', 'blue', 'red', 'grey'].map((bin) => (
            <button
              key={bin}
              onClick={() => setSelectedBin(bin)}
              className={`text-xs px-3 py-1.5 rounded-lg border capitalize font-medium transition ${
                selectedBin === bin
                  ? 'bg-emerald-600 border-emerald-500 text-white shadow'
                  : 'bg-slate-800/60 border-slate-700 text-slate-300 hover:text-white'
              }`}
            >
              {bin === 'all' ? t.allBins : `${bin} Bin`}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {filteredCategories.map((cat, idx) => (
          <div
            key={idx}
            className="bg-slate-900 border border-slate-800 rounded-2xl p-5 hover:border-slate-700 transition"
          >
            <div className="flex items-start justify-between">
              <div>
                <h3 className="font-bold text-white text-base">{cat.name}</h3>
                <span className="text-xs font-mono text-emerald-400">
                  {cat.bin_color} Bin
                </span>
              </div>
              <span className="text-xs font-bold px-3 py-1 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
                {cat.id}
              </span>
            </div>

            <p className="text-xs text-slate-300 mt-3 leading-relaxed">
              {cat.description}
            </p>

            <div className="mt-4 pt-4 border-t border-slate-800/80 space-y-2">
              <div>
                <span className="text-[11px] font-semibold text-slate-400 block mb-1">
                  {t.instructions}:
                </span>
                <p className="text-xs text-slate-200 bg-slate-950 p-2.5 rounded-lg border border-slate-800/80">
                  {cat.disposal_instructions}
                </p>
              </div>

              {cat.examples && (
                <div>
                  <span className="text-[11px] font-semibold text-slate-400 block mb-1">
                    {t.examples}:
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {cat.examples.map((ex, i) => (
                      <span
                        key={i}
                        className="text-[11px] bg-slate-800/80 text-slate-300 px-2 py-0.5 rounded border border-slate-700"
                      >
                        {ex.replace('_', ' ')}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
