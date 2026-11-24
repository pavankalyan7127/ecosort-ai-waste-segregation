import React, { useEffect, useState } from 'react';
import { Layers } from 'lucide-react';

export default function CategoryLegend({ language }) {
  const [categories, setCategories] = useState([]);

  useEffect(() => {
    fetch(`http://localhost:8000/api/knowledge/categories?lang=${language}`)
      .then((res) => res.json())
      .then((data) => setCategories(data))
      .catch((err) => {
        // Fallback local data if backend is offline
        const fallback = [
          {
            id: 'wet_organic',
            bin_color: 'Green',
            name: language === 'ta' ? 'ஈர / மக்கும் கழிவு' : 'Wet / Organic Waste',
            disposal_instructions:
              language === 'ta'
                ? 'பச்சை நிற தொட்டியில் போடவும்.'
                : 'Deposit in Green Bin.'
          },
          {
            id: 'dry_recyclable',
            bin_color: 'Blue',
            name: language === 'ta' ? 'உலர் / மறுசுழற்சி கழிவு' : 'Dry / Recyclable Waste',
            disposal_instructions:
              language === 'ta'
                ? 'நீல நிற தொட்டியில் போடவும்.'
                : 'Deposit in Blue Bin.'
          },
          {
            id: 'hazardous_sanitary',
            bin_color: 'Red',
            name: language === 'ta' ? 'ஆபத்தான / சுகாதார கழிவு' : 'Domestic Hazardous / Sanitary',
            disposal_instructions:
              language === 'ta'
                ? 'சிவப்பு நிற தொட்டியில் போடவும்.'
                : 'Deposit in Red Bin.'
          },
          {
            id: 'e_waste',
            bin_color: 'Grey / Black',
            name: language === 'ta' ? 'மின்னணு கழிவு (இ-கழிவு)' : 'Electronic Waste (E-Waste)',
            disposal_instructions:
              language === 'ta'
                ? 'சாம்பல்/கருப்பு தொட்டி அல்லது மையத்தில் ஒப்படைக்கவும்.'
                : 'Special e-waste bin/counter.'
          }
        ];
        setCategories(fallback);
      });
  }, [language]);

  const getBorder = (color) => {
    switch (color?.toLowerCase()) {
      case 'green':
        return 'border-l-4 border-l-emerald-500';
      case 'blue':
        return 'border-l-4 border-l-blue-500';
      case 'red':
        return 'border-l-4 border-l-rose-500';
      default:
        return 'border-l-4 border-l-slate-400';
    }
  };

  return (
    <div className="bg-slate-800/40 border border-slate-700/60 rounded-2xl p-5 shadow">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-2">
        <Layers className="w-3.5 h-3.5 text-emerald-400" />
        {language === 'ta' ? 'கழிவு வகைப்பாடு வழிகாட்டி' : 'Standard Segregation Schema'}
      </h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
        {categories.map((cat, i) => (
          <div
            key={i}
            className={`bg-slate-900/90 p-3 rounded-xl border border-slate-800 ${getBorder(
              cat.bin_color
            )}`}
          >
            <div className="font-semibold text-slate-200">{cat.name}</div>
            <div className="text-[11px] font-mono text-emerald-400 mt-0.5">
              {cat.bin_color} Bin
            </div>
            <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">
              {cat.disposal_instructions}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
