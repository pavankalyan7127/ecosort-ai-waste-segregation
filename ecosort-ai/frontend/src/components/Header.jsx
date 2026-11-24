import React from 'react';
import { NavLink } from 'react-router-dom';
import { Languages, Cpu, LayoutDashboard, BookOpen } from 'lucide-react';

export default function Header({ language, setLanguage }) {
  const t = {
    en: {
      kiosk: "Kiosk Feed",
      guide: "Waste Guide",
      subtitle: "Autonomous Multilingual Edge Waste Segregation Kiosk"
    },
    ta: {
      kiosk: "கியோஸ்க் திரை",
      guide: "கழிவு வழிகாட்டி",
      subtitle: "சுயாதீன பலமொழி விளிம்பு நிலை கழிவுப் பிரிப்பு கியோஸ்க்"
    }
  }[language];

  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50 px-6 py-4">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center font-bold text-xl text-slate-950 shadow-lg shadow-emerald-500/20">
            🌱
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xl font-bold tracking-tight text-white">EcoSort AI</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-950 text-emerald-400 border border-emerald-800/60 font-medium">
                Bharat AI-SoC '26-27
              </span>
            </div>
            <p className="text-xs text-slate-400">
              {t.subtitle}
            </p>
          </div>
        </div>

        {/* Navigation Links via React Router */}
        <nav className="flex items-center gap-1 bg-slate-950/70 p-1 rounded-xl border border-slate-800">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                isActive
                  ? 'bg-emerald-600 text-white shadow'
                  : 'text-slate-400 hover:text-white'
              }`
            }
          >
            <LayoutDashboard className="w-3.5 h-3.5" />
            <span>{t.kiosk}</span>
          </NavLink>

          <NavLink
            to="/guide"
            className={({ isActive }) =>
              `flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                isActive
                  ? 'bg-emerald-600 text-white shadow'
                  : 'text-slate-400 hover:text-white'
              }`
            }
          >
            <BookOpen className="w-3.5 h-3.5" />
            <span>{t.guide}</span>
          </NavLink>
        </nav>

        {/* Status & Language Toggle */}
        <div className="flex items-center gap-3">
          <div className="hidden sm:flex items-center gap-2 text-xs bg-slate-800/80 px-3 py-1.5 rounded-lg border border-slate-700/60 text-slate-300">
            <Cpu className="w-4 h-4 text-cyan-400" />
            <span>RPi 5 + AI HAT+ 2</span>
            <span className="inline-block w-2 h-2 rounded-full bg-emerald-400 animate-pulse ml-1" />
            <span className="text-slate-400">Edge</span>
          </div>

          <div className="flex items-center gap-1 bg-slate-800 p-1 rounded-lg border border-slate-700">
            <Languages className="w-3.5 h-3.5 text-slate-400 ml-1.5" />
            <button
              onClick={() => setLanguage('en')}
              className={`text-xs px-2.5 py-1 rounded transition font-medium ${
                language === 'en'
                  ? 'bg-emerald-600 text-white shadow'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              EN
            </button>
            <button
              onClick={() => setLanguage('ta')}
              className={`text-xs px-2.5 py-1 rounded transition font-medium ${
                language === 'ta'
                  ? 'bg-emerald-600 text-white shadow'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              தமிழ்
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
