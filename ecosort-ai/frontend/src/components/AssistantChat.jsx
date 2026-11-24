import React, { useState } from 'react';
import { Bot, Send, Mic, Volume2, Loader2 } from 'lucide-react';
import { chatAssistant } from '../api/client.js';

export default function AssistantChat({ language, detectedItems }) {
  const [messages, setMessages] = useState([
    {
      sender: 'assistant',
      text:
        language === 'ta'
          ? 'வணக்கம்! நான் EcoSort AI உதவியாளர். கழிவுப் பிரிப்பு குறித்த உங்கள் சந்தேகங்களைக் கேளுங்கள்.'
          : 'Hello! I am your EcoSort AI Assistant. Show me an item or ask how to sort and dispose of waste properly.'
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const t = {
    en: {
      title: "Conversational Kiosk Assistant",
      subtitle: "Powered by Llama 3.2 Instruct via ExecuTorch",
      placeholder: "Ask disposal question (e.g., 'How to recycle milk packets?')...",
      voiceNote: "Voice I/O via IndicConformer & Indic-TTS ready for hardware deployment"
    },
    ta: {
      title: "உரையாடல் கியோஸ்க் உதவியாளர்",
      subtitle: "ExecuTorch வழியாக Llama 3.2 Instruct மூலம் இயங்குகிறது",
      placeholder: "கழிவுப் பிரிப்பு கேள்வியைக் கேளுங்கள்...",
      voiceNote: "குரல் வழி உள்ளீடு மற்றும் வெளியீடு விரைவில்"
    }
  }[language];

  const handleSend = async (e) => {
    e?.preventDefault();
    if (!input.trim() || isLoading) return;

    const userText = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { sender: 'user', text: userText }]);
    setIsLoading(true);

    try {
      const itemsList = Array.isArray(detectedItems)
        ? detectedItems
        : (detectedItems?.detected_items || []);
      const labels = itemsList.map((d) => d.item_name || d.detected_label || d.label || '').filter(Boolean);

      const data = await chatAssistant(
        userText,
        language,
        labels
      );
      setMessages((prev) => [...prev, { sender: 'assistant', text: data.response }]);
    } catch (err) {
      console.warn('Backend assistant offline, fallback invoked:', err);
      const fallbackReply =
        language === 'ta'
          ? 'கழிவுப் பொருட்களைப் பிரித்து போடுவது தூய்மையான சுற்றுச்சூழலுக்கு மிக முக்கியமானது. நீல நிற தொட்டியில் உலர் கழிவுகளையும், பச்சை நிற தொட்டியில் மக்கும் கழிவுகளையும் போடவும்.'
          : 'Segregating waste at source is essential. Dry recyclables belong in the Blue Bin, while biodegradable organic waste belongs in the Green Bin.';
      setMessages((prev) => [...prev, { sender: 'assistant', text: fallbackReply }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-slate-800/60 border border-slate-700/80 rounded-2xl p-5 shadow-xl flex flex-col h-[480px]">
      <div className="flex items-center justify-between pb-3 border-b border-slate-700/60 mb-3">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-emerald-500/20 text-emerald-400">
            <Bot className="w-4 h-4" />
          </div>
          <div>
            <h2 className="font-semibold text-slate-200 text-sm">{t.title}</h2>
            <p className="text-[11px] text-slate-400">{t.subtitle}</p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 text-xs text-slate-400">
          <button
            title={t.voiceNote}
            className="p-1.5 rounded-lg bg-slate-700/60 hover:bg-slate-700 text-slate-300 transition"
          >
            <Mic className="w-3.5 h-3.5 text-rose-400" />
          </button>
          <button
            title={t.voiceNote}
            className="p-1.5 rounded-lg bg-slate-700/60 hover:bg-slate-700 text-slate-300 transition"
          >
            <Volume2 className="w-3.5 h-3.5 text-cyan-400" />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto space-y-3 pr-1 text-xs">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${
              msg.sender === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            <div
              className={`max-w-[85%] rounded-2xl px-4 py-2.5 leading-relaxed ${
                msg.sender === 'user'
                  ? 'bg-emerald-600 text-white rounded-br-none'
                  : 'bg-slate-900 border border-slate-700 text-slate-200 rounded-bl-none shadow'
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-slate-900 border border-slate-700 text-slate-400 rounded-2xl rounded-bl-none px-4 py-2.5 flex items-center gap-2 text-xs">
              <Loader2 className="w-3.5 h-3.5 animate-spin text-emerald-400" />
              <span>Llama 3.2 ExecuTorch generating...</span>
            </div>
          </div>
        )}
      </div>

      <form onSubmit={handleSend} className="mt-3 flex items-center gap-2 pt-2 border-t border-slate-700/60">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={t.placeholder}
          className="flex-1 bg-slate-900 border border-slate-700 rounded-xl px-3.5 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500"
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white p-2 rounded-xl transition shadow"
        >
          <Send className="w-3.5 h-3.5" />
        </button>
      </form>
    </div>
  );
}
