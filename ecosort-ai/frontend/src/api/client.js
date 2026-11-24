import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export const detectWaste = async (formData) => {
  const response = await api.post('/vision/detect', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const chatAssistant = async (message, language = 'en', detectedItems = []) => {
  const response = await api.post('/assistant/chat', {
    message,
    language,
    detected_items: detectedItems,
  });
  return response.data;
};

export const getCategories = async (lang = 'en') => {
  const response = await api.get(`/knowledge/categories?lang=${lang}`);
  return response.data;
};

export const translateText = async (text, srcLang = 'en', tgtLang = 'ta') => {
  const response = await api.post('/translation/translate', {
    text,
    src_lang: srcLang,
    tgt_lang: tgtLang,
  });
  return response.data;
};

export default api;
