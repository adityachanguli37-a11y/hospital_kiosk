(function () {
  const SUPPORTED_LANGUAGES = new Set(['en', 'hi', 'kn', 'ml']);
  const DEFAULT_LANGUAGE = 'en';
  const STORAGE_KEY = 'hospital_language';
  const THEME_STORAGE_KEY = 'hospital_theme';
  const TRANSLATION_CACHE = new Map();
  const PENDING_REQUESTS = new Map();
  let googleWidgetLoaded = false;
  let currentLanguage = normalizeLanguage(readStoredLanguage() || DEFAULT_LANGUAGE);
  let currentTheme = normalizeTheme(document.documentElement.getAttribute('data-theme') || readStoredTheme() || detectPreferredTheme());

  const ALERT_MESSAGES = {
    invalid_phone: 'Please enter a valid 10-digit phone number',
    invalid_aadhaar: 'Aadhaar number must be exactly 12 digits',
    invalid_date: 'Please choose a valid consultation date.',
    face_missing: 'Please capture your face before registering',
    queue_help: 'Please check the registration desk for your queue status, or use the chatbot for assistance.',
    location_help: (params) => {
      if (params && params.location) {
        return `Please follow the signs or ask staff for directions to ${params.location}. Use the chatbot for more details.`;
      }
      return 'Please follow the signs or ask staff for directions. Use the chatbot for more details.';
    }
  };

  function normalizeLanguage(language) {
    const value = (language || DEFAULT_LANGUAGE).toString().trim().toLowerCase();
    return SUPPORTED_LANGUAGES.has(value) ? value : DEFAULT_LANGUAGE;
  }

  function normalizeTheme(theme) {
    return (theme || 'light').toString().trim().toLowerCase() === 'dark' ? 'dark' : 'light';
  }

  function detectPreferredTheme() {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function readStoredLanguage() {
    try {
      return window.localStorage.getItem(STORAGE_KEY);
    } catch (error) {
      return null;
    }
  }

  function storeLanguage(language) {
    try {
      window.localStorage.setItem(STORAGE_KEY, language);
    } catch (error) {
      // Ignore storage failures in private browsing or restricted contexts.
    }
  }

  function readStoredTheme() {
    try {
      return window.localStorage.getItem(THEME_STORAGE_KEY);
    } catch (error) {
      return null;
    }
  }

  function storeTheme(theme) {
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch (error) {
      // Ignore storage failures in private browsing or restricted contexts.
    }
  }

  function updateThemeSwitcher() {
    const switcher = document.getElementById('themeSwitcher');
    if (switcher && switcher.value !== currentTheme) {
      switcher.value = currentTheme;
    }
  }

  function applyTheme(theme, persist = true) {
    currentTheme = normalizeTheme(theme);
    document.documentElement.setAttribute('data-theme', currentTheme);
    document.documentElement.style.colorScheme = currentTheme;
    updateThemeSwitcher();

    if (persist) {
      storeTheme(currentTheme);
    }

    return currentTheme;
  }

  function readCookie(name) {
    const escapedName = name.replace(/([.$?*|{}()[\]\\/+^])/g, '\\$1');
    const match = document.cookie.match(new RegExp(`(?:^|; )${escapedName}=([^;]*)`));
    return match ? decodeURIComponent(match[1]) : '';
  }

  function writeCookie(name, value) {
    document.cookie = `${name}=${encodeURIComponent(value)}; path=/`;
  }

  function ensureHiddenWidgetContainer() {
    let container = document.getElementById('google_translate_element');
    if (container) {
      return container;
    }

    container = document.createElement('div');
    container.id = 'google_translate_element';
    container.className = 'notranslate';
    container.setAttribute('translate', 'no');
    container.setAttribute('aria-hidden', 'true');
    container.style.cssText = [
      'position:absolute',
      'left:-9999px',
      'top:0',
      'width:1px',
      'height:1px',
      'overflow:hidden'
    ].join(';');
    document.body.appendChild(container);
    return container;
  }

  function injectGoogleTranslateStyles() {
    if (document.getElementById('google-translate-style')) {
      return;
    }

    const style = document.createElement('style');
    style.id = 'google-translate-style';
    style.textContent = `
      .goog-te-banner-frame.skiptranslate,
      .goog-tooltip,
      #goog-gt-tt,
      .goog-te-balloon-frame {
        display: none !important;
      }
      body {
        top: 0 !important;
      }
      .goog-logo-link,
      .goog-te-gadget span {
        display: none !important;
      }
    `;
    document.head.appendChild(style);
  }

  function loadGoogleTranslateScript() {
    if (document.getElementById('google-translate-script')) {
      return;
    }

    const script = document.createElement('script');
    script.id = 'google-translate-script';
    script.async = true;
    script.src = 'https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
    document.head.appendChild(script);
  }

  function updateLanguageSwitcher() {
    const switcher = document.getElementById('languageSwitcher');
    if (switcher && switcher.value !== currentLanguage) {
      switcher.value = currentLanguage;
    }
  }

  function syncGoogleWidget(attempt = 0) {
    document.documentElement.lang = currentLanguage;
    updateLanguageSwitcher();
    writeCookie('googtrans', `/en/${currentLanguage}`);

    const select = document.querySelector('.goog-te-combo');
    if (select) {
      if (select.value !== currentLanguage) {
        select.value = currentLanguage;
        select.dispatchEvent(new Event('change'));
      }
      return;
    }

    if (attempt < 20) {
      window.setTimeout(() => syncGoogleWidget(attempt + 1), 250);
    }
  }

  function formatChatMarkup(text) {
    let formatted = String(text || '');
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    formatted = formatted.replace(/\u2022\s*(.*?)(?=\n|$)/g, '<li>$1</li>');
    if (formatted.includes('<li>')) {
      formatted = formatted.replace(/(<li>.*?<\/li>)+/g, '<ul>$&</ul>');
    }
    return formatted;
  }

  async function translateHospitalTextAsync(text, targetLanguage = currentLanguage) {
    const cleanText = String(text || '');
    const normalizedTarget = normalizeLanguage(targetLanguage);
    if (!cleanText || normalizedTarget === 'en') {
      return cleanText;
    }

    const cacheKey = `${normalizedTarget}::${cleanText}`;
    if (TRANSLATION_CACHE.has(cacheKey)) {
      return TRANSLATION_CACHE.get(cacheKey);
    }

    if (PENDING_REQUESTS.has(cacheKey)) {
      return PENDING_REQUESTS.get(cacheKey);
    }

    const request = fetch('/api/translate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: cleanText,
        target_language: normalizedTarget,
        source_language: 'auto',
      }),
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Translation request failed: ${response.status}`);
        }
        const payload = await response.json();
        const translatedText = payload && payload.translated_text ? String(payload.translated_text) : cleanText;
        TRANSLATION_CACHE.set(cacheKey, translatedText);
        return translatedText;
      })
      .catch((error) => {
        console.error('[translateHospitalTextAsync]', error);
        return cleanText;
      })
      .finally(() => {
        PENDING_REQUESTS.delete(cacheKey);
      });

    PENDING_REQUESTS.set(cacheKey, request);
    return request;
  }

  async function retranslateDynamicContent() {
    const targetLanguage = currentLanguage;

    const messageNodes = document.querySelectorAll('.message.notranslate[data-original-message]');
    for (const node of messageNodes) {
      if (currentLanguage !== targetLanguage) {
        return;
      }

      const body = node.querySelector('.message-body');
      if (!body) {
        continue;
      }

      const originalMessage = node.dataset.originalMessage || '';
      const sender = node.dataset.messageSender || 'bot';
      if (sender === 'bot') {
        const translatedMessage = await translateHospitalTextAsync(originalMessage, targetLanguage);
        if (currentLanguage !== targetLanguage) {
          return;
        }
        body.innerHTML = formatChatMarkup(translatedMessage);
      } else {
        body.textContent = originalMessage;
      }
    }

    const faceMessage = document.getElementById('faceMessage');
    if (faceMessage && faceMessage.dataset.originalMessage) {
      const translatedMessage = await translateHospitalTextAsync(faceMessage.dataset.originalMessage, targetLanguage);
      if (currentLanguage !== targetLanguage) {
        return;
      }
      faceMessage.innerHTML = translatedMessage;
    }

    const questionButtons = document.querySelectorAll('#suggestedQuestions .suggested-question[data-original-question]');
    for (const button of questionButtons) {
      if (currentLanguage !== targetLanguage) {
        return;
      }

      const originalQuestion = button.dataset.originalQuestion || button.textContent || '';
      const translatedQuestion = await translateHospitalTextAsync(originalQuestion, targetLanguage);
      if (currentLanguage !== targetLanguage) {
        return;
      }
      button.textContent = translatedQuestion;
    }
  }

  async function refreshPageTranslation(force = false) {
    syncGoogleWidget();
    if (force) {
      await retranslateDynamicContent();
    }
  }

  async function localizeHospitalAlert(key, params = {}) {
    const alertFactory = ALERT_MESSAGES[key];
    const message = typeof alertFactory === 'function' ? alertFactory(params) : (alertFactory || key);
    const targetLanguage = currentLanguage;
    const translatedMessage = await translateHospitalTextAsync(message, targetLanguage);
    alert(translatedMessage);
  }

  async function setHospitalLanguage(language) {
    currentLanguage = normalizeLanguage(language);
    storeLanguage(currentLanguage);
    await refreshPageTranslation(true);
  }

  async function setHospitalTheme(theme) {
    applyTheme(theme, true);
  }

  function initLanguageSwitcher() {
    const switcher = document.getElementById('languageSwitcher');
    if (!switcher) {
      return;
    }

    switcher.value = currentLanguage;
    switcher.addEventListener('change', (event) => {
      void setHospitalLanguage(event.target.value);
    });
  }

  function initThemeSwitcher() {
    const switcher = document.getElementById('themeSwitcher');
    if (!switcher) {
      return;
    }

    switcher.value = currentTheme;
    switcher.addEventListener('change', (event) => {
      void setHospitalTheme(event.target.value);
    });
  }

  function googleTranslateElementInit() {
    ensureHiddenWidgetContainer();
    googleWidgetLoaded = true;

    if (!window.google || !window.google.translate || !window.google.translate.TranslateElement) {
      return;
    }

    new window.google.translate.TranslateElement({
      pageLanguage: 'en',
      autoDisplay: false,
      includedLanguages: Array.from(SUPPORTED_LANGUAGES).join(','),
      layout: window.google.translate.TranslateElement.InlineLayout.SIMPLE,
    }, 'google_translate_element');

    syncGoogleWidget();
  }

  function bootstrapTranslator() {
    injectGoogleTranslateStyles();
    ensureHiddenWidgetContainer();
    applyTheme(currentTheme, false);
    initLanguageSwitcher();
    initThemeSwitcher();

    const storedCookieLanguage = readCookie('googtrans');
    const storedCookieMatch = storedCookieLanguage.match(/\/en\/([a-z-]+)/i);
    if (!readStoredLanguage() && storedCookieMatch && storedCookieMatch[1]) {
      currentLanguage = normalizeLanguage(storedCookieMatch[1]);
    }

    syncGoogleWidget();
    loadGoogleTranslateScript();

    if (currentLanguage !== 'en') {
      void refreshPageTranslation(true);
    }
  }

  window.translateHospitalText = translateHospitalTextAsync;
  window.translateHospitalTextAsync = translateHospitalTextAsync;
  window.localizeHospitalAlert = localizeHospitalAlert;
  window.refreshPageTranslation = refreshPageTranslation;
  window.setHospitalLanguage = setHospitalLanguage;
  window.setHospitalTheme = setHospitalTheme;
  window.getHospitalLanguage = () => currentLanguage;
  window.getHospitalTheme = () => currentTheme;
  window.googleTranslateElementInit = googleTranslateElementInit;
  window.formatHospitalChatMessage = formatChatMarkup;

  document.addEventListener('DOMContentLoaded', bootstrapTranslator);
})();
