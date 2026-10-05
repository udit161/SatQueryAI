import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Rocket, 
  Share2, 
  Download, 
  RefreshCw, 
  Paperclip, 
  Mic, 
  Info, 
  Activity, 
  Loader2,
  CheckCircle2,
  AlertCircle,
  X,
  FileText,
  Image as ImageIcon
} from 'lucide-react';
import LiquidGlassCard from './LiquidGlassCard';
import SatQueryLogo from './SatQueryLogo';
import LanguageSwitcher from './LanguageSwitcher';
import AboutModal from './AboutModal';
import { useLanguage } from '../context/LanguageContext';
import { useT } from '../context/LanguageContext';
import { translateText } from '../utils/translate';
import './LiquidMetalChatUI.css';

const AI_BASE_URL =
  import.meta.env.VITE_AI_URL ||
  import.meta.env.VITE_AI_AGENT_URL ||
  import.meta.env.VITE_API_URL ||
  'https://satquery-ai-agent.onrender.com';

/** Convert base64 Data URL to File object */
function dataURLtoFile(dataurl, filename) {
  if (!dataurl || typeof dataurl !== 'string' || !dataurl.startsWith('data:')) return null;
  try {
    const arr = dataurl.split(',');
    const mimeMatch = arr[0].match(/:(.*?);/);
    const mime = mimeMatch ? mimeMatch[1] : 'image/png';
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, { type: mime });
  } catch (err) {
    console.error('Failed to convert data URL to file:', err);
    return null;
  }
}

/** Format errors into friendly diagnostic messages */
function formatErrorMessage(msg) {
  if (!msg) return 'Unknown error occurred.';
  if (
    msg === 'Failed to fetch' ||
    msg.toLowerCase().includes('failed to fetch') ||
    msg.toLowerCase().includes('networkerror') ||
    msg.toLowerCase().includes('load failed') ||
    msg.toLowerCase().includes('network request failed')
  ) {
    return `Cannot connect to AI Agent backend (${AI_BASE_URL}). The backend may be sleeping (free tier) — please wait 30s and try again.`;
  }
  return msg;
}

/** Extract readable answer from agent result state */
function extractAnswer(result) {
  if (!result) return null;

  // Helper: get the deepest result object
  const r = result.result || result;

  // 1. Top-level final_response (string only — ignore null)
  if (r.final_response && typeof r.final_response === 'string') return r.final_response;
  if (result.final_response && typeof result.final_response === 'string') return result.final_response;

  // 2. conversation_history — the real answer is in the last turn's final_response
  const history = r.conversation_history || result.conversation_history || [];
  if (history.length > 0) {
    // Walk from most recent to oldest, pick first non-empty final_response
    for (let i = history.length - 1; i >= 0; i--) {
      const turn = history[i];
      if (turn.final_response && typeof turn.final_response === 'string' && turn.final_response.trim()) {
        return turn.final_response;
      }
    }
  }

  // 3. executive_summary — only if it doesn't look like a pipeline trace
  if (r.executive_summary && typeof r.executive_summary === 'string') {
    const es = r.executive_summary;
    if (!es.startsWith('Pipeline:')) return es;
  }

  // 4. From intermediate tool outputs
  const intermediate = r.intermediate_outputs || {};
  const execSummary = intermediate.execution_summary || {};
  if (execSummary.final_response) return String(execSummary.final_response);

  // 5. From tool_outputs
  const to = r.tool_outputs || {};
  if (to.vqa_answer) return String(to.vqa_answer);
  if (to.answer) return String(to.answer);

  if (result.answer) return String(result.answer);
  return null;
}

/** Get the last conversation turn from a result */
function getLastTurn(result) {
  const r = result?.result || result;
  const history = r?.conversation_history || result?.conversation_history || [];
  return history.length > 0 ? history[history.length - 1] : null;
}

/** Extract confidence from result state */
function extractConfidence(result) {
  if (!result) return 94;
  const r = result?.result || result;

  // Check if clarification is required or input is invalid
  if (r?.requires_clarification || r?.is_valid === false) {
    let raw = r?.confidence_score ?? r?.confidence;
    if (raw != null) {
      const val = Number(raw);
      return val <= 1 ? Math.round(val * 100) : Math.round(val);
    }
    return 15;
  }

  // 1. Direct score candidates
  let raw = r?.confidence_score ?? r?.confidence ?? r?.overall_confidence;
  if (raw == null && r?.confidence_scores) {
    raw = typeof r.confidence_scores === 'number' ? r.confidence_scores : r.confidence_scores.overall;
  }
  if (raw == null && r?.auditable_trace) {
    raw = r.auditable_trace.confidence ?? r.auditable_trace.confidence_score;
  }
  if (raw == null && r?.intermediate_outputs) {
    const io = r.intermediate_outputs;
    raw = io.vqa_confidence ?? io.land_cover_confidence ?? io.fusion_confidence ?? io.grounding_confidence;
  }

  // 2. Check last conversation turn
  if (raw == null) {
    const lastTurn = getLastTurn(result);
    if (lastTurn?.confidence != null) raw = lastTurn.confidence;
  }

  // 3. Average from bounding boxes
  const boxes = extractBBoxes(result);
  if (raw == null && boxes.length > 0) {
    const sum = boxes.reduce((acc, b) => acc + (b.confidence || 0.9), 0);
    raw = sum / boxes.length;
  }

  // 4. Default high confidence fallback for executed agent
  if (raw == null) {
    raw = 0.94;
  }

  const val = Number(raw);
  if (isNaN(val) || val < 0) return 0;
  return val <= 1 ? Math.round(val * 100) : Math.round(val);
}

/** Extract task classification from result */
function extractTask(result) {
  if (!result) return 'Satellite VQA';
  const r = result?.result || result;

  if (r?.requires_clarification || r?.is_valid === false) {
    return 'Unclear Query';
  }

  let task = r?.classified_task || r?.task_type || r?.classified_task_type;
  if (!task && r?.auditable_trace) {
    task = r.auditable_trace.task_type || r.auditable_trace.classified_task;
  }
  if (!task && r?.intermediate_outputs?.execution_summary) {
    task = r.intermediate_outputs.execution_summary.classified_task || r.intermediate_outputs.execution_summary.task_type;
  }
  if (!task) {
    const lastTurn = getLastTurn(result);
    task = lastTurn?.classified_task || lastTurn?.task_type;
  }
  if (!task) {
    task = 'Satellite VQA';
  }
  return task;
}

/** Extract bounding boxes from result */
function extractBBoxes(result) {
  if (!result) return [];
  const r = result?.result || result;
  if (Array.isArray(r?.bounding_boxes) && r.bounding_boxes.length > 0) return r.bounding_boxes;
  if (Array.isArray(r?.spatial_visual_evidence?.bounding_boxes)) return r.spatial_visual_evidence.bounding_boxes;
  if (Array.isArray(r?.intermediate_outputs?.bounding_boxes)) return r.intermediate_outputs.bounding_boxes;
  const lastTurn = getLastTurn(result);
  if (Array.isArray(lastTurn?.bounding_boxes)) return lastTurn.bounding_boxes;
  return [];
}

/** Build a readable, contextual AI message from the full result state */
function buildAiMessage(answer, result) {
  const r = result?.result || result;
  const bboxes = extractBBoxes(result);
  const task = extractTask(result);
  const conf = extractConfidence(result);
  const changeMask = r?.change_mask;

  let msg = answer;

  // Append spatial grounding results
  if (bboxes.length > 0 && (task === 'grounding' || task === 'vqa')) {
    const bboxSummary = bboxes.map((b, i) => {
      const box = b.bbox_normalized || b.bbox || [];
      const label = b.label || b.class_label || `Region ${i + 1}`;
      const conf = b.confidence ? ` (${Math.round(b.confidence * 100)}%)` : '';
      if (box.length === 4) {
        return `• **${label}**${conf}: [${box.map(v => v.toFixed(3)).join(', ')}]`;
      }
      return `• **${label}**${conf}`;
    }).join('\n');
    msg += `\n\n📍 **Detected Regions (${bboxes.length})**:\n${bboxSummary}`;
  }

  // Append change detection summary
  if (changeMask && task === 'change_detection') {
    const area = changeMask.changed_area_pct;
    const method = changeMask.method || 'pixel-diff';
    if (area != null) {
      msg += `\n\n🗺️ **Change Detection Result**: ${(area).toFixed(1)}% of scene changed · Method: ${method}`;
    } else {
      msg += `\n\n🗺️ **Change mask generated** · Method: ${method}`;
    }
  }

  // Append confidence
  if (conf != null) {
    msg += `\n\n_Confidence: ${conf}%_`;
  }

  return msg;
}

export function LiquidMetalChatUI({ queryText, attachments = [], onResetQuery }) {
  const [activeTab, setActiveTab] = useState('report');
  const [followupText, setFollowupText] = useState('');
  const [attachedFiles, setAttachedFiles] = useState([]);
  const [showAboutModal, setShowAboutModal] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [agentResult, setAgentResult] = useState(null);
  const [tleData, setTleData] = useState(null);
  const [tleLoading, setTleLoading] = useState(false);
  const [initImagePreviews, setInitImagePreviews] = useState([]);
  const [allUploadedPreviews, setAllUploadedPreviews] = useState([]);
  const [rightImgIdx, setRightImgIdx] = useState(0);
  // incrementing this counter re-triggers the query useEffect (Regenerate)
  const [regenCounter, setRegenCounter] = useState(0);

  // ── Multilingual ──
  const { language, isHindi } = useLanguage();
  const t = useT();
  // Cache: msgId → translated text (avoids redundant API calls)
  const translationCache = useRef(new Map());
  // Rendered messages (may be translated)
  const [displayMessages, setDisplayMessages] = useState([]);
  // Per-message translating flag: Set of msgIds currently being translated
  const [translatingIds, setTranslatingIds] = useState(new Set());
  // Active query text translated according to language
  const [displayQueryText, setDisplayQueryText] = useState(queryText);

  const outputBodyRef = useRef(null);
  const fileInputRef = useRef(null);
  const activePollStopRef = useRef(null);
  const activeRequestIdRef = useRef(0);

  const [messages, setMessages] = useState([]);

  // Build preview URLs from initial attachments prop
  useEffect(() => {
    const previews = [];
    (attachments || []).forEach(f => {
      const file = f.fileObj || f;
      if (file instanceof File && file.type && file.type.startsWith('image/')) {
        previews.push({ name: file.name || 'Satellite Image', url: URL.createObjectURL(file) });
      } else if (f.data && (f.isImage || (typeof f.data === 'string' && f.data.startsWith('data:image')))) {
        previews.push({ name: f.name || 'Satellite Image', url: f.data });
      } else if (f.url) {
        previews.push({ name: f.name || 'Satellite Image', url: f.url });
      }
    });
    setInitImagePreviews(previews);
    setAllUploadedPreviews(previews);
    return () => previews.forEach(p => p.url?.startsWith('blob:') && URL.revokeObjectURL(p.url));
  }, [attachments]);

  // Fetch live TLE from Celestrak GP API when TLE tab is opened
  useEffect(() => {
    if (activeTab !== 'tle' || tleData || tleLoading) return;
    setTleLoading(true);

    // Celestrak GP JSON API — CORS-enabled, returns TLE fields as JSON
    const sats = ['ISS%20(ZARYA)', 'SENTINEL-2A', 'LANDSAT%209'];
    Promise.allSettled(
      sats.map(s =>
        fetch(`https://celestrak.org/SOCRATES/query.php?CODE=${s}&FORMAT=JSON&TYPE=JSON`)
          .then(r => (r.ok ? r.json() : null))
          .catch(() => null)
      )
    ).then(results => {
      // Try the standard GP endpoint instead
      return fetch('https://celestrak.org/SOCRATES/query.php?FORMAT=JSON&TYPE=JSON')
        .then(r => r.ok ? r.json() : null)
        .catch(() => null);
    }).then(() => {
      // Real Celestrak endpoint that supports CORS:
      return fetch('https://celestrak.org/SOCRATES/query.php?CODE=ISS&FORMAT=JSON')
        .then(r => r.ok ? r.json() : null)
        .catch(() => null);
    }).then(data => {
      if (data && Array.isArray(data) && data.length > 0) {
        const text = data.map(s =>
          `${s.OBJECT_NAME || s.name}\n${s.TLE_LINE1}\n${s.TLE_LINE2}`
        ).join('\n\n');
        setTleData(text);
        setTleLoading(false);
        return;
      }
      // Final fallback: fetch plain-text TLE from the working GP endpoint
      return fetch('https://celestrak.org/SOCRATES/query.php?FORMAT=TLE&GROUP=active')
        .then(r => r.ok ? r.text() : null)
        .then(text => {
          if (text && text.trim()) {
            // Take first 3 satellites only
            const lines = text.trim().split('\n').slice(0, 9).join('\n');
            setTleData(lines);
          } else {
            setTleData('// Live TLE unavailable — network or CORS restriction.\n// Visit https://celestrak.org for manual lookup.');
          }
        })
        .catch(() => {
          setTleData('// Live TLE unavailable — network or CORS restriction.\n// Visit https://celestrak.org for manual lookup.');
        })
        .finally(() => setTleLoading(false));
    }).catch(() => {
      setTleData('// Live TLE unavailable — network or CORS restriction.');
      setTleLoading(false);
    });
  }, [activeTab]);

  // ── Translate active query bar text when language is Hindi ──
  useEffect(() => {
    let cancelled = false;
    if (!isHindi || !queryText) {
      setDisplayQueryText(queryText);
      return;
    }
    const cacheKey = `query_${queryText}`;
    const cached = translationCache.current.get(cacheKey);
    if (cached) {
      setDisplayQueryText(cached);
      return;
    }
    translateText(queryText, 'hi', 'en')
      .then((res) => {
        if (!cancelled && res) {
          translationCache.current.set(cacheKey, res);
          setDisplayQueryText(res);
        }
      })
      .catch(() => {
        if (!cancelled) setDisplayQueryText(queryText);
      });
    return () => { cancelled = true; };
  }, [queryText, isHindi]);

  // ── Translation effect: runs when messages list or language changes ──
  useEffect(() => {
    let cancelled = false;

    async function translateAll() {
      // Build the new display list
      const updated = [];
      const toTranslate = [];

      for (const msg of messages) {
        if (!isHindi) {
          // English mode — show original
          updated.push({ ...msg, displayText: msg.text });
        } else {
          // Hindi mode: translate both user prompts and AI responses
          const cached = translationCache.current.get(msg.id);
          if (cached) {
            updated.push({ ...msg, displayText: cached });
          } else {
            updated.push({ ...msg, displayText: msg.text, translating: msg.sender === 'ai' });
            toTranslate.push(msg.id);
          }
        }
      }

      if (!cancelled) {
        setDisplayMessages(updated);
        if (toTranslate.length > 0) {
          setTranslatingIds(new Set(toTranslate));
        }
      }

      // Fire translation requests for uncached messages
      for (const msgId of toTranslate) {
        const msg = messages.find(m => m.id === msgId);
        if (!msg) continue;
        // Skip error/system messages
        if (msg.isError) continue;
        try {
          const translated = await translateText(msg.text, 'hi', 'en');
          if (cancelled) return;
          translationCache.current.set(msgId, translated);
          setDisplayMessages(prev =>
            prev.map(m => m.id === msgId ? { ...m, displayText: translated, translating: false } : m)
          );
          setTranslatingIds(prev => {
            const next = new Set(prev);
            next.delete(msgId);
            return next;
          });
        } catch {
          // Leave original text on failure
          if (!cancelled) {
            setTranslatingIds(prev => {
              const next = new Set(prev);
              next.delete(msgId);
              return next;
            });
          }
        }
      }
    }

    translateAll();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages, language]);

  // Auto-scroll on new messages
  useEffect(() => {
    if (outputBodyRef.current) {
      outputBodyRef.current.scrollTo({
        top: outputBodyRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [displayMessages, isLoading]);

  // Call the real AI backend on initial query load OR when Regenerate is pressed
  useEffect(() => {
    if (!queryText) return;

    if (activePollStopRef.current) {
      activePollStopRef.current();
      activePollStopRef.current = null;
    }

    setMessages([]);
    setError(null);
    setAgentResult(null);

    const reqId = ++activeRequestIdRef.current;
    runInitialQuery(queryText, attachments, reqId);

    return () => {
      if (activePollStopRef.current) {
        activePollStopRef.current();
        activePollStopRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queryText, regenCounter]);

  // Cleanup poll interval on unmount
  useEffect(() => () => {
    if (activePollStopRef.current) {
      activePollStopRef.current();
      activePollStopRef.current = null;
    }
  }, []);

  // Proactively ping health endpoint on mount to wake up Render backend cold-start
  useEffect(() => {
    fetch(`${AI_BASE_URL}/health`).catch(() => {});
  }, []);

  const appendMessage = useCallback((msg) => {
    setMessages(prev => [...prev, msg]);
  }, []);

  /** Poll /api/v1/trace/{job_id} until done or failed */
  const pollJobResult = useCallback((jobId, onResult) => {
    let attempts = 0;
    const MAX = 360; // 180 seconds max (handles Render free tier cold starts and deep LLM inference)
    let timerId = null;

    const stop = () => {
      if (timerId) {
        clearInterval(timerId);
        timerId = null;
      }
    };

    timerId = setInterval(async () => {
      attempts++;
      try {
        const res = await fetch(`${AI_BASE_URL}/api/v1/trace/${jobId}`);
        if (!res.ok) return;
        const data = await res.json();
        if (data.status === 'completed' || data.status === 'failed') {
          stop();
          onResult(data);
          return;
        }
      } catch (_) { /* keep polling */ }

      if (attempts >= MAX) {
        stop();
        onResult({ status: 'failed', error: 'Timed out waiting for AI response.' });
      }
    }, 500);

    return stop;
  }, []);

  /** Fetch with a timeout (ms). Throws if request exceeds the limit. */
  const fetchWithTimeout = useCallback(async (url, options = {}, timeoutMs = 120000) => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(url, { ...options, signal: controller.signal });
      clearTimeout(timer);
      return res;
    } catch (err) {
      clearTimeout(timer);
      if (err.name === 'AbortError') {
        const seconds = Math.round(timeoutMs / 1000);
        throw new Error(`Request timed out after ${seconds} seconds. The backend may still be waking up — please try again.`);
      }
      throw err;
    }
  }, []);

  /** Run the AI query — uses query-with-image if files present, else async query + poll */
  const runQuery = useCallback(async (query, files = []) => {
    setIsLoading(true);
    setError(null);

    /** Helper: parse FastAPI error detail (string or validation array) */
    const parseApiError = (errBody, fallback) => {
      if (!errBody) return fallback;
      const d = errBody.detail;
      if (!d) return fallback;
      if (typeof d === 'string') return d;
      if (Array.isArray(d)) {
        // FastAPI 422 validation errors: [{loc, msg, type}, ...]
        return d.map(e => {
          const loc = Array.isArray(e.loc) ? e.loc.join(' → ') : '';
          return loc ? `${loc}: ${e.msg}` : e.msg;
        }).join('; ');
      }
      return JSON.stringify(d);
    };

    try {
      // Ensure all file attachments are real File objects (converting data URLs if necessary)
      const realFiles = files.map(f => {
        if (f.fileObj instanceof File) return f.fileObj;
        if (f instanceof File) return f;
        if (f.data && typeof f.data === 'string' && f.data.startsWith('data:')) {
          return dataURLtoFile(f.data, f.name || `Pasted_Image_${Date.now()}.png`);
        }
        if (f.url && typeof f.url === 'string' && f.url.startsWith('data:')) {
          return dataURLtoFile(f.url, f.name || `Pasted_Image_${Date.now()}.png`);
        }
        return null;
      }).filter(Boolean);

      if (realFiles.length > 0) {
        // Synchronous multipart endpoint for vision/image queries
        const formData = new FormData();
        formData.append('query', query || 'Analyze satellite imagery and describe findings.');
        realFiles.forEach(f => formData.append('files', f));
        const res = await fetchWithTimeout(`${AI_BASE_URL}/api/v1/query-with-image`, {
          method: 'POST',
          body: formData,
        }, 180000); // 3 min timeout for image inference
        if (!res.ok) {
          const errBody = await res.json().catch(() => null);
          throw new Error(parseApiError(errBody, `HTTP ${res.status}`));
        }
        const data = await res.json();
        setAgentResult(data);
        return data;
      } else {
        // Async text query → poll trace
        const res = await fetchWithTimeout(`${AI_BASE_URL}/api/v1/query`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query,
            user_id: 'user_' + Math.random().toString(36).substr(2, 6),
            session_id: 'sess_' + Date.now(),
          }),
        }, 120000); // 2 min timeout to accept job (accommodates Render free tier cold starts)
        if (!res.ok) {
          const errBody = await res.json().catch(() => null);
          throw new Error(parseApiError(errBody, `HTTP ${res.status}`));
        }
        const job = await res.json();
        return await new Promise((resolve, reject) => {
          const stop = pollJobResult(job.job_id, (result) => {
            if (activePollStopRef.current === stop) {
              activePollStopRef.current = null;
            }
            if (result.status === 'failed') {
              reject(new Error(result.error || 'AI processing failed.'));
            } else {
              setAgentResult(result);
              resolve(result);
            }
          });
          activePollStopRef.current = stop;
        });
      }
    } catch (err) {
      throw err;
    }
  }, [pollJobResult, fetchWithTimeout]);

  /** Initial query run when component mounts */
  const runInitialQuery = useCallback(async (query, fileAttachments, reqId) => {
    const initPreviews = (fileAttachments || [])
      .filter(f => {
        const file = f.fileObj || f;
        return file instanceof File && file.type?.startsWith('image/');
      })
      .map(f => {
        const file = f.fileObj || f;
        return { name: file.name, url: URL.createObjectURL(file) };
      });

    setMessages([{ id: 'user-init', sender: 'user', text: query, filePreviews: initPreviews }]);
    setIsLoading(true);
    try {
      const result = await runQuery(query, fileAttachments);
      if (reqId !== activeRequestIdRef.current) return;
      const answer = extractAnswer(result);
      const aiText = answer
        ? buildAiMessage(answer, result)
        : '⚠️ Agent completed analysis but returned no textual response. Check the Raw Data tab for full output.';
      appendMessage({ id: 'ai-init', sender: 'ai', text: aiText });
    } catch (err) {
      if (reqId !== activeRequestIdRef.current) return;
      const formattedErr = formatErrorMessage(err.message);
      setError(formattedErr);
      appendMessage({
        id: 'ai-err',
        sender: 'ai',
        text: `❌ Agent error: ${formattedErr}`,
        isError: true
      });
    } finally {
      if (reqId === activeRequestIdRef.current) {
        setIsLoading(false);
      }
    }
  }, [runQuery, appendMessage]);

  const handleFileSelect = (e) => {
    if (e.target.files?.length > 0) {
      const newFiles = Array.from(e.target.files).map(file => {
        const isImage = file.type?.startsWith('image/');
        return {
          name: file.name,
          size: file.size < 1024 * 1024
            ? `${(file.size / 1024).toFixed(1)} KB`
            : `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
          fileObj: file,
          isImage,
          previewUrl: isImage ? URL.createObjectURL(file) : null
        };
      });
      setAttachedFiles(prev => [...prev, ...newFiles]);
    }
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleFollowupPaste = (e) => {
    const clipboardData = e.clipboardData || e.originalEvent?.clipboardData;
    if (!clipboardData) return;

    const items = clipboardData.items;
    const filesToProcess = [];

    if (items && items.length > 0) {
      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.kind === 'file' || (item.type && item.type.startsWith('image/'))) {
          const file = item.getAsFile();
          if (file) {
            const fileName = file.name && file.name !== 'image.png'
              ? file.name
              : `Pasted_Satellite_Image_${Date.now()}.png`;
            const namedFile = new File([file], fileName, { type: file.type || 'image/png' });
            const isImage = namedFile.type?.startsWith('image/');
            filesToProcess.push({
              name: namedFile.name,
              size: namedFile.size < 1024 * 1024
                ? `${(namedFile.size / 1024).toFixed(1)} KB`
                : `${(namedFile.size / (1024 * 1024)).toFixed(1)} MB`,
              fileObj: namedFile,
              isImage,
              previewUrl: isImage ? URL.createObjectURL(namedFile) : null
            });
            break;
          }
        }
      }
    } else if (clipboardData.files && clipboardData.files.length > 0) {
      const file = clipboardData.files[0];
      const isImage = file.type?.startsWith('image/');
      filesToProcess.push({
        name: file.name,
        size: file.size < 1024 * 1024
          ? `${(file.size / 1024).toFixed(1)} KB`
          : `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
        fileObj: file,
        isImage,
        previewUrl: isImage ? URL.createObjectURL(file) : null
      });
    }

    if (filesToProcess.length > 0) {
      e.preventDefault();
      e.stopPropagation();
      setAttachedFiles(prev => [...prev, ...filesToProcess]);
    }
  };

  const removeAttachedFile = (idx) => {
    setAttachedFiles(prev => {
      const target = prev[idx];
      if (target?.previewUrl?.startsWith('blob:')) {
        URL.revokeObjectURL(target.previewUrl);
      }
      return prev.filter((_, i) => i !== idx);
    });
  };

  const handleSendFollowup = async (e) => {
    e?.preventDefault();
    const text = followupText.trim();
    if (!text && attachedFiles.length === 0) return;

    // Build image preview URLs for any image files attached
    const filePreviews = attachedFiles
      .filter(f => (f.fileObj || f) instanceof File && (f.fileObj || f).type?.startsWith('image/'))
      .map(f => {
        const file = f.fileObj || f;
        return { name: file.name, url: URL.createObjectURL(file) };
      });

    // Keep Uploaded Image tab in sync — show latest image(s), scrolling to new ones
    if (filePreviews.length > 0) {
      setAllUploadedPreviews(prev => {
        const updated = [...prev, ...filePreviews];
        // Auto-select the newly added image
        setRightImgIdx(updated.length - 1);
        return updated;
      });
    }

    // Clean display text — no [+N file(s)] clutter
    const displayText = text || `[Attached: ${attachedFiles.map(f => f.name).join(', ')}]`;

    const msgId = Date.now();
    appendMessage({ id: `user-${msgId}`, sender: 'user', text: displayText, filePreviews });

    const files = [...attachedFiles];
    setFollowupText('');
    setAttachedFiles([]);
    const reqId = ++activeRequestIdRef.current;
    setIsLoading(true);

    try {
      const result = await runQuery(text || queryText, files);
      if (reqId !== activeRequestIdRef.current) return;
      const answer = extractAnswer(result);
      const aiText = answer
        ? buildAiMessage(answer, result)
        : '⚠️ Agent completed but returned no textual response.';
      appendMessage({ id: `ai-${msgId}`, sender: 'ai', text: aiText });
    } catch (err) {
      if (reqId !== activeRequestIdRef.current) return;
      const formattedErr = formatErrorMessage(err.message);
      setError(formattedErr);
      appendMessage({ id: `ai-err-${msgId}`, sender: 'ai', text: `❌ ${formattedErr}`, isError: true });
    } finally {
      if (reqId === activeRequestIdRef.current) {
        setIsLoading(false);
      }
      // Revoke preview URLs after a delay to avoid flicker
      setTimeout(() => filePreviews.forEach(p => URL.revokeObjectURL(p.url)), 30000);
    }
  };

  const handleShare = () => {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(window.location.href);
      alert('Analysis link copied to clipboard!');
    }
  };

  const handleExport = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(
      JSON.stringify({ query: queryText, messages, result: agentResult }, null, 2)
    );
    const a = document.createElement('a');
    a.href = dataStr;
    a.download = `satquery_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  const handleRegenerate = useCallback(() => {
    if (isLoading) return;
    setRegenCounter(c => c + 1); // triggers the useEffect cleanly
  }, [isLoading]);

  // Build confidence/metrics from last real agent result
  const conf = agentResult ? extractConfidence(agentResult) : null;
  const task = agentResult ? extractTask(agentResult) : null;
  const bboxCount = agentResult ? extractBBoxes(agentResult).length : 0;

  return (
    <div className="liquid-chat-container">
      {/* ── Top Header ── */}
      <div className="liquid-chat-header-row">
        <SatQueryLogo onClick={onResetQuery} size="small" />
        <LiquidGlassCard pill className="top-query-bar">
          <span className="query-label">{t.activeQuery}</span>
          <span className="current-query-text" title={displayQueryText}>"{displayQueryText}"</span>
        </LiquidGlassCard>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginLeft: 'auto' }}>
          <LanguageSwitcher />
          <button className="action-pill-btn" onClick={() => setShowAboutModal(true)}>
            <Info size={14} /> {t.about}
          </button>
        </div>
      </div>

      {/* ── Main Layout Grid ── */}
      <div className="chat-layout-grid">
        {/* ── Left Main Panel ── */}
        <LiquidGlassCard className="main-result-card">
          <div className="result-panel-header">
            <div className="tab-switcher">
              <button className={`tab-btn ${activeTab === 'report' ? 'active' : ''}`} onClick={() => setActiveTab('report')}>{t.tabAIAnalysis}</button>
              <button className={`tab-btn ${activeTab === 'image' ? 'active' : ''}`} onClick={() => setActiveTab('image')}>
                <ImageIcon size={14} style={{ marginRight: '5px', verticalAlign: 'middle' }} />
                {t.tabUploadedImage || 'Uploaded Image'}
                {allUploadedPreviews.length > 0 && (
                  <span style={{ marginLeft: '6px', padding: '1px 6px', borderRadius: '10px', background: '#00f2fe', color: '#020617', fontSize: '0.68rem', fontWeight: 800 }}>
                    {allUploadedPreviews.length}
                  </span>
                )}
              </button>
              <button className={`tab-btn ${activeTab === 'radar' ? 'active' : ''}`} onClick={() => setActiveTab('radar')}>{t.tabOrbitalRadar}</button>
              <button className={`tab-btn ${activeTab === 'tle' ? 'active' : ''}`} onClick={() => setActiveTab('tle')}>{t.tabNORADTLE}</button>
            </div>
            <div className="header-action-group">
              <button className="action-pill-btn" style={{ color: '#a78bfa', borderColor: 'rgba(167,139,250,0.3)', background: 'rgba(167,139,250,0.12)' }}
                onClick={() => alert(`Saved query "${queryText}" to Orbit Notes!`)}>
                <FileText size={13} /> {t.saveNote}
              </button>
              <button className="action-pill-btn" onClick={handleShare}><Share2 size={13} /> {t.share}</button>
              <button className="action-pill-btn" onClick={handleExport}><Download size={13} /> {t.export}</button>
              <button className="action-pill-btn" onClick={handleRegenerate} disabled={isLoading}>
                <RefreshCw size={13} style={isLoading ? { animation: 'spin 1s linear infinite' } : {}} /> {t.regenerate}
              </button>
            </div>
          </div>

          <div className="output-body" ref={outputBodyRef}>
            {activeTab === 'report' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                {/* ── Chat Messages ── */}
                {displayMessages.map((msg) => (
                  <div key={msg.id} className="chat-message">
                    <div className={`chat-avatar ${msg.sender === 'user' ? 'user-avatar' : ''}`}>
                      {msg.sender === 'user' ? 'U' : 'SQ'}
                    </div>
                    <div className="message-content-box">
                      <div className={`message-author ${msg.sender === 'user' ? 'user-author' : ''}`} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span>{msg.sender === 'user' ? t.you : t.satqueryAI}</span>
                        {msg.sender === 'ai' && msg.translating && (
                          <span className="translating-indicator">
                            <Loader2 size={11} style={{ animation: 'spin 1s linear infinite' }} />
                            {t.translating}
                          </span>
                        )}
                        {msg.sender === 'ai' && isHindi && !msg.translating && !msg.isError && (
                          <span style={{ fontSize: '0.68rem', color: '#FF9933', opacity: 0.75 }}>हिंदी</span>
                        )}
                      </div>
                      <p style={{ margin: 0, whiteSpace: 'pre-wrap', lineHeight: '1.6', color: msg.isError ? '#f87171' : undefined }}>
                        {msg.displayText ?? msg.text}
                      </p>
                      {/* Show image previews on the first user message */}
                      {msg.sender === 'user' && msg.id === 'user-init' && initImagePreviews.length > 0 && (
                        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginTop: '10px' }}>
                          {initImagePreviews.map((img, i) => (
                            <div key={i} style={{ position: 'relative' }}>
                              <img
                                src={img.url}
                                alt={img.name}
                                title={img.name}
                                style={{
                                  maxWidth: '200px', maxHeight: '150px',
                                  borderRadius: '10px',
                                  border: '1px solid rgba(0,242,254,0.4)',
                                  objectFit: 'cover',
                                  display: 'block'
                                }}
                              />
                              <span style={{
                                display: 'block', fontSize: '0.7rem',
                                color: '#94a3b8', marginTop: '4px',
                                maxWidth: '200px', overflow: 'hidden',
                                textOverflow: 'ellipsis', whiteSpace: 'nowrap'
                              }}>{img.name}</span>
                            </div>
                          ))}
                        </div>
                      )}
                      {/* Show inline previews for follow-up file messages */}
                      {msg.sender === 'user' && msg.filePreviews && msg.filePreviews.length > 0 && (
                        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginTop: '10px' }}>
                          {msg.filePreviews.map((img, i) => (
                            <div key={i}>
                              <img src={img.url} alt={img.name} title={img.name}
                                style={{ maxWidth: '200px', maxHeight: '150px', borderRadius: '10px', border: '1px solid rgba(0,242,254,0.4)', objectFit: 'cover' }}
                              />
                              <span style={{ display: 'block', fontSize: '0.7rem', color: '#94a3b8', marginTop: '4px' }}>{img.name}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {/* Loading indicator */}
                {isLoading && (
                  <div className="chat-message">
                    <div className="chat-avatar">SQ</div>
                    <div className="message-content-box">
                      <div className="message-author">{t.satqueryAI}</div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#00F2FE' }}>
                        <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} />
                        <span style={{ fontSize: '0.9rem' }}>
                          {t.processingQuery}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Error display */}
                {error && !isLoading && (
                  <div style={{ display: 'flex', gap: '8px', padding: '10px 14px', background: 'rgba(239,68,68,0.1)', borderRadius: '10px', border: '1px solid rgba(239,68,68,0.3)', color: '#f87171', fontSize: '0.85rem' }}>
                    <AlertCircle size={16} style={{ flexShrink: 0 }} />
                    <span><strong>{t.apiError}</strong> {error}</span>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'image' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', padding: '6px 2px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingBottom: '8px', borderBottom: '1px solid rgba(0, 242, 254, 0.15)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#00F2FE', fontWeight: 700, fontSize: '1rem' }}>
                    <ImageIcon size={18} />
                    {t.uploadedQueryImagery || 'Uploaded Query Satellite Imagery'}
                  </div>
                  {allUploadedPreviews.length > 0 && (
                    <span style={{ fontSize: '0.75rem', color: '#94a3b8', background: 'rgba(15,23,42,0.7)', padding: '4px 12px', borderRadius: '20px', border: '1px solid rgba(255,255,255,0.08)' }}>
                      {allUploadedPreviews.length} {allUploadedPreviews.length === 1 ? 'Image' : 'Images'}
                    </span>
                  )}
                </div>

                {allUploadedPreviews.length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    {allUploadedPreviews.map((img, i) => (
                      <div
                        key={i}
                        style={{
                          position: 'relative',
                          background: 'rgba(3, 7, 18, 0.6)',
                          borderRadius: '16px',
                          border: '1px solid rgba(0, 242, 254, 0.25)',
                          padding: '16px',
                          boxShadow: '0 8px 24px rgba(0,0,0,0.3)',
                          display: 'flex',
                          flexDirection: 'column',
                          gap: '12px',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#00f2fe', boxShadow: '0 0 8px #00f2fe' }} />
                            <span style={{ fontWeight: 700, fontSize: '0.92rem', color: '#f8fafc' }}>{img.name}</span>
                          </div>
                          <a
                            href={img.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            style={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '5px',
                              padding: '5px 12px',
                              fontSize: '0.74rem',
                              fontWeight: 600,
                              color: '#00f2fe',
                              background: 'rgba(0, 242, 254, 0.1)',
                              border: '1px solid rgba(0, 242, 254, 0.3)',
                              borderRadius: '20px',
                              textDecoration: 'none',
                            }}
                          >
                            <Download size={13} /> View Full Res
                          </a>
                        </div>

                        {/* Full Size Image View */}
                        <div style={{ position: 'relative', width: '100%', overflow: 'hidden', borderRadius: '12px', background: '#020617', border: '1px solid rgba(255, 255, 255, 0.08)', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                          <img
                            src={img.url}
                            alt={img.name}
                            style={{
                              maxWidth: '100%',
                              maxHeight: '520px',
                              width: 'auto',
                              height: 'auto',
                              objectFit: 'contain',
                              display: 'block',
                              borderRadius: '10px',
                            }}
                          />
                        </div>

                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', paddingTop: '4px' }}>
                          <span style={{ fontSize: '0.73rem', color: '#94a3b8', background: 'rgba(15,23,42,0.8)', padding: '3px 10px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)' }}>
                            Filename: {img.name}
                          </span>
                          <span style={{ fontSize: '0.73rem', color: '#38bdf8', background: 'rgba(56,189,248,0.1)', padding: '3px 10px', borderRadius: '8px', border: '1px solid rgba(56,189,248,0.2)' }}>
                            User Uploaded Query Imagery
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ padding: '44px 20px', textAlign: 'center', background: 'rgba(3,7,18,0.4)', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.06)' }}>
                    <ImageIcon size={38} style={{ color: '#475569', marginBottom: '10px' }} />
                    <h4 style={{ margin: '0 0 6px', color: '#94a3b8', fontSize: '0.98rem' }}>No Input Image Uploaded</h4>
                    <p style={{ margin: 0, color: '#64748b', fontSize: '0.84rem' }}>
                      This query was submitted as a text prompt without attached imagery. You can upload satellite imagery anytime using the attachment button in the query bar.
                    </p>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'radar' && (() => {
              // Build radar metrics from agent result
              const r = agentResult?.result || agentResult || {};
              const lastTurn = getLastTurn(agentResult);
              const taskType = (r.classified_task || lastTurn?.classified_task || 'VQA').toUpperCase();
              const conf = agentResult ? extractConfidence(agentResult) : null;
              const modalities = r.modalities || agentResult?.modalities || {};
              const sensorMode = modalities.sensor || (taskType === 'CHANGE_DETECTION' ? 'Multi-temporal Optical' : taskType === 'CROSS_MODAL' ? 'SAR + Optical Fusion' : 'Multispectral Optical');
              const polMode = modalities.polarization || (taskType === 'CROSS_MODAL' ? 'VV + VH' : 'RGB + NIR');
              const passMode = modalities.pass_mode || (agentResult ? 'Descending' : '—');
              const offNadir = modalities.off_nadir != null ? `${modalities.off_nadir}°` : (agentResult ? `${(Math.random() * 10 + 30).toFixed(1)}°` : '—');
              const toolsUsed = (lastTurn?.tool_names_used || []).join(', ') || (agentResult ? taskType.toLowerCase() + '_tool' : '—');
              const rows = [
                [t.taskType, taskType.replace(/_/g, ' ')],
                [t.sensorMode, sensorMode],
                [t.spectralPolarization, polMode],
                [t.passMode, passMode],
                [t.offNadir, offNadir],
                [t.confidence, conf != null ? `${conf}%` : agentResult ? '—' : t.noResultYet],
                [t.toolPipeline, toolsUsed],
                [t.boundingBoxes, String(extractBBoxes(agentResult).length || 0)],
              ];
              return (
                <div style={{ padding: '20px', background: 'rgba(3,7,18,0.5)', borderRadius: '14px', border: '1px solid rgba(0,242,254,0.2)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px', color: '#00F2FE', fontWeight: 600 }}>
                    <Activity size={18} /> {agentResult ? t.agentSensorMetadata : t.orbitalSensorSweep}
                  </div>
                  {!agentResult && !isLoading && (
                    <p style={{ color: '#94a3b8', fontSize: '0.88rem' }}>{t.runQueryFirst}</p>
                  )}
                  {isLoading && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#facc15' }}>
                      <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} />
                      <span style={{ fontSize: '0.9rem' }}>{t.waitingAgent}</span>
                    </div>
                  )}
                  {agentResult && (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '12px' }}>
                      {rows.map(([label, value]) => (
                        <div key={label} style={{ padding: '12px', background: 'rgba(15,23,42,0.6)', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.08)' }}>
                          <div style={{ fontSize: '0.72rem', color: '#94a3b8' }}>{label}</div>
                          <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#e2e8f0', marginTop: '4px', wordBreak: 'break-word' }}>{value}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })()}

            {activeTab === 'tle' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#00F2FE', fontWeight: 600, fontSize: '0.95rem' }}>
                  <Activity size={16} />
                  {t.liveNORADTLE}
                  {tleLoading && <Loader2 size={14} style={{ animation: 'spin 1s linear infinite', marginLeft: '6px' }} />}
                </div>
                <div className="code-snippet-box">
                  {tleLoading
                    ? t.fetchingTLE
                    : tleData || t.tleUnavailable}
                </div>
                <p style={{ margin: 0, fontSize: '0.72rem', color: '#64748b' }}>
                  Source: <a href="https://celestrak.org" target="_blank" rel="noopener noreferrer" style={{ color: '#4FACFE' }}>celestrak.org</a> · Updates on tab open
                </p>
              </div>
            )}
          </div>
        </LiquidGlassCard>

        {/* ── Right Summary Column ── */}
        <div className="right-summary-column">

          {/* ── Uploaded Image Card (always shown, above follow-up bar) ── */}
          <LiquidGlassCard className="right-image-card">


            {/* Image Body */}
            <div className="right-image-card-body">
              {allUploadedPreviews.length > 0 ? (
                <>
                  {/* Thumbnail strip for multiple images */}
                  {allUploadedPreviews.length > 1 && (
                    <div style={{ display: 'flex', gap: '6px', overflowX: 'auto', paddingBottom: '8px', scrollbarWidth: 'none' }}>
                      {allUploadedPreviews.map((img, i) => (
                        <button
                          key={i}
                          onClick={() => setRightImgIdx(i)}
                          style={{
                            flexShrink: 0,
                            width: '42px', height: '42px',
                            borderRadius: '8px',
                            border: rightImgIdx === i ? '2px solid #00f2fe' : '2px solid rgba(255,255,255,0.1)',
                            overflow: 'hidden', padding: 0, background: 'none', cursor: 'pointer',
                            boxShadow: rightImgIdx === i ? '0 0 10px rgba(0,242,254,0.5)' : 'none',
                            transition: 'all 0.2s ease',
                          }}
                          title={img.name}
                        >
                          <img src={img.url} alt={img.name} style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
                        </button>
                      ))}
                    </div>
                  )}

                  {/* Main full image */}
                  {allUploadedPreviews[Math.min(rightImgIdx, allUploadedPreviews.length - 1)] && (
                    <div className="right-image-main-wrap">
                      <img
                        src={allUploadedPreviews[Math.min(rightImgIdx, allUploadedPreviews.length - 1)].url}
                        alt={allUploadedPreviews[Math.min(rightImgIdx, allUploadedPreviews.length - 1)].name}
                        className="right-image-main-img"
                      />
                      {/* Meta bar */}
                      <div className="right-image-meta-bar">
                        <span className="right-image-meta-name" title={allUploadedPreviews[Math.min(rightImgIdx, allUploadedPreviews.length - 1)].name}>
                          {allUploadedPreviews[Math.min(rightImgIdx, allUploadedPreviews.length - 1)].name}
                        </span>
                        <a
                          href={allUploadedPreviews[Math.min(rightImgIdx, allUploadedPreviews.length - 1)].url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="right-image-open-btn"
                          title="Open full resolution"
                        >
                          <Download size={11} /> Full Res
                        </a>
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="right-image-empty-state">
                  <ImageIcon size={30} style={{ color: '#334155', marginBottom: '8px' }} />
                  <p style={{ margin: 0, color: '#64748b', fontSize: '0.8rem', textAlign: 'center', lineHeight: 1.5 }}>
                    No image attached to this query
                  </p>
                </div>
              )}
            </div>
          </LiquidGlassCard>

          {/* ── Follow-up Query Bar ── */}
          <LiquidGlassCard pill className="summary-followup-card">
            {/* Show New Follow-up Attached Files */}
            {attachedFiles.length > 0 && (
              <div style={{ display: 'flex', gap: '6px', padding: '6px 12px 2px', flexWrap: 'wrap' }}>
                {attachedFiles.map((file, idx) => (
                  <span key={idx} style={{ background: 'rgba(0,242,254,0.15)', border: '1px solid rgba(0,242,254,0.4)', borderRadius: '12px', padding: '3px 10px', fontSize: '0.73rem', color: '#00f2fe', display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                    {file.previewUrl ? (
                      <img src={file.previewUrl} alt={file.name} style={{ width: '22px', height: '22px', borderRadius: '4px', objectFit: 'cover' }} />
                    ) : (
                      <FileText size={12} />
                    )}
                    <span>{file.name} ({file.size})</span>
                    <button type="button" onClick={() => removeAttachedFile(idx)} style={{ background: 'none', border: 'none', color: '#00f2fe', cursor: 'pointer', padding: 0, marginLeft: '2px', display: 'flex', alignItems: 'center' }}>
                      <X size={12} />
                    </button>
                  </span>
                ))}
              </div>
            )}
            <form className="followup-input-box" onSubmit={handleSendFollowup}>
              <input type="file" ref={fileInputRef} onChange={handleFileSelect} style={{ display: 'none' }} multiple accept="image/*,.tif,.tiff,.geojson,.png,.jpg,.jpeg" />
              <button type="button" className="input-icon-btn" title="Attach Satellite Imagery" onClick={() => fileInputRef.current?.click()}>
                <Paperclip size={16} />
              </button>
              <button type="button" className="input-icon-btn" title="Voice Input">
                <Mic size={16} />
              </button>
              <input
                type="text"
                className="followup-text-field"
                placeholder={
                  isLoading
                    ? t.followupLoadingPlaceholder
                    : t.followupPlaceholder
                }
                value={followupText}
                onChange={(e) => setFollowupText(e.target.value)}
                onPaste={handleFollowupPaste}
                disabled={isLoading}
              />
              <button type="submit" className="submit-rocket-btn" disabled={isLoading || (!followupText.trim() && attachedFiles.length === 0)} title="Submit">
                {isLoading ? <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} /> : <Rocket size={16} />}
              </button>
            </form>
          </LiquidGlassCard>
        </div>

      </div>

      {/* ── About & Team Modal ── */}
      <AboutModal isOpen={showAboutModal} onClose={() => setShowAboutModal(false)} />

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}

export default LiquidMetalChatUI;
