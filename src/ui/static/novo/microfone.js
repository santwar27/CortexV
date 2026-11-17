"use strict";
(() => {
const mic = document.querySelector('#mic'), input = document.querySelector('#message-input');
const toast = message => window.cortexProjetos.toast(message);
let active = null;
function release(run) {
  clearTimeout(run.timer);
  run.stream?.getTracks().forEach(track => track.stop());
  window.cortexCore.setLive('idle');
  mic.setAttribute('aria-label', 'Gravar voz');
}
function fail(run, message) {
  if (active !== run) return;
  run.failed = true;
  release(run); active = null;
  try { if (run.recorder?.state !== 'inactive') run.recorder?.stop(); } catch {}
  mic.disabled = false;
  toast(message);
}
function stop() {
  const run = active;
  if (!run) return;
  mic.disabled = true;
  try {
    if (run.recorder && run.recorder.state !== 'inactive') run.recorder.stop();
    else if (!run.recorder) { active = null; mic.disabled = false; }
  } catch { fail(run, 'Não consegui transcrever.'); }
  finally { release(run); }
}
async function transcribe(blob) {
  const placeholder = input.placeholder;
  mic.disabled = true; input.placeholder = 'Transcrevendo…';
  try {
    if (!blob.size) { toast('Não entendi nada na gravação.'); return; }
    const response = await fetch('/ui/voice/transcribe', {
      method: 'POST', credentials: 'same-origin',
      headers: {'Content-Type': blob.type || 'audio/webm'},
      body: blob, signal: AbortSignal.timeout(120000)
    });
    const data = await response.json();
    if (!response.ok) {
      window.cortexCore.setLive('idle');
      toast(response.status === 413 && data.error === 'too_long' ? 'A gravação passou de 1 minuto.' :
        response.status === 413 && data.error === 'too_large' ? 'A gravação ficou grande demais.' :
        response.status === 400 ? 'Não consegui ler o áudio.' :
        response.status === 503 ? 'A transcrição local não está disponível agora.' : 'Não consegui transcrever.');
      return;
    }
    const text = typeof data.text === 'string' ? data.text.trim() : '';
    if (!text) { toast('Não entendi nada na gravação.'); return; }
    input.value += (input.value ? ' ' : '') + text;
    input.dispatchEvent(new Event('input', {bubbles: true}));
    input.focus();
  } catch {
    window.cortexCore.setLive('idle');
    toast('Não consegui transcrever.');
  } finally {
    // A mode chosen during transcription owns its newer placeholder.
    if (input.placeholder === 'Transcrevendo…') input.placeholder = placeholder;
    mic.disabled = false;
  }
}
async function start() {
  if (mic.disabled) return;
  if (active) { stop(); return; }
  if (window.cortexConversa.inFlight) { toast('Espere a resposta terminar para gravar.'); return; }
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    toast('O microfone só funciona por HTTPS ou no próprio Mac.'); return;
  }
  const run = {chunks: []};
  active = run; mic.disabled = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    run.stream = stream;
    // Permission can resolve after leaving the page or cancelling this run.
    if (active !== run) { stream.getTracks().forEach(track => track.stop()); return; }
    if (window.cortexConversa.inFlight) { fail(run, 'Espere a resposta terminar para gravar.'); return; }
    const recorder = run.recorder = new MediaRecorder(stream);
    recorder.addEventListener('start', () => {
      if (active !== run || recorder.state !== 'recording') return;
      window.cortexCore.setLive('listening');
      mic.setAttribute('aria-label', 'Parar gravação');
      mic.disabled = false;
      run.timer = setTimeout(stop, 60000);
    });
    recorder.addEventListener('dataavailable', event => {
      if (!run.failed && event.data.size) run.chunks.push(event.data);
    });
    recorder.addEventListener('error', () => fail(run, 'Não consegui transcrever.'));
    recorder.addEventListener('stop', () => {
      if (active !== run) return;
      release(run); active = null;
      if (run.failed) { mic.disabled = false; return; }
      transcribe(new Blob(run.chunks, {type: recorder.mimeType || run.chunks[0]?.type || 'audio/webm'}));
    });
    stream.getTracks().forEach(track => track.addEventListener('ended', stop, {once: true}));
    recorder.start();
  } catch { fail(run, 'Sem permissão para usar o microfone.'); }
}
mic.addEventListener('click', start);
window.addEventListener('pagehide', () => {
  if (!active) return;
  active.failed = true; active.chunks = [];
  stop();
});
window.cortexMicrofone = {stop};
})();
