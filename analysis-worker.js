
'use strict';

/* Lightweight audio analysis in a Web Worker. */
self.onmessage = function (event) {
  const d = event.data || {};
  const id = d.id;
  const samples = d.samples;
  const sr = d.sampleRate;
  const duration = d.duration;

  try {
    if (!samples || !sr || samples.length < sr * 2) {
      throw new Error('Трек слишком короткий для оценки.');
    }

    const frame = Math.max(256, Math.floor(sr * 0.046));
    const hop = Math.max(128, Math.floor(frame / 2));
    const rms = [];
    let maxPeak = 0;

    for (let i = 0; i < samples.length; i++) {
      const v = samples[i];
      maxPeak = Math.max(maxPeak, Math.abs(v));
    }

    for (let start = 0; start + frame < samples.length; start += hop) {
      let sum = 0;
      for (let j = start; j < start + frame; j++) {
        sum += samples[j] * samples[j];
      }
      rms.push(Math.sqrt(sum / frame));
    }

    const onset = [];
    for (let i = 1; i < rms.length; i++) {
      onset.push(Math.max(0, rms[i] - rms[i - 1]));
    }

    const mean = onset.reduce((a, b) => a + b, 0) /
      Math.max(1, onset.length);

    const envRate = sr / hop;
    const minLag = Math.max(1, Math.floor(envRate * 60 / 180));
    const maxLag = Math.min(
      onset.length - 2,
      Math.ceil(envRate * 60 / 65)
    );

    let bestLag = 0;
    let bestCorr = -1;

    for (let lag = minLag; lag <= maxLag; lag++) {
      let corr = 0;
      let normA = 0;
      let normB = 0;

      for (let i = lag; i < onset.length; i++) {
        const a = onset[i] - mean;
        const b = onset[i - lag] - mean;
        corr += a * b;
        normA += a * a;
        normB += b * b;
      }

      const score = corr / Math.sqrt(normA * normB + 1e-12);

      if (score > bestCorr) {
        bestCorr = score;
        bestLag = lag;
      }
    }

    let bpm = bestLag ? 60 * envRate / bestLag : 0;
    if (bpm > 180) bpm /= 2;
    if (bpm > 0 && bpm < 68) bpm *= 2;

    const avgRms = rms.reduce((a, b) => a + b, 0) /
      Math.max(1, rms.length);
    const energy = Math.round(
      Math.max(0, Math.min(100, avgRms * 280))
    );

    self.postMessage({
      id,
      type: 'progress',
      progress: 0.9
    });

    self.postMessage({
      id,
      type: 'result',
      result: {
        bpm: bpm ? Math.round(bpm * 10) / 10 : null,
        bpmConfidence: Math.max(0, Math.min(1, bestCorr)),
        energy,
        peak: Math.round(maxPeak * 100),
        duration: Math.round(duration * 10) / 10,
        note: 'Approximate audio analysis.'
      }
    });
  } catch (e) {
    self.postMessage({
      id,
      type: 'error',
      error: e.message || String(e)
    });
  }
};
