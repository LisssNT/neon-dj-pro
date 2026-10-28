/* Local, browser-side music feature analysis. This is DSP, not a pretrained AI model. */
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wlenR = Math.cos(ang), wlenI = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wr = 1, wi = 0;
      for (let j = 0; j < len / 2; j++) {
        const uR = re[i + j], uI = im[i + j];
        const vR = re[i + j + len / 2] * wr - im[i + j + len / 2] * wi;
        const vI = re[i + j + len / 2] * wi + im[i + j + len / 2] * wr;
        re[i + j] = uR + vR; im[i + j] = uI + vI;
        re[i + j + len / 2] = uR - vR; im[i + j + len / 2] = uI - vI;
        const nwr = wr * wlenR - wi * wlenI; wi = wr * wlenI + wi * wlenR; wr = nwr;
      }
    }
  }
}
const clamp = (x,a,b)=>Math.max(a,Math.min(b,x));
function norm(a){let m=0;for(const x of a)if(x>m)m=x;if(m>0)for(let i=0;i<a.length;i++)a[i]/=m;return a;}
function movingAvg(a, r=2){return a.map((_,i)=>{let s=0,n=0;for(let j=Math.max(0,i-r);j<=Math.min(a.length-1,i+r);j++){s+=a[j];n++}return s/n})}
function estimateTempo(onsets, hopSeconds) {
  if(onsets.length < 100) return {bpm:null, confidence:0, candidates:[]};
  const o=movingAvg(norm(onsets.slice()),2); const mean=o.reduce((a,b)=>a+b,0)/o.length;
  for(let i=0;i<o.length;i++) o[i]-=mean;
  const minLag=Math.max(2,Math.floor(60/(190*hopSeconds))), maxLag=Math.min(Math.floor(60/(55*hopSeconds)),Math.floor(o.length/2));
  const vals=[];
  for(let lag=minLag;lag<=maxLag;lag++) {let sum=0,den=0;for(let i=lag;i<o.length;i++){sum+=o[i]*o[i-lag];den+=Math.abs(o[i])+Math.abs(o[i-lag])}vals.push({lag,score:sum/(den+1e-9),bpm:60/(lag*hopSeconds)});}
  vals.sort((a,b)=>b.score-a.score);
  const cands=[];for(const p of vals){if(cands.every(q=>Math.abs(q.bpm-p.bpm)>3))cands.push(p);if(cands.length===5)break;}
  if(!cands.length)return {bpm:null,confidence:0,candidates:[]};
  // Prefer common musical tempo range; compare half/double tempo equivalence.
  let chosen=cands[0];
  const equivalents=[];for(const p of vals){for(const bpm of [p.bpm,p.bpm*2,p.bpm/2])if(bpm>=65&&bpm<=175)equivalents.push({bpm,score:p.score,lag:p.lag});}
  equivalents.sort((a,b)=>b.score-a.score);
  if(equivalents.length){chosen=equivalents[0];for(const e of equivalents){if(e.score>=chosen.score*.94&&e.bpm>=85&&e.bpm<=145){chosen=e;break;}}}
  const scoreValues=vals.map(x=>x.score).sort((a,b)=>a-b);
  const medianScore=scoreValues[Math.floor(scoreValues.length/2)]||0;
  const maxScore=Math.max(1e-9,vals[0].score);
  const relativePeak=clamp((chosen.score-medianScore)/(maxScore-medianScore+1e-9),0,1);
  return {bpm:Math.round(chosen.bpm*10)/10,confidence:relativePeak,candidates:cands.map(x=>({bpm:Math.round(x.bpm*10)/10,score:Math.round(x.score*1000)/1000}))};
}
function guessKey(chroma){
 const major=[6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88];
 const minor=[6.33,2.68,3.52,5.38,2.6,3.53,2.54,4.75,3.98,2.69,3.34,3.17];
 const names=['C','C♯/D♭','D','D♯/E♭','E','F','F♯/G♭','G','G♯/A♭','A','A♯/B♭','B'];
 const c=chroma.slice(); const avg=c.reduce((a,b)=>a+b,0)/12;for(let i=0;i<12;i++)c[i]-=avg;
 let top={score:-Infinity};
 for(let shift=0;shift<12;shift++)for(const mode of ['major','minor']){const p=mode==='major'?major:minor;let a=0,b=0,d=0;for(let i=0;i<12;i++){a+=c[(i+shift)%12]*p[i];b+=c[(i+shift)%12]**2;d+=p[i]**2;}const score=a/Math.sqrt((b*d)||1);if(score>top.score)top={score,key:names[shift],mode};}
 return {key:top.key,mode:top.mode,confidence:clamp((top.score+0.2)/1.6,0,1)};
}
function getGenre({bpm,energy,lowRatio,centroid,flatness,onsetDensity}){
 const e=energy||0, low=lowRatio||0, c=centroid||0, flat=flatness||0, d=onsetDensity||0, b=bpm||100;
 const scores={House:0,Techno:0,EDM:0,'Hip-Hop / Trap':0,Phonk:0,'Drum & Bass':0,Pop:0,'Ambient / Downtempo':0};
 if(b>=118&&b<=134){scores.House+=.45;scores.EDM+=.3;}if(b>=125&&b<=150){scores.Techno+=.45;scores.EDM+=.2;}if(b>=160&&b<=185)scores['Drum & Bass']+=.7;
 if(b>=60&&b<=105){scores['Hip-Hop / Trap']+=.4;scores.Phonk+=.2;} if(low>.32){scores.Phonk+=.3;scores['Hip-Hop / Trap']+=.2;scores.House+=.12;}if(e>.55&&c>2400)scores.EDM+=.2;if(e>.5&&d>.35)scores.Techno+=.18;if(e<.25){scores['Ambient / Downtempo']+=.65;scores.Pop-=.1;}if(flat>.35)scores['Hip-Hop / Trap']+=.08;if(b>=90&&b<125&&e>.25&&e<.65)scores.Pop+=.28;
 let best=Object.entries(scores).sort((a,b)=>b[1]-a[1])[0];const total=Object.values(scores).reduce((a,v)=>a+Math.max(0,v),0)||1;return {genre:best[0],confidence:clamp(best[1]/(total*.72),.22,.77),alternatives:Object.entries(scores).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([genre,score])=>({genre,score:Math.max(0,score)}))};
}
function sectionMap(energy, low, onset, centroid, frameSec, duration){
 const win=Math.max(1,Math.round(8/frameSec));const chunks=[];
 for(let i=0;i<energy.length;i+=win){const end=Math.min(energy.length,i+win);let a=0,l=0,o=0,c=0,n=end-i;for(let j=i;j<end;j++){a+=energy[j];l+=low[j];o+=onset[j];c+=centroid[j];}chunks.push({start:i*frameSec,end:end*frameSec,energy:a/n,low:l/n,onset:o/n,centroid:c/n});}
 if(!chunks.length)return [];
 const eVals=chunks.map(x=>x.energy).sort((a,b)=>a-b), med=eVals[Math.floor(eVals.length/2)]||0, max=Math.max(...eVals,1e-9);
 return chunks.map((x,i)=>{const prev=chunks[Math.max(0,i-1)],next=chunks[Math.min(chunks.length-1,i+1)];const trend=(next.energy-prev.energy);let label='Groove';
 if(x.start<20&&x.energy<med*.9)label='Intro';else if(x.energy>med*1.28&&x.onset>chunks.reduce((s,q)=>s+q.onset,0)/chunks.length*1.08)label='High energy';else if(x.energy<med*.72)label='Breakdown';else if(trend>max*.12)label='Build-up';else if(trend<-max*.12)label='Break / release';
 return {start:Math.round(x.start),end:Math.min(Math.round(x.end),Math.round(duration)),label,energy:Math.round(x.energy*1000)/10,low:Math.round(x.low*1000)/10,onset:Math.round(x.onset*1000)/10,centroid:Math.round(x.centroid)};});
}
self.onmessage = (ev) => {
 const {id,samples,sampleRate,duration}=ev.data;
 try {
  const N=2048, H=512, frameSec=H/sampleRate, frames=Math.max(0,Math.floor((samples.length-N)/H));
  if(frames<20)throw new Error('Недостаточно данных для анализа.');
  const win=new Float32Array(N);for(let i=0;i<N;i++)win[i]=.5-.5*Math.cos(2*Math.PI*i/(N-1));
  const re=new Float64Array(N),im=new Float64Array(N),prev=new Float32Array(N/2+1),onset=new Float32Array(frames),rms=new Float32Array(frames),low=new Float32Array(frames),mid=new Float32Array(frames),high=new Float32Array(frames),centroid=new Float32Array(frames),flatness=new Float32Array(frames),chroma=new Float64Array(12);
  const progressEvery=Math.max(1,Math.floor(frames/30));
  for(let f=0;f<frames;f++){
   const off=f*H;let ss=0;for(let i=0;i<N;i++){const v=samples[off+i]||0;re[i]=v*win[i];im[i]=0;ss+=v*v;}rms[f]=Math.sqrt(ss/N);fft(re,im);
   let flux=0,tot=0,lo=0,mi=0,hi=0,cent=0,geo=0,arith=0,bc=0;
   for(let k=1;k<=N/2;k++){const mag=Math.hypot(re[k],im[k]);const val=Math.log1p(mag*18);const diff=val-prev[k];if(diff>0)flux+=diff;prev[k]=val;tot+=mag;const hz=k*sampleRate/N;if(hz<150)lo+=mag;else if(hz<2000)mi+=mag;else hi+=mag;cent+=hz*mag;arith+=mag;geo+=Math.log(mag+1e-9);bc++;if(hz>=35&&hz<5000){const midi=69+12*Math.log2(hz/440);const pc=((Math.round(midi)%12)+12)%12;chroma[pc]+=mag*Math.pow(Math.min(1,hz/100),.2);}}
   onset[f]=flux/(N/2);low[f]=lo/(tot+1e-9);mid[f]=mi/(tot+1e-9);high[f]=hi/(tot+1e-9);centroid[f]=cent/(tot+1e-9);flatness[f]=Math.exp(geo/(bc||1))/(arith/(bc||1)+1e-9);
   if(f%progressEvery===0)self.postMessage({id,type:'progress',progress:f/frames});
  }
  const onsetArr=Array.from(onset),tempo=estimateTempo(onsetArr,frameSec),energyAvg=rms.reduce((s,v)=>s+v,0)/rms.length, lowAvg=low.reduce((s,v)=>s+v,0)/low.length, centroidAvg=centroid.reduce((s,v)=>s+v,0)/centroid.length, flatAvg=flatness.reduce((s,v)=>s+v,0)/flatness.length;
  const onsetMean=onset.reduce((s,v)=>s+v,0)/onset.length, onsetStd=Math.sqrt(onset.reduce((s,v)=>s+(v-onsetMean)**2,0)/onset.length);const onsetDensity=onset.filter(v=>v>onsetMean+onsetStd*.8).length/onset.length;
  const key=guessKey(Array.from(chroma)),genre=getGenre({bpm:tempo.bpm,energy:energyAvg*2,lowRatio:lowAvg,centroid:centroidAvg,flatness:flatAvg,onsetDensity});
  const sections=sectionMap(Array.from(rms),Array.from(low),onsetArr,Array.from(centroid),frameSec,duration);
  const beatPeriod=tempo.bpm?60/tempo.bpm:null;const beatOffset=tempo.bpm?(()=>{let max=-1e9,idx=0;const lag=Math.max(1,Math.round(beatPeriod/frameSec));for(let i=0;i<Math.min(lag,Math.floor(onset.length*.15));i++){let s=0,n=0;for(let j=i;j<onset.length;j+=lag){s+=onset[j];n++;}if(s/(n||1)>max){max=s/(n||1);idx=i;}}return Math.round(idx*frameSec*1000)/1000;})():null;
  const result={duration:Math.round(duration*10)/10,bpm:tempo.bpm,bpmConfidence:tempo.confidence,tempoCandidates:tempo.candidates,key,genre,energy:Math.round(clamp(energyAvg*3.8,0,1)*100),lowRatio:Math.round(lowAvg*100),spectralCentroid:Math.round(centroidAvg),spectralFlatness:Math.round(flatAvg*1000)/1000,onsetDensity:Math.round(onsetDensity*100),beatOffset,sections,features:{rms:Array.from(rms).filter((_,i)=>i%Math.max(1,Math.floor(rms.length/100))===0).map(v=>Math.round(v*1000)/1000),low:Array.from(low).filter((_,i)=>i%Math.max(1,Math.floor(low.length/100))===0).map(v=>Math.round(v*1000)/1000)}};
  self.postMessage({id,type:'result',result});
 }catch(error){self.postMessage({id,type:'error',error:String(error&&error.message||error)});}
};
