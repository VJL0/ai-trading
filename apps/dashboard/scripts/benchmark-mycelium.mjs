// Run against the isolated fixture server, never the trading execution service.
// Example: node scripts/benchmark-mycelium.mjs http://127.0.0.1:4180 /tmp/mycelium-benchmark.json
import { chromium } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import os from 'node:os';
const baseURL = process.argv[2] ?? 'http://127.0.0.1:4180';
const output = process.argv[3] ?? '/tmp/mycelium-benchmark.json';
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const report = { at: new Date().toISOString(), browser: browser.version(), platform: `${os.platform()} ${os.release()} ${os.arch()}`, cpu: os.cpus()[0]?.model, memoryGB: Math.round(os.totalmem()/2**30), synthetic: true, viewport:{width:1512,height:1100}, dpr:2, cases:[] };
try {
  for (const count of [27, 512]) {
    const page = await browser.newPage({viewport:report.viewport,deviceScaleFactor:report.dpr});
    await page.addInitScript(() => {
      window.__myceliumBenchmark = {longTasks:[],frames:[]};
      let last=0;
      const tick=t=>{if(last)window.__myceliumBenchmark.frames.push(t-last);last=t;requestAnimationFrame(tick);};
      requestAnimationFrame(tick);
      if(PerformanceObserver.supportedEntryTypes.includes('longtask'))new PerformanceObserver(list=>{for(const e of list.getEntries())window.__myceliumBenchmark.longTasks.push(e.duration);}).observe({type:'longtask'});
    });
    const errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto(baseURL);
    await page.locator('.mycelium[data-gpu="WebGPU online"]').waitFor({timeout:20000});
    if (count === 512) {
      await page.request.post('http://127.0.0.1:18766/__mock/mycelium-stress');
      await page.waitForTimeout(6500);
      if (!(await page.locator('.myc-foot').innerText()).includes('131,072 samples')) throw new Error('Stress fixture did not reach the full memory capacity');
    }
    await page.waitForTimeout(3000);
    const samples=[];
    for(let i=0;i<12;i++) { await page.waitForTimeout(1000); samples.push(await page.locator('.myc-foot').innerText()); }
    const adapter=await page.locator('.myc-foot span').last().getAttribute('title');
    await page.screenshot({path:output.replace(/\.json$/,`-${count}.png`)});
    const mainThread=await page.evaluate(()=>{
      const {longTasks,frames}=window.__myceliumBenchmark;
      frames.sort((a,b)=>a-b);
      return {longTaskCount:longTasks.length,longestTaskMs:Math.max(0,...longTasks),frameIntervalP95Ms:frames[Math.floor(frames.length*.95)]};
    });
    report.cases.push({requestedRecords:count,adapter,samples,mainThread,errors});
    await page.close();
  }
} finally { await browser.close(); await writeFile(output,JSON.stringify(report,null,2)+'\n'); }
console.log(JSON.stringify(report,null,2));
