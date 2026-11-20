// Deliberately destroys the first WebGPU device, then verifies retry and history preservation.
import { chromium } from '@playwright/test';
import {writeFile} from 'node:fs/promises';
const browser=await chromium.launch({channel:'chrome',headless:true});
try {
 const page=await browser.newPage({viewport:{width:1512,height:1100}});
 let injected=false;
 await page.route('**/assets/worker-*.js',async route=>{
  const response=await route.fetch();
  let body=await response.text();
  if(!injected){injected=true;body=`const originalRequestDevice=GPUAdapter.prototype.requestDevice; GPUAdapter.prototype.requestDevice=async function(...args){const device=await originalRequestDevice.apply(this,args);setTimeout(()=>device.destroy(),2500);return device;};\n`+body;}
  await route.fulfill({response,body});
 });
 await page.goto(process.argv[2] ?? 'http://127.0.0.1:4180');
 await page.locator('.mycelium[data-gpu="WebGPU online"]').waitFor({timeout:15000});
 await page.getByRole('button',{name:'Retry WebGPU'}).waitFor({timeout:10000});
 const reason=await page.locator('.myc-empty').innerText();
 const eventsBefore=await page.locator('.myc-event').count();
 await page.getByRole('button',{name:'Retry WebGPU'}).click();
 await page.locator('.mycelium[data-gpu="WebGPU online"]').waitFor({timeout:15000});
 await page.waitForTimeout(1500);
 const report={injected,reason,eventsBefore,eventsAfter:await page.locator('.myc-event').count(),canvases:await page.locator('.myc-surface canvas').count(),status:await page.locator('.mycelium').getAttribute('data-gpu')};
 if(report.eventsBefore !== report.eventsAfter || report.canvases !== 1 || report.status !== 'WebGPU online') throw new Error('Device recovery invariant failed');
 await writeFile(process.argv[3] ?? '/tmp/mycelium-recovery.json',JSON.stringify(report,null,2)+'\n');
 console.log(report);
}finally{await browser.close();}
