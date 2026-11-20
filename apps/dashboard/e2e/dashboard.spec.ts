import {expect, test} from '@playwright/test';
const MOCK=process.env.LAT_E2E_MOCK_URL??'http://127.0.0.1:18765';
test.beforeEach(async({page})=>{await page.request.post(`${MOCK}/__mock/reset`);});
test('read-only operations, live SSE, health and no credential storage',async({page})=>{
 const errors:string[]=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');
 await expect(page.getByRole('heading',{name:'Trading operations'})).toBeVisible();
 await expect(page.getByTestId('readonly-tag')).toContainText('read-only');
 await expect(page.getByTestId('mode-banner')).toHaveAttribute('data-label','SIMULATED');
 await expect(page.getByTestId('stream-indicator')).toHaveAttribute('data-state','open');
 for(const name of ['Order lifecycle','System health'])await expect(page.getByRole('heading',{name,exact:true})).toBeVisible();
 expect(await page.locator('input[type=password]').count()).toBe(0);
 const requests=(await (await page.request.get(`${MOCK}/__mock/requests`)).json()).requests;
 expect(requests.length).toBeGreaterThan(5);
 for(const r of requests){expect(r.method).toBe('GET');expect(r.headers['x-lat-operator-token']).toBeUndefined();expect(r.headers['x-lat-agent-token']).toBeUndefined();}
 expect(await page.evaluate(()=>({...localStorage}))).toEqual({});
 expect(errors).toEqual([]);
});
test('streams a complete order lifecycle and resynchronizes after reconnect',async({page})=>{
 await page.goto('/');
 await expect(page.getByTestId('stream-indicator')).toHaveAttribute('data-state','open');
 await page.request.post(`${MOCK}/__mock/play`);
 const order=page.getByTestId('lifecycle-lat-sim-01jbdashfixture00000filled1');
 await expect(order).toHaveAttribute('data-state','FILLED',{timeout:20000});
 await page.request.post(`${MOCK}/__mock/disconnect`);
 await expect(page.getByTestId('stream-indicator')).toHaveAttribute('data-state','open');
 await expect(order).toHaveAttribute('data-state','FILLED');
});
test('phone layout and reduced motion',async({page})=>{
 await page.setViewportSize({width:390,height:844});
 await page.emulateMedia({reducedMotion:'reduce'});
 await page.goto('/');
 await expect(page.getByTestId('mode-banner')).toHaveAttribute('data-label','SIMULATED');
 expect(await page.evaluate(()=>document.documentElement.scrollWidth-document.documentElement.clientWidth)).toBeLessThanOrEqual(1);
 expect(await page.evaluate(()=>matchMedia('(prefers-reduced-motion: reduce)').matches)).toBe(true);
});
