// Local read-only fixture API with real SSE and deterministic test-only stream controls.
import {createServer} from 'node:http';
import {readFileSync, readdirSync} from 'node:fs';
const dir = new URL('../fixtures/', import.meta.url);
const fixtures = Object.fromEntries(readdirSync(dir).filter(f => f.endsWith('.json')).map(f => [f.slice(0,-5), JSON.parse(readFileSync(new URL(f,dir)))]));
const clients = new Set();
let state, requests, timers = [];
const clone = v => structuredClone(v);
const reset = () => { timers.forEach(clearTimeout); timers=[]; state=clone(fixtures); requests=[]; };
reset();
const status = () => ({...state.status, server_time:new Date().toISOString()});
const emit = (event,data) => { for (const res of clients) res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`); };
const send = (res, code, data) => {res.writeHead(code, {'content-type':'application/json'}); res.end(JSON.stringify(data));};
const routes = {status, health:()=>state.health, activity:()=>state.activity, 'activity/current':()=>state.activity_current,
  performance:()=>state.performance, sessions:()=>state.sessions, orders:()=>state.orders, audit:()=>state.audit};
const server = createServer((req,res) => {
 const url = new URL(req.url,'http://127.0.0.1');
 if(url.pathname === '/__mock/reset') {reset(); return send(res,200,{});}
 if(url.pathname === '/__mock/requests') return send(res,200,{requests});
 if(url.pathname === '/__mock/disconnect') {for(const c of clients)c.end(); return send(res,200,{});}
 if(url.pathname === '/__mock/mycelium-stress') {
   const template=fixtures.activity.events.find(e=>e.event==='decision');
   for(let i=0;i<540;i++) timers.push(setTimeout(()=>{
     const data={...template, seq:10000+i, ts:new Date(Date.parse('2026-11-05T14:29:56Z')+i*7).toISOString(),
       causal_id:`benchmark-cycle-${Math.floor(i/8)}`, proposer:`BENCHMARK strategy ${i%8}`, confidence:String(0.1+(i%9)/10),
       status:i%5===0?'REJECTED':'PROPOSED', risk_failures:i%5===0?['synthetic risk rejection']:[], reason:'SYNTHETIC BENCHMARK RECORD'};
     emit('decision',data);
   }, i*10));
   return send(res,200,{synthetic:true,events:540});
 }
 if(url.pathname === '/__mock/play') {
   let elapsed=0;
   for(const frame of fixtures.live_script.cycle_filled) {
     elapsed+=frame.delay_ms;
     timers.push(setTimeout(()=>{Object.assign(state,clone(frame.state??{})); for(const e of frame.events)emit(e.event,e.data);},elapsed));
   }
   return send(res,200,{});
 }
 requests.push({method:req.method,path:url.pathname,headers:req.headers});
 if(req.method!=='GET')return send(res,405,{detail:'read only'});
 if(url.pathname==='/api/v1/stream') {
   res.writeHead(200,{'content-type':'text/event-stream','cache-control':'no-cache','connection':'keep-alive'});
   clients.add(res); res.write(`event: status\ndata: ${JSON.stringify(status())}\n\n`);
   const timer=setInterval(()=>{res.write(`event: heartbeat\ndata: ${JSON.stringify({time:new Date().toISOString(),simulated_time:state.status.feed.simulated_time})}\n\n`);},1000);
   req.on('close',()=>{clearInterval(timer);clients.delete(res);}); return;
 }
 const key=url.pathname.replace('/api/v1/','');
 if(routes[key]) return send(res,200,routes[key]());
 if(key.startsWith('orders/') && fixtures[`order_${key.slice(7)}`])return send(res,200,fixtures[`order_${key.slice(7)}`]);
 return send(res,404,{detail:'unknown route'});
});
server.listen(Number(process.env.LAT_MOCK_API_PORT??18765),'127.0.0.1');
