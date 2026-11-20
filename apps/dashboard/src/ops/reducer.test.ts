import {describe,expect,it} from 'vitest';
import {initialLiveOpsState, liveOpsReducer} from './reducer';
import frames from '../../fixtures/live_script.json';
import current from '../../fixtures/activity_current.json';
import type {ActivityCurrentResponse, StreamEventName} from '../api/types';

describe('stream convergence',()=>{
 it('deduplicates deliveries and recovers buffered events across a reconnect snapshot',()=>{
   let state=initialLiveOpsState();
   const events=frames.cycle_filled.flatMap<{event: string; data: unknown}>(f=>f.events);
   for(const e of events)state=liveOpsReducer(state,{type:'stream',name:e.event as StreamEventName,data:e.data,at:1000});
   const rows=state.rows;
   for(const e of events)state=liveOpsReducer(state,{type:'stream',name:e.event as StreamEventName,data:e.data,at:2000});
   expect(state.rows).toEqual(rows);
   expect(state.counters.duplicatesIgnored).toBeGreaterThan(0);
   let recovered=liveOpsReducer(initialLiveOpsState(),{type:'resyncStart'});
   for(const e of events)recovered=liveOpsReducer(recovered,{type:'stream',name:e.event as StreamEventName,data:e.data,at:1000});
   recovered=liveOpsReducer(recovered,{type:'resyncApply',current:current as ActivityCurrentResponse,recent:[],at:1000});
   expect(recovered.decision).toEqual(state.decision);
   expect(recovered.lastSeq).toBe(state.lastSeq);
   expect(recovered.resync.active).toBe(false);
 });
 it('does not turn malformed events into trading facts',()=>{
   const state=liveOpsReducer(initialLiveOpsState(),{type:'stream',name:'decision',data:{action:'BUY'},at:1});
   expect(state.decision).toBeNull();
   expect(state.rows).toHaveLength(0);
 });
});
