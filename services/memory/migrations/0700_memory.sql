-- Durable, transactional outbox. Retrieval/indexing never runs on the execution writer.
CREATE TABLE memory_outbox (
 seq INTEGER PRIMARY KEY AUTOINCREMENT,
 source TEXT NOT NULL,
 source_id TEXT NOT NULL,
 event_at TEXT NOT NULL,
 recorded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
 body TEXT NOT NULL,
 UNIQUE(source,source_id)
);
CREATE TRIGGER memory_outbox_no_update BEFORE UPDATE ON memory_outbox
BEGIN SELECT RAISE(ABORT, 'memory outbox is append-only'); END;
CREATE TRIGGER memory_outbox_no_delete BEFORE DELETE ON memory_outbox
BEGIN SELECT RAISE(ABORT, 'memory outbox is append-only'); END;
CREATE TRIGGER memory_capture_audit_events AFTER INSERT ON audit_events
BEGIN
 INSERT OR IGNORE INTO memory_outbox(source,source_id,event_at,body)
 VALUES('audit_events',NEW.event_id,NEW.ts,json_object('event_type',NEW.event_type,'causal_id',NEW.causal_id,'model_version',NEW.model_version,'dataset_version',NEW.dataset_version,'mode',NEW.mode,'payload',json(NEW.payload_json)));
END;
INSERT OR IGNORE INTO memory_outbox(source,source_id,event_at,body)
 SELECT 'audit_events',event_id,ts,json_object('event_type',event_type,'causal_id',causal_id,'model_version',model_version,'dataset_version',dataset_version,'mode',mode,'payload',json(payload_json)) FROM audit_events;
CREATE TRIGGER memory_capture_market_snapshots AFTER INSERT ON market_snapshots
BEGIN
 INSERT OR IGNORE INTO memory_outbox(source,source_id,event_at,body)
 VALUES('market_snapshots',NEW.snapshot_id,NEW.data_asof,json_object('symbol',NEW.symbol,'is_simulated',NEW.is_simulated,'payload',json(NEW.payload_json)));
END;
INSERT OR IGNORE INTO memory_outbox(source,source_id,event_at,body)
 SELECT 'market_snapshots',snapshot_id,data_asof,json_object('symbol',symbol,'is_simulated',is_simulated,'payload',json(payload_json)) FROM market_snapshots;
CREATE TRIGGER memory_capture_quant_signals AFTER INSERT ON quant_signals
BEGIN
 INSERT OR IGNORE INTO memory_outbox(source,source_id,event_at,body)
 VALUES('quant_signals',NEW.signal_id,NEW.generated_at,json_object('symbol',NEW.symbol,'strategy_id',NEW.strategy_id,'dataset_version',NEW.dataset_version,'net_edge_bps',NEW.net_edge_bps,'payload',json(NEW.payload_json)));
END;
INSERT OR IGNORE INTO memory_outbox(source,source_id,event_at,body)
 SELECT 'quant_signals',signal_id,generated_at,json_object('symbol',symbol,'strategy_id',strategy_id,'dataset_version',dataset_version,'net_edge_bps',net_edge_bps,'payload',json(payload_json)) FROM quant_signals;
CREATE TRIGGER memory_capture_model_runs AFTER INSERT ON model_runs
BEGIN
 INSERT OR IGNORE INTO memory_outbox(source,source_id,event_at,body)
 VALUES('model_runs',NEW.run_id,NEW.started_at,json_object('model_id',NEW.model_id,'status',NEW.status,'estimated_cost_usd',NEW.estimated_cost_usd,'payload',json(NEW.payload_json)));
END;
INSERT OR IGNORE INTO memory_outbox(source,source_id,event_at,body)
 SELECT 'model_runs',run_id,started_at,json_object('model_id',model_id,'status',status,'estimated_cost_usd',estimated_cost_usd,'payload',json(payload_json)) FROM model_runs;
CREATE TRIGGER memory_capture_research_findings AFTER INSERT ON research_findings
BEGIN
 INSERT OR IGNORE INTO memory_outbox(source,source_id,event_at,body)
 VALUES('research_findings',NEW.finding_id,NEW.created_at,json_object('symbol',NEW.symbol,'recommendation',NEW.recommendation,'model_run_id',NEW.model_run_id,'payload',json(NEW.payload_json)));
END;
INSERT OR IGNORE INTO memory_outbox(source,source_id,event_at,body)
 SELECT 'research_findings',finding_id,created_at,json_object('symbol',symbol,'recommendation',recommendation,'model_run_id',model_run_id,'payload',json(payload_json)) FROM research_findings;
CREATE TRIGGER memory_capture_proposals AFTER INSERT ON proposals
BEGIN
 INSERT OR IGNORE INTO memory_outbox(source,source_id,event_at,body)
 VALUES('proposals',NEW.proposal_id,NEW.created_at,json_object('symbol',NEW.symbol,'action',NEW.action,'causal_id',NEW.causal_id,'payload',json(NEW.payload_json)));
END;
INSERT OR IGNORE INTO memory_outbox(source,source_id,event_at,body)
 SELECT 'proposals',proposal_id,created_at,json_object('symbol',symbol,'action',action,'causal_id',causal_id,'payload',json(payload_json)) FROM proposals;
CREATE TRIGGER memory_capture_risk_decisions AFTER INSERT ON risk_decisions
BEGIN
 INSERT OR IGNORE INTO memory_outbox(source,source_id,event_at,body)
 VALUES('risk_decisions',NEW.decision_id,NEW.evaluated_at,json_object('verdict',NEW.verdict,'causal_id',NEW.causal_id,'payload',json(NEW.payload_json)));
END;
INSERT OR IGNORE INTO memory_outbox(source,source_id,event_at,body)
 SELECT 'risk_decisions',decision_id,evaluated_at,json_object('verdict',verdict,'causal_id',causal_id,'payload',json(payload_json)) FROM risk_decisions;
CREATE TRIGGER memory_capture_order_events AFTER INSERT ON order_events
BEGIN
 INSERT OR IGNORE INTO memory_outbox(source,source_id,event_at,body)
 VALUES('order_events',NEW.event_id,NEW.received_ts,json_object('client_order_id',NEW.client_order_id,'event_type',NEW.event_type,'gateway',NEW.gateway,'payload',json(NEW.payload_json)));
END;
INSERT OR IGNORE INTO memory_outbox(source,source_id,event_at,body)
 SELECT 'order_events',event_id,received_ts,json_object('client_order_id',client_order_id,'event_type',event_type,'gateway',gateway,'payload',json(payload_json)) FROM order_events;
CREATE TRIGGER memory_capture_fills AFTER INSERT ON fills
BEGIN
 INSERT OR IGNORE INTO memory_outbox(source,source_id,event_at,body)
 VALUES('fills',NEW.fill_id,NEW.ts,json_object('client_order_id',NEW.client_order_id,'symbol',NEW.symbol,'side',NEW.side,'quantity',NEW.quantity,'price',NEW.price,'gateway',NEW.gateway));
END;
INSERT OR IGNORE INTO memory_outbox(source,source_id,event_at,body)
 SELECT 'fills',fill_id,ts,json_object('client_order_id',client_order_id,'symbol',symbol,'side',side,'quantity',quantity,'price',price,'gateway',gateway) FROM fills;

CREATE TRIGGER memory_capture_activity AFTER INSERT ON activity_events
BEGIN
 INSERT OR IGNORE INTO memory_outbox(source,source_id,event_at,body)
 VALUES('activity_events',CAST(NEW.seq AS TEXT),NEW.ts,
 json_object('kind',NEW.kind,'causal_id',NEW.causal_id,'message',NEW.message,'payload',json(NEW.payload_json)));
END;
INSERT OR IGNORE INTO memory_outbox(source,source_id,event_at,body)
 SELECT 'activity_events',CAST(seq AS TEXT),ts,
 json_object('kind',kind,'causal_id',causal_id,'message',message,'payload',json(payload_json)) FROM activity_events;
