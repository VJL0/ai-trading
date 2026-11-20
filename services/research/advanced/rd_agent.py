"""RD-Agent(Q) proposals feed the existing journal, budget and evidence gates."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from packages.core.config import Settings
from packages.core.secrets import SecretStore
from services.ai.adapters import build_sol_model

from ..autonomous.engine import Journal
from ..autonomous.learners import Hypothesis, available
from .foundation import ROOT


class ExperimentProposal(BaseModel):
    hypothesis: str
    reason: str
    concise_reason: str
    concise_observation: str
    concise_justification: str
    concise_knowledge: str
    family: str
    capacity: int = Field(default=32, ge=8, le=128)
    alpha: float = Field(default=10, gt=0, le=1000)
    transform: str = 'identity'
    pretrain_epochs: int = Field(default=0, ge=0, le=3)
    meta_steps: int = Field(default=0, ge=0, le=4)


def rd_step(directory: Path):
    env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'TMPDIR', 'LANG')}
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    completed = subprocess.run([str(ROOT/'services/research/rd_env/.venv/bin/python'), '-m',
                               'services.research.advanced.rd_worker', str(directory)],
                              cwd=ROOT, env=env, capture_output=True, text=True, timeout=120, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-3000:])


async def propose(journal: Journal, dataset: str, directory: Path, kind='model') -> str:
    settings = Settings()
    if settings.ai_transport != 'real':
        raise ValueError('RD-Agent live proposals require the existing LAT_AI_TRANSPORT=real configuration')
    await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=False)
    history = journal.rows(dataset)
    context = {'available_families': available(), 'allowed_transforms': ['identity', 'signed_log', 'interactions'],
               'objective': 'incremental net risk-adjusted returns after costs; independent validation required',
               'development_results': [{'spec': row['spec'], 'result': row['result']} for row in history[-12:]],
               'constraints': 'No protected holdout data; no generated Python. Propose one registered family.'}
    (directory/'request.json').write_text(json.dumps({'context': context, 'kind': kind}))
    await asyncio.to_thread(rd_step, directory)
    prompts = json.loads((directory/'prompts.json').read_text())
    model = build_sol_model(settings, SecretStore(ROOT), transport='real')
    agent = Agent(model, output_type=ExperimentProposal, instructions=prompts['system'])
    response = await agent.run(prompts['user'], usage_limits=UsageLimits(request_limit=1, total_tokens_limit=12000))
    proposal = response.output
    if proposal.family not in available() or not available()[proposal.family]['available']:
        raise ValueError('RD-Agent proposed an unavailable model')
    if proposal.transform not in context['allowed_transforms']:
        raise ValueError('RD-Agent proposed an unsupported factor transform')
    (directory/'response.json').write_text(proposal.model_dump_json(indent=2))
    (directory/'usage.json').write_text(json.dumps(vars(response.usage()), default=str))
    await asyncio.to_thread(rd_step, directory)
    spec = Hypothesis(proposal.family, capacity=proposal.capacity, alpha=proposal.alpha, transform=proposal.transform,
                      pretrain_epochs=proposal.pretrain_epochs, meta_steps=proposal.meta_steps)
    job = journal.enqueue(dataset, spec)
    journal.event('RD_AGENT', proposal.reason, job)
    return job
