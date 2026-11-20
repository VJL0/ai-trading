# pyright: reportMissingImports=false
"""Microsoft RD-Agent(Q) proposal generation with the application's existing LLM transport.

Two-phase prompt/response exchange keeps broker secrets and protected data out of RD-Agent.
Generated source code is never executed: experiment proposals become allowlisted Hypotheses.
"""
import json
import sys
from pathlib import Path

import rdagent.components.proposal as proposal
from rdagent.core.proposal import Trace
from rdagent.core.scenario import Scenario
from rdagent.scenarios.qlib.proposal.factor_proposal import QlibFactorHypothesisGen
from rdagent.scenarios.qlib.proposal.model_proposal import QlibModelHypothesisGen

root = Path(sys.argv[1])
request = json.loads((root/'request.json').read_text())


class ResearchScenario(Scenario):
    background = 'Local US equities research; reversible PAPER deployments only; no order authority.'
    rich_style_description = background

    def get_scenario_all_desc(self, **kwargs):
        return self.background + '\n' + json.dumps(request['context'])

    def get_runtime_environment(self):
        return '48 GB Apple Silicon; bounded PyTorch MPS/MLX/CPU workers; frozen holdout inaccessible.'


class PromptReady(Exception):
    pass


class ApplicationBackend:
    def build_messages_and_create_chat_completion(self, user_prompt, system_prompt, **kwargs):
        (root/'prompts.json').write_text(json.dumps({'user': user_prompt, 'system': system_prompt}))
        if not (root/'response.json').exists():
            raise PromptReady
        return (root/'response.json').read_text()


proposal.APIBackend = ApplicationBackend
scenario = ResearchScenario()
generator = (QlibFactorHypothesisGen if request['kind'] == 'factor' else QlibModelHypothesisGen)(scenario)
try:
    hypothesis = generator.gen(Trace(scenario))
except PromptReady:
    pass
else:
    (root/'hypothesis.json').write_text(json.dumps(vars(hypothesis)))
