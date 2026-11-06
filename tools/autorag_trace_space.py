"""Reversible TRACE-space adapter for the unmodified upstream ReasoningAgent.

The upstream TrialConfig is a document-RAG schema. We use several fields as
explicit *transport slots*, never to construct a document RAG pipeline; the
corresponding TRACE config is executed by the trajectory evaluator. This
mapping and its departures from upstream semantics must accompany all results.
"""
from agentic_autorag.config.models import ProjectConfig, TrialConfig
from trimem.optimizer import session_space_agentenv_v2 as space

PLANS = (*space.BASE_PLANS, 'P1+Rank')
ACTOR = 'hosted_vllm/Qwen/Qwen3.8-27B-FP8'


class TraceProject(ProjectConfig):
    def validate_trial(self, trial):
        errors = super().validate_trial(trial)
        if trial.graph_query_mode not in space.SEMFILTER_PLANS and trial.reranker != 'threshold_0.6':
            errors.append('threshold must be 0.6 for plans without SemFilter')
        return errors

    def to_agent_prompt(self):
        return super().to_agent_prompt() + '''

TRAJECTORY ADAPTER: this is an experience-graph task, not document RAG. The
following TrialConfig field names are transport slots. Treat their names as
aliases; do not infer that they invoke a document chunker or reranker.
- graph_query_mode = retrieval plan (VGR, VGR+Rank, P2, P2+Rank, P1+Rank).
- graph_top_k = traversal fanout per edge type.
- chunk_token_size = 128 + seed_count. Subtract 128 for the seed count.
- chunk_token_overlap = traversal depth.
- top_k = number of trajectory memories returned.
- embedding_model = outcome policy; it is not an embedding model.
- reranker = threshold_0.4, threshold_0.6 or threshold_0.8 for SemFilter;
  plans without SemFilter must use threshold_0.6.
Both edge types similar and similar_up are fixed; the actor model is fixed.
The cost field called USD in upstream state/history represents mean tokens per
episode, not money. Optimize task quality upward and mean tokens downward.
'''


def project():
    grid = space.grid()
    return TraceProject.model_validate({
        'meta': {'project_name': 'trace-trajectory-source-adapter', 'max_trials': 30},
        'agent': {'optimizer_model': 'gpt-5.6-luna', 'examiner_model': 'unused-train-panel',
                  'judge_model': 'environment-reward', 'optimizer_reasoning_effort': 'low'},
        # This declaration satisfies the upstream graph schema; actual graph access
        # is provided by TRACE, not by the upstream LightRAG builder.
        'graph': {'extraction_model': 'unused-trace-existing-graph'},
        'search_space': {
            'chunking': {'strategies': ['recursive'],
                'chunk_token_size': {'values': [128 + x for x in grid['seed_count']]},
                'chunk_token_overlap': {'values': list(grid['depth'])}},
            'embedding': {'models': list(grid['outcome_policy'])},
            'retrieval': {'index_types': ['graph_only'], 'top_k': {'values': list(grid['top_k'])},
                          'hybrid_alpha': {'min': 0.5, 'max': 0.5}},
            'graph_retrieval': {'graph_query_modes': list(PLANS), 'graph_top_k': {'values': list(grid['fanout'])}},
            'reranker': {'models': ['threshold_0.4', 'threshold_0.6', 'threshold_0.8'],
                         'top_n': {'min': 3, 'max': 3}},
            'generator': {'models': [ACTOR], 'reasoning': False},
            'temperature': {'min': 0, 'max': 0},
        },
    })


def encode(config):
    c = space.validate(config)
    if c['plan'] not in PLANS:
        raise ValueError('plan not in this campaign')
    return TrialConfig(graph_query_mode=c['plan'], graph_top_k=c['fanout'],
                       chunk_token_size=128+c['seed_count'], chunk_token_overlap=c['depth'],
                       embedding_model=c['outcome_policy'], reranker=f"threshold_{c['semfilter_threshold']:.1f}",
                       reranker_top_n=3, top_k=c['top_k'], index_type='graph_only',
                       generator_llm=ACTOR, temperature=0)


def decode(trial):
    errors = project().validate_trial(trial)
    if errors:
        raise ValueError('; '.join(errors))
    return space.validate({'plan': trial.graph_query_mode, 'fanout': trial.graph_top_k,
        'seed_count': trial.chunk_token_size-128, 'depth': trial.chunk_token_overlap,
        'top_k': trial.top_k, 'outcome_policy': trial.embedding_model,
        'semfilter_threshold': float(trial.reranker.removeprefix('threshold_'))})
