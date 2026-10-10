-- Figure 1 semantic-operator state: the LLM result cache and the pending set.
--
-- Both are ordinary heap tables in the SAME database as the vector / graph / relational
-- stores (golden rule 2: one transaction manager, one WAL, no sidecar). The LLM itself
-- stays out of process (ADR-0026); only its verdicts live here.
--
-- The paper writes the pending set as Q (renamed from P to avoid the clash with GEM's
-- policy set P_t, plan D3).

-- One row per distinct semantic-operator invocation. The key covers everything that can
-- change the answer: operator, model, prompt version and the canonical input bytes. A hit
-- is a deterministic table read, so a replay with the LLM switched off reproduces state.
CREATE TABLE IF NOT EXISTS gem_sem_cache (
    op                text NOT NULL,
    model             text NOT NULL,
    prompt_version    text NOT NULL,
    input_hash        text NOT NULL,          -- sha256 of the canonical input JSON
    output            jsonb NOT NULL,
    prompt_tokens     integer NOT NULL DEFAULT 0,
    completion_tokens integer NOT NULL DEFAULT 0,
    llm_seconds       double precision NOT NULL DEFAULT 0,
    created_tx        bigint NOT NULL DEFAULT txid_current(),
    created_at        timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (op, model, prompt_version, input_hash)
);

-- Deferred derivations (Q). A derivation is enqueued by Ingest under a deferred plan and
-- resolved later by a background/batch pass or on read (strong mode).
--   snapshot_seq pins the candidate reads of the derivation to the state that existed
--   when x committed (plan D2), so eager and deferred plans see the same inputs.
CREATE TABLE IF NOT EXISTS gem_pending (
    id            bigserial PRIMARY KEY,
    scope_id      text NOT NULL,
    derivation    text NOT NULL,              -- e.g. 'lesson', 'fixed_by', 'llm_ingest'
    target_ref    text NOT NULL,              -- uid / vid of the item the derivation is for
    trigger       text NOT NULL CHECK (trigger IN ('background','batch','on_read')),
    snapshot_seq  bigint NOT NULL,            -- logical ingest sequence at enqueue time
    payload       jsonb NOT NULL DEFAULT '{}'::jsonb,
    status        text NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending','resolved','failed')),
    enqueued_tx   bigint NOT NULL DEFAULT txid_current(),
    resolved_tx   bigint,
    resolved_by   text,                       -- 'background' | 'batch' | 'on_read'
    error         text,
    UNIQUE (scope_id, derivation, target_ref)
);
CREATE INDEX IF NOT EXISTS gem_pending_open_idx
    ON gem_pending (scope_id, derivation, snapshot_seq) WHERE status = 'pending';
