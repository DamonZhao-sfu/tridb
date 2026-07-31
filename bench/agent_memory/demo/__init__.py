"""The GEM Wikipedia demo — four state-level operators on a real wiki slice.

Plan: ``docs/agent_memory_gem_wiki_demo_plan_v0.1.0.md``.

Three modules, in dependency order:

``wiki_source``  fetches a pinned Wikipedia/Wikidata slice to disk. The ONLY
                 module that touches the network.
``adapter``      pure translation: article -> unit plan, hyperlink/class
                 relation -> typed edge, Wikidata edit comment -> field-level
                 supersession. No I/O at all, which is why it is the part with
                 real unit tests.
``hotpot_link``  resolves HotpotQA gold supporting titles onto slice units so
                 retrieval is graded rather than eyeballed.

The separation is deliberate: a demo that re-fetches at run time is a different
experiment every morning. ``wiki_source`` writes a manifest; everything
downstream reads it.
"""
