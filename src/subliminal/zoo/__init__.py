"""Zoo experiment: inference-time steering peak vs SL trained-student rate.

A thin orchestration layer over the canonical release/v0.1 pipeline. Nothing
here re-implements model loading, activation collection, hook injection, data
generation, filtering, training, or eval — it parameterises the existing
`subliminal.{generate,filter,train,eval,eval_steered,vectors}` library
functions per-animal. See docs/zoo_olmo_experiment.md.
"""
