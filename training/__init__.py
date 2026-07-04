"""Training and evaluation pipeline for the Kimi-Linear (GDN-2) code LM.

Modules
-------
    config      nested dataclasses + YAML loader for an experiment.
    tokenizer   ByteLevel-BPE code tokenizer (trained on the OpenCoder corpus).
    data        OpenCoder -> tokenize/pack -> Grain input pipeline (pretrain & SFT).
    loop        loss, optimizer (Optax), and jitted train/eval steps.
    checkpoint  Orbax checkpoint manager (model + optimizer + step + data iterator).
    train       training entrypoint.
    evaluate    perplexity / generation / HumanEval evaluation entrypoint.
"""
