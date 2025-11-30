# Test query latency on DSB 10x.
python test.py --database dsb --logdir logs
python analyze.py --compute_latency --database dsb --logdir logs

python test_llm_only.py --database dsb --logdir logs_llm_only
python analyze_llm_only.py --compute_latency --database dsb --logdir logs_llm_only

python test_learned_rewrite.py --database dsb --logdir logs_learned_rewrite
python analyze_learned_rewrite.py --compute_latency --database dsb --logdir logs_learned_rewrite
