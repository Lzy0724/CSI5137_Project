python test.py --database tpch --logdir logs
python analyze.py --compute_latency --database tpch --logdir logs --large

python test_llm_only.py --database tpch --logdir logs_llm_only
python analyze_llm_only.py --compute_latency --database tpch --logdir logs_llm_only --large

python test_learned_rewrite.py --database tpch --logdir logs_learned_rewrite
python analyze_learned_rewrite.py --compute_latency --database tpch --logdir logs_learned_rewrite --large