# Test query latency on Calcite (uni).
python test.py --database calcite10 --logdir logs
python analyze.py --compute_latency --database calcite10 --logdir logs

python test_llm_only.py --database calcite10 --logdir logs_llm_only
python analyze_llm_only.py --compute_latency --database calcite10 --logdir logs_llm_only

python test_learned_rewrite.py --database calcite10 --logdir logs_learned_rewrite
python analyze_learned_rewrite.py --compute_latency --database calcite10 --logdir logs_learned_rewrite
