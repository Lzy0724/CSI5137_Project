import os

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 获取当前脚本 (config.py) 所在的目录，即 my_rewriter 目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (my_rewriter 的上一级)
project_root = os.path.dirname(current_dir)

# 使用绝对路径定义 CACHE_PATH (假设 cache 在项目根目录)
CACHE_PATH = os.path.join(project_root, 'cache')

# 使用绝对路径定义 CASE_RULES_PATH
# 假设这个 jsonl 文件在 my_rewriter 目录下：
CASE_RULES_PATH = os.path.join(current_dir, 'stackoverflow-rewrite-rules-query-optimization.jsonl')

# 【注意】：如果你的 jsonl 文件实际上是在项目根目录（LLM4Rewrite 下），请改用下面这行：
# CASE_RULES_PATH = os.path.join(project_root, 'stackoverflow-rewrite-rules-query-optimization.jsonl')


def init_llms(model_type: str = '', load_model=True) -> dict[str, str]:
    if 'open' in model_type:
        if load_model:
            Settings.embed_model = HuggingFaceEmbedding(
                model_name='gte-Qwen2-1.5B-instruct',
                max_length=131072
            )
        embed_dim = 1536
    else:
        if load_model:
            Settings.embed_model = OpenAIEmbedding(
                model="text-embedding-3-small"
            )
        embed_dim = 1536
    
    if 'open' in model_type:
        if load_model:
            Settings.llm = OpenAI(
                model="gpt-5",
                api_key=os.getenv("OPENAI_API_KEY"),
            )
    elif 'gpt3' in model_type:
        if load_model:
            Settings.llm = OpenAI(
                model="gpt-3.5-turbo-0125"
            )
    else:
        if load_model:
            Settings.llm = OpenAI(
                model="gpt-4o"
            )
    from my_rewriter.prompts import GEN_CASE_REWRITE_SYS_PROMPT, GEN_CASE_REWRITE_USER_PROMPT, SELECT_CASE_RULE_SYS_PROMPT, SELECT_CASE_RULE_USER_PROMPT, CLUSTER_REWRITE_SYS_PROMPT, CLUSTER_REWRITE_USER_PROMPT, SUMMARIZE_REWRITE_SYS_PROMPT, SUMMARIZE_REWRITE_USER_PROMPT, SELECT_RULES_SYS_PROMPT, SELECT_RULES_USER_PROMPT, ARRANGE_RULE_SETS_SYS_PROMPT, ARRANGE_RULE_SETS_USER_PROMPT, ARRANGE_RULES_SYS_PROMPT, ARRANGE_RULES_USER_PROMPT, REARRANGE_RULES_SYS_PROMPT, REARRANGE_RULES_USER_PROMPT, SELECT_ARRANGE_RULES_SYS_PROMPT, SELECT_ARRANGE_RULES_USER_PROMPT, RAG_SELECT_ARRANGE_RULES_SYS_PROMPT, RAG_SELECT_ARRANGE_RULES_USER_PROMPT
    return {
        'GEN_CASE_REWRITE_SYS_PROMPT': GEN_CASE_REWRITE_SYS_PROMPT,
        'GEN_CASE_REWRITE_USER_PROMPT': GEN_CASE_REWRITE_USER_PROMPT,
        'SELECT_CASE_RULE_SYS_PROMPT': SELECT_CASE_RULE_SYS_PROMPT,
        'SELECT_CASE_RULE_USER_PROMPT': SELECT_CASE_RULE_USER_PROMPT,
        'CLUSTER_REWRITE_SYS_PROMPT': CLUSTER_REWRITE_SYS_PROMPT,
        'CLUSTER_REWRITE_USER_PROMPT': CLUSTER_REWRITE_USER_PROMPT,
        'SUMMARIZE_REWRITE_SYS_PROMPT': SUMMARIZE_REWRITE_SYS_PROMPT,
        'SUMMARIZE_REWRITE_USER_PROMPT': SUMMARIZE_REWRITE_USER_PROMPT,
        'SELECT_RULES_SYS_PROMPT': SELECT_RULES_SYS_PROMPT,
        'SELECT_RULES_USER_PROMPT': SELECT_RULES_USER_PROMPT,
        'ARRANGE_RULE_SETS_SYS_PROMPT': ARRANGE_RULE_SETS_SYS_PROMPT,
        'ARRANGE_RULE_SETS_USER_PROMPT': ARRANGE_RULE_SETS_USER_PROMPT,
        'ARRANGE_RULES_SYS_PROMPT': ARRANGE_RULES_SYS_PROMPT,
        'ARRANGE_RULES_USER_PROMPT': ARRANGE_RULES_USER_PROMPT,
        'REARRANGE_RULES_SYS_PROMPT': REARRANGE_RULES_SYS_PROMPT,
        'REARRANGE_RULES_USER_PROMPT': REARRANGE_RULES_USER_PROMPT,
        'SELECT_ARRANGE_RULES_SYS_PROMPT': SELECT_ARRANGE_RULES_SYS_PROMPT,
        'SELECT_ARRANGE_RULES_USER_PROMPT': SELECT_ARRANGE_RULES_USER_PROMPT,
        'RAG_SELECT_ARRANGE_RULES_SYS_PROMPT': RAG_SELECT_ARRANGE_RULES_SYS_PROMPT,
        'RAG_SELECT_ARRANGE_RULES_USER_PROMPT': RAG_SELECT_ARRANGE_RULES_USER_PROMPT,
        'EMBED_DIM': embed_dim
    }

def init_db_config(database: str) -> dict[str, str]:
    return {
        'host': 'localhost',
        'port': 5432,
        'user': 'postgres',
        'password': 'Lzy990724@',
        'dbname': database,
        'db': 'postgresql'
    }