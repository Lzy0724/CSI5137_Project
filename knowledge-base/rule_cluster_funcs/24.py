import sqlglot
from sqlglot import exp

def can_be_optimized_by_non_deterministic_function(sql_query, table_schema=None, indexes=None, dialect=None):
    """
    Checks if the SQL query can be optimized for non-deterministic functions
    
    :param sql_query: SQL query string
    :param table_schema: Table schema string (currently not used directly for checks)
    :param indexes: Indexes schema string  (currently not used directly for checks)
    :return: Boolean indicating potential for optimization
    """
    # Parse the SQL query using sqlglot
    try:
        parsed_query = sqlglot.parse_one(sql_query, dialect=dialect)
    except Exception as e:
        print(f"Error parsing SQL query: {str(e)}")
        return False
    
    # Todo: add more non-deterministic SQL functions
    NON_DETERMINISTIC_FUNCS = []

    # Helper function to check if the node is a non-deterministic function
    def is_non_deterministic_function(node):
        # 旧代码：if isinstance(node, NONDETERMINISTIC): return True
        # 新代码：直接删除上面那两行，只保留下面的逻辑

        if isinstance(node, exp.Anonymous) and node.name.upper() in NON_DETERMINISTIC_FUNCS:
            return True
        return False

    # Traverse the AST from the root
    for node in parsed_query.find_all(exp.Func):
        if is_non_deterministic_function(node):
            return True
        
    return False