from prompt_list import *
import json
import time
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
import openai
import re
import requests
import random
from sentence_transformers import SentenceTransformer, util
import os
import numpy as np
import math
import torch.nn as nn
from SPARQLWrapper import SPARQLWrapper, JSON # 用于 SPARQL 查询
from difflib import SequenceMatcher
import ast 


def safe_json_loads(raw_text, default=None):
    if default is None:
        default = {}
    if isinstance(raw_text, dict):
        return raw_text
    if not raw_text or not isinstance(raw_text, str):
        return default
    try:
        return json.loads(raw_text)
    except Exception:
        try:
            return ast.literal_eval(raw_text)
        except Exception:
            return default


def normalize_text(text):
    if text is None:
        return ""
    return str(text).strip()


def cosine_similarity_safe(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return 0.0
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def squash_score(value, lower=-1.0, upper=1.0):
    return max(lower, min(upper, float(value)))

# 定义用于控制台输出着色的常量
color_yellow = "\033[93m"
color_green = "\033[92m"
color_red= "\033[91m"
color_end = "\033[0m"

# 全局 Sentence Transformer 模型实例
model = None

# SPARQL 端点 (应与 freebase_func.py 中的一致)
SPARQLPATH = "http://127.0.0.1:8890/sparql"

# 实体/关系映射的缓存 (可通过 SPARQL 查询动态填充)
entity_id_to_name_cache = {} # 实体 ID -> 名称 缓存
entity_name_to_id_cache = {} # 实体 名称 -> ID 缓存
relation_id_to_name_cache = {} # 关系 ID -> 名称 缓存
relation_name_to_id_cache = {} # 关系 名称 -> ID 缓存



# 自适应beam_width，依据路径多样性
def adaptive_beam_width(next_beam_candidates, base_beam, alpha=5, max_beam=None):
    """
    根据候选路径多样性动态调整 beam 宽度

    :param next_beam_candidates: 候选路径列表
    :param base_beam: 最小 beam 宽度
    :param alpha: 调整系数，控制 beam 扩展幅度
    :param max_beam: 最大 beam 宽度限制
    :return: 计算后的自适应 beam 宽度
    """
    if not next_beam_candidates:
        return base_beam

    # 计算多样性得分（基于subtask）
    diversity_scores = [
        len(set(s.get('previous_subtasks', []) + [s.get('subtask', '')]))
        for s in next_beam_candidates
    ]
    avg_diversity = np.mean(diversity_scores) / 10.0  # 归一化到[0,1]

    # 自适应 beam 宽度
    adaptive_width = int(base_beam + alpha * avg_diversity)
    if max_beam:
        adaptive_width = min(adaptive_width, max_beam)

    return max(adaptive_width, 1)  # 至少为1



# --- 实体/关系映射 ---
def execute_sparql_util(sparql_query):
    try:
        sparql = SPARQLWrapper(SPARQLPATH)
        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        return results["results"]["bindings"]
    except Exception as e:
        print(f"{color_red}SPARQL query failed in utils: {e}{color_end}")
        print(f"Query: {sparql_query}")
        return [] 

def get_entity_id_from_name(entity_name):
    if not entity_name or not isinstance(entity_name, str):
        return None

    # 1. 检查缓存
    if entity_name in entity_name_to_id_cache:
        return entity_name_to_id_cache[entity_name]

    # 2. 用 SPARQL 查询
    sparql_name_query = """PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?entity WHERE { ?entity ns:type.object.name "%s"@en } LIMIT 1"""
    # 用法：sparql_quick_probe % 1

    sparql_query_formatted = sparql_name_query % entity_name.replace('"', '\\"')
    results = execute_sparql_util(sparql_query_formatted)

    if results and 'entity' in results[0]:
        entity_uri = results[0]['entity']['value']
        entity_id = entity_uri.replace("http://rdf.freebase.com/ns/", "")
        # 更新缓存
        entity_name_to_id_cache[entity_name] = entity_id
        entity_id_to_name_cache[entity_id] = entity_name
        return entity_id

    print(f"{color_yellow}Warning: Could not find ID for name '{entity_name}'. Returning None.{color_end}")
    entity_name_to_id_cache[entity_name] = None
    return None

# 将实体ID转换为可读名称或类型
def get_entity_name_from_id(entity_id):
    if not entity_id or not isinstance(entity_id, str):
        return str(entity_id)

    entity_id_clean = entity_id.replace("ns:", "").replace("http://rdf.freebase.com/ns/", "")
    if entity_id_clean in entity_id_to_name_cache:
        return entity_id_to_name_cache[entity_id_clean]

    sparql_id = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?tailEntity
    WHERE {
      {
        ns:%s ns:type.object.name ?tailEntity .
      }
      UNION
      {
        ns:%s <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .
      }
    } LIMIT 1
    """ % (entity_id_clean, entity_id_clean)

    results = execute_sparql_util(sparql_id)
    if results and 'tailEntity' in results[0]:
        val = results[0]['tailEntity']['value']
        entity_id_to_name_cache[entity_id_clean] = val
        return val

    # 查询不到时返回原始 ID
    entity_id_to_name_cache[entity_id_clean] = entity_id_clean
    return entity_id_clean

# 将实体ID转换为可读名称或类型
def get_entity_name_selected(entity_id):
    if not entity_id or not isinstance(entity_id, str):
        return str(entity_id)

    entity_id_clean = entity_id.replace("ns:", "").replace("http://rdf.freebase.com/ns/", "")
    if entity_id_clean in entity_id_to_name_cache:
        return entity_id_to_name_cache[entity_id_clean]

    sparql_id = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?tailEntity
    WHERE {
      {
        ns:%s ns:type.object.name ?tailEntity .
      }
      UNION
      {
        ns:%s <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .
      }
    } LIMIT 1
    """ % (entity_id_clean, entity_id_clean)

    results = execute_sparql_util(sparql_id)
    if results and 'tailEntity' in results[0]:
        val = results[0]['tailEntity']['value']
        entity_id_to_name_cache[entity_id_clean] = val
        return val

    # 查询不到时返回False
    return False

# 将关系名称转换为关系ID
def get_relation_id(relation_name):
    if not relation_name or not isinstance(relation_name, str):
        return None

    relation_name = str(relation_name).strip()

    if relation_name.startswith("ns:"):
        return relation_name[3:]
    if relation_name.startswith("http://rdf.freebase.com/ns/"):
        return relation_name[len("http://rdf.freebase.com/ns/"):]

    # 检查缓存
    if relation_name in relation_name_to_id_cache:
        return relation_name_to_id_cache[relation_name]

    # 将空格、下划线都转为点，拼成 Freebase 标准ID
    # rel_id = relation_name.replace(' ', '.').replace('_', '.')
    rel_id = relation_name.strip()
    # 验证关系ID格式
    if not rel_id or '.' not in rel_id:
        print(f"{color_yellow}Warning: Invalid relation ID format: {rel_id}{color_end}")
        return None
        
    # 更新缓存
    relation_name_to_id_cache[relation_name] = rel_id
    relation_id_to_name_cache[rel_id] = relation_name
    return rel_id

def get_relation_name(relation_id):
    """将关系 ID 转换为更易读的名称。"""
    if not relation_id or not isinstance(relation_id, str):
        return str(relation_id) # 无效输入返回其字符串表示
    
    # 清理 ID
    relation_id_clean = relation_id.replace("ns:", "").replace("http://rdf.freebase.com/ns/", "")
    
    # 检查缓存
    if relation_id_clean in relation_id_to_name_cache:
        return relation_id_to_name_cache[relation_id_clean]
        
    # 简单的转换：用空格替换点
    name = relation_id_clean.replace('.', ' ')
    relation_id_to_name_cache[relation_id_clean] = name # 更新 ID->名称 缓存
    relation_name_to_id_cache[name] = relation_id_clean # 更新 名称->ID 缓存
    return name

# --- 路径/状态格式化 ---
def build_state_uid(state):
    path = state.get("path", []) or []
    path_key = "->".join(
        f"{step[0]}|{step[1]}|{step[2]}"
        for step in path
        if isinstance(step, (tuple, list)) and len(step) >= 3
    )
    entity_id = state.get("current_entity_id", "")
    subtask = normalize_text(state.get("subtask", ""))
    depth = state.get("depth", len(path))
    return f"{entity_id}##{depth}##{subtask}##{path_key}"


def summarize_observation_stats(observation):
    stats = {
        "relation_count": 0,
        "entity_count": 0,
        "unique_entity_count": 0,
        "avg_branching": 0.0,
        "non_empty_ratio": 0.0,
    }
    if not isinstance(observation, dict) or not observation:
        return stats

    relation_count = len(observation)
    entities = []
    non_empty = 0
    for tails in observation.values():
        if isinstance(tails, list) and tails:
            non_empty += 1
            entities.extend(tails)
        elif tails:
            non_empty += 1
            entities.append(tails)

    entity_count = len(entities)
    unique_entity_count = len(set(map(str, entities)))
    stats["relation_count"] = relation_count
    stats["entity_count"] = entity_count
    stats["unique_entity_count"] = unique_entity_count
    stats["avg_branching"] = entity_count / max(1, relation_count)
    stats["non_empty_ratio"] = non_empty / max(1, relation_count)
    return stats


def get_path_text(path):
    if not path:
        return ""
    return " | ".join(
        f"{get_entity_name_from_id(step[0])} -> {get_relation_name(step[1])} -> {get_entity_name_from_id(step[2])}"
        for step in path
        if isinstance(step, (tuple, list)) and len(step) >= 3
    )


def get_relation_sequence(path):
    if not path:
        return []
    return [
        step[1]
        for step in path
        if isinstance(step, (tuple, list)) and len(step) >= 3 and step[1]
    ]


def get_relation_path_text(path):
    relation_names = [get_relation_name(rel_id) for rel_id in get_relation_sequence(path)]
    return " ; ".join(relation_names)


def format_path_for_prompt(path):
    if not path:
        return "[]"

    formatted_steps = []
    for step in path:
        if isinstance(step, tuple) and len(step)  >= 2:
            head_entity_id = step[0]
            relation_id = step[1]
            tail_entity_id = step[2] if len(step) > 2 else "?" 

            head_entity_name = get_entity_name_from_id(head_entity_id)
            relation_name = get_relation_name(relation_id)
            tail_entity_name = get_entity_name_from_id(tail_entity_id) if tail_entity_id != "?" else "?"
           
            formatted_steps.append((head_entity_name, relation_name, tail_entity_name))
        else:
            print(f"Warning: Unexpected path step format in final path: {step}")
    return formatted_steps




def format_observation_for_prompt(observation, use_names=False, max_items=5):
    if not observation:
        return "None"

    lines = []
    for relation, tail_list in list(observation.items())[:max_items]:
        if not tail_list:
            continue
        tail_names = []
        for tail in tail_list:
            if use_names:
                tail_name = get_entity_name_from_id(tail)
                tail_names.append(tail_name if tail_name else tail)
            else:
                tail_names.append(tail)
        line = f"{relation}: {', '.join(tail_names)}"
        lines.append(line)

    if len(observation) > max_items:
        lines.append("...")

    return "\n".join(lines)




# --- LLM 交互 ---
def run_llm(prompt, temperature, max_tokens, openai_api_keys, engine="gpt-4.1", print_in=True, print_out=True):
    if print_in:
        print(f"{color_green}--- LLM Prompt -->\n{prompt}\n---{color_end}")
    result = "" 
    token_num = {"total": 0, "input": 0, "output": 0} 
    
    if not openai_api_keys:
            print(f"{color_red}Error: OpenAI API key is missing.{color_end}")
            return result, token_num 

    try:
        client = openai.OpenAI(api_key=openai_api_keys)
        messages = [
            {"role": "system", "content": "You are an AI assistant performing structured tasks based on instructions. Output only the requested format (e.g., JSON) without explanations."}, # 系统消息，指导 LLM 行为
            {"role": "user", "content": prompt} 
        ]

        completion = client.chat.completions.create(
            model=engine,         
            messages=messages,    
            temperature=temperature,
            max_tokens=max_tokens,   
            frequency_penalty=0.0, 
            presence_penalty=0.0   
        )

        result = completion.choices[0].message.content.strip()
        try:
            if result.startswith("{") or result.startswith("["):
                parsed = ast.literal_eval(result)  
                result = json.dumps(parsed, ensure_ascii=False)  
        except Exception as convert_err:
            print(f"{color_yellow}Warning: Failed to convert LLM output to JSON: {convert_err}{color_end}")

        token_num = {
            "total": completion.usage.total_tokens,
            "input": completion.usage.prompt_tokens,
            "output": completion.usage.completion_tokens
        }

        if print_out:
            print(f"{color_yellow}--- LLM Response <--\n{result}\n---{color_end}")

    except openai.AuthenticationError as e:
         print(f"{color_red}OpenAI API Error: Authentication failed. Check your API key. {e}{color_end}")
    except openai.RateLimitError as e:
         print(f"{color_red}OpenAI API Error: Rate limit exceeded. Please wait and try again. {e}{color_end}")
         time.sleep(5)
    except openai.APIConnectionError as e:
         print(f"{color_red}OpenAI API Error: Connection issue. Check network. {e}{color_end}")
    except Exception as e:
        print(f"{color_red}Error during LLM call: {e}{color_end}")

    return result, token_num



# --- PPO 相关函数 ---
# 使用 SentenceTransformer 为推理路径生成嵌入向量
def get_path_embedding(path, model_st):
    global model 

    if model_st is None:
        print(f"{color_red}Error: Sentence Transformer model not loaded in utils.{color_end}")
        return None 

    # 空路径返回零向量
    if not path:
        return np.zeros(model_st.get_sentence_embedding_dimension())

    # 将路径格式化为单一字符串（使用实体/关系名称）
    path_string_representation = ""
    formatted_steps = []
    for step in path:
        if isinstance(step, tuple) and len(step) == 3:
            head_id, rel_id, tail_ids = step
            head_name = get_entity_name_from_id(head_id)
            rel_name = get_relation_name(rel_id)
            if isinstance(tail_ids, list):
                tail_names = [get_entity_name_from_id(tid) for tid in tail_ids]
                tail_str = ", ".join(tail_names) if tail_names else "[]"
            else:
                 tail_name = get_entity_name_from_id(tail_ids)
                 tail_str = tail_name
            formatted_steps.append(f"{head_name} --[{rel_name}]--> {tail_str}")
        elif isinstance(step, str): 
             name = get_entity_name_from_id(step) if step.startswith('m.') else step
             formatted_steps.append(f"Start: {name}")
        else:
             formatted_steps.append(str(step)) 
             
    # 用分隔符连接所有步骤
    path_string_representation = " | ".join(formatted_steps)


    # 如果路径字符串为空，返回零向量
    if not path_string_representation:
        return np.zeros(model_st.get_sentence_embedding_dimension())

    # 尝试编码路径字符串
    try:
        if model_st is None: model_st = model 
        if model_st is None: raise ValueError("SentenceTransformer model not available.")

        # 使用模型进行编码
        embedding = model_st.encode(path_string_representation, convert_to_numpy=True)
        return embedding
    except Exception as e:
        # 编码失败时打印错误并返回零向量
        print(f"{color_red}Error encoding path '{path_string_representation[:100]}...': {e}{color_end}")
        return np.zeros(model_st.get_sentence_embedding_dimension())

# 将当前推理状态字典转换为 PPO Agent 需要的固定大小的向量
def _deprecated_convert_state_to_ppo_input(state_dict, args, model_st):
    global model 
    if model_st is None: model_st = model 
    if model_st is None:
        print(f"{color_red}Error: Sentence Transformer model not available for state conversion.{color_end}")
        return np.zeros(args.state_dim) 

    state_dim = args.state_dim 
    embedding_dim = model_st.get_sentence_embedding_dimension() 

    # --- 从状态字典中提取组件 ---
    question = state_dict.get('query', '') 
    path = state_dict.get('path', []) 
    observation = state_dict.get('observation') 
    subtask = state_dict.get('subtask', '') 
    current_entity_id = state_dict.get('current_entity_id', None) 

    # --- 特征工程 (将状态信息编码为向量) ---
    # 1. 问题嵌入 (768维)
    q_embedding = model_st.encode(question, convert_to_numpy=True) if question else np.zeros(embedding_dim)
    q_feature = np.zeros(768) # 固定为768维
    q_len = min(len(q_embedding), 768)
    q_feature[:q_len] = q_embedding[:q_len]

    # 2. 子任务嵌入 (64维)
    st_embedding = model_st.encode(subtask, convert_to_numpy=True) if subtask else np.zeros(embedding_dim)
    st_feature = np.zeros(64) # 固定为64维
    st_len = min(len(st_embedding), 64)
    st_feature[:st_len] = st_embedding[:st_len]

    # 3. 当前实体嵌入 (64维)
    current_entity_name = get_entity_name_from_id(current_entity_id) if current_entity_id else ""
    ce_embedding = model_st.encode(current_entity_name, convert_to_numpy=True) if current_entity_name else np.zeros(embedding_dim)
    ce_feature = np.zeros(64) # 固定为64维
    ce_len = min(len(ce_embedding), 64)
    ce_feature[:ce_len] = ce_embedding[:ce_len]

    # 4. 路径特征 (5维)
    path_feature = np.zeros(5)
    path_feature[0] = len(path) # 路径长度
    path_feature[1] = len(set([step[0] for step in path if isinstance(step, tuple)])) # 唯一头实体数
    path_feature[2] = len(set([step[1] for step in path if isinstance(step, tuple)])) # 唯一关系数
    path_feature[3] = len(set([step[2] for step in path if isinstance(step, tuple)])) # 唯一尾实体数
    path_feature[4] = sum(1 for step in path if isinstance(step, tuple)) # 有效步骤数

    # 5. 观察结果特征 (5维)
    obs_feature = np.zeros(5)
    if isinstance(observation, list):
        obs_feature[0] = len(observation) # 结果数量
        obs_feature[1] = len(observation[0]) if observation else 0 # 变量数量
        obs_feature[2] = sum(1 for obs in observation if obs) # 非空结果数
        obs_feature[3] = len(set(str(obs) for obs in observation)) # 唯一结果数
        obs_feature[4] = 1 if any(obs for obs in observation) else 0 # 是否有结果

    # 6. 占位符特征 (10维)
    placeholder_feature = np.zeros(10)

    # 将所有特征向量拼接成一个单一的状态向量
    state_vector = np.concatenate([
        q_feature,      # 768维
        st_feature,     # 64维
        ce_feature,     # 64维
        path_feature,   # 5维
        obs_feature,    # 5维
        placeholder_feature  # 10维
    ])

    # 验证最终向量维度
    if state_vector.shape[0] != state_dim:
        print(f"{color_red}Error: State vector dimension mismatch. Expected {state_dim}, got {state_vector.shape[0]}.{color_end}")
        # 如果维度不匹配，返回零向量
        return np.zeros(state_dim)

    return state_vector


# --- 数据集处理 ---
def prepare_dataset(dataset_name):
    print(f"Utils: Loading dataset: {dataset_name}")
    datas = [] 
    question_string = 'question'
    question_id_key = 'id'      

    # 根据数据集名称确定文件路径和键名
    data_file_path = None
    if dataset_name == 'cwq':
        data_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cwq.json') 
        question_string = 'question'
        question_id_key = 'ID' 
    elif dataset_name == 'webqsp':
        data_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'WebQSP.json')
        question_string = 'ProcessedQuestion'
        question_id_key = 'QuestionId' 
    elif dataset_name == 'grailqa':
        data_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'grailqa.json')
        question_string = 'question'
        question_id_key = 'qid' 
    else:
        print(f"{color_red}Error: Dataset '{dataset_name}' not recognized. Choose from {{cwq, webqsp, grailqa}}.{color_end}")
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"Utils: Attempting to load from: {data_file_path}")
    if data_file_path and os.path.exists(data_file_path):
        try:
            with open(data_file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f) 
                if dataset_name == 'webqsp' and 'Questions' in raw_data:
                    datas = raw_data['Questions']
                else: 
                    datas = raw_data
            print(f"Utils: Loaded {len(datas)} items.")

        except FileNotFoundError:
            print(f"{color_red}Error: Dataset file not found at {data_file_path}{color_end}")
            raise 
        except json.JSONDecodeError:
            print(f"{color_red}Error: Could not decode JSON from {data_file_path}. Check format.{color_end}")
            raise 
        except Exception as e:
            print(f"{color_red}Error loading dataset from {data_file_path}: {e}{color_end}")
            raise 
            
    # 如果指定了文件路径但文件不存在
    elif data_file_path:
         print(f"{color_red}Error: Dataset file path specified but not found: {data_file_path}{color_end}")
         raise FileNotFoundError(f"Dataset file not found: {data_file_path}")

    # 数据加载后的验证：检查第一个数据项是否包含所需的键
    if datas:
        first_item = datas[0]
        
        if question_string not in first_item:
             print(f"{color_red}Error: Question text key '{question_string}' not found in dataset item: {first_item}{color_end}")
        
             raise KeyError(f"Expected question key '{question_string}' not found.") 
        if question_id_key not in first_item:
             print(f"{color_red}Error: Question ID key '{question_id_key}' not found in dataset item: {first_item}{color_end}")
             raise KeyError(f"Expected question ID key '{question_id_key}' not found.")
    else:
        print(f"{color_red}Warning: No data loaded from dataset file.{color_end}")


    # 返回加载的数据列表、问题 ID 键名和问题文本键名
    return datas, question_id_key, question_string

# --- 结果保存与杂项 ---
def save_2_jsonl(result, file_path, question_id_key):
    output_dict = result.copy() 
    if "timestamp" not in output_dict:
         output_dict["timestamp"] = time.time()

    essential_keys = [
        question_id_key, "question", "answer", "reasoning_paths",
        "central_entities_ids", "central_entities_names", "subtask_formulated",
        "total_token_usage", "time_taken"
    ]
    for key in essential_keys:
        if key not in output_dict:
            output_dict[key] = None 

    # 添加旧格式的键以实现可能的向后兼容性（通过映射新键）
    output_dict["results"] = output_dict.get("answer") # "results" 映射到 "answer"
    output_dict["reasoning_chains"] = output_dict.get("reasoning_paths") # "reasoning_chains" 映射到 "reasoning_paths"
    total_tokens = output_dict.get("total_token_usage", {}).get('total', 0) 
    max_len = output_dict.get("max_length", 1) 
    output_dict["call_num"] = total_tokens / max_len if max_len > 0 else 0 
    output_dict["total_token"] = total_tokens
    output_dict["input_token"] = output_dict.get("total_token_usage", {}).get('input', 0) # 输入 token 数
    output_dict["output_token"] = output_dict.get("total_token_usage", {}).get('output', 0) # 输出 token 数
    output_dict["time"] = output_dict.get("time_taken", 0) 

    question_identifier = output_dict.get(question_id_key)
 
    if question_identifier is None:
        print(f"{color_red}Error: Cannot save result, missing identifier for key '{question_id_key}'. Result: {output_dict}{color_end}")
        return

    # 使用 try-except 处理文件 IO 操作
    try:
        # 高效读取现有数据
        existing_data = {} 
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
       
                        item_id = item.get(question_id_key)
                        if item_id is not None:
                            existing_data[item_id] = item 
                        else:
                         
                             print(f"{color_yellow}Skipping line in {file_path} lacking identifier '{question_id_key}': {line.strip()}{color_end}")
                
                    except json.JSONDecodeError:
                        print(f"{color_yellow}Skipping invalid JSON line in {file_path}: {line.strip()}{color_end}")

        existing_data[question_identifier] = output_dict

        # 将所有数据写回文件
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in existing_data.values():
                item_serializable = json.loads(json.dumps(item, cls=NumpyEncoder))
                json.dump(item_serializable, f, ensure_ascii=False)
                f.write('\n') 

    except IOError as e:
        print(f"{color_red}I/O Error saving result to {file_path}: {e}{color_end}")
    except Exception as e:
        print(f"{color_red}Unexpected Error saving result to {file_path}: {e}{color_end}")


# 自定义 JSON 编码器，用于处理 NumPy 数据类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): # 如果是 NumPy 整数
            return int(obj)             # 转换为 Python int
        elif isinstance(obj, np.floating): # 如果是 NumPy 浮点数
            return float(obj)           # 转换为 Python float
        elif isinstance(obj, np.ndarray): # 如果是 NumPy 数组
            return obj.tolist()         # 转换为 Python list
        # 对于其他类型，使用基类的默认处理方式
        return super(NumpyEncoder, self).default(obj)

# 从列表的字符串表示中提取字符串列表
def extract_add_ent(string):
    try:
        first_brace_p = string.find('[')
        last_brace_p = string.rfind(']')
        # 确保找到了方括号且顺序正确
        if first_brace_p != -1 and last_brace_p != -1 and last_brace_p > first_brace_p:
            list_string = string[first_brace_p : last_brace_p + 1]
            # 首先尝试标准的 JSON 解析
            try:
                 loaded_list = json.loads(list_string)
                 if isinstance(loaded_list, list):
                      # 确保列表中的所有项都是字符串
                      return [str(item) for item in loaded_list]
            # 如果 JSON 解析失败，尝试使用正则表达式处理格式不规范的列表，例如 ['ent1', 'ent2']
            except json.JSONDecodeError:
                 # 查找单引号或双引号包围的实体
                 entities = [ent.strip().strip("'\"") for ent in re.findall(r"'(.*?)'|\"(.*?)\"", list_string)]
                 # 过滤掉由正则表达式可能产生的空字符串
                 return [ent for ent in entities if ent] 
    # 处理其他可能的异常
    except Exception as e:
        print(f"Error during entity list extraction: {e}")
    # 如果提取失败，返回空列表
    return [] 

# 从类 JSON 字符串中提取 'Add' 标志 (yes/no) 和 'Reason'
def extract_add_and_reason(string):
    flag = False # 默认为 'no' (False)
    reason = "No reason provided." # 默认原因
    try:
         # 使用正则表达式查找 "Add": "yes/no" 和 "Reason": "..." (忽略大小写)
         add_match = re.search(r'"Add"\s*:\s*"?(yes|no)"?', string, re.IGNORECASE)
         reason_match = re.search(r'"Reason"\s*:\s*"(.*?)"', string, re.DOTALL)

         # 如果找到 "Add" 且值为 "yes"
         if add_match and add_match.group(1).lower() == 'yes':
             flag = True
         # 如果找到 "Reason"
         if reason_match:
             reason = reason_match.group(1).strip() 

    except Exception as e:
         print(f"Error extracting add/reason: {e} from: {string[:100]}...")

    return flag, reason

# --- 模型加载 ---
def load_sentence_transformer_model(model_path):
    global model 
    print(f"Utils: Loading Sentence Transformer model from {model_path}...")
    try:
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model directory not found: {model_path}")
        model = SentenceTransformer(model_path)
        print("Utils: Sentence Transformer model loaded successfully.")
        return model 
    except FileNotFoundError as e:
         print(f"{color_red}{e}{color_end}")
         print(f"{color_red}Please ensure the model is downloaded at the specified path.{color_end}")
         raise 
    except Exception as e:
        print(f"{color_red}Error loading Sentence Transformer model: {e}{color_end}")
        print(f"{color_red}PathDiversity reward calculation will be disabled.{color_end}")
        model = None 
        raise 

# --- 判断是否终止 ---
def _deprecated_should_terminate(question, current_path,current_state, observation, args):
    if not current_path or not observation:
        return (False, "Need more exploration to answer the question.")

    # 根据观察结果判断当前路径是否可能找到答案
    prompt = termination_prompt.format(
        question=question,
        current_path=current_path,
        current_state=current_state,
        observation=observation
    )

    response, _ = run_llm(
        prompt=prompt,
        temperature=0.0,
        max_tokens=args.max_length // 8,
        openai_api_keys=args.openai_api_keys
    )
    
    try:
        result = json.loads(response)
        should_end = result.get("termination", "no").lower() == "yes"
        reason = result.get("reason", "No reason provided")
        confidence = float(result.get("confidence", 0.5))
        
        # 只有当置信度足够高时才终止
        if should_end and confidence >= 0.8:
            return (True, reason)
        elif confidence < 0.7:
            # 置信度太低，不终止但记录
            print(f"Low confidence in termination decision ({confidence:.2f}). Continuing reasoning.")
            return (False, "Low confidence in termination decision. Continuing reasoning.")
        else:
            return (False, reason)
    except Exception as e:
        # 解析失败，不终止
        print(f"Error parsing termination decision: {e}")
        return (False, "Failed to parse termination decision")

def context_aware_state_score(question, state, args, model_st):
    if model_st is None:
        return 0.0, {
            "semantic": 0.0,
            "observation": 0.0,
            "depth": 0.0,
            "chain": 0.0,
            "novelty": 0.0,
        }

    question = normalize_text(question)
    subtask = normalize_text(state.get("subtask", ""))
    current_entity_name = get_entity_name_from_id(state.get("current_entity_id", ""))
    path = state.get("path", []) or []
    previous_subtasks = state.get("previous_subtasks", []) or []
    observation = state.get("observation", {})
    path_text = get_path_text(path)

    q_embedding = model_st.encode(question, convert_to_numpy=True) if question else None
    subtask_embedding = model_st.encode(subtask, convert_to_numpy=True) if subtask else None
    entity_embedding = model_st.encode(current_entity_name, convert_to_numpy=True) if current_entity_name else None
    path_embedding = model_st.encode(path_text, convert_to_numpy=True) if path_text else None

    semantic_alignment = 0.0
    if q_embedding is not None:
        semantic_alignment = max(
            cosine_similarity_safe(q_embedding, subtask_embedding),
            cosine_similarity_safe(q_embedding, entity_embedding),
            cosine_similarity_safe(q_embedding, path_embedding),
        )

    obs_stats = summarize_observation_stats(observation)
    observation_score = min(1.0, 0.2 * obs_stats["relation_count"] + 0.1 * obs_stats["avg_branching"])

    target_depth = max(1, getattr(args, "optimal_path_length", 3))
    current_depth = state.get("depth", len(path))
    depth_score = max(0.0, 1.0 - abs(current_depth - target_depth) / max(1, target_depth))

    chain_diversity = len(set(previous_subtasks + ([subtask] if subtask else [])))
    chain_score = min(1.0, chain_diversity / max(1.0, float(target_depth)))

    novelty_score = 1.0
    if previous_subtasks and subtask:
        recent = previous_subtasks[-3:]
        sims = []
        for prev in recent:
            prev_embedding = model_st.encode(prev, convert_to_numpy=True)
            sims.append(cosine_similarity_safe(subtask_embedding, prev_embedding))
        if sims:
            novelty_score = max(0.0, 1.0 - max(sims))

    weights = {
        "semantic": getattr(args, "state_semantic_weight", 0.40),
        "observation": getattr(args, "state_observation_weight", 0.20),
        "depth": getattr(args, "state_depth_weight", 0.15),
        "chain": getattr(args, "state_chain_weight", 0.15),
        "novelty": getattr(args, "state_novelty_weight", 0.10),
    }
    components = {
        "semantic": semantic_alignment,
        "observation": observation_score,
        "depth": depth_score,
        "chain": chain_score,
        "novelty": novelty_score,
    }
    total = sum(weights[name] * components[name] for name in components)
    return squash_score(total, 0.0, 1.0), components


class LightweightPathEvaluator(nn.Module):
    """
    Lightweight path discriminator that complements the state scorer.
    It focuses on question-path semantic plausibility over relation sequences.
    """

    def __init__(self):
        super().__init__()

    def score_path(self, question, path, args, model_st):
        default_components = {
            "question_path": 0.0,
            "question_last_relation": 0.0,
            "path_coherence": 0.0,
            "length_fitness": 0.0,
            "relation_repetition_penalty": 0.0,
        }
        if model_st is None or not path:
            return 0.0, default_components

        relation_ids = get_relation_sequence(path)
        if not relation_ids:
            return 0.0, default_components

        question_text = normalize_text(question)
        relation_names = [get_relation_name(rel_id) for rel_id in relation_ids]
        path_text = " ; ".join(relation_names)
        last_relation_text = relation_names[-1]

        q_embedding = model_st.encode(question_text, convert_to_numpy=True) if question_text else None
        path_embedding = model_st.encode(path_text, convert_to_numpy=True) if path_text else None
        last_relation_embedding = model_st.encode(last_relation_text, convert_to_numpy=True) if last_relation_text else None
        relation_embeddings = model_st.encode(relation_names, convert_to_numpy=True) if relation_names else []

        question_path = max(0.0, cosine_similarity_safe(q_embedding, path_embedding))
        question_last_relation = max(0.0, cosine_similarity_safe(q_embedding, last_relation_embedding))

        path_coherence = 1.0
        if len(relation_embeddings) > 1:
            consecutive_sims = []
            for idx in range(len(relation_embeddings) - 1):
                consecutive_sims.append(
                    max(0.0, cosine_similarity_safe(relation_embeddings[idx], relation_embeddings[idx + 1]))
                )
            if consecutive_sims:
                path_coherence = float(np.mean(consecutive_sims))

        target_hops = max(1, getattr(args, "optimal_path_length", 3))
        current_hops = len(relation_ids)
        length_fitness = max(0.0, 1.0 - abs(current_hops - target_hops) / max(1, target_hops))

        repetition_penalty = 1.0 - (len(set(relation_ids)) / max(1, len(relation_ids)))

        score = (
            0.45 * question_path +
            0.25 * question_last_relation +
            0.15 * path_coherence +
            0.15 * length_fitness -
            0.15 * repetition_penalty
        )
        score = squash_score(score, 0.0, 1.0)
        return score, {
            "question_path": question_path,
            "question_last_relation": question_last_relation,
            "path_coherence": path_coherence,
            "length_fitness": length_fitness,
            "relation_repetition_penalty": repetition_penalty,
        }


def update_tree_statistics(tree_stats, state_uid, reward, parent_uid=None, prior_score=0.0):
    node = tree_stats.setdefault(
        state_uid,
        {
            "visits": 0,
            "value_sum": 0.0,
            "prior": 0.0,
            "parent": parent_uid,
        },
    )
    if parent_uid is not None and not node.get("parent"):
        node["parent"] = parent_uid
    node["prior"] = max(float(prior_score), float(node.get("prior", 0.0)))
    node["visits"] += 1
    node["value_sum"] += float(reward)
    return node


def get_tree_ucb_score(tree_stats, state_uid, parent_uid=None, exploration_weight=1.2, fallback_prior=0.0):
    node = tree_stats.get(state_uid, {})
    visits = float(node.get("visits", 0))
    value_sum = float(node.get("value_sum", 0.0))
    prior = float(node.get("prior", fallback_prior))
    q_value = value_sum / visits if visits > 0 else 0.0

    parent_visits = 1.0
    if parent_uid and parent_uid in tree_stats:
        parent_visits = max(1.0, float(tree_stats[parent_uid].get("visits", 1)))

    u_value = exploration_weight * prior * math.sqrt(parent_visits + 1.0) / (1.0 + visits)
    return q_value + u_value


def derive_pseudo_preference_pairs(scored_states, margin_threshold=0.05):
    valid_states = [s for s in scored_states if isinstance(s.get("score"), dict)]
    if len(valid_states) < 2:
        return []

    ranked = sorted(
        valid_states,
        key=lambda s: s["score"].get("total_reward", 0.0),
        reverse=True,
    )
    preference_pairs = []
    top_limit = min(2, len(ranked))
    bottom_slice = ranked[-top_limit:]
    for winner in ranked[:top_limit]:
        for loser in bottom_slice:
            gap = winner["score"].get("total_reward", 0.0) - loser["score"].get("total_reward", 0.0)
            if gap > margin_threshold:
                preference_pairs.append(
                    {
                        "winner_uid": build_state_uid(winner),
                        "loser_uid": build_state_uid(loser),
                        "margin": float(gap),
                    }
                )
    return preference_pairs


def run_llm(prompt, temperature, max_tokens, openai_api_keys, engine="gpt-4.1", print_in=True, print_out=True):
    if print_in:
        print(f"{color_green}--- LLM Prompt -->\n{prompt}\n---{color_end}")

    result = ""
    token_num = {"total": 0, "input": 0, "output": 0}
    api_key = openai_api_keys or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.gptsapi.net/v1")
    if not api_key:
        print(f"{color_red}Error: OpenAI API key is missing.{color_end}")
        return result, token_num

    try:
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant performing structured knowledge-graph reasoning. Output only the requested format."
            },
            {"role": "user", "content": prompt},
        ]
        if OpenAI is not None:
            client = OpenAI(base_url=base_url, api_key=api_key)
            completion = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result = normalize_text(completion.choices[0].message.content)
            usage = getattr(completion, "usage", None)
            if usage is not None:
                token_num = {
                    "total": getattr(usage, "total_tokens", 0),
                    "input": getattr(usage, "prompt_tokens", 0),
                    "output": getattr(usage, "completion_tokens", 0),
                }
        else:
            openai.api_key = api_key
            openai.base_url = base_url
            completion = openai.ChatCompletion.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result = normalize_text(completion["choices"][0]["message"]["content"])
            usage = completion.get("usage", {})
            token_num = {
                "total": usage.get("total_tokens", 0),
                "input": usage.get("prompt_tokens", 0),
                "output": usage.get("completion_tokens", 0),
            }

        if print_out:
            print(f"{color_yellow}--- LLM Response <--\n{result}\n---{color_end}")
    except Exception as e:
        http_proxy = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY", "")
        https_proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY", "")
        print(f"{color_red}Error during LLM call: {type(e).__name__}: {e}{color_end}")
        print(f"{color_yellow}[LLM DEBUG] base_url={base_url}, model={engine}, http_proxy={http_proxy}, https_proxy={https_proxy}{color_end}")

    return result, token_num


def convert_state_to_ppo_input(state_dict, args, model_st):
    # Canonical PPO state featurizer used by the current EnTic-main search loop.
    global model
    if model_st is None:
        model_st = model
    if model_st is None:
        print(f"{color_red}Error: Sentence Transformer model not available for state conversion.{color_end}")
        return np.zeros(args.state_dim)

    state_dim = args.state_dim
    embedding_dim = model_st.get_sentence_embedding_dimension()
    question = state_dict.get("query", "")
    path = state_dict.get("path", [])
    observation = state_dict.get("observation") or {}
    subtask = state_dict.get("subtask", "")
    current_entity_id = state_dict.get("current_entity_id", None)

    q_embedding = model_st.encode(question, convert_to_numpy=True) if question else np.zeros(embedding_dim)
    q_feature = np.zeros(512)
    q_feature[:min(len(q_embedding), 512)] = q_embedding[:512]

    st_embedding = model_st.encode(subtask, convert_to_numpy=True) if subtask else np.zeros(embedding_dim)
    st_feature = np.zeros(96)
    st_feature[:min(len(st_embedding), 96)] = st_embedding[:96]

    current_entity_name = get_entity_name_from_id(current_entity_id) if current_entity_id else ""
    ce_embedding = model_st.encode(current_entity_name, convert_to_numpy=True) if current_entity_name else np.zeros(embedding_dim)
    ce_feature = np.zeros(96)
    ce_feature[:min(len(ce_embedding), 96)] = ce_embedding[:96]

    path_text = get_path_text(path)
    path_embedding = model_st.encode(path_text, convert_to_numpy=True) if path_text else np.zeros(embedding_dim)
    path_semantic_feature = np.zeros(96)
    path_semantic_feature[:min(len(path_embedding), 96)] = path_embedding[:96]

    path_feature = np.zeros(8)
    path_feature[0] = len(path)
    path_feature[1] = len(set([step[0] for step in path if isinstance(step, tuple)]))
    path_feature[2] = len(set([step[1] for step in path if isinstance(step, tuple)]))
    path_feature[3] = len(set([step[2] for step in path if isinstance(step, tuple)]))
    path_feature[4] = sum(1 for step in path if isinstance(step, tuple))
    path_feature[5] = state_dict.get("depth", len(path))
    path_feature[6] = state_dict.get("backtrack_count", 0)
    path_feature[7] = len(state_dict.get("previous_subtasks", []) or [])

    obs_stats = summarize_observation_stats(observation)
    obs_feature = np.zeros(8)
    obs_feature[0] = obs_stats["relation_count"]
    obs_feature[1] = obs_stats["entity_count"]
    obs_feature[2] = obs_stats["unique_entity_count"]
    obs_feature[3] = obs_stats["avg_branching"]
    obs_feature[4] = obs_stats["non_empty_ratio"]
    obs_feature[5] = 1.0 if observation else 0.0
    obs_feature[6] = len(state_dict.get("candidate_relations", []) or [])
    obs_feature[7] = float(state_dict.get("relation_score", 0.0))

    chain_score, chain_components = context_aware_state_score(question, state_dict, args, model_st)
    chain_feature = np.zeros(8)
    chain_feature[0] = chain_score
    chain_feature[1] = chain_components["semantic"]
    chain_feature[2] = chain_components["observation"]
    chain_feature[3] = chain_components["depth"]
    chain_feature[4] = chain_components["chain"]
    chain_feature[5] = chain_components["novelty"]
    chain_feature[6] = float(state_dict.get("state_score", 0.0))
    chain_feature[7] = float(state_dict.get("mcts_score", 0.0))

    state_vector = np.concatenate([
        q_feature,
        st_feature,
        ce_feature,
        path_semantic_feature,
        path_feature,
        obs_feature,
        chain_feature,
    ])

    if state_vector.shape[0] != state_dim:
        print(f"{color_red}Error: State vector dimension mismatch. Expected {state_dim}, got {state_vector.shape[0]}.{color_end}")
        return np.zeros(state_dim)

    return state_vector


def should_terminate(question, current_path, current_state, observation, args):
    # Canonical termination helper used by EnTic-main/freebase_func.py.
    if not current_path or not observation:
        return (False, "Need more exploration to answer the question.")

    prompt = termination_prompt.format(
        question=question,
        current_path=current_path,
        current_state=current_state,
        observation=observation
    )

    response, _ = run_llm(
        prompt=prompt,
        temperature=0.0,
        max_tokens=args.max_length // 8,
        openai_api_keys=args.openai_api_keys,
        engine=getattr(args, "LLM_type", "gpt-4.1"),
        print_in=False,
        print_out=False,
    )

    result = safe_json_loads(response, default={})
    should_end = str(result.get("termination", "no")).lower() == "yes"
    reason = result.get("reason", "No reason provided")
    confidence = float(result.get("confidence", 0.5))
    if should_end and confidence >= 0.8:
        return (True, reason)
    if confidence < 0.7:
        return (False, "Low confidence in termination decision. Continuing reasoning.")
    return (False, reason)


def append_jsonl(path, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, cls=NumpyEncoder)
        f.write("\n")

def ts():
    return time.time()
