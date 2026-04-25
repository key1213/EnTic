from SPARQLWrapper import SPARQLWrapper, JSON 
import json 
import numpy as np
from utils import * 
from prompt_list import * 
from sentence_transformers import util
import torch        
import copy
import traceback


SPARQLPATH = "http://127.0.0.1:8890/sparql" 

# --------------------------- SPARQL 查询模板 ---------------------------
# 查询实体作为主语的所有出边关系
sparql_head_relations = """PREFIX ns: <http://rdf.freebase.com/ns/> SELECT DISTINCT ?relation WHERE { ns:%s ?relation ?x . }"""
# 查询给定头实体和关系的尾实体
sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?tailEntity WHERE { ns:%s ns:%s ?tailEntity . }"""
# 查询实体名称或同义 URI
sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\n
SELECT DISTINCT ?tailEntity\n
WHERE {\n
  {\n
    ?entity ns:type.object.name ?tailEntity .\n
    FILTER(?entity = ns:%s)\n
  }\n
  UNION\n
  {\n
    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n
    FILTER(?entity = ns:%s)\n
  }\n
}"""


# --------------------------- SPARQL 执行与辅助函数 ---------------------------
# 执行SPARQL查询
def execute_sparql(sparql_query):
    sparql_query = str(sparql_query).strip()
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]

# 提取关系ID
def replace_relation_prefix(relations):
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/","") for relation in relations]

# 提取实体ID
def replace_entities_prefix(entities):
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/","") for entity in entities]

# 对实体 ID 进行字符串转义
def escape_entity_id_for_sparql(entity_id):
    if not entity_id or not isinstance(entity_id, str):
        return ""
    return entity_id.replace("'", "\\'").replace('"', '\\"')

# 判断是否是合法实体ID
def is_valid_entity(eid):
    return isinstance(eid, str) and eid.startswith(('m.', 'g.', '/en/'))

def normalize_score(value, min_val=0.0, max_val=1.0):
    return max(min_val, min(max_val, value))

def evaluate_accuracy(question, subtask, args):
    """评估 Subtask 的语义准确性（通过 LLM 打分）"""
    prompt = evaluation_prompt.format(question=question, subtask=subtask)
    try:
        response, _ = run_llm(
            prompt=prompt,
            temperature=0.0,
            max_tokens=args.max_length,
            openai_api_keys=args.openai_api_keys,
            print_in=False,
            print_out=False
        )
        try:
            json_response = json.loads(response)
            score = float(json_response.get("score", 0.0))
        except json.JSONDecodeError:
            score_str = ''.join(filter(lambda x: x.isdigit() or x == '.', response))
            score = float(score_str) if score_str else 0.0
        return normalize_score(score, -1.0, 1.0)
    except Exception:
        return 0.0

def estimate_optimal_length(question, args):
    prompt = estimate_optimal_path_prompt.format(question=question)
    try:
        response, _ = run_llm(
            prompt=prompt,
            temperature=0.0,
            max_tokens=args.max_length,
            openai_api_keys=args.openai_api_keys
        )
        json_response = json.loads(response)
        answer = int(json_response.get("answer", 0.0))
        # 默认最佳路径长度为1
        return answer if answer else 1
    except Exception:
        return 1

def evaluate_efficiency(current_len, optimal_len):
    """评估路径跳数是否接近最优"""
    if optimal_len == 0:
        return 0.0
    diff_ratio = max(0, current_len - optimal_len) / optimal_len
    return normalize_score(1.0 - diff_ratio)

def evaluate_diversity(sub_paths, model_st):
    """基于语义嵌入计算路径之间的差异性"""
    if not model_st or len(sub_paths) <= 1:
        return 0.0

    embeddings = [get_path_embedding(p, model_st) for p in sub_paths if p]
    valid_embeddings = [e for e in embeddings if e is not None and np.any(e)]

    if len(valid_embeddings) < 2:
        return 0.0

    similarity_sum, pair_count = 0.0, 0
    for i in range(len(valid_embeddings)):
        for j in range(i + 1, len(valid_embeddings)):
            sim = util.cos_sim(valid_embeddings[i], valid_embeddings[j])[0][0].item()
            similarity_sum += sim
            pair_count += 1

    avg_similarity = similarity_sum / pair_count if pair_count > 0 else 1.0
    return normalize_score(1.0 - avg_similarity)

# --------------------------- EnTic: 初始实体提取 ---------------------------
def extract_central_entities(question, args):
    print("Step 1: Extracting Central Entities...")
    response = ""
    token_num = {'total': 0, 'input': 0, 'output': 0}
    try:
        prompt = central_entity_prompt.format(question=question)
        response, token_num = run_llm(
            prompt=prompt,
            temperature=0.0,
            max_tokens=args.max_length,
            openai_api_keys=args.openai_api_keys
        )
    except Exception as e:
        print(f"{color_red}Step 1 Exception:{e}{color_end}")

    extracted_entity_names = []
    try:
        json_response = json.loads(response)
        entities_raw = json_response.get("entities", [])
        extracted_entity_names = [e for e in entities_raw if isinstance(e, str) and e]
    except Exception:
        extracted_entity_names = []

    if not extracted_entity_names:
        print(f"{color_red}Failed to extract valid central entities from LLM response.{color_end}")
        return [], [], token_num

    central_entity_ids = [get_entity_id_from_name(name) for name in extracted_entity_names]
    central_entity_ids = [eid for eid in central_entity_ids if eid]
    print(f"Extracted Names: {extracted_entity_names}")
    print(f"Mapped IDs: {central_entity_ids}")
    return extracted_entity_names, central_entity_ids, token_num

# --------------------------- EnTic: 主推理循环 -------------------------------
def _deprecated_beam_search_reasoning_tao(question, central_entity_ids, args, ppo_agent, model_st):
    # 检查是否有中心实体
    if not central_entity_ids:
        print("Error: No central entities provided to start reasoning.")
        return [], [],{'total': 0, 'input': 0, 'output': 0},[]

    # 初始化 token 统计与 PPO 经验
    total_token_num = {'total': 0, 'input': 0, 'output': 0}
    all_experiences = []

    # 初始化状态去重结构
    visited = set()
    visited_stack = []
    visited_entities = set()
    subtask_history = {}

    # 初始化搜索状态
    state_history = []
    current_depth = 0
    current_beam = []

    # 每个中心实体作为初始推理起点
    for entity_id in central_entity_ids:
        initial_state = {
            'query': question,
            'path': [],
            'observation': None,
            'subtask': None,
            'previous_subtasks': [],
            'current_entity_id': entity_id,
            'parent_state_index': None,
            'action_from_parent': None,
            'backtrack_count': 0,
            'depth': 0,
            'error': None,
            'error_details': None
        }
        state_history.append(initial_state)
        current_beam.append(initial_state)
        visited_entities.add(entity_id)
    print(f"{color_yellow}current_beam for step:{current_beam} {color_end}")
    # 记录最后一次单步推理状态即完整的path路径
    last_current_beam = []
    # 开始 beam search 推理
    while current_depth < args.depth and current_beam:
        print(f"\nDepth {current_depth}: Processing {len(current_beam)} states...")
        next_beam_candidates = []
        current_visited_in_round = []
        # 对current_beam进行遍历
        for state_index, state in enumerate(current_beam):
            try:
                current_state = copy.deepcopy(state)

                # 执行单步推理
                step_result = step(current_state, args, state_index, model_st)  

                # 若 terminated 给出建议为True则停止单步推理进行最终推理，若子任务为空则继续循环
                if step_result.get('terminated'):
                    current_state['terminated'] = True
                    continue
                subtask = step_result.get('subtask')
                if not subtask:
                    continue

                # 获取单步推理子任务
                subtask = step_result.get('subtask')

                # 调用llm的输入输出
                for k in total_token_num:
                    total_token_num[k] += step_result['token_num'][k]

                # 获取当前状态向量用于 PPO 决策
                state_vector = convert_state_to_ppo_input(current_state, args, model_st)
                num_candidates = len(step_result['candidate_relations'])

                if num_candidates == 0:
                    current_state['terminated'] = True
                    continue

                # 使用 PPO 选择一个动作 eg:sports.sports_team.championships
                chosen_action_index, log_prob = ppo_agent.select_action(state_vector, num_candidates)
                if chosen_action_index < 0 or chosen_action_index >= num_candidates:
                    current_state['backtrack_count'] += 1
                    continue
                chosen_relation_id = step_result['candidate_relations'][chosen_action_index]
                print(f"[DEBUG] PPO-chosen relation: {chosen_relation_id}")

                # 获取候选关系列表
                next_states = step_result.get('next_states', [])

                # 如果根本没有下一个状态，就打印一次警告
                if not next_states:
                    print(f"{color_yellow}Warning: no next_states, skipping expansion.{color_end}")
                else:
                    print(f"[DEBUG] Filtered next_states by PPO relation: {len(next_states)}")
                if chosen_action_index < 0 or chosen_action_index >= len(step_result['candidate_relations']):
                    print(f"[DEBUG] Invalid chosen_action_index: {chosen_action_index}. It should be between 0 and {len(step_result['candidate_relations']) - 1}")
                else:
                    print(f"[DEBUG] Valid chosen_action_index: {chosen_action_index}")

                # 对应动作的尾实体ID列表
                observation_list = step_result['observations'].get(chosen_relation_id, [])
                if not isinstance(observation_list, list):
                    observation_list = [observation_list]
                if chosen_action_index < 0 or chosen_action_index >= len(step_result['candidate_relations']):
                    print(f"[DEBUG] Invalid chosen_action_index: {chosen_action_index}. It should be between 0 and {len(step_result['candidate_relations']) - 1}")

                # 保留所有合法尾实体
                if len(observation_list) == 0:
                    print(f"[DEBUG] No valid observations for relation {chosen_relation_id}.")
                else:
                    observation_entities = [eid for eid in observation_list if is_valid_entity(eid)]
                    if not observation_entities:
                        current_state['backtrack_count'] += 1
                        continue

                # 获取全部 next_states
                all_next_states = step_result.get('next_states', [])

                # 根据当前 PPO 策略选中的 relation 过滤 next_states
                next_states = [
                    s for s in all_next_states
                    if s.get('action_from_parent') == chosen_relation_id
                ]
                if not next_states:
                    print(f"{color_yellow}Warning: no valid next_states for relation {chosen_relation_id}.{color_end}")
                    current_state['backtrack_count'] += 1
                    continue

                # 将对应state的尾节点ID和子任务组成键值对存储，重复则跳过本次循环
                for new_state in next_states:
                    new_state['parent_state_index'] = state_index
                    tail_id = new_state.get('current_entity_id')
                    subtask_text = new_state.get("subtask") or subtask
                    key = (tail_id, subtask_text)
                    if key in visited:
                        print(f"[Beam] Skipping duplicate (entity, subtask): {key}")
                        continue

                    visited.add(key)
                    visited_entities.add(tail_id)
                    subtask_history.setdefault((tail_id, new_state['depth']), set()).add(subtask_text)
                    next_beam_candidates.append(new_state)

                    #PPO经验收集
                    next_state_vector = convert_state_to_ppo_input(new_state, args, model_st)
                    done = new_state['depth'] >= args.depth
                    experience = {
                        'state': state_vector,
                        'action': chosen_action_index,
                        'reward': 0.0,
                        'next_state': next_state_vector,
                        'log_prob': log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob,
                        'done': done
                    }
                    all_experiences.append(experience)
            except Exception as e:
                print(f"something went wrong:{e}")
                error_stack = traceback.format_exc()
                state['error'] = "State processing failed"
                state['error_details'] = {
                    'error_message': str(e),
                    'error_type': type(e).__name__,
                    'error_stack': error_stack
                }
                state['backtrack_count'] += 1
                continue

        # 按照语义多样性排序，保留 top-K 路径 剪枝操作
        args.beam_width = adaptive_beam_width(next_beam_candidates,4)
        next_beam_candidates.sort(
            key=lambda x: len(set(x.get('previous_subtasks', []) + [x.get('subtask', '')])),
            reverse=True
        )
        # 给current_beam赋值，不为空则继续单步推理
        current_beam = next_beam_candidates[:args.beam_width]
        if current_beam:
            last_current_beam = current_beam
        state_history.extend(current_beam)
        visited_stack.append(current_visited_in_round)
        current_depth += 1

    # 第二阶段：评估所有推理路径
    print("\nStep 2: Evaluating all reasoning paths...")
    current_beam = last_current_beam
    all_valid_paths = [s['path'] for s in current_beam if s.get('path')]
    evaluated_paths = []
    subtasks = []
    for state in current_beam:
        subtasks.append(state['subtask'])
        if state.get('path'):
            state_vec = convert_state_to_ppo_input(state, args, model_st)
            score = evaluate_path(
                state,
                question,
                args,
                model_st,
                all_experiences,
                state_vec,
                all_paths=all_valid_paths
            )
            state['score'] = score

            if score.get('total_reward', 0.0) >= args.backtrack_threshold:
                evaluated_paths.append(state)
    # 第三阶段：对低分路径进行回溯
    if not evaluated_paths:
        print("\nNo valid paths found. Attempting backtracking...")
        backtrack_success = False
        max_try = 3  # 最多尝试 3 个不同的历史状态进行回溯
        try_count = 0

        # 将历史状态按分数从高到低排序
        def get_total_reward(state):
            score = state.get('score')
            if isinstance(score, dict):
                return score.get('total_reward', 0.0)
            return 0.0

        all_states = sorted(state_history, key=get_total_reward, reverse=True)

        # 尝试从多个低分路径进行回溯
        for state in reversed(all_states):
            if try_count >= max_try:
                break

            if state['backtrack_count'] < args.max_backtrack:
                parent_index = state.get('parent_state_index')
                parent_state = state_history[parent_index] if parent_index is not None else None
                if parent_state:
                    # 增加回溯计数
                    state['backtrack_count'] += 1

                    # 回溯成功，设置父状态为新的 beam 起点
                    current_beam = [parent_state]
                    current_depth = len(parent_state.get('path', []))

                    if visited_stack:
                        last_visited = visited_stack.pop()
                        for k in last_visited:
                            visited.discard(k)
                    backtrack_success = True
                    try_count += 1

                    print(f"Backtracked to depth {current_depth} from state with score {get_total_reward(state):.3f}")

                    if state.get('error'):
                        print(f"Previous error: {state['error']}")
                        if state.get('error_details'):
                            print(f"Error details: {state['error_details']}")

                    continue  # 回溯一次后返回主循环进行下一轮搜索

        if not backtrack_success:
            print("No valid states to backtrack to. Ending search.")
            return [], all_experiences, total_token_num,[]

    final_paths = [s['path'] for s in evaluated_paths if s.get('path')]
    return final_paths, all_experiences, total_token_num,list(set(subtasks))

# --------------------------- EnTic: 单步推理 ---------------------------
def step(state, args, current_state_index, model_st=None):
    print(f"{color_green}Step 2.1: Expanding state at depth {state.get('depth', 0)} for entity {state.get('current_entity_id', 'unknown')}.{color_end}")
    token_num = {'total': 0, 'input': 0, 'output': 0}
    question = state['query']
    current_entity_id = state['current_entity_id']
    current_entity_name = get_entity_name_from_id(current_entity_id)
    current_path = state['path']
    last_step = current_path[-1] if current_path else None
    last_observation = state.get('observation', {})

    # 构造路径文本用于生成子任务
    path_description = []
    for step_ in current_path:
        head_entity_name = get_entity_name_from_id(step_[0])
        relation_name = get_relation_name(step_[1])
        tail_entity_name = get_entity_name_from_id(step_[2]) if len(step_) > 2 else "?"
        path_description.append(f"{head_entity_name} --{relation_name}--> {tail_entity_name}")

    prompt_subtask = formulate_subtask_prompt.format(
        question=question,
        current_path=path_description,
        previous_action=get_relation_name(last_step[1]) if last_step else "None",
        previous_observation=format_observation_for_prompt(last_observation, use_names=True),
        current_entity_name=current_entity_name,
        current_entity_id=current_entity_id
    )

    # 调整温度用于去重控制
    subtask_temp = args.temperature_reasoning
    if state.get('subtask_repeat_count', 0) > 0:
        subtask_temp = min(0.9, subtask_temp + 0.2)

    # 请求 LLM 获取子任务
    subtask_response, tn_subtask = run_llm(
        prompt=prompt_subtask,
        temperature=subtask_temp,
        max_tokens=args.max_length // 4,
        openai_api_keys=args.openai_api_keys
    )
    # 将String类型llm输出内容转换成json格式
    response_json = safe_json_loads(subtask_response or "", default={})
    token_num = {k: token_num[k] + tn_subtask[k] for k in token_num}

    # 解析子任务
    previous_subtask = response_json.get('subtask')
    try:
        subtask = response_json.get("subtask", question) if isinstance(response_json, dict) else question
    except Exception as e:
        print(f"{color_red}Step 2.1 Subtask Parsing Exception: {e}{color_end}")
        subtask = question
    prev_subtasks = state.get('previous_subtasks', [])
    if model_st is not None and subtask and prev_subtasks:
        new_emb = model_st.encode(subtask, convert_to_tensor=True)
        for prev in prev_subtasks[-3:]:
            prev_emb = model_st.encode(prev, convert_to_tensor=True)
            cos_sim = util.cos_sim(new_emb, prev_emb).item()
            if cos_sim > 0.9:
                return {
                    'candidate_relations': [],
                    'observations': {},
                    'subtask': subtask,
                    'token_num': token_num,
                    'terminated': True,
                    'termination_reason': "Semantically repeated subtask."
                }

    # 当前子任务未重复，更新子任务列表
    state['subtask'] = subtask
    if previous_subtask:
        state.setdefault('previous_subtasks', []).append(previous_subtask)

    # 检查连续重复次数
    if previous_subtask == subtask:
        state['subtask_repeat_count'] = state.get('subtask_repeat_count', 0) + 1
    else:
        state['subtask_repeat_count'] = 0
    if state['subtask_repeat_count'] >= 2:
        return {
            'candidate_relations': [],
            'observations': {},
            'subtask': subtask,
            'token_num': token_num,
            'terminated': True,
            'termination_reason': f"Stuck in repetitive subtask '{subtask}'"
        }
    # 查询当前实体的所有可用关系
    query = sparql_head_relations % escape_entity_id_for_sparql(current_entity_id)
    results = execute_sparql(query)
    possible_relation_ids = replace_relation_prefix(results)
    if not possible_relation_ids:
        print(f"{color_yellow}Step 2.1 Warning: no outgoing relations found for entity {current_entity_id}.{color_end}")
        return {
            'candidate_relations': [],
            'observations': {},
            'subtask': subtask,
            'token_num': token_num,
            'terminated': True
        }

    # 让 LLM 选择最相关的关系
    possible_relation_names = [rel_id for rel_id in possible_relation_ids if rel_id is not None]

    k = args.beam_width
    possible_relations_str = "; ".join(possible_relation_names)
    prompt_select_relation = select_relevant_relations_prompt.format(
        k=k,
        question=question,
        subtask=subtask,
        topic_entity=current_entity_name,
        possible_relations=possible_relations_str
    )

    select_relation_response, tn_select = run_llm(
        prompt=prompt_select_relation,
        temperature=args.temperature_reasoning,
        max_tokens=args.max_length // 4,
        openai_api_keys=args.openai_api_keys
    )
    token_num = {k: token_num[k] + tn_select[k] for k in token_num}

    # 提取 selected relations
    selected_relations_names = []
    relation_score_map = {}
    if select_relation_response:
        response_json = safe_json_loads(select_relation_response, default={})
        print(f"{color_green}Step 2.2 Relation Selection Response: {response_json}{color_end}")
        if isinstance(response_json, dict):
            llm_selection = response_json.get('selected_relations', [])
            relation_scores = response_json.get('relation_scores', [])
            if isinstance(llm_selection, list):
                selected_relations_names = [
                    str(name).strip()
                    for name in llm_selection
                    if isinstance(name, (str, int, float))
                ]
            else:
                print(f"{color_yellow}Step 2.2 Warning: selected_relations is not a list: {llm_selection}{color_end}")

            if isinstance(relation_scores, list):
                for item in relation_scores:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        rel_name = str(item[0]).strip()
                        try:
                            relation_score_map[rel_name] = float(item[1])
                        except Exception:
                            continue

    # 与 SPARQL 返回的交集
    selected_valid_relation_names = [name for name in selected_relations_names if name in possible_relation_names]
    candidate_relations_ids = [
        get_relation_id(name)
        for name in selected_valid_relation_names
        if get_relation_id(name) is not None
    ]

    candidate_relations_ids = candidate_relations_ids[:k]
    if not candidate_relations_ids and possible_relation_ids:
        candidate_relations_ids = possible_relation_ids[:k]
        print(f"{color_yellow}Step 2.2 Warning: LLM relation selection failed. Falling back to top {k} relations from SPARQL results.{color_end}")

    print(f"{color_green}Step 2.2 Selected valid relation names: {selected_valid_relation_names}{color_end}")

    # 查询每条关系对应的所有尾实体，返回为列表
    observations = {}
    # 过滤没有对应尾实体的candidate_relations_ids
    filter_candidate_relations_ids = []
    for rel_id in candidate_relations_ids:
        query = sparql_tail_entities_extract % (
            escape_entity_id_for_sparql(current_entity_id),
            rel_id
        )
        print(f"[SPARQL] Query: {query}")
        results = execute_sparql(query)
        # print(f"[SPARQL] Results for {rel_id}: {results}")
        tail_ids = replace_entities_prefix(results)
        tail_values = [eid for eid in tail_ids if is_valid_entity(eid)]
        if not tail_values:  # 如果尾实体为空，则跳过
            print(f"{color_yellow}Warning: no valid next_statesNo valid tail entities for relation {rel_id}. Skipping.{color_end}")
            continue
        filter_candidate_relations_ids.append(rel_id)
        observations[rel_id] = tail_values
        print(f"Observation for relation '{get_relation_name(rel_id) if get_relation_name(rel_id) else rel_id}': {tail_values}")
    candidate_relations_ids = filter_candidate_relations_ids

    next_states = []
    for rel_id in candidate_relations_ids:
        tail_entities = observations.get(rel_id, [])
        # print(f"[Debug] Relation {rel_id} -> Tail Entities: {tail_entities}")
        for tail_id in tail_entities:
            if is_valid_entity(tail_id):
                entity_name = get_entity_name_selected(tail_id)
                if entity_name :
                    next_state = {
                        'query': question,
                        'path': current_path + [(current_entity_id, rel_id, tail_id)],
                        'observation': observations,
                        'subtask': subtask,
                        'previous_subtasks': state.get('previous_subtasks', []),
                        'current_entity_id': tail_id,
                        'current_entity_name': entity_name,
                        'parent_state_index': current_state_index,
                        'state_history_index': None,
                        'action_from_parent': rel_id,
                        'backtrack_count': 0,
                        'depth': state.get('depth', 0) + 1,
                        'error': None,
                        'error_details': None,
                        'terminated': False,
                        'relation_score': relation_score_map.get(rel_id, relation_score_map.get(get_relation_name(rel_id), 0.0))
                    }
                    next_states.append(next_state)

    termination_result = should_terminate(
        question=question,
        current_path=current_path,
        observation=observations,
        current_state = state,
        args=args,
    )

    selected_candidate_relations_ids = []
    for next_state in next_states:
        if next_state['action_from_parent'] not in selected_candidate_relations_ids:
            selected_candidate_relations_ids.append(next_state['action_from_parent'])
    print(f"{color_green}Step 2.3 Expansion Summary: {len(selected_candidate_relations_ids)} candidate relations, {len(next_states)} next states, terminated={termination_result[0]}.{color_end}")
    # 返回 step 结果
    return {
        'candidate_relations': selected_candidate_relations_ids,
        'observations': observations,
        'subtask': subtask,
        'token_num': token_num,
        'terminated': termination_result[0],
        'termination_reason': termination_result[1],
        'next_states': next_states,
        'relation_score_map': relation_score_map
    }

def _deprecated_evaluate_path(state, question, args, model_st, all_experiences=None, state_vec=None, all_paths=None):
    """整合调用各项评估指标，自动检查合法性并更新 PPO 经验"""
    path = state.get('path')
    subtask = state.get('subtask')

    if not isinstance(path, list) or not path or not all(
            isinstance(step, (list, tuple)) and len(step) == 3 for step in path):
        print(f"{color_yellow}Skipping invalid path during evaluation: {path}{color_end}")
        return {'accuracy': 0.0, 'efficiency': 0.0, 'diversity': 0.0, 'total_reward': 0.0}

    acc = evaluate_accuracy(question, subtask, args)
    print(f"Accuracy: {acc:.3f}")
    optimal_len = estimate_optimal_length(question, args)
    eff = evaluate_efficiency(len(path), optimal_len)
    if all_paths and isinstance(all_paths, list) and len(all_paths) > 1:
        other_paths = [p for p in all_paths if p != path]
        div = evaluate_diversity([path] + other_paths, model_st)
    else:
        div = 0.0

    total = args.w1 * acc + args.w2 * eff + args.w3 * div

    if all_experiences and state_vec is not None:
        for exp in all_experiences:
            if exp['done'] and np.allclose(exp['next_state'], state_vec, atol=1e-6):
                exp['reward'] = total

    return {
        'accuracy': acc,
        'efficiency': eff,
        'diversity': div,
        'total_reward': total
    }

# --- EnTic: 最终答案生成 ---
def apply_pseudo_state_refinement(all_experiences, preference_pairs, args):
    if not all_experiences or not preference_pairs:
        return

    reward_updates = {}
    refine_weight = getattr(args, "pseudo_refine_weight", 0.2)
    for pair in preference_pairs:
        margin = float(pair.get("margin", 0.0))
        reward_updates[pair["winner_uid"]] = reward_updates.get(pair["winner_uid"], 0.0) + refine_weight * margin
        reward_updates[pair["loser_uid"]] = reward_updates.get(pair["loser_uid"], 0.0) - refine_weight * margin

    for exp in all_experiences:
        uid = exp.get("next_state_uid")
        if uid in reward_updates:
            exp["reward"] = float(exp.get("reward", 0.0)) + reward_updates[uid]
            exp["pseudo_refined"] = True


def evaluate_path(state, question, args, model_st, path_evaluator=None, all_experiences=None, state_vec=None, all_paths=None):
    # Canonical path evaluator for the DAMR-aligned PoG pipeline.
    path = state.get('path')
    subtask = state.get('subtask')

    if not isinstance(path, list) or not path or not all(
        isinstance(step, (list, tuple)) and len(step) == 3 for step in path
    ):
        print(f"{color_yellow}Skipping invalid path during evaluation: {path}{color_end}")
        return {
            'accuracy': 0.0,
            'efficiency': 0.0,
            'diversity': 0.0,
            'state_value': 0.0,
            'path_value': 0.0,
            'relation_value': 0.0,
            'total_reward': 0.0
        }

    acc = evaluate_accuracy(question, subtask, args)
    optimal_len = estimate_optimal_length(question, args)
    eff = evaluate_efficiency(len(path), optimal_len)
    if all_paths and isinstance(all_paths, list) and len(all_paths) > 1:
        other_paths = [p for p in all_paths if p != path]
        div = evaluate_diversity([path] + other_paths, model_st)
    else:
        div = 0.0

    state_value = state.get("state_score")
    if state_value is None:
        state_value, _ = context_aware_state_score(question, state, args, model_st)
    path_value = float(state.get("path_score", 0.0))
    if path_evaluator is not None and path_value == 0.0:
        path_value, path_components = path_evaluator.score_path(question, path, args, model_st)
        state["path_score"] = path_value
        state["path_components"] = path_components
    relation_value = float(state.get("relation_score", 0.0))

    total = (
        args.w1 * acc +
        args.w2 * eff +
        args.w3 * div +
        getattr(args, "state_reward_weight", 0.35) * state_value +
        getattr(args, "path_reward_weight", 0.15) * path_value +
        getattr(args, "relation_reward_weight", 0.15) * relation_value
    )

    state_uid = build_state_uid(state)
    if all_experiences:
        for exp in all_experiences:
            if exp.get("next_state_uid") == state_uid:
                exp["reward"] = total
                exp["done"] = True

    return {
        'accuracy': acc,
        'efficiency': eff,
        'diversity': div,
        'state_value': state_value,
        'path_value': path_value,
        'relation_value': relation_value,
        'total_reward': total
    }


def beam_search_reasoning_tao(question, central_entity_ids, args, ppo_agent, model_st, path_evaluator=None):
    # Canonical search loop kept active until true rollout MCTS replaces beam expansion.
    if not central_entity_ids:
        print(f"{color_red}Step 2 Exception: No central entities provided to start reasoning.{color_end}")
        return [], [], {'total': 0, 'input': 0, 'output': 0}, [], {}

    total_token_num = {'total': 0, 'input': 0, 'output': 0}
    all_experiences = []
    visited = set()
    tree_stats = {}
    state_history = []
    current_beam = []
    search_diagnostics = {
        "expanded_candidates": 0,
        "scored_candidates": 0,
        "avg_state_score": 0.0,
        "avg_path_score": 0.0,
        "avg_simulation_score": 0.0,
        "avg_mcts_score": 0.0,
        "final_state_count": 0,
        "path_evaluator_enabled": bool(path_evaluator is not None),
    }

    for entity_id in central_entity_ids:
        initial_state = {
            'query': question,
            'path': [],
            'observation': None,
            'subtask': None,
            'previous_subtasks': [],
            'current_entity_id': entity_id,
            'parent_state_index': None,
            'action_from_parent': None,
            'backtrack_count': 0,
            'depth': 0,
            'error': None,
            'error_details': None,
            'state_score': 0.0,
            'path_score': 0.0,
            'mcts_score': 0.0,
            'relation_score': 0.0,
        }
        current_beam.append(initial_state)
        state_history.append(initial_state)
        update_tree_statistics(tree_stats, build_state_uid(initial_state), 0.0, prior_score=0.0)

    last_current_beam = current_beam[:]
    current_depth = 0
    print(f"{color_green}Step 2: Starting search with {len(current_beam)} root states. Path evaluator enabled: {bool(path_evaluator is not None)}.{color_end}")
    while current_depth < args.depth and current_beam:
        print(f"\nDepth {current_depth}: Processing {len(current_beam)} states...")
        next_beam_candidates = []

        for state_index, state in enumerate(current_beam):
            try:
                current_state = copy.deepcopy(state)
                parent_uid = build_state_uid(current_state)
                update_tree_statistics(
                    tree_stats,
                    parent_uid,
                    reward=current_state.get("state_score", 0.0),
                    prior_score=current_state.get("state_score", 0.0),
                )

                step_result = step(current_state, args, state_index, model_st)
                for key in total_token_num:
                    total_token_num[key] += step_result['token_num'][key]

                if step_result.get('terminated'):
                    current_state['terminated'] = True
                    continue

                relation_ids = step_result.get('candidate_relations', [])
                next_states = step_result.get('next_states', [])
                if not relation_ids or not next_states:
                    continue

                state_vector = convert_state_to_ppo_input(current_state, args, model_st)
                action_probs = ppo_agent.get_action_probs(state_vector, len(relation_ids))
                relation_prob_map = {
                    relation_ids[idx]: float(action_probs[idx])
                    for idx in range(min(len(relation_ids), len(action_probs)))
                }

                relation_score_map = step_result.get("relation_score_map", {})
                for next_state in next_states:
                    state_uid = build_state_uid(next_state)
                    if state_uid in visited:
                        continue
                    visited.add(state_uid)

                    rel_id = next_state.get("action_from_parent")
                    relation_prior = float(
                        relation_score_map.get(rel_id, relation_score_map.get(get_relation_name(rel_id), 0.0))
                    )
                    policy_prior = float(relation_prob_map.get(rel_id, 0.0))
                    state_score, state_components = context_aware_state_score(question, next_state, args, model_st)
                    path_score, path_components = (
                        path_evaluator.score_path(question, next_state.get("path", []), args, model_st)
                        if path_evaluator is not None
                        else (0.0, {})
                    )
                    prior_score = (
                        getattr(args, "state_score_weight", 0.5) * state_score +
                        getattr(args, "path_score_weight", 0.2) * path_score +
                        getattr(args, "policy_score_weight", 0.3) * policy_prior +
                        getattr(args, "relation_score_weight", 0.1) * relation_prior
                    )
                    mcts_score = get_tree_ucb_score(
                        tree_stats,
                        state_uid,
                        parent_uid=parent_uid,
                        exploration_weight=getattr(args, "mcts_exploration_weight", 1.2),
                        fallback_prior=prior_score,
                    )

                    next_state["state_score"] = state_score
                    next_state["state_components"] = state_components
                    next_state["path_score"] = path_score
                    next_state["path_components"] = path_components
                    next_state["policy_score"] = policy_prior
                    next_state["relation_score"] = relation_prior
                    next_state["simulation_score"] = prior_score
                    next_state["mcts_score"] = mcts_score
                    next_state["parent_state_uid"] = parent_uid
                    next_state["state_uid"] = state_uid
                    next_beam_candidates.append(next_state)
                    search_diagnostics["expanded_candidates"] += 1
                    search_diagnostics["scored_candidates"] += 1
                    search_diagnostics["avg_state_score"] += float(state_score)
                    search_diagnostics["avg_path_score"] += float(path_score)
                    search_diagnostics["avg_simulation_score"] += float(prior_score)
                    search_diagnostics["avg_mcts_score"] += float(mcts_score)

                    action_index = relation_ids.index(rel_id) if rel_id in relation_ids else 0
                    next_state_vector = convert_state_to_ppo_input(next_state, args, model_st)
                    all_experiences.append({
                        'state': state_vector,
                        'action': action_index,
                        'reward': 0.0,
                        'next_state': next_state_vector,
                        'log_prob': float(np.log(max(policy_prior, 1e-8))),
                        'done': False,
                        'next_state_uid': state_uid,
                        'parent_state_uid': parent_uid,
                    })
            except Exception as e:
                print(f"{color_red}Step 2 Search Exception at depth {current_depth}: {e}{color_end}")
                error_stack = traceback.format_exc()
                state['error'] = "State processing failed"
                state['error_details'] = {
                    'error_message': str(e),
                    'error_type': type(e).__name__,
                    'error_stack': error_stack
                }
                state['backtrack_count'] = state.get('backtrack_count', 0) + 1

        if not next_beam_candidates:
            break

        beam_width = adaptive_beam_width(next_beam_candidates, args.beam_width, max_beam=args.beam_width + 2)
        next_beam_candidates.sort(
            key=lambda x: (
                x.get('mcts_score', 0.0),
                x.get('simulation_score', 0.0),
                x.get('state_score', 0.0),
                x.get('path_score', 0.0),
                x.get('policy_score', 0.0),
                len(set(x.get('previous_subtasks', []) + [x.get('subtask', '')])),
            ),
            reverse=True
        )

        current_beam = next_beam_candidates[:beam_width]
        if current_beam:
            last_current_beam = current_beam[:]
            print(
                f"{color_green}Step 2 Depth {current_depth} Summary: kept {len(current_beam)}/{len(next_beam_candidates)} states, "
                f"avg_state={np.mean([s.get('state_score', 0.0) for s in current_beam]):.4f}, "
                f"avg_path={np.mean([s.get('path_score', 0.0) for s in current_beam]):.4f}.{color_end}"
            )
        state_history.extend(current_beam)
        current_depth += 1

    print("\nStep 2: Evaluating all reasoning paths...")
    current_beam = last_current_beam
    all_valid_paths = [s['path'] for s in current_beam if s.get('path')]
    evaluated_states = []
    subtasks = []
    for state in current_beam:
        subtasks.append(state.get('subtask'))
        if state.get('path'):
            score = evaluate_path(
                state,
                question,
                args,
                model_st,
                path_evaluator=path_evaluator,
                all_experiences=all_experiences,
                state_vec=None,
                all_paths=all_valid_paths
            )
            state['score'] = score
            evaluated_states.append(state)
            update_tree_statistics(
                tree_stats,
                build_state_uid(state),
                reward=score.get("total_reward", 0.0),
                parent_uid=state.get("parent_state_uid"),
                prior_score=state.get("state_score", 0.0),
            )

    preference_pairs = derive_pseudo_preference_pairs(
        evaluated_states,
        margin_threshold=getattr(args, "pseudo_refine_margin", 0.05),
    )
    apply_pseudo_state_refinement(all_experiences, preference_pairs, args)

    final_states = [
        state for state in evaluated_states
        if state.get('score', {}).get('total_reward', 0.0) >= args.backtrack_threshold
    ]
    if not final_states and evaluated_states:
        final_states = sorted(
            evaluated_states,
            key=lambda s: s.get('score', {}).get('total_reward', 0.0),
            reverse=True
        )[:1]

    final_paths = [s['path'] for s in final_states if s.get('path')]
    scored_count = max(1, int(search_diagnostics["scored_candidates"]))
    search_diagnostics["avg_state_score"] = float(search_diagnostics["avg_state_score"] / scored_count)
    search_diagnostics["avg_path_score"] = float(search_diagnostics["avg_path_score"] / scored_count)
    search_diagnostics["avg_simulation_score"] = float(search_diagnostics["avg_simulation_score"] / scored_count)
    search_diagnostics["avg_mcts_score"] = float(search_diagnostics["avg_mcts_score"] / scored_count)
    search_diagnostics["final_state_count"] = int(len(final_states))
    search_diagnostics["final_path_count"] = int(len(final_paths))
    print(
        f"{color_green}Step 2 Completed: final_paths={search_diagnostics['final_path_count']}, "
        f"avg_state={search_diagnostics['avg_state_score']:.4f}, "
        f"avg_path={search_diagnostics['avg_path_score']:.4f}, "
        f"avg_sim={search_diagnostics['avg_simulation_score']:.4f}.{color_end}"
    )
    return final_paths, all_experiences, total_token_num, list(set([s for s in subtasks if s])), search_diagnostics


def generate_answer(question, reasoning_paths, args):
    print("Step 3: Generating Final Answer...")

    if not reasoning_paths:
        print(f"{color_yellow}Warning: No reasoning paths found to generate answer.{color_end}")
        return "Could not find an answer based on the reasoning paths.", {'total': 0, 'input': 0, 'output': 0}

    formatted_paths = [format_path_for_prompt(path) for path in reasoning_paths[:5]]
    paths_str = json.dumps(formatted_paths)

    if len(paths_str) > 2000:
        paths_str = paths_str[:2000] + '...]'

    prompt = answer_prompt.format(
        question=question,
        reasoning_paths=paths_str,
    )

    response, token_stat = run_llm(
        prompt=prompt,
        temperature=args.temperature_answer,
        max_tokens=args.max_length // 4,
        openai_api_keys=args.openai_api_keys
    )

    try:
        answer_json = safe_json_loads(response, default={})
        answer = answer_json.get('answer', response)
    except Exception as e:
        print(f"{color_red}Step 3 Exception: {e}{color_end}")
        answer = response

    print(f"{color_green}Step 3 Generated Answer: {answer}{color_end}")
    return answer, token_stat
