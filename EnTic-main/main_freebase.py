from tqdm import tqdm
import argparse
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from utils import *
from freebase_func import *

os.environ['HF_HOME'] = '/root/huggingface/'

color_yellow = "\033[93m"
color_green = "\033[92m"
color_red= "\033[91m"
color_end = "\033[0m"

# --- 数据集过滤 ---
def repeat_unanswer(dataset_name, datas, question_id_key, model_name, results_dir):
    answered_set = set()
    new_data = []        
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, f'EnTic{dataset_name}_{model_name}.jsonl')
    if os.path.exists(file_path):
        print(f"Checking existing results in: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    try:
                        data = json.loads(line) 
                        q_id = data.get(question_id_key)
                        if q_id is not None:
                            answered_set.add(q_id) 
                        else:
                            print(f"{color_yellow}Warning: Line missing ID key '{question_id_key}' in {file_path}: {line.strip()}{color_end}")
                    except json.JSONDecodeError:
                         print(f"{color_yellow}Warning: Skipping invalid JSON line in {file_path}: {line.strip()}{color_end}")
        except Exception as e:
             print(f"{color_red}Error reading existing results file {file_path}: {e}. Proceeding without filtering.{color_end}")
             answered_set = set() 
    print(f"Found {len(answered_set)} already answered questions.")

    # 遍历原始数据集，过滤掉已回答的问题
    for x in datas:
        q_id = x.get(question_id_key)
        if q_id is not None and q_id not in answered_set:
            new_data.append(x)
        elif q_id is None:
             print(f"{color_yellow}Warning: Data item missing ID key '{question_id_key}'. Skipping: {x}{color_end}")

    print(f"Returning {len(new_data)} unanswered questions.")
    return new_data 

# 定义支持动态字段的类
class GenericItem:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)  # 动态设置属性

    def __repr__(self):
        return f"GenericItem({self.__dict__})"

# 根据问题 ID 从数据集中检索单个问题数据
def get_one_data(datas, question_id_key, target_question_id):
    for data in datas:
        if data.get(question_id_key) == target_question_id:
            return [data]
    return [] 

# --- PPO Agent 实现 ---
class Actor(nn.Module):
    """PPO Actor 网络: 将状态映射到动作概率。"""
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128), # 输入层 (状态维度 -> 128)
            nn.ReLU(),                 # ReLU 激活函数
            nn.Linear(128, 128),       # 隐藏层 (128 -> 128)
            nn.ReLU(),
            nn.Linear(128, action_dim) # 输出层 (128 -> 动作维度)，输出 logits
        )

    def forward(self, state):
        """前向传播，返回动作的 logits。"""
        return self.network(state)

class Critic(nn.Module):
    """PPO Critic 网络: 将状态映射到一个价值估计。"""
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128), # 输入层 (状态维度 -> 128)
            nn.ReLU(),
            nn.Linear(128, 128),       # 隐藏层 (128 -> 128)
            nn.ReLU(),
            nn.Linear(128, 1)          # 输出层 (128 -> 1)，输出单个状态价值
        )

    def forward(self, state):
        """前向传播，返回状态价值。"""
        return self.network(state)

# 用于 EnTic 反馈模块的 PPO Agent
class PPOAgent:
    def __init__(self, args):
        print("Initializing PPO Agent...")
        self.metrics_path = "/root/autodl-fs/EnTic-main-main/Results/ppo_metrics.jsonl"
        self.global_update_step = 0
        self.args = args 
        self.action_dim = args.beam_width 
        self.state_dim = args.state_dim 

        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.learning_rate)

        # PPO 超参数
        self.gamma = args.gamma             # 折扣因子
        self.ppo_epochs = args.ppo_epochs   # 每次更新时训练的轮数
        self.clip_epsilon = args.clip_epsilon # PPO 裁剪参数
        self.batch_size = args.batch_size     # 训练更新的批次大小

        # 添加GAE和KL散度相关参数
        self.lambda_gae = args.lambda_gae  # GAE的lambda参数
        self.beta = args.beta  # KL散度约束系数

    def select_action(self, state_vector, num_candidates):
        """根据当前状态使用 Actor 网络选择一个动作（候选关系的索引）。"""
        if num_candidates <= 0:
             print(f"{color_red}Error: PPO select_action called with zero candidates!{color_end}")
             return 0, torch.tensor(0.0) 

        # 将状态向量转换为 FloatTensor，并添加 batch 维度
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0) 

        # 在不计算梯度的情况下获取 Actor 网络的输出 (logits)
        with torch.no_grad(): 
            action_logits = self.actor(state_tensor)

        # --- 对 Logits 进行掩码，只考虑有效的候选动作 ---
        if num_candidates < self.action_dim: 
             mask = torch.ones(self.action_dim) * -1e18 
             mask[:num_candidates] = 0 
             masked_logits = action_logits + mask 
        else:
             masked_logits = action_logits

        # 基于掩码后的 logits 创建分类（Categorical）概率分布
        dist = Categorical(logits=masked_logits)

        # 从分布中采样一个动作索引
        action_index = dist.sample()

        # 计算采样到的动作的对数概率 log(pi(a|s))
        log_prob = dist.log_prob(action_index)

        # 返回动作索引（整数）和对数概率（张量）
        return action_index.item(), log_prob.squeeze(0) 

    def get_action_probs(self, state_vector, num_candidates):
        if num_candidates <= 0:
            return np.array([])

        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
        with torch.no_grad():
            action_logits = self.actor(state_tensor)

        if num_candidates < self.action_dim:
            mask = torch.ones(self.action_dim) * -1e18
            mask[:num_candidates] = 0
            masked_logits = action_logits + mask
        else:
            masked_logits = action_logits

        probs = torch.softmax(masked_logits, dim=-1).squeeze(0).cpu().numpy()
        return probs[:num_candidates]

    def update_policy(self, experiences_to_train_on):
        """使用收集到的经验更新Actor和Critic网络，支持GAE和KL散度约束。"""
        if not experiences_to_train_on:
            print("PPO Update: No experiences in memory.")
            return
            
        if len(experiences_to_train_on) < self.batch_size:
            print(f"PPO Update: Not enough experiences ({len(experiences_to_train_on)} < {self.batch_size}) for a full batch. Skipping update.")
            return
            
        print(f"PPO Update: Updating policy with {len(experiences_to_train_on)} experiences...")
        
        # 将经验列表转换为Tensors
        states = torch.FloatTensor(np.array([exp['state'] for exp in experiences_to_train_on]))
        actions = torch.LongTensor(np.array([exp['action'] for exp in experiences_to_train_on]))
        rewards = torch.FloatTensor(np.array([exp['reward'] for exp in experiences_to_train_on]))
        next_states = torch.FloatTensor(np.array([exp['next_state'] for exp in experiences_to_train_on]))
        old_log_probs = torch.FloatTensor(np.array([exp['log_prob'] for exp in experiences_to_train_on]))
        dones = torch.FloatTensor(np.array([exp['done'] for exp in experiences_to_train_on])).unsqueeze(1)
        
        # 计算GAE优势
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            
            # 计算TD误差
            td_errors = rewards.unsqueeze(1) + self.gamma * next_values * (1 - dones) - values
            
            # 计算GAE优势
            advantages = torch.zeros_like(td_errors)
            running_advantage = 0
            for t in reversed(range(len(td_errors))):
                if t == len(td_errors) - 1:
                    running_advantage = td_errors[t]
                else:
                    running_advantage = td_errors[t] + self.gamma * self.lambda_gae * running_advantage
                advantages[t] = running_advantage
            
            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 计算回报
            returns = advantages + values
        # --- 新增：本次 update 的累计量 ---
        epoch_policy_losses, epoch_value_losses = [], []
        epoch_entropies, epoch_kls, epoch_clipfracs = [], [], []

        # PPO训练循环
        for _ in range(self.ppo_epochs):
            # 随机打乱数据
            indices = np.arange(len(experiences_to_train_on))
            np.random.shuffle(indices)
            
            # 小批量训练
            for start in range(0, len(experiences_to_train_on), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # 获取当前批次的数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 计算当前策略的动作概率
                current_logits = self.actor(batch_states)
                current_dist = Categorical(logits=current_logits)
                current_log_probs = current_dist.log_prob(batch_actions)

                # 小批次训练开始处添加这句以计算熵
                current_entropy = current_dist.entropy().mean()

                # 计算策略比率
                ratio = torch.exp(current_log_probs - batch_old_log_probs)

                # 计算 clipfrac（有多少样本触发裁剪区域）
                clipfrac = (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean()

                # 计算PPO裁剪目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算KL散度
                kl_div = torch.mean(batch_old_log_probs - current_log_probs)
                
                # 添加KL散度约束
                policy_loss = policy_loss + self.beta * kl_div



                # 计算价值损失
                current_values = self.critic(batch_states)
                value_loss = nn.MSELoss()(current_values, batch_returns)

                # 反向传播与优化后，累计度量
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropies.append(current_entropy.item())
                epoch_kls.append(kl_div.item())
                epoch_clipfracs.append(clipfrac.item())
                # 更新Actor网络
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_optimizer.step()

                # 更新Critic网络
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()
                
        print("PPO policy update finished.")

        # --- 一个 update_policy() 完成后，写出平均指标 ---
        if len(epoch_policy_losses) > 0:
            metrics = {
                "t": ts(),
                "update_step": int(self.global_update_step),
                "policy_loss": float(np.mean(epoch_policy_losses)),
                "value_loss": float(np.mean(epoch_value_losses)),
                "policy_entropy": float(np.mean(epoch_entropies)),  # ← 策略熵
                "approx_kl": float(np.mean(epoch_kls)),  # ← 近似 KL
                "clipfrac": float(np.mean(epoch_clipfracs)),
                "gamma": float(self.gamma),
                "clip_eps": float(self.clip_epsilon),
                "beta_kl": float(self.beta),
                "batch_size": int(self.batch_size),
                "ppo_epochs": int(self.ppo_epochs)
            }
            try:
                from utils import append_jsonl
                append_jsonl(self.metrics_path, metrics)
            except Exception as e:
                print(f"[Metrics] Failed to log metrics: {e}")
            self.global_update_step += 1

# --- 主执行函数 ---
def main():
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="Run EnTic framework with PPO Feedback.")
    
    # 数据集和路径参数
    parser.add_argument("--dataset", type=str, default="cwq", choices=["cwq", "webqsp", "grailqa"], help="Choose the dataset.")
    parser.add_argument("--results_dir", type=str, default="/root/autodl-fs/EnTic-main-main/Results", help="Directory to save results.")
    parser.add_argument("--model_path", type=str, default='/root/huggingface/models--sentence-transformers--msmarco-distilbert-base-tas-b/snapshots/b12d9352e776979147078a8975a4885042984fd1/', help="Path to Sentence Transformer model.")
    
    # LLM 配置参数
    parser.add_argument("--LLM_type", type=str, default="gpt-4.1", help="Base LLM model for generation/evaluation.")
    parser.add_argument("--openai_api_keys", type=str, default=os.environ.get("OPENAI_API_KEY", ""), help="OpenAI API key. Reads from env var OPENAI_API_KEY if not set.")
    parser.add_argument("--max_length", type=int, default=2048, help="Max tokens for LLM responses.")
    parser.add_argument("--temperature_reasoning", type=float, default=0.3, help="Temperature for LLM during reasoning (subtask, relation selection).")
    parser.add_argument("--temperature_answer", type=float, default=0.3, help="Temperature for LLM during final answer generation.")

    # EnTic 推理参数
    parser.add_argument("--depth", type=int, default=5, help="Max reasoning depth (number of TAO steps).")
    parser.add_argument("--beam_width", type=int, default=3, help="Beam search width (k); also PPO action dim.")
    parser.add_argument("--max_path_length", type=int, default=5, help="Maximum length of a reasoning path before termination.")

    # EnTic 评估参数
    parser.add_argument('--optimal_path_length', type=int, default=3, help='Target optimal path length for Efficiency reward.')
    parser.add_argument('--alpha', type=float, default=0.5, help='Reward weight for partially correct subtasks (Accuracy).')
    parser.add_argument('--w1', type=float, default=1.0, help='Weight for Accuracy reward.')
    parser.add_argument('--w2', type=float, default=0.3, help='Weight for Efficiency reward.') # 示例：降低效率权重
    parser.add_argument('--w3', type=float, default=0.2, help='Weight for Diversity reward.') # 示例：降低多样性权重

    # EnTic 反馈 (PPO) 参数
    parser.add_argument('--state_dim', type=int, default=512 + 96 + 96 + 96 + 8 + 8 + 8, help='Dimension of the state vector for PPO (adjust based on utils.convert_state_to_ppo_input).') 
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for PPO optimizers.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for PPO rewards.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for PPO training updates.')
    parser.add_argument('--ppo_epochs', type=int, default=5, help='Number of epochs to train PPO per update.')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='Clipping epsilon for PPO policy loss.')
    parser.add_argument('--mcts_exploration_weight', type=float, default=1.2, help='Exploration constant used in state-MCTS style UCB scoring.')
    parser.add_argument('--state_score_weight', type=float, default=0.5, help='Weight of context-aware state score when ranking next states.')
    parser.add_argument('--path_score_weight', type=float, default=0.2, help='Weight of lightweight path plausibility score during simulation ranking.')
    parser.add_argument('--policy_score_weight', type=float, default=0.3, help='Weight of PPO policy prior when ranking next states.')
    parser.add_argument('--relation_score_weight', type=float, default=0.1, help='Weight of LLM relation prior when ranking next states.')
    parser.add_argument('--state_reward_weight', type=float, default=0.35, help='Reward weight assigned to context-aware state value.')
    parser.add_argument('--path_reward_weight', type=float, default=0.15, help='Reward weight assigned to lightweight path plausibility.')
    parser.add_argument('--relation_reward_weight', type=float, default=0.15, help='Reward weight assigned to relation prior score.')
    parser.add_argument('--pseudo_refine_weight', type=float, default=0.2, help='Reward shaping weight from pseudo chain-state preference refinement.')
    parser.add_argument('--pseudo_refine_margin', type=float, default=0.05, help='Minimum reward gap required to form a pseudo preference pair.')
    parser.add_argument('--state_semantic_weight', type=float, default=0.40, help='Context-aware state scorer: semantic alignment weight.')
    parser.add_argument('--state_observation_weight', type=float, default=0.20, help='Context-aware state scorer: observation richness weight.')
    parser.add_argument('--state_depth_weight', type=float, default=0.15, help='Context-aware state scorer: depth fitness weight.')
    parser.add_argument('--state_chain_weight', type=float, default=0.15, help='Context-aware state scorer: chain consistency weight.')
    parser.add_argument('--state_novelty_weight', type=float, default=0.10, help='Context-aware state scorer: novelty weight.')

    #ppo评估
    # --- 新增：度量记录 ---
    parser.add_argument("--metrics_path", type=str,
                        default="/root/autodl-fs/EnTic-main-main/Results/ppo_metrics.jsonl",
                        help="Where to append PPO training metrics (JSONL).")
    parser.add_argument("--metrics_eval_path", type=str,
                        default="/root/autodl-fs/EnTic-main-main/Results/ppo_eval_metrics.jsonl",
                        help="Where to append per-question eval metrics (JSONL).")

    # 添加回溯推理相关参数
    parser.add_argument('--backtrack_threshold', type=float, default=1.0,
                       help='Score threshold below which paths are considered for backtracking.')
    parser.add_argument('--max_backtrack', type=int, default=3, 
                       help='Maximum number of times a path can be backtracked.')

    # 添加PPO训练相关参数
    parser.add_argument('--lambda_gae', type=float, default=0.95, 
                       help='Lambda for Generalized Advantage Estimation.')
    parser.add_argument('--beta', type=float, default=0.01, 
                       help='KL divergence constraint coefficient.')
    parser.add_argument('--disable_path_evaluator', action='store_true', help='Disable the lightweight path evaluator for ablation.')
    parser.add_argument('--path_evaluator_warmup', type=int, default=0, help='Enable the path evaluator only after this many processed questions.')
    parser.add_argument('--experiment_tag', type=str, default='state_path_sim', help='Tag written into result and metric files for ablation tracking.')
    parser.add_argument('--max_questions', type=int, default=100, help='Maximum number of questions to process in one run.')

    # 执行控制参数
    parser.add_argument('--question_id', type=str, default=None, help='Process only a single question specified by its ID.')
    parser.add_argument('--skip_answered', action='store_true', help='Skip questions already found in the results file.')

    args = parser.parse_args()

    # --- 初始化 ---
    # 加载 Sentence Transformer 模型 
    try:
        model_st = load_sentence_transformer_model(args.model_path)
    except Exception:
         print(f"{color_red}Failed to load Sentence Transformer model. Exiting.{color_end}")
         return 

    # 初始化 PPO Agent
    if args.beam_width <= 0 : raise ValueError("beam_width must be positive")
    print(f"Setting PPO action_dim = beam_width = {args.beam_width}")
    ppo_agent = PPOAgent(args)
    base_path_evaluator = None if args.disable_path_evaluator else LightweightPathEvaluator()
    print(
        f"{color_green}Experiment Setup: tag={args.experiment_tag}, "
        f"path_evaluator_enabled={not args.disable_path_evaluator}, "
        f"path_evaluator_warmup={args.path_evaluator_warmup}, "
        f"max_questions={args.max_questions}.{color_end}"
    )

    # 加载数据集
    try:
        datas, question_id_key, question_text_key = prepare_dataset(args.dataset)
    except Exception as e:
        print(f"{color_red}Failed to load dataset '{args.dataset}': {e}. Exiting.{color_end}")
        return

    # --- 数据过滤 ---
    if args.question_id: 
        print(f"Processing only question with ID: {args.question_id}")
        datas_to_process = get_one_data(datas, question_id_key, args.question_id)
        if not datas_to_process:
             print(f"{color_red}Error: Question with ID '{args.question_id}' not found.{color_end}")
             return
    else: 
        datas_to_process = datas 
        if args.skip_answered:
            datas_to_process = repeat_unanswer(args.dataset, datas_to_process, question_id_key, args.LLM_type.replace('/','_'), args.results_dir)

    all_training_experiences = []

    # --- 主循环：处理每个问题 ---
    total_questions = len(datas_to_process) 
    print(f"\nStarting processing for {total_questions} questions...")
    successful_questions = 0 
    failed_questions = 0    

    time_circle = 1
    # 使用 tqdm 显示进度条,对 datas_to_process 进行迭代
    for i, data_item in enumerate(tqdm(datas_to_process, desc="Processing Questions")):
        if time_circle > args.max_questions:
            break
        start_time = time.time()
        question_id = data_item.get(question_id_key)
        question_text = data_item.get(question_text_key)

        # 如果缺少 ID 或文本，跳过此数据项
        if question_id is None or question_text is None:
            print(f"{color_yellow}Warning: Skipping data item missing ID ('{question_id_key}') or Text ('{question_text_key}'): {data_item}{color_end}")
            failed_questions += 1
            continue

        print(f"\n\n===== Processing Question {i+1}/{total_questions}: {question_id} =====")
        print(f"Q: {question_text}")

        # 初始化当前问题的变量
        total_token_usage = {'total': 0, 'input': 0, 'output': 0} 
        final_reasoning_paths = [] 
        answer = "Processing failed."
        central_entity_names = []
        central_entity_ids = []
        question_experiences = []
        subtasks = []
        search_diagnostics = {}
        tn_entity = {'total': 0, 'input': 0, 'output': 0}
        tn_reasoning = {'total': 0, 'input': 0, 'output': 0}
        tn_answer = {'total': 0, 'input': 0, 'output': 0}
        result_list_wrong = ["Insufficient information to provide an answer.", "Processing failed.",
                       "Could not find an answer based on the reasoning paths.",
                       "Insufficient information to provide an answer."]
        try:
            # 步骤 1: 提取中心实体
            central_entity_names, central_entity_ids, tn_entity = extract_central_entities(question_text, args)
            print(f"# 步骤 1: 提取中心实体: central_entity_names{central_entity_names},central_entity_ids:{central_entity_ids},tn_entity:{tn_entity}")
            total_token_usage = {k: total_token_usage.get(k, 0) + tn_entity.get(k, 0) for k in total_token_usage}
            # 步骤 2, 3, 4a: 执行 EnTic 推理循环
            active_path_evaluator = base_path_evaluator if i >= args.path_evaluator_warmup else None
            print(
                f"{color_green}Step 2 Config: active_path_evaluator={bool(active_path_evaluator is not None)}, "
                f"beam_width={args.beam_width}, depth={args.depth}.{color_end}"
            )
            final_reasoning_paths, question_experiences, tn_reasoning, subtasks, search_diagnostics = beam_search_reasoning_tao(
                question=question_text,
                central_entity_ids=central_entity_ids,
                args=args,
                ppo_agent=ppo_agent,
                model_st=model_st,
                path_evaluator=active_path_evaluator,
            )
            all_training_experiences.extend(question_experiences)
            # 步骤 4b: 反馈 - PPO 策略更新
            # if len(all_training_experiences) >= args.batch_size:
            #     print("\n--- Starting PPO Policy Update ---")
            #     # 更新策略
            #     ppo_agent.update_policy(all_training_experiences)
            #     all_training_experiences = []

            # 步骤 5: 生成最终答案
            answer, tn_answer = generate_answer(question_text, final_reasoning_paths,args)
            total_token_usage = {k: total_token_usage.get(k, 0) + tn_answer.get(k, 0) for k in total_token_usage}
            successful_questions += 1

        except Exception as e:
            print(f"{color_red}Question Processing Exception [{question_id}]: {e}{color_end}")
            failed_questions += 1 

        finally:
            time_taken = time.time() - start_time
            # === 新增：记录“平均回报/单题耗时/Token”等评测指标 ===
            try:
                # 仅统计本题刚收集的经验（question_experiences）
                # 有些 exp 的 reward 可能在 evaluate_path 后才被写入，这里只取已写入的那些
                q_rewards = [exp.get('reward', 0.0) for exp in (question_experiences or []) if
                             isinstance(exp.get('reward', None), (int, float))]
                avg_return = float(np.mean(q_rewards)) if len(q_rewards) > 0 else 0.0

                eval_metrics = {
                    "t": time.time(),
                    "qid": question_id,
                    "dataset": args.dataset,
                    "experiment_tag": args.experiment_tag,
                    "path_evaluator_enabled": bool((base_path_evaluator is not None) and (i >= args.path_evaluator_warmup)),
                    "avg_return": avg_return,  # ← 平均回报
                    "expanded_candidates": int(search_diagnostics.get("expanded_candidates", 0)),
                    "avg_state_score": float(search_diagnostics.get("avg_state_score", 0.0)),
                    "avg_path_score": float(search_diagnostics.get("avg_path_score", 0.0)),
                    "avg_simulation_score": float(search_diagnostics.get("avg_simulation_score", 0.0)),
                    "avg_mcts_score": float(search_diagnostics.get("avg_mcts_score", 0.0)),
                    "final_path_count": int(search_diagnostics.get("final_path_count", len(final_reasoning_paths))),
                    "time_taken": float(time_taken),
                    "input_token": int(total_token_usage.get("input", 0)),
                    "output_token": int(total_token_usage.get("output", 0)),
                    "total_token": int(total_token_usage.get("total", 0)),
                    "call_num": float(total_token_usage.get("total", 0) / max(1, args.max_length))
                }
                from utils import append_jsonl
                append_jsonl(args.metrics_eval_path, eval_metrics)
            except Exception as e:
                print(f"{color_red}Eval Metrics Exception [{question_id}]: {e}{color_end}")

            time_taken = time.time() - start_time
            # --- 结果保存 ---
            result_data = {
                question_id_key: question_id, 
                'question': question_text,  
                "answer": answer,             
                "reasoning_paths": final_reasoning_paths,
                "central_entities_ids": central_entity_ids if 'central_entity_ids' in locals() else [],
                "central_entities_names": central_entity_names if 'central_entity_names' in locals() else [],
                "subtasks": subtasks,
                "search_diagnostics": search_diagnostics,
                "experiment_tag": args.experiment_tag,
                "path_evaluator_enabled": bool((base_path_evaluator is not None) and (i >= args.path_evaluator_warmup)),
                "total_token_usage": total_token_usage,
                "time_taken": time_taken,             
                "max_length": args.max_length      
            }

            # 再次定义结果文件路径用于保存
            results_file = os.path.join(args.results_dir, f'EnTic{args.dataset}_{args.LLM_type.replace("/","_")}.jsonl')
            try:
                save_2_jsonl(result_data, results_file, question_id_key)
                print(f"{color_green}Result for question {question_id} saved to {results_file}{color_end}")
            except Exception as save_e:
                print(f"{color_red}Result Saving Exception [{question_id}]: {save_e}{color_end}")

            print(f"{color_green}Question Summary [{question_id}]: time={time_taken:.2f}s, tokens={total_token_usage}, diagnostics={search_diagnostics}{color_end}")
            time_circle += 1


    # --- 处理完成，打印总结信息 ---
    print("\n===== Processing Summary =====")
    print(f"Total questions targeted: {total_questions}")
    print(f"Successfully processed: {successful_questions}")
    print(f"Failed processing: {failed_questions}")
    success_rate = (successful_questions / total_questions * 100) if total_questions > 0 else 0
    print(f"Success rate: {success_rate:.2f}%")
    print("============================")


# --- 程序入口 ---
if __name__ == '__main__':
    os.environ.setdefault("OPENAI_API_KEY", "")

    if not os.environ.get("OPENAI_API_KEY"):
         temp_parser = argparse.ArgumentParser(add_help=False) 
         temp_parser.add_argument("--openai_api_keys", type=str, default="")
         temp_args, _ = temp_parser.parse_known_args() 
         if not temp_args.openai_api_keys:
              print(f"{color_red}Error: OpenAI API key not found. Set OPENAI_API_KEY environment variable or use --openai_api_keys argument.{color_end}")
              exit(1)

    main()
