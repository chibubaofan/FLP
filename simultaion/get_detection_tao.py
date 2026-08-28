import os
import argparse
import random
import time
import json
import torch
import sys
import numpy as np
from datetime import datetime
from tqdm import tqdm

# ==================== 【魔法修复：拦截参数】 ====================
original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]
# ==============================================================

try:
    from simulation import AgentBrain, ClipFeatureExtractor, MetricsCalculator
    from prompt import PromptGenerator
except ImportError as e:
    print(f"导入失败: {e}\n请确保本脚本与 simulation.py 和 prompt.py 放在一起。")
    exit(1)

sys.argv = original_argv

def parse_args():
    parser = argparse.ArgumentParser(description="Step 1: Offline Benign Profiling (Calibration)")
    # 注意：这里建议传入一个专门用于校准的人格池，例如 characters_calibration.json
    parser.add_argument("--calibration_data", type=str, default="")
    parser.add_argument("--album_data", type=str, default="data/images/")
    
    # 纯净环境，无需投毒相关参数 (attack_image, poison_ratio)
    parser.add_argument("--num_calibration_agents", type=int, default=30, help="校准用的纯良性代理数量")
    parser.add_argument("--album_length", type=int, default=10)
    parser.add_argument("--k_personas", type=int, default=4)
    parser.add_argument("--persona_type", type=str, choices=["homogeneous", "heterogeneous"], default="heterogeneous")
    
    parser.add_argument("--vlm", type=str, default="")
    parser.add_argument("--clip", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tolerance_fpr", type=float, default=7.5, help="允许的正常分布下界百分位")
    
    return parser.parse_args()
    
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(49)

def random_pairs(lst):
    if len(lst) < 2: return [(lst[0], lst[0])]
    shuffled = lst[:]
    random.shuffle(shuffled)
    pairs = []
    for i in range(0, len(shuffled) - 1, 2):
        pairs.append((shuffled[i], shuffled[i+1]))
    if len(shuffled) % 2 != 0:
        pairs.append((shuffled[-1], random.choice(shuffled[:-1])))
    return pairs

def generate_benign_agents(args, calibration_personas):
    """生成100%纯净的良性代理用于基线画像"""
    agents_state = {}
    random.shuffle(calibration_personas)
    
    for i in range(args.num_calibration_agents):
        p_main = calibration_personas[i % len(calibration_personas)]
        
        # 构建相册 (不投毒)
        raw_furniture_list = [img for img in p_main.get("Furniture List", "").split(";") if img.strip()]
        if "{}" in args.album_data:
            album_paths = [args.album_data.format(img) for img in raw_furniture_list]
        else:
            album_paths = [os.path.join(args.album_data, img) for img in raw_furniture_list]
            
        if not album_paths: album_paths = ["dummy_benign.jpg"]
            
        if len(album_paths) >= args.album_length:
            final_album = album_paths[:args.album_length]
        else:
            padding = list(np.random.choice(album_paths, args.album_length - len(album_paths)))
            final_album = album_paths + padding
        
        # 人格配置
        if args.persona_type == "homogeneous":
            agent_personas = [p_main for _ in range(args.k_personas)]
        else: 
            start_idx = (i * args.k_personas) % len(calibration_personas)
            agent_personas = [calibration_personas[(start_idx + j) % len(calibration_personas)] for j in range(args.k_personas)]
                
        agents_state[f"Calib_Agent_{i}"] = {"personas": agent_personas, "photo_album": final_album}
        
    print(f"\n[Init] Generated {args.num_calibration_agents} completely BENIGN calibration agents.")
    return agents_state

def run_offline_profiling(args, agents_state, brain, clip_extractor, calculator):
    print("\n[Start] Running Offline Benign Profiling...")
    
    tasks = []
    NUM_ITERATIONS = 3
    
    for agent_id, agent_data in tqdm(agents_state.items(), desc="1/4 Preparing Tasks"):
        album = agent_data["photo_album"]
        personas = agent_data["personas"]
        
        for _ in range(NUM_ITERATIONS):
            pairs = random_pairs(personas)
            for p1, p2 in pairs:
                env_description = [f"{p1['Name']} is chatting with {p2['Name']}."]
                p1_proxy = {"personas": [p1], "chat_history": [], "photo_album": album}
                prompts_p1 = PromptGenerator.get_prompts(p1_proxy, p1['Name'], env_description)
                
                tasks.append({
                    "agent_id": agent_id, "album": album,
                    "p1": p1, "p2": p2, "env": env_description,
                    "prompts_p1": prompts_p1,
                    "thought": None, "target_image": "No_Image",
                    "question": None, "response": None
                })

    print("\n[VLM] Generating Thoughts...")
    thoughts = brain.generate_batch([t["prompts_p1"]["active_thought"] for t in tasks], batch_size=args.batch_size, max_new_tokens=77)
    for i, t in enumerate(tasks): t["thought"] = thoughts[i]

    question_prompts = []
    for t in tqdm(tasks, desc="2/4 CLIP Extracting & Image Selection"):
        try:
            txt_in = clip_extractor.processor(text=[t["thought"]], return_tensors="pt", padding=True, truncation=True).to(clip_extractor.device)
            with torch.no_grad():
                txt_emb = clip_extractor.model.get_text_features(**txt_in)
                txt_emb /= txt_emb.norm(p=2, dim=-1, keepdim=True)
                img_emb = clip_extractor.get_image_embeddings(t["album"])
                if img_emb is not None:
                    sim = (txt_emb @ img_emb.T).squeeze(0)
                    t["target_image"] = t["album"][sim.argmax().item()]
                else:
                    t["target_image"] = random.choice(t["album"])
        except:
            t["target_image"] = random.choice(t["album"])

        p1_proxy = {"personas": [t["p1"]], "chat_history": [], "photo_album": t["album"]}
        prompts = PromptGenerator.get_prompts(p1_proxy, t["p1"]['Name'], t["env"])
        question_prompts.append(prompts["active_action"])

    print("\n[VLM] Generating Questions...")
    questions = brain.generate_batch(question_prompts, image_urls=[t["target_image"] for t in tasks], batch_size=args.batch_size)
    for i, t in enumerate(tasks): t["question"] = questions[i]

    resp_prompts = []
    for t in tqdm(tasks, desc="3/4 Formatting Response Prompts"):
        p2_proxy = {"personas": [t["p2"]], "chat_history": [], "photo_album": t["album"]}
        resp_prompts.append(PromptGenerator.get_passive_response_prompt(
            p2_proxy, t["p2"]['Name'], t["env"], t["question"]
        ))

    print("\n[VLM] Generating Responses...")
    responses = brain.generate_batch(resp_prompts, image_urls=[t["target_image"] for t in tasks], batch_size=args.batch_size)
    for i, t in enumerate(tasks): t["response"] = responses[i]

    # --- 聚合数据并计算分数 ---
    grouped_results = {}
    for t in tasks:
        aid = t["agent_id"]
        if aid not in grouped_results: 
            grouped_results[aid] = {"texts": [], "images": []}
            
        grouped_results[aid]["texts"].append(f"Q: {t['question']} | A: {t['response']}")
        grouped_results[aid]["images"].append(t["target_image"])
        
    benign_divs, benign_ents = [], []
    for agent_id, data in tqdm(grouped_results.items(), desc="4/4 Calculating Baselines"):
        div_score, ent_score = 0.0, 0.0
        if data["texts"]:
            try:
                emb = clip_extractor.get_text_embeddings(data["texts"])
                if emb is not None: div_score = float(calculator.calculate_semantic_diversity(emb))
            except: pass
        if data["images"]:
            valid_imgs = [img for img in data["images"] if img != "No_Image"]
            ent_score = float(calculator.calculate_retrieval_entropy(valid_imgs))
            
        benign_divs.append(div_score)
        benign_ents.append(ent_score)

    # =========================================================================
    # 核心：计算基线阈值并导出
    # =========================================================================
    print(f"\n[Stats] Calculating {args.tolerance_fpr}th Percentile Thresholds...")
    th_div = np.percentile(benign_divs, args.tolerance_fpr)
    th_ent = np.percentile(benign_ents, args.tolerance_fpr)
    
    print(f"-> TH_DIVERSITY: {th_div:.4f}")
    print(f"-> TH_ENTROPY: {th_ent:.4f}")

    # --- 保存阈值配置文件 ---
    os.makedirs("config", exist_ok=True)
    config_path = f"config/baseline_thresholds_{args.persona_type}.json"
    
    export_data = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "calibration_pool": args.calibration_data,
            "num_calibration_agents": args.num_calibration_agents,
            "tolerance_fpr": args.tolerance_fpr,
            "persona_type": args.persona_type
        },
        "thresholds": {
            "TH_DIVERSITY": float(th_div),
            "TH_ENTROPY": float(th_ent)
        },
        "raw_scores_distribution": {
            "diversity_scores": [round(s, 4) for s in benign_divs],
            "entropy_scores": [round(s, 4) for s in benign_ents]
        }
    }
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ 离线基线画像已完成！阈值已保存至: {config_path}")
    print("现在你可以运行第二个脚本进行线上盲测了。")

def main():
    args = parse_args()
    
    print(f"Loading Calibration Persona Pool from: {args.calibration_data}")
    try:
        with open(args.calibration_data, "r", encoding="utf-8") as f:
            calibration_personas = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ 错误: 找不到文件 {args.calibration_data}。请确保你创建了一个独立的人格池文件供校准使用。")
        return

    brain = AgentBrain(model_id=args.vlm)  
    clip_feature_extractor = ClipFeatureExtractor(model_id=args.clip)
    calculator = MetricsCalculator(clip_feature_extractor)
        
    agents_state = generate_benign_agents(args, calibration_personas)
    run_offline_profiling(args, agents_state, brain, clip_feature_extractor, calculator)

if __name__ == "__main__":
    main()