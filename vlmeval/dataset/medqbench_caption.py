"""
MedQ-Bench Caption Dataset
医学图像低层次描述能力评测（描述生成）
"""

import os
import pandas as pd
from .image_base import ImageBaseDataset
from ..smp import *
import re
import warnings
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# 判分prompt模板
PROMPT_TEMPLATES = {
    'completeness': (
        '#System: You are a helpful assistant.\n'
        '#User: Evaluate whether the description [MLLM DESC] completely includes the low-level visual information in the reference description [GOLDEN DESC]. '
        'Please rate score 2 for completely or almost completely including reference information, 0 for not including at all, 1 for including part of the information or similar description. '
        'Please only provide the result in the following format: Score:'
    ),
    'preciseness': (
        '#System: You are a helpful assistant.\n'
        '#User: The precision metric evaluates whether the low-level description is consistent with the reference and reasonably aligned with the final quality judgment. '
        'Minor wording differences or small omissions that do not change the overall meaning should still be considered consistent. '
        'Only penalize clear contradictions with the reference, such as describing blur for clear, noisy for clean, motion-free for motion artifacts, noise-free for low-dose noise, etc. '
        'Evaluate whether output [MLLM DESC] reasonably reflects reference [GOLDEN DESC]. '
        'Please rate score 2 for overall consistency and no major contradictions with the quality conclusion, '
        '1 for partial consistency or very few minor contradictions, and 0 for obvious contradictions or misalignment with the quality conclusion. '
        'Please only provide the result in the following format: Score:'
    ),
    'consistency': (
        '#System: You are a helpful assistant.\n'
        '#User: Evaluate the internal consistency between the reasoning path (description of image problems) and the final quality judgment in [MLLM DESC]. '
        'The reasoning should logically support the final quality conclusion. For example, if many serious problems are described, the final quality should be "reject"; '
        'if minor problems are described, it should be "usable"; if no or very few problems are described, it should be "good". '
        'Compare with the reference [GOLDEN DESC] to understand the expected reasoning-conclusion relationship. '
        'Please rate score 2 for highly consistent reasoning and conclusion, 1 for partially consistent with minor logical gaps, '
        'and 0 for major inconsistency between described problems and quality judgment. '
        'Please only provide the result in the following format: Score:'
    ),
    'quality_accuracy': (
        '#System: You are a helpful assistant.\n'
        '#User: Evaluate the accuracy of the final quality judgment in [MLLM DESC] compared to the reference [GOLDEN DESC]. '
        'The quality levels have a progressive relationship: reject < usable < good. Consider the distance between predicted and reference quality: '
        'Please rate score 2 for exactly matching the reference quality level, 1 for adjacent level difference (e.g., usable vs good, or reject vs usable), '
        'and 0 for distant level difference (reject vs good) or completely incorrect quality assessment. '
        'Please only provide the result in the following format: Score:'
    ),

}

class MedQBench_Caption_Scorer:
    def __init__(self, data, judge_model, n_rounds=1, nproc=4, sleep=0.5, target_metrics=None):
        self.data = data
        self.judge_model = judge_model
        self.n_rounds = n_rounds
        self.nproc = nproc
        self.sleep = sleep  # 控制API速率
        self.target_metrics = target_metrics if target_metrics is not None else list(PROMPT_TEMPLATES.keys())

    def build_prompt(self, metric, mllm_desc, golden_desc):
        prompt = PROMPT_TEMPLATES[metric]
        prompt = prompt.replace('MLLM DESC', mllm_desc).replace('GOLDEN DESC', golden_desc)
        return prompt

    def _safe_print_prompt(self, prompt, max_chars=200):
        try:
            # 尝试直接打印
            print("JUDGE PROMPT:", prompt[:max_chars] + ("..." if len(prompt) > max_chars else ""))
        except UnicodeEncodeError:
            # 如果出现编码错误，使用repr
            print("JUDGE PROMPT (repr):", repr(prompt[:max_chars]))
        except Exception as e:
            # 其他错误，打印基本信息
            print(f"JUDGE PROMPT (error printing): {type(prompt)}, length: {len(prompt)}")

    def ask_judge(self, prompt):
        for _ in range(3):
            try:
                self._safe_print_prompt(prompt)
                resp = self.judge_model.generate(prompt)
                print("JUDGE RESPONSE:", resp)
                score = self.parse_score_from_response(resp)
                print("PARSED SCORE:", score)
                if score is not None:
                    return score
            except Exception as e:
                print("JUDGE ERROR:", e)
                time.sleep(self.sleep)
        return None

    @staticmethod
    def parse_score_from_response(resp):
        # 只提取“Score: x”中的x
        import re
        import json as _json
        if isinstance(resp, dict):
            resp = str(resp)
        text = str(resp).strip()
        match = re.search(r'Score\s*[:：]\s*([0-2](?:\.\d+)?)', text)
        if match:
            try:
                return float(match.group(1))
            except:
                return None
        # 兼容仅返回数字的情况，如 "2" 或 {"score":2}
        try:
            # 尝试JSON解析
            j = _json.loads(text)
            if isinstance(j, (int, float)) and 0 <= float(j) <= 2:
                return float(j)
            if isinstance(j, dict):
                for k in ['score', 'Score']:
                    if k in j and 0 <= float(j[k]) <= 2:
                        return float(j[k])
        except Exception:
            pass
        # 直接匹配裸数字
        match2 = re.fullmatch(r'\s*([0-2](?:\.\d+)?)\s*', text)
        if match2:
            try:
                return float(match2.group(1))
            except:
                return None
        return None

    def score_one(self, line):
        mllm_desc = str(line['prediction'])
        golden_desc = str(line['description'])
        result = {}
        # 只对目标指标进行打分
        for metric in self.target_metrics:
            scores = []
            for _ in range(self.n_rounds):
                prompt = self.build_prompt(metric, mllm_desc, golden_desc)
                score = self.ask_judge(prompt)
                scores.append(score)
                time.sleep(self.sleep)
            # 过滤None
            scores = [x for x in scores if x is not None]
            result[metric] = sum(scores)/len(scores) if scores else None
            result[f'{metric}_scores'] = scores
        return result

    def compute_scores(self, use_threading=False):
        # 多线程加速（可选）
        results = []
        # 动态生成默认异常结果（只针对目标指标）
        default_result = {metric: None for metric in self.target_metrics}
        
        if use_threading:
            with ThreadPoolExecutor(max_workers=self.nproc) as executor:
                future2idx = {executor.submit(self.score_one, line): i for i, line in self.data.iterrows()}
                for future in as_completed(future2idx):
                    idx = future2idx[future]
                    try:
                        res = future.result()
                    except Exception as e:
                        res = default_result.copy()
                    results.append((idx, res))
        else:
            for i, line in self.data.iterrows():
                try:
                    res = self.score_one(line)
                except Exception as e:
                    res = default_result.copy()
                results.append((i, res))
        # 按原顺序排序
        results = sorted(results, key=lambda x: x[0])
        return [x[1] for x in results]

class MedqbenchCaptionDataset(ImageBaseDataset):
    """MedQ-Bench Caption Dataset"""
    TYPE = 'Caption'
    DATASET_URL = {
        'MedqbenchCaption': './data/v0.2/medqbench_description.tsv',
        'MedqbenchCaption_dev': './data/v0.2/medqbench_description_dev.tsv',
        'MedqbenchCaption_test': './data/v0.2/medqbench_description_test.tsv',
    }
    DATASET_MD5 = {
        'MedqbenchCaption': None,
        'MedqbenchCaption_dev': None,
        'MedqbenchCaption_test': None,
    }

    @classmethod
    def supported_datasets(cls):
        return ['MedqbenchCaption', 'MedqbenchCaption_dev', 'MedqbenchCaption_test']

    def __init__(self, dataset='MedqbenchCaption', data_path=None, **kwargs):
        if data_path is not None:
            self.custom_data_path = data_path
        else:
            self.custom_data_path = None
        super().__init__(dataset, **kwargs)

    def load_data(self, dataset):
        if hasattr(self, 'custom_data_path') and self.custom_data_path is not None:
            data_path = self.custom_data_path
        else:
            data_path = self.DATASET_URL.get(dataset, None)
            if data_path is None:
                raise ValueError(f"Dataset configuration not found for {dataset}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        print(f"Loading MedQ-Bench Caption data file: {data_path}")
        data = load(data_path)
        if 'index' in data.columns:
            data['index'] = data['index'].astype(str)

        if 'question' not in data:
            data['question'] = [
            """As a medical image quality assessment expert, provide a concise description focusing on low-level appearance of the image in details. Conclude with "Overall, the quality of this image is [good/usable/reject]".""" for _ in range(len(data))
            ]
        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs.append(dict(type='image', value=tgt_path))
        msgs.append(dict(type='text', value=question))
        return msgs

    @classmethod
    def _test_judge_availability(cls, judge_kwargs):
        """测试judge模型API是否可用"""
        try:
            from vlmeval.dataset.utils import build_judge
            import requests
            
            # 构建judge模型
            judge_model = build_judge(**judge_kwargs)
            
            # 测试API连接
            if hasattr(judge_model, 'keywords') and 'api_base' in judge_model.keywords:
                api_base = judge_model.keywords.get('api_base')
                api_key = judge_model.keywords.get('key')
                
                if api_base and api_key:
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    # 获取模型名称
                    model_id = judge_model.keywords.get('model', 'unknown')
                    
                    payload = {
                        "model": model_id,
                        "messages": [
                            {"role": "user", "content": "Hello, please respond with 'API is working' if you can see this message."}
                        ],
                        "temperature": 0,
                        "max_tokens": 100
                    }
                    
                    response = requests.post(api_base, headers=headers, json=payload, timeout=30)
                    
                    if response.status_code == 200:
                        return True, "Judge模型API可用"
                    else:
                        error_msg = response.json().get('error', {}).get('message', 'Unknown error')
                        return False, f"Judge模型API请求失败: {error_msg}"
                else:
                    return False, "Judge模型缺少API配置"
            else:
                # 对于本地模型，直接返回可用
                return True, "本地Judge模型可用"
                
        except Exception as e:
            return False, f"Judge模型连接错误: {e}"

    # 修改evaluate方法，支持自动judge判分
    
    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """
        自动化GPT判分，四项指标：completeness, preciseness, consistency, quality_accuracy。
        支持1次打分平均。若已有分数字段则直接聚合，否则自动调用judge。
        judge_kwargs可传model, api_base, api_key等。
        """
        data = load(eval_file)
        
        # 检查是否有prediction列，如果没有则无法进行评测
        if 'prediction' not in data.columns:
            print("警告：Excel文件中没有找到prediction列，无法进行评测")
            return {"error": "No prediction column found"}
        
        lt = len(data)
        # 使用当前定义的指标
        metrics = list(PROMPT_TEMPLATES.keys())
        
        # 直接进行judge评测
        from vlmeval.dataset.utils import build_judge
        nproc = judge_kwargs.pop('nproc', 4)
        n_rounds = judge_kwargs.pop('n_rounds', 1)  # TODO：多次评估的次数，默认为1
        sleep = judge_kwargs.pop('sleep', 0.5)
        use_threading = judge_kwargs.pop('use_threading', False)  # 是否使用多线程
        
        try:
            judge_model = build_judge(**judge_kwargs)
            scorer = MedQBench_Caption_Scorer(data, judge_model, n_rounds=n_rounds, nproc=nproc, sleep=sleep, target_metrics=metrics)
            score_results = scorer.compute_scores(use_threading=use_threading)
            
            # 写回data
            # 预创建分数列表列，避免dtype冲突
            for m in metrics:
                col = f'{m}_scores'
                if col not in data.columns:
                    data[col] = [None] * lt
            
            # 更新评测结果
            for i, res in enumerate(score_results):
                for k, v in res.items():
                    if isinstance(v, list):
                        # 存为JSON字符串，后续用safe_avg_list_col解析
                        data.at[i, k] = json.dumps(v, ensure_ascii=False)
                    else:
                        data.at[i, k] = v
            
            # 保存更新后的数据
            dump(data, eval_file)
            
        except Exception as e:
            print(f"Judge评测过程中出现错误: {e}")
            return {"error": f"Judge evaluation failed: {e}"}
        
        # 聚合分数
        def avg(lst):
            lst = [x for x in lst if isinstance(x, (int, float))]
            return sum(lst)/len(lst) if lst else None

        def safe_avg_list_col(data, col):
            import ast
            if col not in data:
                return [None] * len(data)
            values = []
            for x in data[col]:
                if isinstance(x, list):
                    values.append(avg(x))
                elif isinstance(x, str):
                    x_str = x.strip()
                    # 支持JSON或python字面量形式
                    if (x_str.startswith('[') and x_str.endswith(']')):
                        try:
                            parsed = json.loads(x_str)
                        except Exception:
                            try:
                                parsed = ast.literal_eval(x_str)
                            except Exception:
                                parsed = None
                        values.append(avg(parsed) if isinstance(parsed, list) else None)
                    else:
                        values.append(None)
                else:
                    values.append(None)
            return values

        # 逐指标收集/补全分数
        metric_to_scores = {}
        for m in metrics:
            col = m
            if col in data:
                scores_col = list(data[col])
                # 若全为None或NaN，则尝试从列表列聚合
                valid_scores = [x for x in scores_col if isinstance(x, (int, float)) and pd.notna(x)]
                if len(valid_scores) == 0:
                    scores_col = safe_avg_list_col(data, f'{m}_scores')
                    data[col] = scores_col
            else:
                scores_col = safe_avg_list_col(data, f'{m}_scores')
                data[col] = scores_col
            metric_to_scores[m] = scores_col

        # 统计平均分
        metric_avgs = {}
        for m, scs in metric_to_scores.items():
            valid = [x for x in scs if isinstance(x, (int, float)) and pd.notna(x)]
            metric_avgs[m] = (sum(valid) / len(valid)) if len(valid) else 0.0

        # 汇总结果
        result = metric_avgs
        
        # 保存详细分数
        score_file = eval_file.replace('.xlsx', '_score.json')
        dump(result, score_file)
        
        # 打印整体结果
        summary_str = ', '.join([f"{m.capitalize()}: {metric_avgs[m]:.4f}" for m in metrics])
        print(f"\nMedQ-Bench Caption评测完成！\n{summary_str}")
        print(f"结果保存到{score_file}")
         
        return result

if __name__ == "__main__":
    try:
        dataset = MedqbenchCaptionDataset()
        print(f"MedQ-Bench Caption数据集加载成功! 样本数: {len(dataset)}")
        if len(dataset) > 0:
            sample = dataset[0]
            prompt = dataset.build_prompt(sample)
            print("Prompt构建成功!")
    except Exception as e:
        print(f"测试出错: {e}") 