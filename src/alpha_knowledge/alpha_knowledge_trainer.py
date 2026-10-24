import torch
import numpy as np
import json
import os
import pandas as pd
from alpha_knowledge.alpha_pool import AlphaKnowledgePool
from alphagen.data.expression_knowledge_graph import ExpressionKnowledgeGraph, ExpressionNode
from alphagen.data import expression as exp_module
from alphagen.data.tree import ExpressionParser, InvalidExpressionException
from alphagen_qlib.stock_data import StockData
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr, batch_ret, batch_sharpe_ratio, batch_max_drawdown
from gan.utils.builder import exprs2tensor

from typing import List, Tuple, Optional, Set, Any
from openai import OpenAI
from utils.llm import OpenAIModel
from utils.prompt import *
from utils.tracing import RunTracer
from tqdm import tqdm
from alphagen.data.dimension import DimensionCalculator, _is_close

import re
import time



def remove_linearly_dependent_rows(x, y, to_pred, tol=1e-10):
    """
    Remove linearly dependent rows using efficient rank detection for speed.
    
    Args:
        x: Training factor matrix (n_samples, n_factors)
        y: Target matrix (n_samples, n_targets)
        to_pred: Prediction factor matrix (n_stocks, n_factors)
        tol: Tolerance for linear independence
    
    Returns:
        x_filtered: Filtered training matrix
        y_filtered: Filtered target matrix  
        to_pred: Original prediction matrix (unchanged)
        selected_rows: List of selected row indices
    """
    if x.shape[0] <= x.shape[1]:
        # If we have fewer samples than features, keep all samples
        return x, y, to_pred, list(range(x.shape[0]))
    
    # For efficiency, only check for linear dependence if we have many more samples than features
    # This is the common case where linear dependence in rows matters
    sample_ratio = x.shape[0] / x.shape[1]
    
    if sample_ratio < 5:  # Not enough samples to worry about row dependence
        return x, y, to_pred, list(range(x.shape[0]))
    
    try:
        # Use SVD on transposed matrix to find rank efficiently
        U, S, Vh = torch.linalg.svd(x.T, full_matrices=False)
        
        # Effective rank based on singular values
        rank = torch.sum(S > tol * S[0]).item()
        
        if rank >= min(x.shape[0], x.shape[1]):
            # Matrix is full rank, no need to remove rows
            return x, y, to_pred, list(range(x.shape[0]))
        
        # If rank deficient, use QR decomposition to find independent rows
        Q, R = torch.linalg.qr(x.T, mode='reduced')
        diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
        pivot_mask = torch.abs(diag_R) > tol
        
        if not torch.any(pivot_mask):
            selected_rows = [0]
        else:
            selected_rows = torch.where(pivot_mask)[0].tolist()
            if len(selected_rows) == 0:
                selected_rows = [0]
                
    except:
        # If SVD/QR fails, keep all rows
        return x, y, to_pred, list(range(x.shape[0]))
    
    # Filter matrices
    x_filtered = x[selected_rows]
    y_filtered = y[selected_rows] if y is not None else None
    
    return x_filtered, y_filtered, to_pred, selected_rows


def remove_linearly_dependent_cols(x, to_pred, tol=1e-10):
    """
    Remove linearly dependent columns (factors) using QR decomposition with pivoting for speed.
    
    Args:
        x: Training factor matrix (n_samples, n_factors)
        to_pred: Prediction factor matrix (n_stocks, n_factors)
        tol: Tolerance for linear independence
    
    Returns:
        x_filtered: Filtered training matrix
        to_pred_filtered: Filtered prediction matrix
        selected_factors: List of selected factor indices
    """
    if x.shape[1] <= 1:
        return x, to_pred, list(range(x.shape[1]))
    
    # Use SVD for more robust rank detection (faster than iterative QR)
    try:
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        
        # Find columns corresponding to significant singular values
        rank = torch.sum(S > tol * S[0]).item()  # Relative tolerance
        
        if rank == 0:
            selected_factors = [0]
        else:
            # Use the first 'rank' columns as they correspond to largest singular values
            selected_factors = list(range(min(rank, x.shape[1])))
            
    except:
        # Fallback to QR decomposition if SVD fails
        Q, R = torch.linalg.qr(x, mode='reduced')
        diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
        pivot_mask = torch.abs(diag_R) > tol
        
        if not torch.any(pivot_mask):
            selected_factors = [0]
        else:
            selected_factors = torch.where(pivot_mask)[0].tolist()
            if len(selected_factors) == 0:
                selected_factors = [0]
    
    # Filter matrices
    x_filtered = x[:, selected_factors]
    to_pred_filtered = to_pred[:, selected_factors]
    
    return x_filtered, to_pred_filtered, selected_factors


def calculate_vif(x):
    """
    Calculate Variance Inflation Factor for each feature.
    VIF > 10 indicates multicollinearity issues.
    """
    n_features = x.shape[1]
    vif_scores = torch.zeros(n_features)
    
    for i in range(n_features):
        # Regression of feature i on all other features
        y_i = x[:, i]
        x_others = torch.cat([x[:, :i], x[:, i+1:]], dim=1)
        
        if x_others.shape[1] == 0:
            vif_scores[i] = 1.0
            continue
            
        try:
            # Add constant term
            ones = torch.ones(x_others.shape[0], 1, device=x.device)
            x_others_const = torch.cat([x_others, ones], dim=1)
            
            # Solve regression
            coef = torch.linalg.lstsq(x_others_const, y_i.unsqueeze(1), rcond=1e-15).solution
            y_pred = x_others_const @ coef
            
            # Calculate R-squared
            ss_res = torch.sum((y_i.unsqueeze(1) - y_pred) ** 2)
            ss_tot = torch.sum((y_i - torch.mean(y_i)) ** 2)
            r_squared = 1 - ss_res / ss_tot
            
            # VIF = 1 / (1 - R^2)
            vif_scores[i] = 1.0 / (1.0 - torch.clamp(r_squared, max=0.999))
            
        except:
            vif_scores[i] = float('inf')
    
    return vif_scores


def remove_multicollinearity_vif(x, to_pred, vif_threshold=10.0):
    """
    Remove factors with high VIF to address multicollinearity.
    
    Args:
        x: Training factor matrix (n_samples, n_factors)
        to_pred: Prediction factor matrix (n_stocks, n_factors)
        vif_threshold: VIF threshold above which factors are removed
    
    Returns:
        x_filtered: Filtered training matrix
        to_pred_filtered: Filtered prediction matrix
        selected_factors: List of selected factor indices
    """
    if x.shape[1] <= 1:
        return x, to_pred, list(range(x.shape[1]))
    
    selected_factors = list(range(x.shape[1]))
    
    while len(selected_factors) > 1:
        # Calculate VIF for current factors
        x_current = x[:, selected_factors]
        vif_scores = calculate_vif(x_current)
        
        # Find factor with highest VIF
        max_vif_idx = torch.argmax(vif_scores)
        max_vif = vif_scores[max_vif_idx]
        
        # If max VIF is below threshold, stop
        if max_vif <= vif_threshold:
            break
            
        # Remove the factor with highest VIF
        selected_factors.pop(max_vif_idx.item())
    
    # Filter matrices
    x_filtered = x[:, selected_factors]
    to_pred_filtered = to_pred[:, selected_factors]
    
    return x_filtered, to_pred_filtered, selected_factors

def chunk_batch_spearmanr(x, y, chunk_size=60):
    n_days = len(x)
    spearmanr_list= []
    for i in range(0, n_days, chunk_size):
        spearmanr_list.append(batch_spearmanr(x[i:i+chunk_size], y[i:i+chunk_size]))
    spearmanr_list = torch.cat(spearmanr_list, dim=0)
    return spearmanr_list


def get_tensor_metrics(x, y, risk_free_rate=0.0, args: Any = None):
    # Ensure tensors are 2D (days, stocks)
    if x.dim() > 2: x = x.squeeze(-1)
    if y.dim() > 2: y = y.squeeze(-1)

    ic_s = batch_pearsonr(x, y)
    ric_s = chunk_batch_spearmanr(x, y, chunk_size=args.chunk_size)
    ret_s = batch_ret(x, y)

    ic_s = torch.nan_to_num(ic_s, nan=0.)
    ric_s = torch.nan_to_num(ric_s, nan=0.)
    ret_s = torch.nan_to_num(ret_s, nan=0.) / args.label_days
    ic_s_mean = ic_s.mean().item()
    ic_s_std = ic_s.std().item() if ic_s.std().item() > 1e-6 else 1.0
    ric_s_mean = ric_s.mean().item()
    ric_s_std = ric_s.std().item() if ric_s.std().item() > 1e-6 else 1.0
    ret_s_mean = (ret_s).mean().item()
    ret_s_std = (ret_s).std().item() if (ret_s).std().item() > 1e-6 else 1.0

    # Calculate Sharpe Ratio and Maximum Drawdown for ret series
    ret_sharpe = batch_sharpe_ratio(ret_s, risk_free_rate).item()
    ret_mdd = batch_max_drawdown(ret_s).item()

    result = dict(
        ic=ic_s_mean,
        ic_std=ic_s_std,
        icir=ic_s_mean / ic_s_std,
        ric=ric_s_mean,
        ric_std=ric_s_std,
        ricir=ric_s_mean / ric_s_std,
        ret=ret_s_mean * len(ret_s) / 3,
        ret_std=ret_s_std,
        retir=ret_s_mean / ret_s_std,
        ret_sharpe=ret_sharpe,
        ret_mdd=ret_mdd,
    )
    return result, ret_s

class AlphaKnowledgeLogger:
    def __init__(self, test_data: StockData, target: exp_module.Expression, log_dir: str, args: Any):
        self.test_data = test_data
        self.target = target
        self.log_dir = log_dir
        self.args = args
    
    def load_test_res(self, pool: AlphaKnowledgePool, file: Any) -> dict:
        state = pool.state
        exprs = state.get('exprs', [])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.cuda)
        if self.args.instruments == 'sp500':
            QLIB_PATH = os.getenv('QLIB_PATH_SP500', 'PATH/TO/data/qlib_data/us_data_qlib')
        else:
            QLIB_PATH = os.getenv('QLIB_PATH_CN', 'PATH/TO/.qlib/qlib_data/cn_data')
        device = torch.device(self.args.device)
        close = exp_module.Feature(exp_module.FeatureType.CLOSE)
        target = exp_module.Ref(close, - 20) / close - 1
        train_end_time = f'2020-12-31'
        valid_start_time = f'2021-01-01'
        valid_end_time = f'2022-06-30'
        test_start_time = f'2022-07-01'
        test_end_time = f'2025-06-30'
        data_all = StockData(instrument=self.args.instruments, start_time='2010-01-01', end_time=test_end_time, qlib_path=QLIB_PATH, device=device)
        data_valid = StockData(instrument=self.args.instruments, start_time=valid_start_time, end_time=valid_end_time, qlib_path=QLIB_PATH, device=device)
        data_test = StockData(instrument=self.args.instruments, start_time=test_start_time, end_time=test_end_time, qlib_path=QLIB_PATH, device=device)
        
        fct_tensor = exprs2tensor(exprs, data_all, normalize=True)
        tgt_tensor = exprs2tensor([target], data_all, normalize=False)

        # 3. Pre-calculate daily metrics for all factors
        ic_list, ric_list, ret_list = [], [], []
        print("Pre-calculating daily metrics for each factor...")
        for i in tqdm(range(fct_tensor.shape[-1])):
            factor_slice = fct_tensor[..., i]
            target_slice = tgt_tensor[..., 0]
            ic_s = batch_pearsonr(factor_slice, target_slice)
            ric_s = chunk_batch_spearmanr(factor_slice, target_slice, chunk_size=self.args.chunk_size)
            #ret_s = batch_ret(factor_slice, target_slice)
            ic_list.append(torch.nan_to_num(ic_s, nan=0.))
            ric_list.append(torch.nan_to_num(ric_s, nan=0.))
            #ret_list.append(torch.nan_to_num(ret_s, nan=0.))

        ic_s = torch.stack(ic_list, dim=-1)
        ric_s = torch.stack(ric_list, dim=-1)
        #ret_s = torch.stack(ret_list, dim=-1)
        torch.cuda.empty_cache()

        # 4. Main adaptive combination loop
        pred_list = []
        shift = self.args.label_days + 1  # To avoid lookahead bias
        
        valid_test_days = data_valid.n_days + data_test.n_days
        start_day = len(fct_tensor) - valid_test_days
        
        print("Starting adaptive combination process...")
        pbar = tqdm(range(start_day, len(fct_tensor)))
        for cur in pbar:
            # Define rolling window for evaluation
            # begin = 0 if not np.isfinite(self.args.window) else max(0, cur - self.args.window - shift)
            begin = 0
            # Slice metrics for the current window
            cur_ic = ic_s[begin:cur-shift]
            cur_ric = ric_s[begin:cur-shift]
            
            # Calculate performance metrics over the window
            ic_mean = cur_ic.mean(dim=0)
            ic_std = cur_ic.std(dim=0)
            ric_mean = cur_ric.mean(dim=0)
            ric_std = cur_ric.std(dim=0)

            icir = ic_mean / ic_std
            ricir = ric_mean / ric_std
            
            # Filter and select best factors
            metrics_df = pd.DataFrame({
                'ric': ric_mean.cpu().numpy(),
                'ricir': ricir.cpu().numpy()
            })
            good_factors = metrics_df[(metrics_df['ric'].abs() > self.args.threshold_ric) & (metrics_df['ricir'].abs() > self.args.threshold_ricir)]
            if len(good_factors) < 1:
                good_factors = metrics_df.reindex(metrics_df.ricir.abs().sort_values(ascending=False).index).iloc[:1]
            
            good_idx = good_factors.iloc[:self.args.n_factors].index.to_list()
            
            # Prepare data for linear regression
            x = fct_tensor[begin:cur-shift, :, good_idx]
            y = tgt_tensor[begin:cur-shift, :, :]
            to_pred = fct_tensor[cur, :, good_idx]
            y = y.reshape(-1, y.shape[-1])
            x = x.reshape(-1, x.shape[-1])
            
            # Filter out NaNs
            valid_mask = torch.isfinite(y)[:, 0]
            y = y[valid_mask]
            x = x[valid_mask]
            
            to_pred = torch.nan_to_num(to_pred, nan=0.)
            
            # Remove linearly dependent columns (factors) for speed
            x, to_pred, selected_factors = remove_linearly_dependent_cols(x, to_pred, tol=self.args.linear_dep_tol)
            
            # Remove linearly dependent rows (samples) for speed  
            x, y, to_pred, selected_rows = remove_linearly_dependent_rows(x, y, to_pred, tol=self.args.linear_dep_tol)
            
            # Add constant for intercept
            ones = torch.ones_like(x[..., 0:1])
            x = torch.cat([x, ones], dim=-1)
            ones_pred = torch.ones_like(to_pred[..., 0:1])
            to_pred = torch.cat([to_pred, ones_pred], dim=-1)
            
            # Train regression and predict with improved stability
            try:
                # Check condition number before solving
                coef = torch.linalg.lstsq(x, y).solution
                
                pred = to_pred @ coef
                
            except Exception as e:
                print(f"Warning: Regression failed with error {e}, using zero prediction")
                # Handle singular matrix case
                pred = torch.zeros_like(to_pred[:, 0:1])

            pred_list.append(pred[:, 0])
            
            # Update progress bar description with running IC
            if len(pred_list) > 1:
                running_preds = torch.stack(pred_list, dim=0)
                running_targets = tgt_tensor[start_day:cur+1, :, 0]
                running_ic = batch_pearsonr(running_preds, running_targets).mean().item()
                pbar.set_description(f"Running IC: {running_ic:.4f}, Factors selected: {len(good_idx)}")


        # 5. Evaluate and display results
        print("\n" + "="*50)
        print("Adaptive combination finished. Calculating final metrics...")
        
        all_pred = torch.stack(pred_list, dim=0)
        
        # Slice predictions and targets for validation and test sets
        pred_valid = all_pred[:data_valid.n_days]
        pred_test = all_pred[data_valid.n_days:]
        
        tgt_valid = tgt_tensor[start_day : start_day + data_valid.n_days, :, 0]
        tgt_test = tgt_tensor[start_day + data_valid.n_days :, :, 0]
        
        # Calculate metrics
        valid_results, _ = get_tensor_metrics(pred_valid.to(device), tgt_valid.to(device), 0, self.args)
        test_results, ret_s = get_tensor_metrics(pred_test.to(device), tgt_test.to(device), 0, self.args)
        ret_s = ret_s.cpu().numpy()
        # Format and print results
        results_df = pd.DataFrame([valid_results, test_results], index=['Validation', 'Test'])
        print("\n--- Final Performance Metrics ---")
        
        # Print with full precision and no truncation
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(results_df.round(4))
        
        # Also print in a more parseable format
        file.write("\n--- Parseable Format ---")
        file.write(f"{'Dataset':<12} {'IC':>8} {'IC_STD':>8} {'ICIR':>8} {'RIC':>8} {'RIC_STD':>8} {'RICIR':>8} {'RET':>8} {'RET_STD':>8} {'RETIR':>8} {'RET_SR':>8} {'RET_MDD':>8}\n")
        for index, row in results_df.iterrows():
            file.write(f"{index:<12} {row['ic']:>8.4f} {row['ic_std']:>8.4f} {row['icir']:>8.4f} {row['ric']:>8.4f} {row['ric_std']:>8.4f} {row['ricir']:>8.4f} {row['ret']:>8.4f} {row['ret_std']:>8.4f} {row['retir']:>8.4f} {row['ret_sharpe']:>8.4f} {row['ret_mdd']:>8.4f}")
        
        file.write("="*50)
        return {index: {k: float(v) for k, v in row.items()} for index, row in results_df.iterrows()}
    
    def log(self, pool: AlphaKnowledgePool, graph: ExpressionKnowledgeGraph, iteration: int) -> dict:
        print(f"=== Logging at iteration {iteration} ===")
        pool_path = os.path.join(self.log_dir, f'pool_{iteration}.json')
        with open(pool_path, 'w') as f:
            json.dump(pool.to_dict(), f, indent=4)
        state = pool.state
        exprs = state.get('exprs', [])
        target_test = pool._normalize_by_day(self.target.evaluate(self.test_data))
        n = len(exprs)
        print('---------------------------------------------')
        alphas = []
        for i in range(n):
            expr = state['exprs'][i]
            expr_str = str(expr)
            ic_ret = state['ics_ret'][i]

            # Calculate test IC metrics
            value_test = pool._normalize_by_day(expr.evaluate(self.test_data))
            pearson_corrs_test = batch_pearsonr(value_test, target_test)
            ic_test = pearson_corrs_test.mean().item()
            icir_test = ic_test / (pearson_corrs_test.std().item() + 1e-6)
            ic_test_abs = float(np.abs(ic_test))
            icir_test_abs = float(np.abs(icir_test))

            # Update node test metrics if available
            if i < len(pool.expr2node):
                node = pool.expr2node[i]
                if node is not None:
                    node.test_ic = ic_test_abs
                    node.test_icir = icir_test_abs

            print(f'> Alpha #{i}: ic={ic_ret:.4f}, test_ic={ic_test:.4f}, test_icir={icir_test:.4f}, expr={expr_str}')
            alphas.append({"expression": expr_str, "ic": float(ic_ret), "test_ic": float(ic_test), "test_icir": float(icir_test)})
        if pool.size > 0:
            print(f'>> Best single ic: {np.max(pool.single_ics[:pool.size]):.4f}')
        print('---------------------------------------------')
        with open(os.path.join(self.log_dir, f'res.txt'), 'a+') as f:
            combination = self.load_test_res(pool, f)
        return {"iteration": iteration, "alphas": alphas, "combination": combination}



    

class AlphaKnowledgeTrainer:
    def __init__(self, pool: AlphaKnowledgePool, initial_expressions: List[Tuple[str, str, str]], test_data:StockData, target:exp_module.Expression, args: Any):
        self.pool = pool
        self.test_data = test_data
        self.target = target
        self.parser = ExpressionParser()
        self.graph = ExpressionKnowledgeGraph(depth_limit = args.depth_limit, expression_size_limit=args.max_length)
        self.pool.attach_knowledge_graph(self.graph)
        for (expr, topic, explanation) in initial_expressions:
            ic, mutic, icir = self.pool.get_ic_icir_mutics(self.parser.parse(expr.strip().replace('%d', '10')))

            expression_node = self.graph.build_expression_node(expr.strip(), topic, explanation, args.max_length, ic, icir)
            if expression_node is None:
                print(f"Initial expression {expr} is invalid, skipping.")
                continue
            res = self.pool.try_new_expr(self.parser.parse(expr.replace('%d', '10')), topic, explanation, expression_node)
            if res:
                inserted = self.graph.insert(expression_node=expression_node, parent_node=None)
                if inserted:
                    node = self.graph.search(expr.strip())
                    # if node is not None:
                    #     self.pool.register_expression_node(node)
        model_name = os.getenv("OPENAI_MODEL_NAME", "your_model_name")
        log_dir = os.path.join(
        'data/knowledge_logs',
        f'pool_{args.pool_capacity}',
        f'kg_dag_and_bayesian_icir_and_mutl_new_no_decay_{model_name.replace("/", "_")}_{args.generate_num}_{args.instruments}_{args.temp}_{args.depth_limit}_{args.max_length}_{args.ic_mut_threshold}_{args.pool_capacity}_{args.search_time}_{args.ic_threshold}_{args.use_res_correlation}_{args.use_semantic_similarity}_{args.use_edit_distance}_{args.separate_leaf_non_leaf}_{args.icir_decay_threshold}_{args.times_decay}_{args.depth_decay}')

        os.makedirs(log_dir, exist_ok=True)
        self.logger = AlphaKnowledgeLogger(test_data, target, log_dir, args)
        # API key, base url and model name come from .env (see .env.example)
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", ""),
            timeout=float(os.getenv("LLM_TIMEOUT", "600")),
        )
        max_tokens = os.getenv("LLM_MAX_TOKENS")
        self.model = OpenAIModel(model_name, "", 400000,
                                 max_tokens=int(max_tokens) if max_tokens else None,
                                 stream=os.getenv("LLM_STREAM", "false").lower() in ("1", "true", "yes"))
        run_id = time.strftime("%Y%m%d-%H%M%S")
        self.tracer = RunTracer(
            session_id=f"alphaprobe-{args.instruments}-{run_id}",
            tags=[f"instruments:{args.instruments}", f"model:{model_name}", f"seed:{args.seed}"],
            metadata={"log_dir": log_dir, "model": model_name, "temp": args.temp,
                      "generate_num": args.generate_num, "pool_capacity": args.pool_capacity,
                      "search_time": args.search_time, "ic_threshold": args.ic_threshold,
                      "ic_new_threshold": args.ic_new_threshold, "top_k": args.top_k},
        )
        print(f"Langfuse tracing: {'on' if self.tracer.enabled else 'off'} (session {self.tracer.session_id})")


## qwen3-235b-a22b-thinking-2507 deepseek-r1

    def _train_one_iteration(self, step: int, args: Any):

        def build_traces(paths: List[ExpressionNode]) -> str:
            if len(paths) <= 1:
                return ""
            res = ""
            for i in range(len(paths)):
                res += f"Expression: {paths[i].payload.source} Explanation: {paths[i].description}"
                if i < len(paths) - 1:
                    res += " -> "
            # print(f"Built traces: {res} for expression {paths[0].payload.source}")
            return res

        with self.tracer.observe("retrieve-parents", as_type="retriever",
                                 input={"pool_size": self.pool.size, "top_k": args.top_k}) as retrieval:
            nodes_to_optimize, exprs_to_optimize = self.pool.search()
            retrieval.update(output=[
                None if node is None else {k: v for k, v in node.as_dict().items() if k not in ("tokens", "children")}
                for node in nodes_to_optimize
            ])
        nodes_str = [node.payload.source for node in nodes_to_optimize]
        print(f"Step {step}: optimizing {len(nodes_to_optimize)} nodes: {nodes_str}")
        stats = {"nodes": len(nodes_to_optimize), "candidates": 0, "parse_failures": 0, "inserted": []}
        iter = 0
        for node, parent_expr in tqdm(zip(nodes_to_optimize, exprs_to_optimize), desc=f"optimizing step {step}"):
            if node is None:
                print(f"Node  not found in graph at iter {iter}.")
                iter += 1
                continue
            iter += 1
            paths = self.graph.path_to_root(node)
            traces = build_traces(paths)
            with self.tracer.observe("evolve-node", as_type="chain",
                                     input={"expression": node.payload.source, "topic": node.topic,
                                            "explanation": node.description, "traces": traces},
                                     metadata={"depth": node.depth, "times": node.times}) as evolve:
                outcome = self._evolve_node(node, parent_expr, traces, args)
                evolve.update(output=outcome)
            stats["candidates"] += len(outcome["candidates"])
            stats["parse_failures"] += int(outcome["parse_failed"])
            stats["inserted"] += [c["expression"] for c in outcome["candidates"] if c["outcome"].startswith("inserted")]
        return stats

    def _evolve_node(self, node: ExpressionNode, parent_expr: exp_module.Expression, traces: str, args: Any) -> dict:
        topic = node.topic
        generation_prompt = PROMPT_FEATURES_AND_OPERATORS + PROMPT_GENERARTION.format(topic=topic, expressions=node.payload.source, traces=traces, num =args.generate_num, explanations = node.description)
        with self.tracer.generation("generate-offspring", model=self.model.model_name,
                                    system_prompt=PROMPT_HEAD, user_prompt=generation_prompt,
                                    model_parameters={"temperature": args.temp, "top_p": 1,
                                                      "max_tokens": self.model.max_tokens, "stream": self.model.stream}) as generation:
            res, finish_reason = self.model.chat_generate(self.client, system_prompt = PROMPT_HEAD, user_prompt = generation_prompt, temperature = args.temp)
            call = self.model.last_call
            generation.update(output=res, usage_details=call.get("usage_details") or None,
                              metadata={"finish_reason": finish_reason, "latency_s": call.get("latency_s"),
                                        "reasoning": call.get("reasoning") or None})
        result = {"parse_failed": False, "candidates": []}
        try:
            # print(res)
            json_res = json.loads(res)
            assert len(json_res["expressions_fixed"]) == args.generate_num
            assert len(json_res["explanations"]) == args.generate_num
        except (json.JSONDecodeError, AssertionError, KeyError, TypeError) as exc:
            print(f"Failed to parse JSON response or invalid number of expressions/explanations: {exc}")
            pattern = re.compile(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```|\{([\s\S]*?)\}',re.MULTILINE)
            candidate = None
            for match in pattern.finditer(res):
                candidate = match.group(1) or f'{{{match.group(2)}}}'
            try:
                json_res = json.loads(candidate) if candidate is not None else None
                assert json_res is not None
                assert len(json_res["expressions"]) >= args.generate_num
                assert len(json_res["explanations"]) >= args.generate_num
            except (json.JSONDecodeError, AssertionError, KeyError, TypeError) as exc:
                print(f"Failed to candidate {candidate}: {exc}")
                result["parse_failed"] = True
                return result
        candidates = result["candidates"]
        with self.tracer.observe("evaluate-offspring", as_type="evaluator",
                                 input={"expressions": json_res["expressions"][:args.generate_num]}) as evaluation:
            parent_ic, parent_icir, parent_mut = self.pool.get_ic_icir_mutics(expr=parent_expr)
            result["parent"] = {"ic": _num(parent_ic), "icir": _num(parent_icir), "max_mut_ic": _max_abs(parent_mut)}
            for i in range(args.generate_num):
                raw_expr = json_res["expressions"][i]
                explanation = json_res["explanations"][i]
                record = {"expression": raw_expr, "outcome": "invalid"}
                candidates.append(record)
                if self.graph.search(raw_expr.strip()) is not None:
                    node = self.graph.search(raw_expr.strip())
                    expr = node.payload.expression
                    print(f"Expression {raw_expr} already exists in the graph, not inserting.")
                    ic, icir, _ = self.pool.get_ic_icir_mutics(expr)
                    mut_ic = self.pool.get_mutl_ic(expr=expr)
                    record.update(outcome="duplicate", ic=_num(ic), icir=_num(icir), max_mut_ic=_num(mut_ic))
                    if ic is not None:
                        if np.abs(parent_icir) < np.abs(icir) and np.abs(ic) > args.ic_threshold:
                            print(f"Expression {str(expr)} has higher ICIR {icir:.4f} than parent ICir {parent_icir:.4f}, while {str(expr)} ic {ic:.4f}, parent IC {parent_ic:.4f}. Try to insert early generated expression.")
                            accepted = self.pool.try_new_expr(expr, node.topic, node.description, node)
                            record["outcome"] = "duplicate_reinserted" if accepted else "duplicate_pool_rejected"
                        elif (np.abs(parent_icir) >= np.abs(icir) and mut_ic is not None
                              and np.abs(mut_ic) < args.ic_new_threshold
                              and np.abs(ic) > args.ic_threshold
                              and np.abs(icir) > np.abs(parent_icir) * args.icir_decay_threshold
                              and np.max(np.abs(mut_ic)) < np.max(np.abs(parent_mut))):
                            print(f"Expression {str(expr)} has sufficient lower MutIC {mut_ic:.4f}, with parent mutic {np.max(np.abs(parent_mut)):.4f}. Try to insert early generated expression.")
                            accepted = self.pool.try_new_expr(expr, node.topic, node.description, node)
                            record["outcome"] = "duplicate_reinserted" if accepted else "duplicate_pool_rejected"
                    continue
                expr_obj: Optional[exp_module.Expression] = None
                if raw_expr.__contains__("%d"):
                    used_windows = [5,10, 20, 30]
                    max_ic, max_windows = 0.0, 5
                    expr_obj = None
                    for w in used_windows:
                        expr_ = raw_expr.replace("%d", str(w))
                        try:
                            expr_obj = self.parser.parse(expr_)
                            ic = self.pool.get_ic(expr_obj)
                        except (InvalidExpressionException, ValueError) as exc:
                            print(f"Expression {expr_} is invalid: {exc}")
                            ic = None

                        if ic is not None and np.abs(ic) > max_ic:
                            max_ic = np.abs(ic)
                            max_windows = w
                    expr_ = raw_expr.replace("%d", str(max_windows))
                    record["window"] = max_windows
                    try:
                        expr_obj = self.parser.parse(expr_)
                    except (InvalidExpressionException, ValueError) as exc:
                        print(f"Expression {expr_} is invalid: {exc}")
                        record["error"] = str(exc)
                else:
                    try:
                        expr_obj = self.parser.parse(raw_expr)
                        max_ic = self.pool.get_ic(expr_obj)
                    except (InvalidExpressionException, ValueError) as exc:
                        print(f"Expression {raw_expr} is invalid: {exc}")
                        record["error"] = str(exc)
                if expr_obj is not None:
                    ic, icir, mut_ics = self.pool.get_ic_icir_mutics(expr_obj)
                    record.update(ic=_num(ic), icir=_num(icir), max_mut_ic=_max_abs(mut_ics))
                    expression_node = self.graph.build_expression_node(raw_expr.strip(), topic, explanation, args.max_length, np.abs(ic), np.abs(icir))
                    if expression_node is None:
                        print(f"Generated expression {raw_expr} is invalid, skipping.")
                        record["outcome"] = "invalid_node"
                        continue
                    record["outcome"] = "below_threshold"
                    if ic is not None and np.abs(ic) > args.ic_threshold and np.abs(icir) >np.abs(parent_icir):
                        res = self.pool.try_new_expr(expr_obj, topic, explanation, expression_node)
                        record["outcome"] = "pool_rejected"
                        if res:
                            print(f"Inserted new expression {str(expr_obj)} with ICIR {icir:.4f} higher than parent ICIR {parent_icir:.4f}, while current ic {ic:.4f}, parent IC {parent_ic:.4f}.")
                            self.graph.insert(expression_node=expression_node, parent_node=node)
                            record["outcome"] = "inserted_higher_icir"
                    elif (ic is not None and
                          np.max(np.abs(mut_ics)) < args.ic_new_threshold and
                          np.abs(ic) > args.ic_threshold and
                          np.abs(icir) > np.abs(parent_icir) * args.icir_decay_threshold and
                          np.max(np.abs(mut_ics)) < np.max(np.abs(parent_mut))):
                        res = self.pool.try_new_expr(expr_obj, topic, explanation, expression_node)
                        record["outcome"] = "pool_rejected"
                        if res:
                            print(f"Inserted new expression {str(expr_obj)} with sufficient lower MutIC {np.max(np.abs(mut_ics)):.4f} than parent expression mutic . Current ic {ic:.4f}, parent IC {parent_ic:.4f}. Current icir {icir:.4f}, parent ICIR {parent_icir:.4f}.")
                            self.graph.insert(expression_node=expression_node, parent_node=node)
                            record["outcome"] = "inserted_lower_mutic"
            evaluation.update(output=candidates)
        return result

    def _log_status(self, iteration: int, args: Any):
        with self.tracer.observe("evaluate-pool", as_type="evaluator", input={"pool_size": self.pool.size}) as observation:
            metrics = self.logger.log(self.pool, self.graph, iteration)
            observation.update(output=metrics)
        for split in ("Validation", "Test"):
            for name in ("ic", "icir", "ric", "ricir", "ret_sharpe"):
                value = metrics.get("combination", {}).get(split, {}).get(name)
                self.tracer.score(f"{split.lower()}_{name}", value)

    def train(self, args: Any):
        iter_times = args.search_time
        try:
            for it in range(iter_times):
                print(f"=== Iteration {it+1}/{iter_times} ===")
                with self.tracer.iteration(it + 1, input={"step": it + 1, "pool_size": self.pool.size}) as root:
                    stats = self._train_one_iteration(step=it + 1, args=args)
                    best_ic = float(np.max(np.abs(self.pool.single_ics[:self.pool.size]))) if self.pool.size > 0 else None
                    root.update(output={**stats, "pool_size": self.pool.size, "best_single_ic": best_ic})
                    self.tracer.score("candidates", stats["candidates"])
                    self.tracer.score("inserted", len(stats["inserted"]))
                    self.tracer.score("insert_rate", len(stats["inserted"]) / stats["candidates"] if stats["candidates"] else 0.0)
                    self.tracer.score("parse_failure_rate", stats["parse_failures"] / stats["nodes"] if stats["nodes"] else 0.0)
                    self.tracer.score("pool_size", self.pool.size)
                    self.tracer.score("best_single_ic", best_ic)
                    if (it + 1) % args.log_freq == 0 or it == iter_times - 1:
                        self._log_status(it + 1, args)
                self.tracer.flush()
        finally:
            self.tracer.flush()


def _num(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _max_abs(values: Any) -> Optional[float]:
    try:
        return None if values is None or len(values) == 0 else float(np.max(np.abs(values)))
    except (TypeError, ValueError):
        return None
