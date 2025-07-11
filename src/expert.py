import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import random
import expert_functions
import logging
from sentence_transformers import SentenceTransformer

import re
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Expert:
    """
    Expert system skeleton
    """
    def __init__(self, args, inquiry, options):
        # Initialize the expert with necessary parameters and the initial context or inquiry
        self.args = args
        self.inquiry = inquiry
        self.options = options

    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        raise NotImplementedError
    
    def ask_question(self, patient_state, prev_messages):
        # Generate a question based on the current patient state
        kwargs = {
            "patient_state": patient_state,
            "inquiry": self.inquiry,
            "options_dict": self.options,
            "messages": prev_messages,
            "independent_modules": self.args.independent_modules,
            "model_name": self.args.expert_model_question_generator,
            "use_vllm": self.args.use_vllm,
            "use_api": self.args.use_api,
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_tokens,
            "top_p": self.args.top_p,
            "top_logprobs": self.args.top_logprobs,
            "api_account": self.args.api_account
        }
        return expert_functions.question_generation(**kwargs)
    
    def get_abstain_kwargs(self, patient_state):
        kwargs = {
            "max_depth": self.args.max_questions,
            "patient_state": patient_state,
            "rationale_generation": self.args.rationale_generation,
            "inquiry": self.inquiry,
            "options_dict": self.options,
            "abstain_threshold": self.args.abstain_threshold,
            "self_consistency": self.args.self_consistency,
            "model_name": self.args.expert_model,
            "use_vllm": self.args.use_vllm,
            "use_api": self.args.use_api,
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_tokens,
            "top_p": self.args.top_p,
            "top_logprobs": self.args.top_logprobs,
            "api_account": self.args.api_account
        }
        return kwargs


class RandomExpert(Expert):
    """
    Below is an example Expert system that randomly asks a question or makes a choice based on the current patient state.
    This should be replaced with a more sophisticated expert system that can make informed decisions based on the patient state.
    """

    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        initial_info = patient_state['initial_info']  # not use because it's random
        history = patient_state['interaction_history']  # not use because it's random

        # randomly decide to ask a question or make a choice
        abstain = random.random() < 0.5
        toy_question = "Can you describe your symptoms more?"
        toy_decision = self.choice(patient_state)
        conf_score = random.random()/2 if abstain else random.random()

        return {
            "type": "question" if abstain else "choice",
            "question": toy_question,
            "letter_choice": toy_decision,
            "confidence": conf_score,  # Optional confidence score
            "urgent": True,  # Example of another optional flag
            "additional_info": "Check for any recent changes."  # Any other optional data
        }

    def choice(self, patient_state):
        # Generate a choice or intermediate decision based on the current patient state
        # randomly choose an option
        return random.choice(list(self.options.keys()))


class BasicExpert(Expert):
    def respond(self, patient_state):
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.implicit_abstention_decision(**kwargs)
        return {
            "type": "question" if abstain_response_dict["abstain"] else "choice",
            "question": abstain_response_dict["atomic_question"],
            "letter_choice": abstain_response_dict["letter_choice"],
            "confidence": abstain_response_dict["confidence"],
            "usage": abstain_response_dict["usage"]
        }


class FixedExpert(Expert):
    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.fixed_abstention_decision(**kwargs)
        if abstain_response_dict["abstain"] == False:
            return {
                "type": "choice",
                "letter_choice": abstain_response_dict["letter_choice"],
                "confidence": abstain_response_dict["confidence"],
                "usage": abstain_response_dict["usage"]
            }

        question_response_dict = self.ask_question(patient_state, abstain_response_dict["messages"])
        abstain_response_dict["usage"]["input_tokens"] += question_response_dict["usage"]["input_tokens"]
        abstain_response_dict["usage"]["output_tokens"] += question_response_dict["usage"]["output_tokens"]
        return {
            "type": "question",
            "question": question_response_dict["atomic_question"],
            "letter_choice": abstain_response_dict["letter_choice"],
            "confidence": abstain_response_dict["confidence"],
            "usage": abstain_response_dict["usage"]
        }
        

class BinaryExpert(Expert):
    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.binary_abstention_decision(**kwargs)
        if abstain_response_dict["abstain"] == False:
            return {
                "type": "choice",
                "letter_choice": abstain_response_dict["letter_choice"],
                "confidence": abstain_response_dict["confidence"],
                "usage": abstain_response_dict["usage"]
            }

        question_response_dict = self.ask_question(patient_state, abstain_response_dict["messages"])
        abstain_response_dict["usage"]["input_tokens"] += question_response_dict["usage"]["input_tokens"]
        abstain_response_dict["usage"]["output_tokens"] += question_response_dict["usage"]["output_tokens"]
        return {
            "type": "question",
            "question": question_response_dict["atomic_question"],
            "letter_choice": abstain_response_dict["letter_choice"],
            "confidence": abstain_response_dict["confidence"],
            "usage": abstain_response_dict["usage"]
        }


class NumericalExpert(Expert):
    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.numerical_abstention_decision(**kwargs)
        if abstain_response_dict["abstain"] == False:
            return {
                "type": "choice",
                "letter_choice": abstain_response_dict["letter_choice"],
                "confidence": abstain_response_dict["confidence"],
                "usage": abstain_response_dict["usage"]
            }

        question_response_dict = self.ask_question(patient_state, abstain_response_dict["messages"])
        abstain_response_dict["usage"]["input_tokens"] += question_response_dict["usage"]["input_tokens"]
        abstain_response_dict["usage"]["output_tokens"] += question_response_dict["usage"]["output_tokens"]
        return {
            "type": "question",
            "question": question_response_dict["atomic_question"],
            "letter_choice": abstain_response_dict["letter_choice"],
            "confidence": abstain_response_dict["confidence"],
            "usage": abstain_response_dict["usage"]
        }


class NumericalCutOffExpert(Expert):
    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.numcutoff_abstention_decision(**kwargs)
        if abstain_response_dict["abstain"] == False:
            return {
                "type": "choice",
                "letter_choice": abstain_response_dict["letter_choice"],
                "confidence": abstain_response_dict["confidence"],
                "usage": abstain_response_dict["usage"]
            }

        question_response_dict = self.ask_question(patient_state, abstain_response_dict["messages"])
        abstain_response_dict["usage"]["input_tokens"] += question_response_dict["usage"]["input_tokens"]
        abstain_response_dict["usage"]["output_tokens"] += question_response_dict["usage"]["output_tokens"]
        return {
            "type": "question",
            "question": question_response_dict["atomic_question"],
            "letter_choice": abstain_response_dict["letter_choice"],
            "confidence": abstain_response_dict["confidence"],
            "usage": abstain_response_dict["usage"]
        }


class ScaleExpert(Expert):
    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.scale_abstention_decision(**kwargs)
        if abstain_response_dict["abstain"] == False:
            return {
                "type": "choice",
                "letter_choice": abstain_response_dict["letter_choice"],
                "confidence": abstain_response_dict["confidence"],
                "usage": abstain_response_dict["usage"]
            }

        question_response_dict = self.ask_question(patient_state, abstain_response_dict["messages"])
        abstain_response_dict["usage"]["input_tokens"] += question_response_dict["usage"]["input_tokens"]
        abstain_response_dict["usage"]["output_tokens"] += question_response_dict["usage"]["output_tokens"]
        return {
            "type": "question",
            "question": question_response_dict["atomic_question"],
            "letter_choice": abstain_response_dict["letter_choice"],
            "confidence": abstain_response_dict["confidence"],
            "usage": abstain_response_dict["usage"]
        }
class InfoGainExpert(Expert):
    _model = None
    _tokenizer = None
    _embedding_model = None
    def gini_impurity(self, dist):
        # Gini impurity: 1 - sum(p^2)
        return 1 - torch.sum(dist ** 2).item()

    def semantic_embedding_shift(self, prior, post, embedding_model):
        # Try to get embeddings and compute Euclidean distance, else return 0.0
        try:
            prior_emb = embedding_model(prior)
            post_emb = embedding_model(post)
            return torch.norm(prior_emb - post_emb, p=2).item()
        except Exception:
            return 0.0

    def entropy_constrained_margin_gain(self, prior, post):
        # Margin is the difference between top two probabilities
        def margin(dist):
            top2 = torch.topk(dist, 2).values
            return (top2[0] - top2[1]).item()
        return margin(post) - margin(prior)

    def surprisal_adjusted_belief_shift(self, prior, post, true_idx):
        # sum_i prior_i * (log prior_i - log post_i)
        return torch.sum(prior * (torch.log(prior + 1e-12) - torch.log(post + 1e-12))).item()

    def fisher_information_approx(self, dist):
        # Approximate Fisher information as sum of p*(1-p)
        return torch.sum(dist * (1 - dist)).item()

    def entropy_weighted_confidence_gain(self, prior, post, true_idx):
        # prior confidence in true_idx * (entropy(post) - entropy(prior))
        entropy = lambda d: -torch.sum(d * torch.log(d + 1e-12)).item()
        conf = prior[true_idx].item() if 0 <= true_idx < len(prior) else 0.0
        return conf * (entropy(post) - entropy(prior))

    def cross_entropy_reduction(self, prior, post, true_idx):
        # H(prior, true) - H(post, true)
        # H(p, q) = -sum true_i * log(q_i), but true is one-hot at true_idx
        ce = lambda d: -torch.log(d[true_idx] + 1e-12).item() if 0 <= true_idx < len(d) else 0.0
        return ce(prior) - ce(post)

    def kl_to_truth_reduction(self, prior, post, true_idx):
        # KL(prior || truth) - KL(post || truth), where truth is one-hot at true_idx
        # KL(p || q) = sum_i p_i * (log(p_i) - log(q_i)), but truth is one-hot, so KL(p || truth) = -log(p_true)
        kl = lambda d: -torch.log(d[true_idx] + 1e-12).item() if 0 <= true_idx < len(d) else 0.0
        return kl(prior) - kl(post)
    def __init__(self, args, inquiry, options):
        super().__init__(args, inquiry, options)
        # tokenizer caching
        if InfoGainExpert._tokenizer is None:
            InfoGainExpert._tokenizer = AutoTokenizer.from_pretrained(args.expert_model)
        self.tokenizer = InfoGainExpert._tokenizer

        # 8-bit model caching
        if InfoGainExpert._model is None:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            InfoGainExpert._model = AutoModelForCausalLM.from_pretrained(
                args.expert_model,
                device_map="auto",
                quantization_config=bnb_config,
                torch_dtype=torch.float16
            )
        self.model = InfoGainExpert._model

        # determine device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # embedding model caching
        if InfoGainExpert._embedding_model is None:
            InfoGainExpert._embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.embedding_model = InfoGainExpert._embedding_model

        # log storage
        self.ig_log = []
        

    def respond(self, patient_state):
        import logging
        detail_logger = logging.getLogger("mediq.expert.detail")
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.fixed_abstention_decision(**kwargs)

        context_before = self.format_context(patient_state['interaction_history'])

        # Compute belief before follow-up
        belief_before = self.get_mcq_distribution(context_before)
        detail_logger.info(f"belief_before: {belief_before.tolist()}")

        if abstain_response_dict["abstain"] == False:
            is_correct = abstain_response_dict["letter_choice"] == patient_state.get("answer")
            # Compute entropy for before and after (same in this case)
            entropy_before = -torch.sum(belief_before * torch.log(belief_before + 1e-12)).item()
            entropy_after = entropy_before
            entropy_delta = entropy_after - entropy_before
            detail_logger.info(f"Entropy before: {entropy_before}, Entropy after: {entropy_after}, Entropy delta: {entropy_delta}")
            # Set up dummy values for metrics that require two distributions or an answer
            kl = 0.0
            wasserstein = 0.0
            tv = 0.0
            cosine = 0.0
            euclidean = 0.0
            margin_gain = 0.0
            surprisal_adj = 0.0
            log_odds_gain = 0.0
            # Additional QuID metrics
            true_idx = patient_state.get("answer", 0)
            # gini_drop
            gini_drop = self.gini_impurity(belief_before) - self.gini_impurity(belief_before)
            # semantic embedding shift
            if hasattr(self, "embedding_model") and self.embedding_model is not None:
                sem_shift = self.semantic_embedding_shift(belief_before, belief_before, self.embedding_model)
            else:
                sem_shift = 0.0
            # entropy-constrained margin gain
            ec_margin = self.entropy_constrained_margin_gain(belief_before, belief_before)
            # surprisal-adjusted belief shift
            sab_shift = self.surprisal_adjusted_belief_shift(belief_before, belief_before, true_idx)
            # fisher information
            fisher_info = self.fisher_information_approx(belief_before)
            # normalized KL
            norm_kl = kl / (entropy_before + 1e-8)
            # normalized margin
            norm_margin = margin_gain / (entropy_before + 1e-8)
            # calibrated eig
            eig_cal = (entropy_before - entropy_after) / (entropy_before + 1e-8)
            # entropy-weighted confidence gain
            ewc_gain = self.entropy_weighted_confidence_gain(belief_before, belief_before, true_idx)
            # cross-entropy reduction
            ce_reduction = self.cross_entropy_reduction(belief_before, belief_before, true_idx)
            # kl to truth reduction
            kl_to_truth_reduction = self.kl_to_truth_reduction(belief_before, belief_before, true_idx)
            # delta_surprisal
            delta_surprisal = 0.0
            # semantic relevance and info metrics
            sem_relevance_cosine = 0.0
            sem_relevance_euclidean = 0.0
            sem_relevance_dot = 0.0
            sem_info_cosine = 0.0
            sem_info_euclidean = 0.0
            sem_info_dot = 0.0
            ig_metrics = {
                "kl": kl,
                "wasserstein": wasserstein,
                "tv": tv,
                "cosine": cosine,
                "euclidean": euclidean,
                "margin_gain": margin_gain,
                "surprisal_adj_belief_shift": surprisal_adj,
                "log_odds_gain": log_odds_gain,
                "correct": int(is_correct),
                "entropy_before": entropy_before,
                "entropy_after": entropy_after,
                "entropy_delta": entropy_delta,
                "gini_drop": gini_drop,
                "semantic_shift": sem_shift,
                "entropy_constrained_margin": ec_margin,
                "surprisal_adj_belief_shift_full": sab_shift,
                "fisher_info": fisher_info,
                "normalized_kl": norm_kl,
                "normalized_margin": norm_margin,
                "calibrated_eig": eig_cal,
                "entropy_weighted_conf_gain": ewc_gain,
                "cross_entropy_reduction": ce_reduction,
                "kl_to_truth_reduction": kl_to_truth_reduction,
                "delta_surprisal": delta_surprisal,
                "sem_relevance_cosine": sem_relevance_cosine,
                "sem_relevance_euclidean": sem_relevance_euclidean,
                "sem_relevance_dot": sem_relevance_dot,
                "sem_info_cosine": sem_info_cosine,
                "sem_info_euclidean": sem_info_euclidean,
                "sem_info_dot": sem_info_dot,
            }
            detail_logger.info(f"ig_metrics: {ig_metrics}")
            self.ig_log.append(ig_metrics)
            return {
                "type": "choice",
                "letter_choice": abstain_response_dict["letter_choice"],
                "confidence": abstain_response_dict["confidence"],
                "usage": abstain_response_dict["usage"],
                "ig_metrics": ig_metrics
            }

        question_response_dict = self.ask_question(patient_state, abstain_response_dict["messages"])
        q_text = f"Doctor: {question_response_dict['atomic_question']}"
        if 'interaction_history' in patient_state and len(patient_state['interaction_history']) > 0:
            _, real_response = patient_state['interaction_history'][-1]
            r_text = f"Patient: {real_response}"
        else:
            r_text = "Patient: [RESPONSE]"

        context_after = context_before + "\n" + q_text + "\n" + r_text
        belief_after = self.get_mcq_distribution(context_after)
        detail_logger.info(f"belief_after: {belief_after.tolist()}")

        # Belief shift logging
        if torch.allclose(belief_before, belief_after, atol=1e-5):
            detail_logger.info("No belief shift detected between belief_before and belief_after.")
        else:
            detail_logger.info("Belief shift detected.")

        entropy_before = -torch.sum(belief_before * torch.log(belief_before + 1e-12)).item()
        entropy_after = -torch.sum(belief_after * torch.log(belief_after + 1e-12)).item()
        entropy_delta = entropy_after - entropy_before
        detail_logger.info(f"Entropy before: {entropy_before}, Entropy after: {entropy_after}, Entropy delta: {entropy_delta}")

        # Compute information gain metrics
        kl = F.kl_div(belief_after.log(), belief_before, reduction='batchmean').item()
        wasserstein = torch.cdist(
            belief_before.float().unsqueeze(0),
            belief_after.float().unsqueeze(0),
            p=1
        ).item()
        tv = 0.5 * torch.sum(torch.abs(belief_before - belief_after)).item()
        cosine = F.cosine_similarity(belief_before.unsqueeze(0), belief_after.unsqueeze(0)).item()
        euclidean = torch.norm(belief_before - belief_after, p=2).item()
        margin_gain = (torch.max(belief_after) - torch.max(belief_before)).item()
        surprisal_adj = (torch.sum(belief_before * (belief_before.log() - belief_after.log()))).item()
        log_odds_gain = torch.sum(torch.abs(torch.logit(belief_after) - torch.logit(belief_before))).item()

        is_correct = abstain_response_dict["letter_choice"] == patient_state.get("answer")
        true_idx = patient_state.get("answer", 0)
        # Compute additional QuID metrics
        gini_drop = self.gini_impurity(belief_before) - self.gini_impurity(belief_after)
        if hasattr(self, "embedding_model") and self.embedding_model is not None:
            sem_shift = self.semantic_embedding_shift(belief_before, belief_after, self.embedding_model)
        else:
            sem_shift = 0.0
        ec_margin = self.entropy_constrained_margin_gain(belief_before, belief_after)
        sab_shift = self.surprisal_adjusted_belief_shift(belief_before, belief_after, true_idx)
        fisher_info = self.fisher_information_approx(belief_after)
        norm_kl = kl / (entropy_before + 1e-8)
        norm_margin = margin_gain / (entropy_before + 1e-8)
        eig_cal = (entropy_before - entropy_after) / (entropy_before + 1e-8)
        ewc_gain = self.entropy_weighted_confidence_gain(belief_before, belief_after, true_idx)
        ce_reduction = self.cross_entropy_reduction(belief_before, belief_after, true_idx)
        kl_to_truth_reduction = self.kl_to_truth_reduction(belief_before, belief_after, true_idx)
        delta_surprisal = 0.0
        sem_relevance_cosine = 0.0
        sem_relevance_euclidean = 0.0
        sem_relevance_dot = 0.0
        sem_info_cosine = 0.0
        sem_info_euclidean = 0.0
        sem_info_dot = 0.0
        ig_metrics = {
            "kl": kl,
            "wasserstein": wasserstein,
            "tv": tv,
            "cosine": cosine,
            "euclidean": euclidean,
            "margin_gain": margin_gain,
            "surprisal_adj_belief_shift": surprisal_adj,
            "log_odds_gain": log_odds_gain,
            "correct": int(is_correct),
            "entropy_before": entropy_before,
            "entropy_after": entropy_after,
            "entropy_delta": entropy_delta,
            "gini_drop": gini_drop,
            "semantic_shift": sem_shift,
            "entropy_constrained_margin": ec_margin,
            "surprisal_adj_belief_shift_full": sab_shift,
            "fisher_info": fisher_info,
            "normalized_kl": norm_kl,
            "normalized_margin": norm_margin,
            "calibrated_eig": eig_cal,
            "entropy_weighted_conf_gain": ewc_gain,
            "cross_entropy_reduction": ce_reduction,
            "kl_to_truth_reduction": kl_to_truth_reduction,
            "delta_surprisal": delta_surprisal,
            "sem_relevance_cosine": sem_relevance_cosine,
            "sem_relevance_euclidean": sem_relevance_euclidean,
            "sem_relevance_dot": sem_relevance_dot,
            "sem_info_cosine": sem_info_cosine,
            "sem_info_euclidean": sem_info_euclidean,
            "sem_info_dot": sem_info_dot,
        }
        detail_logger.info(f"ig_metrics: {ig_metrics}")
        self.ig_log.append(ig_metrics)

        abstain_response_dict["usage"]["input_tokens"] += question_response_dict["usage"]["input_tokens"]
        abstain_response_dict["usage"]["output_tokens"] += question_response_dict["usage"]["output_tokens"]
        return {
            "type": "question",
            "question": question_response_dict["atomic_question"],
            "letter_choice": abstain_response_dict["letter_choice"],
            "confidence": abstain_response_dict["confidence"],
            "usage": abstain_response_dict["usage"],
            "ig_metrics": ig_metrics
        }

    def format_context(self, history):
        return "\n".join([f"Doctor: {q}\nPatient: {r}" for q, r in history])

    def get_mcq_distribution(self, context):
        prompt = f"{context}\n\nQ: {self.inquiry}\nChoices:\n" + "\n".join([f"{k}. {v}" for k, v in self.options.items()]) + "\nA:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)
        logits = outputs.logits[0, -1]  # Last token logits

        # Extract logits for A–D
        choice_logits = []
        for option in self.options.keys():
            token_id = self.tokenizer.encode(option, add_special_tokens=False)[0]
            choice_logits.append(logits[token_id])
        probs = F.softmax(torch.tensor(choice_logits), dim=0)
        return probs
    def save_ig_log(self, filepath="ig_vs_accuracy.jsonl"):
        import json
        with open(filepath, "w") as f:
            for entry in self.ig_log:
                json.dump(entry, f)
                f.write("\n")


# LLMJudgeExpert: LLM-based multidimensional question rater
class LLMJudgeExpert(Expert):
    _model = None
    _tokenizer = None
    _embedding_model = None

    def __init__(self, args, inquiry, options, scale=5):
        super().__init__(args, inquiry, options)
        self.scale = scale
        # reuse tokenizer/model caching logic
        if LLMJudgeExpert._tokenizer is None:
            LLMJudgeExpert._tokenizer = AutoTokenizer.from_pretrained(args.expert_model)
        self.tokenizer = LLMJudgeExpert._tokenizer

        if LLMJudgeExpert._model is None:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            LLMJudgeExpert._model = AutoModelForCausalLM.from_pretrained(
                args.expert_model,
                device_map="auto",
                quantization_config=bnb_config,
                torch_dtype=torch.float16
            )
        self.model = LLMJudgeExpert._model

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if LLMJudgeExpert._embedding_model is None:
            LLMJudgeExpert._embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.embedding_model = LLMJudgeExpert._embedding_model

    def rate_question_multidimensional(self, context, question, scale=None, concept=None):
        if scale is None:
            scale = self.scale

        def ask_single_metric(metric_name, prompt_detail, scale):
            prompt = (
                f"You are evaluating a question asked during a game where the goal is to guess a hidden concept. "
                f"For the given context and question, rate the question on a scale from 1 to {scale} for the following aspect:\n\n"
                f"{prompt_detail}\n\n"
                f"Context:\n{context.strip()}\n\n"
                f"Question-Answer: \"{question}\"\n\n"
                f"ONLY respond with a single number from 1 to {scale}. Answer: "
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
            try:
                match = [int(n) for n in re.findall(r'\b\d+\b', decoded)]
                if match:
                    return max(1, min(scale, match[-1]))
            except Exception:
                pass
            return -1

        informativeness = ask_single_metric(
            "informativeness",
            "Informativeness: To what extent does the question reduce uncertainty about the task-relevant latent variable?",
            scale
        )
        relevance = ask_single_metric(
            "relevance",
            "Relevance: Is the question appropriate and grounded in the current context or conversation state?",
            scale
        )
        answerability = ask_single_metric(
            "answerability",
            "Answerability: Is the question well-formed such that an answer can be generated or obtained reliably?",
            scale
        )
        efficiency = ask_single_metric(
            "efficiency",
            "Efficiency: Does the question progress the agent toward task completion with minimal redundancy?",
            scale
        )

        result = {
            'llm_informativeness': informativeness,
            'llm_relevance': relevance,
            'llm_answerability': answerability,
            'llm_efficiency': efficiency
        }
        valid_scores = [v for v in result.values() if v >= 0]
        result['llm_score'] = np.mean(valid_scores) if valid_scores else -1
        return result, None

    def compute_metrics(self, context, prior, object, question, scale=None, multidimensional=False):
        if multidimensional:
            return self.rate_question_multidimensional(context, question, scale, object)
        else:
            return self.rate_question_informativeness(context, question, scale, object)

    def rate_question_informativeness(self, context, question, scale=None, concept=None):
        # Single-dimensional informativeness rating
        if scale is None:
            scale = self.scale
        # reuse the ask_single_metric logic for informativeness only
        result, _ = self.rate_question_multidimensional(context, question, scale, concept)
        # extract only informativeness
        return {'llm_informativeness': result['llm_informativeness']}, None

    def respond(self, patient_state):
        # Format the existing interaction history
        context = "\n".join([f"Doctor: {q}\nPatient: {r}" for q, r in patient_state["interaction_history"]])
        # Ask a follow-up question (or decide to choose), using fixed abstention
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response = expert_functions.fixed_abstention_decision(**kwargs)
        if not abstain_response["abstain"]:
            return {
                "type": "choice",
                "letter_choice": abstain_response["letter_choice"],
                "confidence": abstain_response["confidence"],
                "usage": abstain_response["usage"]
            }
        # Otherwise ask a question
        question_dict = self.ask_question(patient_state, abstain_response["messages"])
        # Compute LLM-based metrics on this question
        metrics, _ = self.compute_metrics(
            context,
            None,
            patient_state.get("answer"),
            question_dict["atomic_question"],
            multidimensional=True
        )
        # Package the response
        abstain_response["usage"]["input_tokens"] += question_dict["usage"]["input_tokens"]
        abstain_response["usage"]["output_tokens"] += question_dict["usage"]["output_tokens"]
        return {
            "type": "question",
            "question": question_dict["atomic_question"],
            "letter_choice": abstain_response["letter_choice"],
            "confidence": abstain_response["confidence"],
            "usage": abstain_response["usage"],
            "llm_metrics": metrics
        }
