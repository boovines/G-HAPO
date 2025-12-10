import re
import math
from math_utils import is_correct as is_correct_hapo

# --- Logic from evaluate_reclor.py ---
def is_correct_textual(response: str, ground_truth: str) -> bool:
    """Checks ReClor (Multiple Choice) answers."""
    gt = ground_truth.strip().upper()
    
    # 1. Check for Boxed Answer
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        prediction = boxed_match.group(1).strip().upper()
        # Clean "Option A" -> "A"
        prediction = re.sub(r'^(OPTION|ANSWER)\s*', '', prediction)
        prediction = prediction.strip("()[]")
        if prediction == gt:
            return True
            
    # 2. Fallback: Last sentence heuristic
    lines = response.strip().split('\n')
    if lines:
        last_line = lines[-1].upper()
        match = re.search(r'(?:ANSWER|OPTION)?\s*[:=]?\s*([A-D])\b', last_line)
        if match and match.group(1) == gt:
             return True

    return False

# --- Logic from evaluate_s1k.py ---
def extract_answer_content(text: str) -> str:
    if not text: return ""
    text = str(text).strip()
    boxed_match = re.search(r'\\boxed\{(.*?)\}', text, re.DOTALL)
    if boxed_match: return boxed_match.group(1).strip()
    final_phrase_match = re.search(r'(?i)the final answer is[:\s]*(.*)', text)
    if final_phrase_match: return final_phrase_match.group(1).strip().rstrip('.')
    answer_colon_match = re.search(r'(?i)\banswer:\s*(.*)', text)
    if answer_colon_match: return answer_colon_match.group(1).strip()
    return text

def normalize_answer(text: str) -> str:
    text = text.replace(r'\,', '').replace(',', '')
    text = text.replace('$', '').replace('\\', '')
    return text.strip().lower()

def is_correct_math(response: str, ground_truth: str) -> bool:
    """Checks s1K (Math/Science) answers."""
    model_ans = extract_answer_content(response)
    gold_ans = extract_answer_content(ground_truth)
    
    model_norm = normalize_answer(model_ans)
    gold_norm = normalize_answer(gold_ans)

    # Float comparison
    try:
        m_val = float(model_norm)
        g_val = float(gold_norm)
        return math.isclose(m_val, g_val, rel_tol=1e-5)
    except ValueError:
        pass

    return model_norm == gold_norm

# --- THE ROUTER ---
def is_correct(response: str, ground_truth: str, use_math_verify: bool = False) -> bool:
    """
    Routes the check to the correct function based on the tag in ground_truth.
    Format expected: "SOURCE:::REAL_GT"
    """
    if ":::" in ground_truth:
        source, real_gt = ground_truth.split(":::", 1)
        
        if source == "RECLOR":
            return is_correct_textual(response, real_gt)
        elif source == "S1K":
            return is_correct_math(response, real_gt)
        # Add more sources here if needed
        
    # Default to HAPO (Original) behavior if no tag found or source is HAPO
    # We strip the tag if it was HAPO::: to be safe
    clean_gt = ground_truth.replace("HAPO:::", "")
    return is_correct_hapo(response, clean_gt, use_math_verify)