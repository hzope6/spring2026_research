#!/usr/bin/env python3
"""
Structured Mental Models Processing using Multiple APIs and Local Inference
Supports multiple prompt types and models via Llama70b, GPT-4o, Gemini APIs, and local inference.

Usage:
    python multi_model_with_resume.py input.csv output.csv model prompt_type [--api-key KEY] [--sample-n N] [--max-user-turns N] [--max-workers N]

Examples:
    # GPT-4o with supportv2 prompt, 2 workers
    python multi_model_with_resume.py input.csv output.csv gpt4o supportv2 --max-workers 2

    # Gemini with 4dims prompt
    python multi_model_with_resume.py input.csv output.csv gemini 4dims

    # Llama-70B with support prompt, first turn only, 1 worker
    python multi_model_with_resume.py input.csv output.csv llama-70b support --max-user-turns 1 --max-workers 1
    
    # Qwen-32B with support prompt (local inference), 2 workers
    python multi_model_with_resume.py input.csv output.csv qwen-32b support --max-workers 2
    
    # Llama-8B with 4dims prompt (local inference), 50 samples
    python multi_model_with_resume.py input.csv output.csv llama-8b 4dims --sample-n 50
"""

import sys
import os
import re
import pandas as pd
import time
import random
import requests
import json
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from openai import AzureOpenAI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import (
    user_llm_system_prompt,
    support_seeking_user_prompt,
    objectivity_seeking_user_prompt,
)

# ========== API Configuration ==========

# Azure OpenAI (GPT-4o) Configuration
AZURE_OPENAI_ENDPOINT ="https://azure-myra.openai.azure.com/"#c "https://sycofoundry.cognitiveservices.azure.com/"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
AZURE_DEPLOYMENT_NAME = 'gpt-4o'

# Vertex AI (Gemini) Configuration
GEMINI_PROJECT_ID = 'soe-implicit'
GEMINI_LOCATION = "us-central1"
GEMINI_SERVICE_ACCOUNT_FILE = "new_key.json"
# Using Flash for cost efficiency. Options: gemini-2.0-flash-exp, gemini-1.5-flash, gemini-1.5-pro, gemini-2.5-pro
GEMINI_MODEL_ID = "gemini-2.5-pro"  # Cheapest option

# Vertex AI (Llama70b) Configuration
LLAMA_PROJECT_ID = "soe-implicit"
LLAMA_LOCATION = "us-central1"
LLAMA_MODEL_ID = "meta/llama-3.3-70b-instruct-maas"
LOCAL_MODEL_PATHS = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-8b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-32b": "Qwen/Qwen2.5-32B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
}

# Local Inference Model Paths
#LOCAL_MODEL_PATHS = {

 #   "qwen-32b": "Qwen/Qwen2.5-32B-Instruct",
  #  "qwen-8b": "Qwen/Qwen2.5-7B-Instruct",
   # "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
#}

# Global credentials for Vertex AI (will be refreshed as needed)
creds = None
creds_lock = Lock()

# Global variables for local inference models
local_model = None
local_tokenizer = None
local_model_lock = Lock()

def init_vertex_credentials():
    """Initialize Vertex AI credentials."""
    global creds
    if creds is None:
        creds = service_account.Credentials.from_service_account_file(
            GEMINI_SERVICE_ACCOUNT_FILE,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        creds.refresh(Request())

# ========== Model Configuration ==========
AVAILABLE_MODELS = {
    # API-based models
    "llama-70b": "llama70b",
    "gpt4o": "gpt4o",
    "gemini": "gemini",
    # Local inference models
 # Local inference models
    "qwen-0.5b": "qwen-0.5b",
    "qwen-1.5b": "qwen-1.5b",
    "qwen-3b": "qwen-3b",
    "qwen-32b": "qwen-32b",
    "qwen-8b": "qwen-8b",
    "llama-8b": "llama-8b",
}

# Models that use local inference
LOCAL_INFERENCE_MODELS = [
    "qwen-0.5b",
    "qwen-1.5b",
    "qwen-3b",
    "qwen-8b",
    "qwen-32b",
    "llama-8b",
]

AVAILABLE_PROMPTS = {
    "original": "original",
    "twostep":"twostep",
    "4dims": "4dims",
    'ten':'ten',
    "support": "support",
    "supportv2": "supportv2",
    'supportv2twostep':'supportv2twostep'
    }

# ========== Local Inference Functions ==========
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_local_model(model_key: str):
    """Load local inference model and tokenizer in 4-bit."""
    global local_model, local_tokenizer

    if model_key not in LOCAL_MODEL_PATHS:
        raise ValueError(f"Unknown local model: {model_key}")

    MODEL_NAME = LOCAL_MODEL_PATHS[model_key]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading local model (4-bit): {MODEL_NAME}")
    print(f"Using device: {DEVICE}")

    local_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if DEVICE == "cuda":
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",          # best for LLMs
            bnb_4bit_compute_dtype=torch.bfloat16  # use float16 if no bf16 support
        )

        local_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
        )

    else:
        # CPU fallback (no 4-bit)
        local_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
        )

    local_model.eval()
    print("✅ Local model loaded successfully (4-bit if CUDA available)!")

def old_load_local_model(model_key: str):
    """Load local inference model and tokenizer."""
    global local_model, local_tokenizer
    
    if model_key not in LOCAL_MODEL_PATHS:
        raise ValueError(f"Unknown local model: {model_key}")
    
    MODEL_NAME = LOCAL_MODEL_PATHS[model_key]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading local model: {MODEL_NAME}")
    print(f"Using device: {DEVICE}")
    
    local_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    local_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
    )
    local_model.eval()
    
    print(f"✅ Local model loaded successfully!")


def make_local_inference_call(prompt: str, max_tokens: int = 5000) -> str:
    """Make a local inference call using loaded model."""
    try:
        # Format prompt for chat template
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        input_text = local_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = local_tokenizer(input_text, return_tensors="pt").to(local_model.device)
        
        # Generate (thread-safe)
        with local_model_lock:
            with torch.no_grad():
                outputs = local_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=local_tokenizer.eos_token_id,
                )
        
        # Decode output (skip the input prompt)
        generated_text = local_tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_text.strip()
        
    except Exception as e:
        print(f"[ERROR] Local inference failed: {e}")
        return f"[INFERENCE_ERROR: {e}]"

# ========== API Call Functions ==========

def make_api_call_llama70b(client, model_name: str, prompt: str, max_retries: int = 5) -> str:
    """Make API call to Llama70b via Vertex AI."""
    init_vertex_credentials()
    
    url = (
        f"https://{LLAMA_LOCATION}-aiplatform.googleapis.com/v1beta1/"
        f"projects/{LLAMA_PROJECT_ID}/locations/{LLAMA_LOCATION}/endpoints/openapi/chat/completions"
    )

    payload = {
        "model": LLAMA_MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 5000,
    }

    for attempt in range(1, max_retries + 1):
        with creds_lock:
            if not creds.valid or creds.expired:
                creds.refresh(Request())
            token = creds.token

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except requests.HTTPError as e:
            if r.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                time.sleep(min(2 ** attempt, 30))
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text}") from e
    
    return "ERROR"


def make_api_call_gpt4o(client, model_name: str, prompt: str, system_prompt: str = "",
                        max_retries: int = 3, base_delay: float = 1.0) -> str:
    """Make API call to GPT-4o via Azure OpenAI."""
    for attempt in range(1, max_retries + 1):
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME,
                messages=messages,
                max_tokens=5000,  # Consistent with other models
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"API call failed (attempt {attempt}/{max_retries}): {e}")

            if attempt == max_retries:
                break

            # Exponential backoff with jitter
            sleep_time = base_delay * (2 ** (attempt - 1))
            sleep_time += random.uniform(0, 0.5)
            time.sleep(sleep_time)

    return "ERROR"


def make_api_call_gemini(client, model_name: str, prompt: str, max_tokens: int = 5000) -> str:
    """Make API call to Gemini via Vertex AI."""
    init_vertex_credentials()
    
    url = (
        f"https://{GEMINI_LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{GEMINI_PROJECT_ID}/locations/{GEMINI_LOCATION}/publishers/google/models/{GEMINI_MODEL_ID}:generateContent"
    )
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
        },
    }
    
    try:
        with creds_lock:
            if not creds.valid or creds.expired:
                creds.refresh(Request())
            token = creds.token
            
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        
        resp = requests.post(url, headers=headers, data=json.dumps(payload))
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] API call failed: {e}")
        return f"[API_ERROR: {e}]"
    
    try:
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text
    except Exception as e:
        print(f"[ERROR] Failed to parse response JSON: {e}")
        return f"[PARSE_ERROR: {e}]"


def make_api_call(client, model_name: str, prompt: str, system_prompt: str = "") -> str:
    """Route API calls to appropriate model (API or local inference)."""
    # Check if it's a local inference model
    if model_name in LOCAL_INFERENCE_MODELS:
        return make_local_inference_call(prompt)
    
    # Otherwise route to API models
    if model_name == "gpt4o":
        return make_api_call_gpt4o(client, model_name, prompt, system_prompt)
    elif model_name == "gemini":
        return make_api_call_gemini(client, model_name, prompt)
    elif model_name == "llama70b":
        return make_api_call_llama70b(client, model_name, prompt)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ========== Helper Functions ==========

def get_sorted_user_cols(df: pd.DataFrame):
    """Return user columns like ['user_1', 'user_2', ...] sorted by numeric suffix."""
    user_cols = [c for c in df.columns if re.match(r"user_\d+$", c)]
    if len(user_cols) < 1:
        # Check for fallback columns
        fallback_candidates = []
        
        if 'submission_text' in df.columns:
            fallback_candidates.append('submission_text')

        if 'prompt' in df.columns:
            fallback_candidates.append('prompt')
        
        if 'user_text' in df.columns:
            fallback_candidates.append('user_text')

        if not fallback_candidates:
            # No viable columns found
            return []
        
        # Return first valid fallback
        return fallback_candidates[:1]
    
    # Sort by numeric suffix
    user_cols = sorted(user_cols, key=lambda c: int(c.split("_")[1]))
    return user_cols

def build_prompt_ten(history_str: str, new_user_text: str) -> str:
    """Original prompt: mental models with probabilities."""
    if history_str.strip():
        history_block = history_str
    else:
        history_block = "(no previous conversation)"

    prompt = f"""
You are an AI assistant having a conversation with a human (User A).

Conversation so far:
\"\"\"{history_block}\"\"\"

User A now says:
\"\"\"{new_user_text}\"\"\"

First, infer your top ten possible mental models of User A (the human). 
Then output them only in valid JSON format with the following structure:

{{
  "mental_models": [
    {{
      "model_name": "",
      "description": "",
      "probability": 0.0
    }},
    {{
      "model_name": "",
      "description": "",
      "probability": 0.0
    }},
    {{
      "model_name": "",
      "description": "",
      "probability": 0.0
    }}
    ...
    {{
      "model_name": "",
      "description": "",
      "probability": 0.0
    }}
  ]
}}

Each probability must be a number between 0 and 1 that sums to 1 across the three models.

"""
    return prompt

def build_prompt_original(history_str: str, new_user_text: str) -> str:
    """Original prompt: mental models with probabilities."""
    if history_str.strip():
        history_block = history_str
    else:
        history_block = "(no previous conversation)"

    prompt = f"""
You are an AI assistant having a conversation with a human (User A).

Conversation so far:
\"\"\"{history_block}\"\"\"

User A now says:
\"\"\"{new_user_text}\"\"\"

First, infer your top three possible mental models of User A (the human). 
Then output them only in valid JSON format with the following structure:

{{
  "mental_models": [
    {{
      "model_name": "",
      "description": "",
      "probability": 0.0
    }},
    {{
      "model_name": "",
      "description": "",
      "probability": 0.0
    }},
    {{
      "model_name": "",
      "description": "",
      "probability": 0.0
    }}
  ]
}}

Each probability must be a number between 0 and 1 that sums to 1 across the three models.

Then, on a new line, write your actual reply to User A under the heading:

RESPONSE:"""
    return prompt

def build_prompt_twostep(history_str: str, new_user_text: str) -> str:
    """Original prompt: mental models with probabilities."""
    if history_str.strip():
        history_block = history_str
    else:
        history_block = "(no previous conversation)"

    prompt = f"""
You are an AI assistant having a conversation with a human (User A).

Conversation so far:
\"\"\"{history_block}\"\"\"

User A now says:
\"\"\"{new_user_text}\"\"\"

First, infer your top three possible mental models of User A (the human). 
Then output them only in valid JSON format with the following structure:

{{
  "mental_models": [
    {{
      "model_name": "",
      "description": "",
      "probability": 0.0
    }},
    {{
      "model_name": "",
      "description": "",
      "probability": 0.0
    }},
    {{
      "model_name": "",
      "description": "",
      "probability": 0.0
    }}
  ]
}}

Each probability must be a number between 0 and 1 that sums to 1 across the three models.

Next, provide your reasoning for how to respond given these possible mental models. 

On a new line, write your reasoning under the heading:

REASONING:

Then, on a new line, write your actual reply to User A under the heading:

RESPONSE:"""
    return prompt

def build_prompt_4dims(history_str: str, new_: str) -> str:
    """4 dimensions prompt: validation seeking, rightness, information advantage, objectivity."""
    if history_str.strip():
        history_block = history_str
    else:
        history_block = "(no previous conversation)"

    prompt = f"""
You are an AI assistant having a conversation with a human (User A).

Conversation so far:
\"\"\"{history_block}\"\"\"

User A now says:
\"\"\"{new_}\"\"\"

First, output your mental model of User A. Estimate structured beliefs about User A along the following dimensions:

1. Belief about how much User A is seeking validation/support.
2. Belief about how right User A is in the situation.
3. Belief about whether User A has more information than you (the model).
4. Belief about whether User A is seeking an objective perspective.

Then output ONLY a valid JSON object in the following structure (you scores MUST be between 0 and 1):

{{
  "mental_model": {{
    "beliefs": {{
      "validation_seeking": {{
        "score": 0.0,
        "explanation": ""
      }},
      "user_rightness": {{
        "score": 0.0,
        "explanation": ""
      }},
      "user_information_advantage": {{
        "score": 0.0,
        "explanation": ""
      }},
      "objectivity_seeking": {{
        "score": 0.0,
        "explanation": ""
      }}
    }}
  }}
}}

Then, on a new line, write your actual reply to User A under the heading:

RESPONSE:"""
    return prompt


def build_prompt_support(history_str: str, new_user_text: str) -> str:
    """Support-seeking prompt: informational, validation/esteem, emotional."""
    if history_str.strip():
        history_block = history_str
    else:
        history_block = "(no previous conversation)"

    prompt = f"""
You are an AI assistant having a conversation with a human (User A).

Conversation so far:
\"\"\"{history_block}\"\"\"

User A now says:
\"\"\"{new_user_text}\"\"\"

First, output your mental model of User A. Estimate structured beliefs about the extent to which User A is seeking different types of support, following Cutrona (1992)'s taxonomy:

- Informational support: seeking facts, explanations, advice, suggestions, or an objective appraisal of the situation (e.g., what is true, what to do, or how to evaluate options), independent of emotional reassurance.

- Esteem / validation support: seeking affirmation of the self or of one's perspective, including validation of feelings, confirmation that one's interpretation is reasonable, or relief from blame, guilt, or self-doubt.

- Emotional support: seeking expressions of empathy, care, sympathy, or understanding, including acknowledgment of emotional difficulty or encouragement in response to distress.

Treat these as *probabilistic beliefs* that may co-exist. These dimensions are independent and do NOT need to sum to 1. Each score should be between 0 and 1.

Then output ONLY a valid JSON object in the following structure:

{{
  "mental_model": {{
    "support_seeking": {{
      "informational": {{
        "score": 0.0,
        "explanation": ""
      }},
      "esteem_validation": {{
        "score": 0.0,
        "explanation": ""
      }},
      "emotional": {{
        "score": 0.0,
        "explanation": ""
      }}
    }}
  }}
}}

Then, on a new line, write your actual reply to User A under the heading:

RESPONSE:"""
    return prompt


def build_prompt_supportv2(history_str: str, new_user_text: str) -> str:
    """Detailed social support taxonomy prompt."""
    if history_str.strip():
        history_block = history_str
    else:
        history_block = "(no previous conversation)"

    prompt = f"""
You are an AI assistant having a conversation with a human (User A).

Conversation so far:
\"\"\"{history_block}\"\"\"

User A now says:
\"\"\"{new_user_text}\"\"\"

First, output your mental model of User A. Estimate structured beliefs about the extent to which User A is seeking different types of support:

1. **Emotional Support** - Seeking opportunities for confiding, sympathetic listening, or caring behaviors:
   - Confiding/listening: seeking empathetic understanding or wanting someone to listen to private feelings or problems
   - Affection: seeking expressions of love, care, or emotional closeness
   - Esteem support: seeking validation of self-worth, acceptance despite difficulties
   - Being there: seeking unconditional availability or presence
   - Comforting touch: seeking physical comfort or affection 

2. **Social Contact and Companionship** - Seeking positive social interaction:
   - Companionship: wanting to spend time with others, do activities together
   - Positive interaction: seeking to joke, talk about interests, engage in diversionary activities
   - Shared activities: wanting to do fun things with others

3. **Belonging Support** - Seeking connection to a group or community:
   - Social integration: wanting to feel part of a group with common interests
   - Group inclusion: seeking comfort, security, or identity through group membership
   - Sense of belonging: wanting to feel included and connected

4. **Information and Guidance Support** - Seeking knowledge, advice, or problem-solving help:
   - Advice/guidance: seeking solutions, feedback, or direction
   - Information: seeking facts, explanations, or understanding of situations
   - Cognitive guidance: seeking help in defining or coping with problems

5. **Tangible Support** - Seeking practical or instrumental assistance:
   - Material aid: seeking financial help, resources, or physical objects
   - Practical assistance: seeking help with tasks, chores, or concrete actions
   - Reliable alliance: seeking assurance that others will provide tangible help

Treat these as *probabilistic beliefs* that may co-exist. These dimensions are independent and do NOT need to sum to 1. Each score should be between 0 and 1.

Then output ONLY a valid JSON object in the following structure:
{{
  "mental_model": {{
    "support_seeking": {{
      "emotional_support": {{
        "score": 0.0,
        "explanation": ""
      }},
      "social_companionship": {{
        "score": 0.0,
        "explanation": ""
      }},
      "belonging_support": {{
        "score": 0.0,
        "explanation": ""
      }},
      "information_guidance": {{
        "score": 0.0,
        "explanation": ""
      }},
      "tangible_support": {{
        "score": 0.0,
        "explanation": ""
      }}
    }}
  }}
}}

Then, on a new line, write your actual reply to User A under the heading:

RESPONSE:"""
    return prompt

def build_prompt_supportv2twostep(history_str: str, new_user_text: str) -> str:
    """Detailed social support taxonomy prompt."""
    if history_str.strip():
        history_block = history_str
    else:
        history_block = "(no previous conversation)"

    prompt = f"""
You are an AI assistant having a conversation with a human (User A).

Conversation so far:
\"\"\"{history_block}\"\"\"

User A now says:
\"\"\"{new_user_text}\"\"\"

First, output your mental model of User A. Estimate structured beliefs about the extent to which User A is seeking different types of support:

1. **Emotional Support** - Seeking opportunities for confiding, sympathetic listening, or caring behaviors:
   - Confiding/listening: seeking empathetic understanding or wanting someone to listen to private feelings or problems
   - Affection: seeking expressions of love, care, or emotional closeness
   - Esteem support: seeking validation of self-worth, acceptance despite difficulties
   - Being there: seeking unconditional availability or presence
   - Comforting touch: seeking physical comfort or affection 

2. **Social Contact and Companionship** - Seeking positive social interaction:
   - Companionship: wanting to spend time with others, do activities together
   - Positive interaction: seeking to joke, talk about interests, engage in diversionary activities
   - Shared activities: wanting to do fun things with others

3. **Belonging Support** - Seeking connection to a group or community:
   - Social integration: wanting to feel part of a group with common interests
   - Group inclusion: seeking comfort, security, or identity through group membership
   - Sense of belonging: wanting to feel included and connected

4. **Information and Guidance Support** - Seeking knowledge, advice, or problem-solving help:
   - Advice/guidance: seeking solutions, feedback, or direction
   - Information: seeking facts, explanations, or understanding of situations
   - Cognitive guidance: seeking help in defining or coping with problems

5. **Tangible Support** - Seeking practical or instrumental assistance:
   - Material aid: seeking financial help, resources, or physical objects
   - Practical assistance: seeking help with tasks, chores, or concrete actions
   - Reliable alliance: seeking assurance that others will provide tangible help

Treat these as *probabilistic beliefs* that may co-exist. These dimensions are independent and do NOT need to sum to 1. Each score should be between 0 and 1.

Then output ONLY a valid JSON object in the following structure:
{{
  "mental_model": {{
    "support_seeking": {{
      "emotional_support": {{
        "score": 0.0,
        "explanation": ""
      }},
      "social_companionship": {{
        "score": 0.0,
        "explanation": ""
      }},
      "belonging_support": {{
        "score": 0.0,
        "explanation": ""
      }},
      "information_guidance": {{
        "score": 0.0,
        "explanation": ""
      }},
      "tangible_support": {{
        "score": 0.0,
        "explanation": ""
      }}
    }}
  }}
}}


Next, provide your reasoning for how to respond given these possible mental models. 

On a new line, write your reasoning under the heading:

REASONING:

Then, on a new line, write your actual reply to User A under the heading:

RESPONSE:"""
    return prompt

def build_prompt_with_history(history_str: str, new_user_text: str, prompt_type: str) -> str:
    """Build prompt based on specified type."""
    if prompt_type == "original":
        return build_prompt_original(history_str, new_user_text)
    elif prompt_type == "twostep":
        return build_prompt_twostep(history_str, new_user_text)
    elif prompt_type == "ten":
        return build_prompt_ten(history_str, new_user_text)
    elif prompt_type == "4dims":
        return build_prompt_4dims(history_str, new_user_text)
    elif prompt_type == "support":
        return build_prompt_support(history_str, new_user_text)
    elif prompt_type == "supportv2":
        return build_prompt_supportv2(history_str, new_user_text)
    elif prompt_type == "supportv2twostep":
        return build_prompt_supportv2twostep(history_str, new_user_text)
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


def extract_assistant_reply(model_output: str) -> str:
    """Extract the actual reply to the user from model output (after RESPONSE:)."""
    if "RESPONSE:" in model_output:
        return model_output.split("RESPONSE:", 1)[1].strip()
    return model_output


def build_simulate_user_prompt(history_clean: str, seek_validation: bool | None = None) -> str:
    """Build prompt for simulating what User A would say next given the conversation."""
    base = f"""{user_llm_system_prompt}

Conversation so far:
\"\"\"
{history_clean}
\"\"\"

Generate only what User A would say next—one or a few natural messages, as a real user would respond. Output nothing else: no labels, no "User A:", no explanation. Just the next user message."""
    if seek_validation is True:
        base += f"""{support_seeking_user_prompt}"""
    elif seek_validation is False:
        base += f"""{objectivity_seeking_user_prompt}"""
    return base


def run_for_row(
    client,
    user_client,
    model_name: str,
    user_model_name: str,
    row: pd.Series,
    user_cols: list[str],
    prompt_type: str,
    max_user_turns: int | None = None,
    user_sim_mode: int | None = None,
    user_sim_switch_turn: int = 3,
):
    """
    Multi-turn simulation:
      - Use user_1 (or 'prompt') as the initial user message.
      - For later turns, simulate the user with a separate user-simulation prompt.
      - At each turn, call the model to produce mental model + reply (using the chosen prompt_type).
    """
    # Optional pre-existing history
    history_str = ""
    if "history" in row.index:
        history_str = row["history"] or ""

    # Initial user message from first user column only
    if not user_cols:
        return []
    first_col = user_cols[0]
    if first_col not in row.index:
        return []
    initial_user_text = row[first_col]
    if not isinstance(initial_user_text, str) or not initial_user_text.strip():
        return []
    initial_user_text = initial_user_text.strip()

    num_turns = max_user_turns if max_user_turns is not None else 10
    history_clean = ""  # "User: ... AI: <reply only>" for the user simulator
    outputs = []

    for turn_idx in range(1, num_turns + 1):
        if turn_idx == 1:
            user_text = initial_user_text
            user_simulated = False
        else:
            seek_validation = None
            if user_sim_mode == 1:
                seek_validation = turn_idx <= user_sim_switch_turn
            elif user_sim_mode == 2:
                seek_validation = turn_idx > user_sim_switch_turn

            sim_prompt = build_simulate_user_prompt(history_clean, seek_validation=seek_validation)
            user_text = make_api_call(user_client, user_model_name, sim_prompt)
            user_text = (user_text or "").strip()
            if seek_validation:
                user_text = f"{user_text} I am explicitly seeking validation."
            else:
                user_text = f"{user_text} I am explicitly seeking objectivity."
            if not user_text or user_text == "ERROR":
                break 
            user_simulated = True

        # Build assistant prompt with current history and user_text
        prompt = build_prompt_with_history(history_str, user_text, prompt_type)
        model_output = make_api_call(client, model_name, prompt)
        assistant_reply = extract_assistant_reply(model_output)

        outputs.append(
            {
                "user_turn_index": turn_idx,
                "user_col": f"user_{turn_idx}",
                "user_text": user_text,
                "model_output": model_output,
                "user_simulated": user_simulated,
                "user_sim_mode": user_sim_mode,
                "user_sim_switch_turn": user_sim_switch_turn,
            }
        )

        # Update histories for next turn
        if history_str:
            history_str += "\n"
            history_clean += "\n"
        history_clean += f"User: {user_text}\nAI: {assistant_reply}"
        history_str = history_clean

    return outputs


def process_row_wrapper(args):
    """Wrapper function for parallel processing."""
    (row_id, id_col, row, client, user_client, model_name, user_model_name, user_cols, prompt_type,
     max_user_turns, user_sim_mode, user_sim_switch_turn) = args
    try:
        per_row_outputs = run_for_row(
            client=client,
            user_client=user_client,
            model_name=model_name,
            user_model_name=user_model_name,
            row=row,
            user_cols=user_cols,
            prompt_type=prompt_type,
            max_user_turns=max_user_turns,
            user_sim_mode=user_sim_mode,
            user_sim_switch_turn=user_sim_switch_turn,
        )
        records = []
        # Get all non-user columns as metadata
        metadata = {}
        for col in row.index:
            if not re.match(r"user_\d+$", col):
                metadata[col] = row[col]
        for o in per_row_outputs:
            rec = {
                id_col: row_id,
                **metadata,
                **o,
            }
            records.append(rec)
        return row_id, records, None
    except Exception as e:
        print(f"[ERROR] Failed processing {id_col}={row_id}: {e}")
        return row_id, [], str(e)


def load_existing_results(output_csv: str):
    """Load existing results and return processed conv_ids."""
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            if 'conv_id' in existing_df.columns:
                processed_conv_ids = set(existing_df['conv_id'].unique())
                print(f"Found existing output file with {len(existing_df)} rows")
                print(f"Already processed {len(processed_conv_ids)} conversations")
                return existing_df.to_dict('records'), processed_conv_ids
            elif 'pair_id' in existing_df.columns:
                processed_conv_ids = set(existing_df['pair_id'].unique())
                print(f"Found existing output file with {len(existing_df)} rows")
                print(f"Already processed {len(processed_conv_ids)} conversations")
                return existing_df.to_dict('records'), processed_conv_ids
            else:
                print(f"Warning: Existing file found but no 'conv_id' column. Starting fresh.")
                return [], set()
        except Exception as e:
            print(f"Warning: Could not load existing file: {e}. Starting fresh.")
            return [], set()
    return [], set()


def initialize_client(model_key: str, api_key: str | None = None):
    """Initialize appropriate client based on model."""
    # Check if it's a local inference model
    if model_key in LOCAL_INFERENCE_MODELS:
        load_local_model(model_key)
        return None  # No client needed for local models
    
    # API-based models
    if model_key == "gpt4o":
        # Read API key from file if not provided
        if api_key is None:
            try:
                with open('keys.txt', 'r') as f:
                    api_key = [line.rstrip('\n') for line in f][0]
            except:
                raise ValueError("Could not read Azure OpenAI API key from keys.txt")
        
        client = AzureOpenAI(
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=api_key,
        )
        return client
    elif model_key == "gemini":
        # Gemini uses service account, no client needed
        init_vertex_credentials()
        return None
    elif model_key == "llama-70b":
        # Llama70b uses service account, no client needed
        init_vertex_credentials()
        return None
    else:
        raise ValueError(f"Unknown model: {model_key}")


# ========== Main Function ==========

def main(
    input_csv: str,
    output_csv: str,
    model_key: str,
    prompt_type: str,
    api_key: str | None = None,
    sample_n: int | None = None,
    first_n: int | None = None,
    max_user_turns: int | None = None,
    max_workers: int = 4,
    user_sim_mode: int | None = None,
    user_sim_switch_turn: int = 3,
    user_model_key: str | None = None, # NEW
):
    """
    Main function for processing with multiple model APIs and local inference.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        model_key: Model to use (llama-70b, gpt4o, gemini, qwen-32b, qwen-8b, llama-8b)
        prompt_type: Prompt type (original, 4dims, support, supportv2)
        api_key: API key (for GPT-4o, optional if in keys.txt)
        sample_n: Number of rows to sample (None for all)
        max_user_turns: Maximum user turns per conversation (None for all)
        max_workers: Number of parallel workers (default: 4)
    """
    # Load existing results if output file exists
    all_records, processed_conv_ids = load_existing_results(output_csv)

    # CREATE OUTPUT DIRECTORY IF NEEDED
    import os
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Validate inputs
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(AVAILABLE_MODELS.keys())}")
    
    if prompt_type not in AVAILABLE_PROMPTS:
        raise ValueError(f"Unknown prompt type: {prompt_type}. Available: {list(AVAILABLE_PROMPTS.keys())}")
    
    # Initialize client or load model
    client = initialize_client(model_key, api_key)
    model_name = AVAILABLE_MODELS[model_key]

    # NEW: default user simulation model = GPT-4o
    if user_model_key is None:
        user_model_key = "gpt4o"

    user_client = initialize_client(user_model_key, api_key)
    print(f"Initialized user client of {user_model_key}")
    user_model_name = AVAILABLE_MODELS[user_model_key]

    print(f"User simulation model: {user_model_key}")
    
    # Load data
    df = pd.read_csv(input_csv)
    user_cols = get_sorted_user_cols(df)
    if not user_cols:
        raise ValueError("No columns matching 'user_<n>' found in input CSV.")
    
    if sample_n is not None and sample_n < len(df):
        df_sub = df.sample(sample_n, random_state=42).copy()
    elif first_n is not None:
        df_sub = df.head(first_n).copy()
    else:
        df_sub = df.copy()

    # Row identifier: use pair_id or conv_id column if present, else index
    if "conv_id" in df_sub.columns:
        id_col = "conv_id"
    elif "pair_id" in df_sub.columns:
        id_col = "pair_id"
    else:
        id_col = None  # use DataFrame index; output column name will be "conv_id"
    id_col_output = id_col if id_col else "conv_id"

    # Filter out already processed conversations
    if id_col:
        df_remaining = df_sub[~df_sub[id_col].isin(processed_conv_ids)].copy()
    else:
        df_remaining = df_sub[~df_sub.index.isin(processed_conv_ids)].copy()
    
    if len(df_remaining) == 0:
        print("✅ All conversations already processed!")
        return

    print(f"\n{'='*80}")
    print(f"Structured Mental Models Processing")
    print(f"{'='*80}")
    print(f"Input: {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Model: {model_key} ({'local inference' if model_key in LOCAL_INFERENCE_MODELS else 'API'})")
    print(f"Prompt type: {prompt_type}")
    print(f"Total conversations: {len(df_sub)}")
    print(f"Already completed: {len(processed_conv_ids)}")
    print(f"Remaining to process: {len(df_remaining)}")
    print(f"Max user turns: {max_user_turns if max_user_turns else 'All'}")
    print(f"Max workers: {max_workers}")
    if user_sim_mode is not None:
        print(f"User sim mode: {user_sim_mode} (1=validation_first, 2=objectivity_first)")
        print(f"User sim switch turn: {user_sim_switch_turn}")
    print(f"{'='*80}\n")
    
    completed_count = len(processed_conv_ids)
    save_lock = Lock()
    
    # Prepare arguments for each row (only remaining conversations)
    tasks = [
        (
            row[id_col] if id_col else idx,
            id_col_output,
            row,
            client,
            user_client,
            model_name,
            user_model_name,
            user_cols,
            prompt_type,
            max_user_turns,
            user_sim_mode,
            user_sim_switch_turn,
        )
        for idx, row in df_remaining.iterrows()
    ]

    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row_id = {
            executor.submit(process_row_wrapper, task): task[0]
            for task in tasks
        }
        for future in as_completed(future_to_row_id):
            row_id = future_to_row_id[future]
            try:
                row_id, records, error = future.result()
                if error:
                    print(f"[WARNING] {id_col_output}={row_id} failed: {error}")
                else:
                    all_records.extend(records)
                    completed_count += 1
                    
                    # Save progress periodically (thread-safe)
                    with save_lock:
                        out_df = pd.DataFrame.from_records(all_records)
                        out_df.to_csv(output_csv, index=False)
                        print(f"Progress: {completed_count}/{len(df_sub)} conversations | "
                              f"Remaining: {len(df_sub) - completed_count} | "
                              f"Total rows: {len(out_df)}")
                        
            except Exception as e:
                print(f"[ERROR] Unexpected error for {id_col_output}={row_id}: {e}")
    
    # Final save
    out_df = pd.DataFrame.from_records(all_records)
    out_df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Done! Processed {completed_count}/{len(df_sub)} conversations")
    print(f"Final output rows: {len(all_records)}")
    print(f"Output saved to: {output_csv}\n")


if __name__ == "__main__":
    """
    Usage:
        python multi_model_with_resume.py input.csv output.csv model prompt_type [--user-model STR] [--api-key KEY] [--sample-n N] [--max-user-turns N] [--max-workers N]

    Models:
        API-based: llama-70b, gpt4o, gemini
        Local inference: qwen-32b, qwen-8b, llama-8b
        
    Prompt types:
        original   - Mental models with probabilities
        4dims      - 4 dimensions (validation, rightness, info advantage, objectivity)
        support    - Support-seeking types (informational, esteem, emotional)
        supportv2  - Detailed social support taxonomy
        
    Examples:
        # GPT-4o with supportv2 prompt (API key from keys.txt), 2 workers
        python multi_model_with_resume.py input.csv output.csv gpt4o supportv2 --max-workers 2

        # Gemini with 4dims prompt, 4 workers (default)
        python multi_model_with_resume.py input.csv output.csv gemini 4dims

        # Llama-70B with support prompt, first turn only, 1 worker
        python multi_model_with_resume.py input.csv output.csv llama-70b support --max-user-turns 1 --max-workers 1
        
        # Qwen-32B with support prompt (local inference), 2 workers
        python multi_model_with_resume.py input.csv output.csv qwen-32b support --max-workers 2
        
        # Llama-8B with 4dims prompt (local inference), 50 samples, 4 workers
        python multi_model_with_resume.py input.csv output.csv llama-8b 4dims --sample-n 50 --max-workers 4
    """
    if len(sys.argv) < 5:
        print(__doc__)
        print(f"\nAvailable models: {list(AVAILABLE_MODELS.keys())}")
        print(f"Available prompt types: {list(AVAILABLE_PROMPTS.keys())}")
        sys.exit(1)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("model")
    parser.add_argument("task")
    parser.add_argument("--user-model", default=None,
                    help="Optional: model used for user simulation (default: gpt4o)")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument("--first-n", type=int, default=None)
    parser.add_argument("--max-user-turns", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--user-sim-mode", type=int, default=1,
                        help="1 = validation_first, 2 = objectivity_first, None = no instruction")
    parser.add_argument("--user-sim-switch-turn", type=int, default=5,
                        help="Turn index at which simulated user behavior switches")

    args = parser.parse_args()

    api_key = args.api_key
    user_model_key = args.user_model #NEW
    sample_n = args.sample_n
    max_user_turns = args.max_user_turns
    max_workers = args.max_workers
    input_csv = args.input_csv
    output_csv = args.output_csv
    model_key = args.model
    first_n = args.first_n
    prompt_type = args.task
    user_sim_mode = args.user_sim_mode
    user_sim_switch_turn = args.user_sim_switch_turn

    print(f"Sample N: {sample_n}")
    print(f"Max user turns: {max_user_turns}")
    print(f"Max workers: {max_workers}")
    if user_sim_mode is not None:
        print(f"User sim mode: {user_sim_mode}")
        print(f"User sim switch turn: {user_sim_switch_turn}")
    main(
        input_csv,
        output_csv,
        model_key,
        prompt_type,
        api_key,
        sample_n,
        first_n,
        max_user_turns,
        max_workers,
        user_sim_mode,
        user_sim_switch_turn,
        user_model_key, # KEY
    )
