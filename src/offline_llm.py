"""
100% Local Offline LLM Reasoning & Conversational Engine
FPGA-Based Multilingual AI Receptionist

Provides strictly local inference:
1. Primary: Local Ollama server (e.g. Qwen2.5:1.5b / Llama3.2:1b / Phi3) running locally on host
2. Fallback: Local Deterministic ReceptionistAgent with Plan-Act-Verify loop and zero hallucinations
3. Zero Cloud Calls, Zero API Keys, 100% Offline
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, List, Tuple

from src.language_config import get_language_profile, normalize_language_name
from src.agent import get_receptionist_agent, ReceptionistAgent


class OfflineLLMEngine:
    """100% Local Offline Multilingual LLM and Reasoning Engine."""

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        preferred_model: str = "qwen2.5:1.5b"
    ):
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.preferred_model = preferred_model
        self.agent = get_receptionist_agent()

    def check_ollama_status(self) -> Dict[str, Any]:
        """
        Checks if local Ollama daemon is running and lists locally installed models.
        """
        status = {
            "online": False,
            "models": [],
            "preferred_model": self.preferred_model,
            "has_preferred_model": False,
            "base_url": self.ollama_base_url,
            "offline_ready": False
        }
        try:
            req = urllib.request.Request(
                f"{self.ollama_base_url}/api/tags",
                headers={"User-Agent": "MultilingualReceptionist/1.0"}
            )
            with urllib.request.urlopen(req, timeout=1.5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    models = [m.get("name", "") for m in data.get("models", [])]
                    status["online"] = True
                    status["models"] = models
                    status["has_preferred_model"] = any(self.preferred_model in m for m in models)
                    status["offline_ready"] = True
        except Exception:
            status["online"] = False
            status["offline_ready"] = True  # Deterministic agent is always ready offline

        return status

    def generate_response(
        self,
        user_text: str,
        language: str = "English",
        history: Optional[List[Dict[str, str]]] = None,
        timeout: int = 15
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generates grounded receptionist response locally.

        Execution Priority:
        1. Run local ReceptionistAgent Plan-Act-Verify loop for grounded facts and verified tool results.
        2. If Ollama is running, optionally use it to polish phrasing in Tamil/Hindi/English with context.
        3. Return grounded local answer with full action trace.

        Returns:
            response_text (str), metadata (dict)
        """
        start_time = time.time()
        profile = get_language_profile(language)

        # 1. Execute deterministic local reasoning agent (Plan-Act-Verify)
        grounded_answer, trace_steps, agent_meta = self.agent.run_agent_loop(
            visitor_text=user_text,
            language=language,
            use_llm=False
        )

        action_trace = [step.to_dict() if hasattr(step, "to_dict") else step for step in trace_steps]
        goal_analysis = {
            "goal": agent_meta.get("goal", "general_query"),
            "entities": agent_meta.get("active_entities", {}),
            "step_count": agent_meta.get("step_count", len(trace_steps))
        }

        # 2. Check if local Ollama model can provide natural conversational flow
        ollama_status = self.check_ollama_status()
        if ollama_status["online"] and ollama_status["models"]:
            active_model = self.preferred_model if ollama_status["has_preferred_model"] else ollama_status["models"][0]
            try:
                system_prompt = (
                    f"You are the professional Multilingual AI Receptionist at RMK Engineering College & FPGA Laboratory. "
                    f"You must communicate strictly in {profile.name}. "
                    f"Answer accurately based ONLY on the following verified college facts: '{grounded_answer}'. "
                    f"Do not invent facts. Keep your response concise (1-2 sentences), courteous, and natural for speech."
                )

                prompt_payload = {
                    "model": active_model,
                    "prompt": f"Visitor: {user_text}\nReceptionist ({profile.name}):",
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.8,
                        "num_predict": 128
                    }
                }

                req = urllib.request.Request(
                    f"{self.ollama_base_url}/api/generate",
                    data=json.dumps(prompt_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )

                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if resp.status == 200:
                        llm_data = json.loads(resp.read().decode("utf-8"))
                        polished_text = llm_data.get("response", "").strip()
                        if polished_text and len(polished_text) > 4:
                            latency_ms = (time.time() - start_time) * 1000
                            return polished_text, {
                                "provider": f"Ollama Local LLM ({active_model})",
                                "grounded_base": grounded_answer,
                                "action_trace": action_trace,
                                "goal_analysis": goal_analysis,
                                "language": profile.name,
                                "latency_ms": latency_ms,
                                "offline": True
                            }
            except Exception:
                pass  # Fall back directly to the verified agent response

        # 3. Return verified agent answer (100% Local, Zero Hallucinations)
        latency_ms = (time.time() - start_time) * 1000
        return grounded_answer, {
            "provider": "Local Receptionist Agent (Grounded Plan-Act-Verify)",
            "action_trace": action_trace,
            "goal_analysis": goal_analysis,
            "language": profile.name,
            "latency_ms": latency_ms,
            "offline": True
        }


# Singleton instance
_offline_llm_engine = None

def get_offline_llm_engine(preferred_model: str = "qwen2.5:1.5b") -> OfflineLLMEngine:
    global _offline_llm_engine
    if _offline_llm_engine is None:
        _offline_llm_engine = OfflineLLMEngine(preferred_model=preferred_model)
    return _offline_llm_engine
