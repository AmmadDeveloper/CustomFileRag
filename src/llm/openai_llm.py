"""
OpenAI LLM implementation using the OpenAI API.
"""
import os
import requests
import json
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from .llm_factory import LLMModel, LLMFactory

# Load environment variables
load_dotenv()

@LLMFactory.register("openai")
class OpenAILLM(LLMModel):
    """
    LLM model that uses OpenAI's API.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the OpenAI LLM model.

        Args:
            api_key: OpenAI API key. If not provided, will try to load from OPENAI_API_KEY env var
            model: OpenAI model to use. Options include:
                  - o1-preview (o1-nano model)
                  - gpt-4o
                  - gpt-4-turbo
                  - gpt-3.5-turbo
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")

        self._model = model

    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generate text based on a prompt using OpenAI.

        Args:
            prompt: The prompt to generate text from
            max_tokens: Maximum number of tokens to generate
                        (For o1 models, this is passed as max_completion_tokens)
            temperature: Temperature for generation (higher = more creative)
                         (Note: o1 models only support the default temperature value)

        Returns:
            Generated text
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }

        # Different models have different parameter requirements
        data = {
            "model": self._model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        # o1 models use max_completion_tokens and don't support custom temperature
        if self._model.startswith("o1-"):
            data["max_completion_tokens"] = max_tokens
            # o1 models only support default temperature (1)
        else:
            # Other models use max_tokens and support custom temperature
            data["max_tokens"] = max_tokens
            data["temperature"] = temperature

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )

        if response.status_code != 200:
            raise Exception(f"Error from OpenAI API: {response.text}")

        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]

    @property
    def model_name(self) -> str:
        """Return the name of the LLM model."""
        return self._model
