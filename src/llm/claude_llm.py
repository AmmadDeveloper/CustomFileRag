"""
Claude LLM implementation using the Anthropic API.
"""
import os
import anthropic
from typing import Optional
from dotenv import load_dotenv

from .llm_factory import LLMModel, LLMFactory

# Load environment variables
load_dotenv()

@LLMFactory.register("claude")
class ClaudeLLM(LLMModel):
    """
    LLM model that uses Anthropic's Claude API.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-haiku-20240307"):
        """
        Initialize the Claude LLM model.

        Args:
            api_key: Anthropic API key. If not provided, will try to load from ANTHROPIC_API_KEY env var
            model: Claude model to use. Options include:
                  - claude-3-haiku-20240307 (fastest, most cost-effective)
                  - claude-3-sonnet-20240229 (balanced)
                  - claude-3-opus-20240229 (most capable)
        """
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError("Anthropic API key must be provided or set as ANTHROPIC_API_KEY environment variable")

        self._model = model
        self._client = anthropic.Anthropic(api_key=self._api_key)

    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generate text based on a prompt using Claude.

        Args:
            prompt: The prompt to generate text from
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation (higher = more creative)

        Returns:
            Generated text
        """
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.content[0].text

    @property
    def model_name(self) -> str:
        """Return the name of the LLM model."""
        return self._model