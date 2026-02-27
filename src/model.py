"""
LLM API wrapper for OpenAI and Anthropic models.
"""

import os
from typing import Optional
import openai
import anthropic


class LLMWrapper:
    """
    Unified wrapper for LLM API calls.
    Supports OpenAI and Anthropic providers.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 64,
    ):
        """
        Initialize LLM wrapper.

        Args:
            provider: "openai" or "anthropic"
            model_name: Model identifier (e.g., "gpt-4o-mini", "claude-3-5-sonnet-20241022")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize client
        if self.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = openai.OpenAI(api_key=api_key)

        elif self.provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            self.client = anthropic.Anthropic(api_key=api_key)

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            temperature: Override default temperature (optional)
            max_tokens: Override default max_tokens (optional)

        Returns:
            Generated text response
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=tokens,
            )
            return response.choices[0].message.content.strip()

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=tokens,
                temperature=temp,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def batch_generate(
        self,
        prompts: list[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> list[str]:
        """
        Generate responses for multiple prompts.
        Currently processes sequentially (can be optimized with async in future).

        Args:
            prompts: List of input prompts
            temperature: Override default temperature (optional)
            max_tokens: Override default max_tokens (optional)

        Returns:
            List of generated responses
        """
        return [self.generate(prompt, temperature, max_tokens) for prompt in prompts]
