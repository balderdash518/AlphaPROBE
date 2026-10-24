import backoff  # for exponential backoff
import openai
import json
import asyncio
import re
import time
from typing import Any
from retrying import retry

@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIConnectionError))
def completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)

async def dispatch_openai_chat_requests(
    client,
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
        stop_words: List of words to stop the model from generating.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        client.chat.completions.create(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop = stop_words
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


THINK_PATTERN = re.compile(r"<think>([\s\S]*?)(?:</think>|$)")


def split_thinking(text: str) -> tuple[str, str]:
    """Split Qwen3-style <think>...</think> blocks out of the answer text."""
    reasoning = "\n".join(m.strip() for m in THINK_PATTERN.findall(text))
    # Some servers drop the opening <think> tag and only emit the closing one.
    if not reasoning and "</think>" in text:
        reasoning, text = text.split("</think>", 1)
    return THINK_PATTERN.sub("", text).strip(), reasoning.strip()


def usage_details(usage) -> dict[str, int]:
    """Langfuse usage buckets from a chat.completions usage object."""
    if usage is None:
        return {}
    details = {"input": usage.prompt_tokens or 0, "output": usage.completion_tokens or 0}
    reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
    if reasoning:
        details["output_reasoning"] = reasoning
    return details


class OpenAIModel:
    def __init__(self, model_name, stop_words, max_new_tokens, max_tokens=None, stream=False) -> None:
        self.model_name = model_name
        self.stop_words = stop_words
        self.max_new_tokens = max_new_tokens
        # Sent with chat_generate only when set; None keeps the server default.
        self.max_tokens = max_tokens
        self.stream = stream
        self.last_call: dict[str, Any] = {}

    def _stream_completion(self, client, **kwargs):
        stream = completions_with_backoff(client, stream=True, stream_options={"include_usage": True}, **kwargs)
        content, reasoning, finish_reason, usage = [], [], None, None
        for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta is not None:
                if delta.content:
                    content.append(delta.content)
                extra = getattr(delta, "reasoning_content", None)
                if extra:
                    reasoning.append(extra)
            if choice.finish_reason:
                finish_reason = choice.finish_reason
        if finish_reason is None:
            # The connection closed before the answer finished; let @retry try again.
            raise RuntimeError("LLM stream ended without finish_reason")
        return "".join(content), "".join(reasoning), finish_reason, usage

    @retry(stop_max_attempt_number=10, wait_fixed=2000)
    def chat_generate(self, client, system_prompt:str, user_prompt:str, temperature = 0.0):
        kwargs = dict(
                model = self.model_name,
                messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                temperature = temperature,
                top_p = 1,
        )
        if self.stop_words:
            kwargs["stop"] = self.stop_words
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        started = time.perf_counter()
        while True:
            try:
                if self.stream:
                    raw_text, reasoning, finish_reason, usage = self._stream_completion(client, **kwargs)
                else:
                    response = completions_with_backoff(client, **kwargs)
                    message = response.choices[0].message
                    raw_text = message.content or ""
                    reasoning = getattr(message, "reasoning_content", None) or ""
                    finish_reason = response.choices[0].finish_reason
                    usage = response.usage
                break
            except openai.BadRequestError as exc:
                # max_tokens is sized close to the context window; a longer prompt than
                # expected overflows it, so shrink the output budget and try again.
                if kwargs.get("max_tokens", 0) <= 4096 or not re.search(r"context|max_tokens|too long", str(exc), re.I):
                    raise
                kwargs["max_tokens"] -= 2048
                print(f"[llm] prompt + max_tokens exceeds the context window; retrying with max_tokens={kwargs['max_tokens']}")
        generated_text, inline_reasoning = split_thinking(raw_text)
        self.last_call = {
            "reasoning": "\n".join(r for r in (reasoning.strip(), inline_reasoning) if r),
            "finish_reason": finish_reason,
            "usage_details": usage_details(usage),
            "latency_s": round(time.perf_counter() - started, 2),
        }
        return generated_text, finish_reason

    def generate(self, client, input_string, temperature = 0.0):
        return self.chat_generate(client, input_string, temperature)
    
    def batch_chat_generate(self, client, messages_list, temperature = 0.0):
        open_ai_messages_list = []
        system_prompt = "You are a helpful assistant. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem."
        for message in messages_list:
            open_ai_messages_list.append(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]
            )
        predictions = asyncio.run(
            dispatch_openai_chat_requests(
                    client, open_ai_messages_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        finish_reason = [x.choices[0].finish_reason.strip() for x in predictions]
        return [x.choices[0].message.content.strip() for x in predictions]

    def batch_generate(self, client, messages_list, temperature = 0.0):
        return self.batch_chat_generate(client, messages_list, temperature)