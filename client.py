import argparse
import yaml
import sys
import re
import asyncio
from openai import AsyncOpenAI

class AsyncLLMClient:
    # Regex to match thinking blocks:
    # 1. <think>...</think> (DeepSeek/Standard)
    # 2. <thought>...</thought> (Variant)
    # 3. <thinking>...</thinking> (Variant)
    # 4. <|thinking|>...</|thinking|> (Qwen3 Special Token Format)
    # We use named groups to ensure the closing tag matches the opening tag style if needed, 
    # but for simplicity and robustness, we match the content greedily between any valid pair.
    # Note: \| matches the literal pipe character.
    THINKING_PATTERN = re.compile(
        r"<(think|thought|thinking|\|thinking\|)>(.*?)</\1>", 
        re.DOTALL | re.IGNORECASE
    )

    def __init__(self, base_url, api_key, model):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def _parse_thinking(self, text):
        """Extracts thinking content and final response from text."""
        match = self.THINKING_PATTERN.search(text)
        if match:
            thinking = match.group(2).strip()
            # Remove the thinking block from the text to get the final response
            final = self.THINKING_PATTERN.sub("", text).strip()
            return thinking, final
        return None, text

    def _print_block(self, title, content):
        print(f"\n{'='*10} {title} {'='*10}")
        print(content)

    async def generate_batch(self, messages, params, extra_body=None):
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                extra_body=extra_body,
                **params
            )
            content = response.choices[0].message.content
            
            thinking, final = self._parse_thinking(content)
            
            if thinking:
                self._print_block("THINKING PROCESS", thinking)
                self._print_block("FINAL RESPONSE", final)
            else:
                self._print_block("RESPONSE", content)
                
            # if hasattr(response, 'usage'):
            #     print(f"Usage: {response.usage}")

        except Exception as e:
            print(f"Error: {e}")

    async def generate_stream(self, messages, params, extra_body=None):
        full_content = ""
        display_buffer = "" 
        state = "NORMAL" 
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                extra_body=extra_body,
                **params
            )
            
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if not content:
                    continue
                
                full_content += content
                display_buffer += content
                
                if state == "NORMAL":
                    tag_found = None
                    # Check for various opening tags
                    if "<think>" in display_buffer:
                        tag_found = "<think>"
                    elif "<thought>" in display_buffer:
                        tag_found = "<thought>"
                    elif "<thinking>" in display_buffer:
                        tag_found = "<thinking>"
                    elif "<|thinking|>" in display_buffer:
                        tag_found = "<|thinking|>"
                        
                    if tag_found:
                        pre, post = display_buffer.split(tag_found, 1)
                        if pre.strip():
                            print(pre, end="", flush=True)
                        print(f"\n{'='*20} THINKING PROCESS {'='*20}\n")
                        state = "THINKING"
                        display_buffer = post 
                    elif "<" in display_buffer:
                        continue
                    else:
                        print(display_buffer, end="", flush=True)
                        display_buffer = ""
                        
                elif state == "THINKING":
                    close_tag_found = None
                    # Check for various closing tags
                    if "</think>" in display_buffer:
                        close_tag_found = "</think>"
                    elif "</thought>" in display_buffer:
                        close_tag_found = "</thought>"
                    elif "</thinking>" in display_buffer:
                        close_tag_found = "</thinking>"
                    elif "</|thinking|>" in display_buffer: # Assuming consistent closing
                        close_tag_found = "</|thinking|>"
                    elif "<|/thinking|>" in display_buffer: # Alternative Qwen style
                        close_tag_found = "<|/thinking|>"
                        
                    if close_tag_found:
                        think_content, post = display_buffer.split(close_tag_found, 1)
                        print(think_content, end="", flush=True)
                        print(f"\n\n{'='*20} FINAL RESPONSE {'='*20}\n")
                        state = "DONE_THINKING"
                        display_buffer = post 
                    elif "</" in display_buffer or "<|" in display_buffer:
                        continue
                    else:
                        print(display_buffer, end="", flush=True)
                        display_buffer = ""
                        
                elif state == "DONE_THINKING":
                    print(display_buffer, end="", flush=True)
                    display_buffer = ""

            if display_buffer:
                print(display_buffer, end="", flush=True)
            print()
            
        except Exception as e:
            print(f"Error during stream: {e}")

def load_template(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

async def main():
    parser = argparse.ArgumentParser(description="Async OpenAI Client for vLLM")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="API Base URL")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API Key")
    parser.add_argument("--model", type=str, required=True, help="Model ID (must match server)")
    parser.add_argument("--template", type=str, default="templates/chat.yaml", help="Path to chat template YAML")
    parser.add_argument("--prompt", type=str, help="Direct user prompt from CLI")
    parser.add_argument("--stream", action="store_true", help="Enable streaming")
    parser.add_argument("--think", action="store_true", help="Force enable thinking logic (sets enable_thinking=True and adds /think)")
    parser.add_argument("--no-think", action="store_true", help="Force disable thinking logic (sets enable_thinking=False and adds /no_think)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum tokens to generate (default: 4096)")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    
    args = parser.parse_args()

    config = load_template(args.template)
    messages = []
    if "system" in config:
        messages.append({"role": "system", "content": config["system"]})
    
    extra_body = {}
    if args.no_think:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    elif args.think:
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}

    if args.prompt:
        prompt_text = args.prompt
        if args.no_think:
            prompt_text = f"/no_think {prompt_text}"
        elif args.think:
            prompt_text = f"/think {prompt_text}"
        messages.append({"role": "user", "content": prompt_text})
    elif "messages" in config:
        msgs = config["messages"]
        if args.no_think:
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i]["role"] == "user":
                    msgs[i]["content"] = f"/no_think {msgs[i]['content']}"
                    break
        elif args.think:
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i]["role"] == "user":
                    msgs[i]["content"] = f"/think {msgs[i]['content']}"
                    break
        messages.extend(msgs)
    else:
        print("Error: No prompt provided.")
        sys.exit(1)
        
    params = config.get("parameters", {})
    if args.max_tokens:
        params["max_tokens"] = args.max_tokens
    if args.temperature:
        params["temperature"] = args.temperature

    client = AsyncLLMClient(args.base_url, args.api_key, args.model)
    
    if args.stream:
        await client.generate_stream(messages, params, extra_body=extra_body)
    else:
        await client.generate_batch(messages, params, extra_body=extra_body)

if __name__ == "__main__":
    asyncio.run(main())
