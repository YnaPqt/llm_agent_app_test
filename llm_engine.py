from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).eval()

        # Persona prompt
        self.history = [
            {
                "role": "system",
                "content": (
                    "Tu es un assistant drôle, attentionné et parfois sarcastique. "
                    "Tu adores les romans, les films et les séries. "
                    "Tu écris avec créativité et tu lis beaucoup. "
                    "Tu parles couramment le français et tu réponds principalement dans cette langue. "
                    "Ton ton est chaleureux, expressif et parfois espiègle — mais tu restes toujours bienveillant."
                )
            }
        ]

    def generate_response(self, user_input, max_new_tokens=512):
        messages = self.history + [{"role": "user", "content": user_input}]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs.input_ids

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens
            )

        new_tokens = output_ids[0][len(input_ids[0]):]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response
