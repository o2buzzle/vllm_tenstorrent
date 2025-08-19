from vllm import ModelRegistry
import os


def register_models_tt():
    llama_text_version = os.getenv("TT_LLAMA_TEXT_VER", "tt_transformers")
    if llama_text_version == "tt_transformers":
        path_llama_text = \
            "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM"
    elif llama_text_version == "llama3_70b_galaxy":
        path_llama_text = \
            "models.demos.llama3_70b_galaxy.tt.generator_vllm:LlamaForCausalLM"
    elif llama_text_version == "llama2_70b":
        path_llama_text = \
            "models.demos.t3000.llama2_70b.tt.generator_vllm:TtLlamaForCausalLM"
    else:
        raise ValueError(
            f"Unsupported TT Llama version: {llama_text_version}, "
            "pick one of [tt_transformers, llama3_70b_galaxy, llama2_70b]")

    # Llama3.1/3.2 - Text
    ModelRegistry.register_model("TTLlamaForCausalLM", path_llama_text)

    # Llama3.2 - Vision
    ModelRegistry.register_model(
        "TTMllamaForConditionalGeneration",
        "models.tt_transformers.tt.generator_vllm:MllamaForConditionalGeneration"
    )

    # Qwen2.5 - Text
    path_qwen_text = "models.tt_transformers.tt.generator_vllm:QwenForCausalLM"
    ModelRegistry.register_model("TTQwen2ForCausalLM", path_qwen_text)
    ModelRegistry.register_model("TTQwen3ForCausalLM", path_qwen_text)

    # Mistral
    ModelRegistry.register_model(
        "TTMistralForCausalLM",
        "models.tt_transformers.tt.generator_vllm:MistralForCausalLM")

    ModelRegistry.register_model(
        "TTQwen2_5_VLForConditionalGeneration",
        "models.demos.qwen25_vl.tt.generator_vllm:Qwen2_5_VLForConditionalGeneration"
    )


def check_tt_model_supported(model):
    supported_models = [
        "meta-llama/Llama-3.1-70B",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "meta-llama/Llama-3.3-70B",
        "meta-llama/Llama-3.3-70B-Instruct",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-Coder-32B",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2.5-72B",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ]
    assert model in supported_models, f"Invalid model: {model}"


def generate_env(model):
    current_env = os.environ.copy()

    # We can auto-update Qwen
    if model.startswith("Qwen/"):
        current_env["HF_MODEL"] = model

    # Not for Llama since they require LLAMA_DIR (impossible to do)
    if model.startswith("meta-llama/"):
        if "LLAMA_DIR" not in current_env:
            raise ValueError(
                "LLAMA_DIR environment variable is required for Llama models. "
                "Please set it to the Meta checkpoint-style weights.")

    os.environ.update(current_env)
