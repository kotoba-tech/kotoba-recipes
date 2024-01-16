from transformers import AutoTokenizer, LlamaTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from megatron_lm.megatron.global_vars import get_args


def get_tokenizer(tokenizer_path: str) -> PreTrainedTokenizer | LlamaTokenizer:
    args = get_args()

    if "Llama" in tokenizer_path:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer  # type: ignore
    elif "Mistral" in tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer  # type: ignore
    elif "Mixtral" in tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer  # type: ignore
    elif "calm2-7b" in tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        return tokenizer  # type: ignore
    elif "japanese-stablelm-base-alpha-7b" in args.base_model:
        tokenizer = LlamaTokenizer.from_pretrained(
            "novelai/nerdstash-tokenizer-v1",
            additional_special_tokens=['▁▁']
        )

        return tokenizer  # type: ignore
    elif "stockmark-13b" in tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path
        )

        return tokenizer  # type: ignore
    elif "plamo-13b" in tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )

        return tokenizer  # type: ignore
    elif "llm-jp-13b-v1.0" in tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path
        )

        return tokenizer  # type: ignore
    elif "ELYZA-japanese-Llama-2-7b" in tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path
        )

        return tokenizer  # type: ignore
    elif "japanese-stablelm-base-ja_vocab-beta-7b" in tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path
        )

        return tokenizer  # type: ignore
    elif "japanese-stablelm-base-beta" in tokenizer_path:
        tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_path,
        )

        return tokenizer  # type: ignore
    else:
        raise NotImplementedError(
            f"Tokenizer {tokenizer_path} is not supported. Please use Llama or Mistral."
        )
