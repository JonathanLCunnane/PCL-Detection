import json
import os
import torch
from transformers import MarianMTModel, MarianTokenizer
from logging import getLogger

LOG = getLogger(__name__)


_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _cache_path(pivot_lang: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"bt_cache_{pivot_lang}.json")


def backtranslate(
    texts: list[str],
    pivot_lang: str = "fr",
    batch_size: int = 16,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.92,
    device: torch.device | None = None,
) -> list[str]:
    """
    Translate texts EN→pivot→EN using Helsinki-NLP MarianMT models.

    Uses temperature sampling (do_sample=True) rather than beam search or greedy
    decoding, which produces more lexically varied paraphrases with no extra VRAM
    cost over greedy (still one candidate per item).  temperature=1.3 / top_k=50
    gives good variety while keeping output fluent.

    Results are cached to data/bt_cache_{pivot_lang}.json. On subsequent calls,
    cached translations are returned instantly; only unseen texts are translated.
    Delete the cache file to regenerate with different sampling settings.

    Returns list of back-translated strings, same length and order as input.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load cache (one file per pivot language)
    cache_path = _cache_path(pivot_lang)
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    unseen = [t for t in texts if t not in cache]

    if unseen:
        fwd_name = f"Helsinki-NLP/opus-mt-en-{pivot_lang}"
        bwd_name = f"Helsinki-NLP/opus-mt-{pivot_lang}-en"

        fwd_tok = MarianTokenizer.from_pretrained(fwd_name)
        fwd_model = MarianMTModel.from_pretrained(fwd_name).to(device).eval()
        bwd_tok = MarianTokenizer.from_pretrained(bwd_name)
        bwd_model = MarianMTModel.from_pretrained(bwd_name).to(device).eval()

        def translate_batch(model, tokenizer, batch_texts):
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, do_sample=True,
                                     temperature=temperature, top_k=top_k,
                                     num_beams=1, top_p=top_p)
            return tokenizer.batch_decode(out, skip_special_tokens=True)

        LOG.info(f"Translating {len(unseen)} unseen texts via {pivot_lang} pivot...")
        # EN → pivot
        pivoted = []
        for i in range(0, len(unseen), batch_size):
            pivoted.extend(translate_batch(fwd_model, fwd_tok, unseen[i:i + batch_size]))
            LOG.info(f"Translated {min(i + batch_size, len(unseen))}/{len(unseen)} texts to {pivot_lang}...")

        LOG.info(f"Translating {len(pivoted)} pivoted texts back to EN...")
        # pivot → EN
        back = []
        for i in range(0, len(pivoted), batch_size):
            back.extend(translate_batch(bwd_model, bwd_tok, pivoted[i:i + batch_size]))
            LOG.info(f"Translated {min(i + batch_size, len(pivoted))}/{len(pivoted)} pivoted texts back to EN...")
        # Update cache
        for orig, bt in zip(unseen, back):
            cache[orig] = bt
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)

        del fwd_model, bwd_model
        torch.cuda.empty_cache()

    return [cache[t] for t in texts]
