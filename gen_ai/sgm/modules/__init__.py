from .encoders.modules import GeneralConditioner

UNCONDITIONAL_CONFIG = {
    "target": "gen_ai.sgm.modules.GeneralConditioner",
    "params": {"emb_models": []},
}
