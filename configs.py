import ml_collections

def get_triq_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.backbone = 'resnet50'
    config.hidden_size = 32
    config.n_quality_levels = 5
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 64
    config.transformer.num_heads = 8
    config.transformer.num_layers = 2
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config