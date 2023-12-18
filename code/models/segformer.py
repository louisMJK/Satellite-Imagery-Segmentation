from transformers import SegformerConfig, SegformerForSemanticSegmentation


def Segformer():
    # B4
    config = SegformerConfig(
        num_channels=18,
        num_labels=14,
        num_encoder_blocks=4, 
        depths=[3, 8, 27, 3],
        hidden_sizes=[64, 128, 320, 512], 
        decoder_hidden_size=768,
        semantic_loss_ignore_index=0
    )
    model = SegformerForSemanticSegmentation(config)
    return model

