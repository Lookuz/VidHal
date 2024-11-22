
def load_model(model):
    """
    Returns a triple of (model, vis_processor, text_processor). If your model does not require any of these, you may return None
    """
    return {
        "random" : (None, None, None)
    }[model]
