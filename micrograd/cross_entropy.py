class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, logits, y_pred):
        max_logit = max(logits, key=lambda x: x.data)
        shifted = [x - max_logit for x in logits]
        
        exps = [x.exp() for x in shifted]
        denom = sum(exps)
        
        log_sum_exp = denom.log()
        loss = log_sum_exp - shifted[y_pred]
        return loss
