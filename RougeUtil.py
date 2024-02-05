from rouge import Rouge


# https://github.com/Diego999/py-rouge
def cal_rouge(hyps, refs):
    evaluator = Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2)
    score = evaluator.get_scores(hyps, refs)
    return score
