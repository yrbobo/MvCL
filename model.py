import torch
from torch import nn
from transformers import RobertaModel
from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()


def RankingLoss(score, summary_score=None, margin=0, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False):
    ones = torch.ones_like(score)
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(score, score, ones)
    # candidate loss
    n = score.size(1)
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss
    if no_gold:
        return TotalLoss
    # gold summary loss
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)
    return TotalLoss


class ReRanker(nn.Module):
    def __init__(self, encoder, pad_token_id):
        super(ReRanker, self).__init__()
        self.encoder = RobertaModel.from_pretrained(encoder)
        self.pad_token_id = pad_token_id

    def forward(self, text_id, candidate_id, summary_id=None, require_gold=True, pooler=False, is_test=False):

        batch_size = text_id.size(0)

        input_mask = text_id != self.pad_token_id

        if pooler:
            doc_emb = self.encoder(text_id, attention_mask=input_mask).pooler_output
        else:
            out = self.encoder(text_id, attention_mask=input_mask)[0]
            doc_emb = out[:, 0, :]

        if require_gold:
            input_mask = summary_id != self.pad_token_id
            if pooler:
                summary_emb = self.encoder(summary_id, attention_mask=input_mask).pooler_output
            else:
                out = self.encoder(summary_id, attention_mask=input_mask)[0]
                summary_emb = out[:, 0, :]
            summary_score = torch.cosine_similarity(summary_emb, doc_emb, dim=-1)

        candidate_num = candidate_id.size(1)
        candidate_id = candidate_id.view(-1, candidate_id.size(-1))
        input_mask = candidate_id != self.pad_token_id
        if pooler:
            out = self.encoder(candidate_id, attention_mask=input_mask).pooler_output
            candidate_emb = out.view(batch_size, candidate_num, -1)
        else:
            out = self.encoder(candidate_id, attention_mask=input_mask)[0]
            candidate_emb = out[:, 0, :].view(batch_size, candidate_num, -1)

        doc_emb = doc_emb.unsqueeze(1).expand_as(candidate_emb)

        outer_score = torch.cosine_similarity(candidate_emb, doc_emb, dim=-1)
        output = {'outer_score': outer_score}

        if not is_test:
            inner_score = torch.zeros((batch_size, candidate_num))
            for bs in range(batch_size):
                for cn in range(candidate_num):
                    avg_sim = 0
                    for _cn in range(candidate_num):
                        if _cn != cn:
                            sim = torch.cosine_similarity(candidate_emb[bs][cn].unsqueeze(0),
                                                          candidate_emb[bs][_cn].unsqueeze(0), dim=-1)
                            avg_sim += sim
                    avg_sim /= (candidate_num - 1)
                    inner_score[bs, cn] = avg_sim

            output['inner_score'] = inner_score

        if require_gold:
            output['summary_score'] = summary_score

            summary_avg_score = torch.zeros(batch_size)
            for bs in range(batch_size):
                avg_sim = 0
                for cn in range(candidate_num):
                    sim = torch.cosine_similarity(summary_emb[bs], candidate_emb[bs][cn], dim=-1)
                    avg_sim += sim
                summary_avg_score[bs] = avg_sim / candidate_num
            output['summary_avg_score'] = summary_avg_score

        # for inference
        output['candidate_emb'] = candidate_emb
        output['doc_emb'] = doc_emb
        return output

