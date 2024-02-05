import argparse
import os
import random
import time
from functools import partial
import numpy as np
from torch import optim, nn
from tqdm import tqdm
from model import ReRanker, RankingLoss
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from DatasetIterator import to_cuda, collate_mp, ReRankingDataset
from RougeUtil import cal_rouge
from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()


def train(args, scorer, dataloader, val_dataloader):
    print(f'args: {args}')
    train_date = time.strftime('%Y-%m-%d-%H-%M-%s', time.localtime())
    with open(f'log/{args.dataset}/{args.dataset}_test_log_{train_date}.txt', 'a') as f:
        f.write(f'args: {args}')
    init_lr = args.max_lr / args.warmup_steps
    optimizer = optim.Adam(scorer.parameters(), lr=init_lr)
    all_step_cnt = 0
    minimum_loss = 1000000
    no_improve = 0
    for epoch in range(args.epoch):
        optimizer.zero_grad()
        avg_loss = 0
        step_cnt = 0
        for idx, batch in tqdm(enumerate(dataloader)):
            to_cuda(batch, 0)
            step_cnt += 1
            output = scorer(batch["src_input_ids"], batch["candidate_ids"], batch["tgt_input_ids"], pooler=args.pooler)
            outer_similarity, inner_similarity, gold_similarity, gold_avg_similarity = \
                output['outer_score'], output['inner_score'], output['summary_score'], output['summary_avg_score']
            ranking_loss_1 = RankingLoss(outer_similarity, gold_similarity, args.outer_margin, args.gold_margin, args.gold_weight, no_gold=args.no_gold)
            ranking_loss_2 = RankingLoss(inner_similarity, gold_avg_similarity, args.inner_margin, args.gold_margin, args.gold_weight, no_gold=args.no_gold)
            loss = args.loss_weight * ranking_loss_1 + (1 - args.loss_weight) * ranking_loss_2
            avg_loss += loss.item()
            loss /= args.accumulate_step

            loss.backward()

            if step_cnt == args.accumulate_step:
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(scorer.parameters(), args.grad_norm)
                step_cnt = 0
                all_step_cnt += 1
                lr = args.max_lr * min(all_step_cnt ** (-0.5), all_step_cnt * (args.warmup_steps ** (-1.5)))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                optimizer.step()
                optimizer.zero_grad()
        del outer_similarity, inner_similarity, gold_similarity, gold_avg_similarity, loss
        print('Epoch {}, loss: {}'.format(epoch, avg_loss / len(dataloader)))

        loss = evaluation(val_dataloader, scorer, args)
        if loss < minimum_loss:
            no_improve = 0
            minimum_loss = loss
            model_path = os.path.join(args.model_save_path, 'best_margin{},{}_{}.pt'.format(args.outer_margin, args.inner_margin, train_date))
            torch.save(scorer.state_dict(), model_path)
            args.model_path = model_path
            print('The best reranking model has been updated!')
            t1, t2, t3 = re_ranker_test(args)
            with open(f'log/{args.dataset}/{args.dataset}_test_log_{train_date}.txt', 'a') as f:
                f.write(f"rouge-1: {t1}, rouge-2: {t2}, rouge-L: {t3}\n")
        else:
            no_improve += 1

        if no_improve >= args.early_stop:
            print('early stop!')
            break


def evaluation(dataloader, scorer, args):
    scorer.eval()
    cnt = 0
    rouge1, rouge2, rougeLsum = 0, 0, 0
    with torch.no_grad():
        for (i, batch) in enumerate(dataloader):
            to_cuda(batch, args.cuda)
            samples = batch["data"]
            output = scorer(batch["src_input_ids"], batch["candidate_ids"], batch["tgt_input_ids"])
            outer_similarity = output['outer_score']
            outer_similarity = outer_similarity.cpu().numpy()
            cand_emb = output['candidate_emb']

            for j in range(outer_similarity.shape[0]):
                cnt += 1
                sample = samples[j]
                sorted_idx = np.argsort(outer_similarity[j])[::-1][:args.filter_num]
                avg_score = np.array([])
                for s_i in sorted_idx:
                    avg = 0
                    for s_ii in sorted_idx:
                        if s_ii != s_i:
                            sim = torch.cosine_similarity(cand_emb[j][s_i], cand_emb[j][s_ii], dim=-1)
                            avg += sim
                    avg_score = np.append(avg_score, (avg / (len(sorted_idx) - 1)).item())

                sents = sample["candidates"][sorted_idx[np.argmax(avg_score)]][0]

                score = cal_rouge(sents, sample["faq"])
                rouge1 += score['rouge-1']['f']
                rouge2 += score['rouge-2']['f']
                rougeLsum += score['rouge-l']['f']
    rouge1 = rouge1 / cnt
    rouge2 = rouge2 / cnt
    rougeLsum = rougeLsum / cnt
    scorer.train()
    loss = 1 - ((rouge1 + rouge2 + rougeLsum) / 3)
    print(f"eval: rouge-1: {rouge1}, rouge-2: {rouge2}, rouge-L: {rougeLsum}")
    return loss


def re_ranker_test(args):
    test_date = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    tokenizer = RobertaTokenizer.from_pretrained(args.model_type)
    collate_fn_test = partial(collate_mp, pad_token_id=tokenizer.pad_token_id, is_test=True)
    test_set = ReRankingDataset("dataset/{}/{}".format(args.dataset, 'test'), args.model_type, is_test=True,
                                maxlen=512, is_sorted=False, maxnum=args.max_num)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                 collate_fn=collate_fn_test)
    model_path = args.model_path
    scorer = ReRanker(args.model_type, tokenizer.pad_token_id)
    scorer = scorer.cuda()
    scorer.load_state_dict(torch.load(model_path, map_location='cuda:{}'.format(args.cuda)))
    scorer.eval()
    print("load model: {}".format(model_path))
    hyps = []
    refs = []
    with torch.no_grad():
        for (i, batch) in tqdm(enumerate(test_dataloader)):
            to_cuda(batch, args.cuda)
            samples = batch["data"]
            output = scorer(batch["src_input_ids"], batch["candidate_ids"], batch["tgt_input_ids"], pooler=args.pooler, is_test=True)
            outer_similarity = output['outer_score']
            outer_similarity = outer_similarity.cpu().numpy()
            cand_emb = output['candidate_emb']

            for j in range(outer_similarity.shape[0]):
                sample = samples[j]
                sorted_idx = np.argsort(outer_similarity[j])[::-1][:args.filter_num]
                avg_score = np.array([])
                for s_i in sorted_idx:
                    avg = 0
                    for s_ii in sorted_idx:
                        if s_ii != s_i:
                            sim = torch.cosine_similarity(cand_emb[j][s_i], cand_emb[j][s_ii], dim=-1)
                            avg += sim
                    avg_score = np.append(avg_score, (avg / (len(sorted_idx) - 1)).item())

                sents = sample["candidates"][sorted_idx[np.argmax(avg_score)]][0]
                hyps.append(sents)
                refs.append(sample["faq"])

    scorer.train()
    rg_score = cal_rouge(hyps, refs)
    rouge1, rouge2, rougeLsum = rg_score['rouge-1']['f'], rg_score['rouge-2']['f'], rg_score['rouge-l']['f']

    print(f"test: rouge-1: {rouge1}, rouge-2: {rouge2}, rouge-L: {rougeLsum}")
    return rouge1, rouge2, rougeLsum


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ReRanker Training")
    parser.add_argument("--dataset", type=str, default="MeQSum",
                        choices=["MeQSum", "CHQ-Summ", "iCliniq", "HealthCareMagic"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--outer_margin", type=float, default=0.01)
    parser.add_argument("--inner_margin", type=float, default=0.01)
    parser.add_argument("--gold_margin", type=float, default=0)
    parser.add_argument("--cand_weight", type=float, default=1)
    parser.add_argument("--gold_weight", type=float, default=1)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--grad_norm", type=int, default=0)
    parser.add_argument("--max_lr", type=float, default=0.002)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--max_num", type=int, default=16)
    parser.add_argument("--model_path", type=str, help="checkpoint position", default='')
    parser.add_argument("--model_save_path", type=str, default="checkpoint/MeQSum")
    parser.add_argument("--accumulate_step", type=int, default=1)
    parser.add_argument("--early_stop", type=int, default=50)
    parser.add_argument("--mod", type=str, default="train")
    parser.add_argument("--model_type", type=str, default="roberta-base")
    parser.add_argument("--gate_threshold", type=float, default=None)
    parser.add_argument("--pooler", action="store_true")
    parser.add_argument("--no_gold", action="store_true")
    parser.add_argument("--filter_num", type=int, default=8)
    parser.add_argument("--loss_weight", type=float, default=0.5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    if args.mod == "train":
        tokenizer = RobertaTokenizer.from_pretrained(args.model_type)
        collate_fn = partial(collate_mp, pad_token_id=tokenizer.pad_token_id, is_test=False)
        collate_fn_val = partial(collate_mp, pad_token_id=tokenizer.pad_token_id, is_test=True)

        train_set = ReRankingDataset("dataset/{}/{}".format(args.dataset, 'train'), args.model_type,
                                     maxlen=args.max_len, maxnum=args.max_num)
        val_set = ReRankingDataset("dataset/{}/{}".format(args.dataset, 'val'), args.model_type, is_test=True,
                                   maxlen=512, is_sorted=False, maxnum=args.max_num)
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                collate_fn=collate_fn)
        val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                    collate_fn=collate_fn_val)

        scorer = ReRanker(args.model_type, tokenizer.pad_token_id)
        scorer = scorer.cuda()
        scorer.train()
        train(args, scorer, dataloader, val_dataloader)
    elif args.mod == "test":
        re_ranker_test(args)
