# Multi-view Contrastive Learning for Medical Question Summarization

This is the code repository for the paper "Multi-view Contrastive Learning for Medical Question Summarization". 

## Requirements

Python >= 3.8

pytorch == 1.10.1

transformers == 4.26.1

rouge==1.0.1

py-rouge == 1.1

## 1. Prepare the datasets

The datasets can be downloaded from the following URLs.

| Dataset         | URLs                                                         |
| --------------- | ------------------------------------------------------------ |
| MeQSum          | https://github.com/abachaa/MeQSum                            |
| CHQ-Summ        | https://github.com/shwetanlp/Yahoo-CHQ-Summ                  |
| iCliniq         | https://drive.google.com/drive/u/1/folders/1FQTsgRYDJajcNlKJXG-FFPKFw4Cf4FzU |
| HealthCareMagic | https://drive.google.com/drive/u/1/folders/1Hq4AiYr96jfOsB8OJMlyDRRUhmr_BYvY |



## 2. Train

You can train the model with the following command:

```bash
# MeQSum Dataset
python main.py --no_gold --dataset MeQSum --batch_size 16 --epoch 55 --outer_margin 0.01 --inner_margin 0.001 --model_save_path checkpoint/MeQSum --accumulate_step 1 --pooler --filter_num 8 --loss_weight 0.4

# CHQ-Summ Dataset
python main.py --no_gold --dataset CHQ-Summ --batch_size 16 --epoch 30 --outer_margin 0.01 --inner_margin 0.01 --model_save_path checkpoint/CHQ-Summ --accumulate_step 1 --pooler --filter_num 8 --loss_weight 0.7

# iCliniq Dataset
python main.py --no_gold --dataset iCliniq --batch_size 16 --epoch 15 --outer_margin 0.001 --inner_margin 0.001 --model_save_path checkpoint/iCliniq --accumulate_step 4 --pooler --filter_num 8 --loss_weight 0.7

# HealthCareMagic Dataset
python main.py --no_gold --dataset HealthCareMagic --batch_size 16 --epoch 5 --outer_margin 0.001 --inner_margin 0.001 --model_save_path checkpoint/HealthCareMagic --accumulate_step 8 --pooler --filter_num 8 --loss_weight 0.7

```



## 3. Test

You can test the trained model with a command like the following:

```bash
python main.py --mod test --dataset MeQSum --model_path reranking_model/MeQSum/your_model.pt --pooler --filter_num 8
```



## Acknowledgement

If this work is useful in your research, please cite our paper.

```
@inproceedings{wei2024multi,
  title={Multi-view Contrastive Learning for Medical Question Summarization},
  author={Sibo Wei, Xueping Peng, Hongjiao Guan, Lina Geng, Ping Jian, Hao Wu, Wenpeng Lu},
  booktitle={Proceedings of the 27th International Conference on Computer Supported Cooperative Work in Design (CSCWD 2024)},
  year={2024},
  organization={IEEE}
}
```

