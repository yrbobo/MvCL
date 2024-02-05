from torch import nn
from transformers import BartForConditionalGeneration


class FtModel(nn.Module):
    def __init__(self):
        super(FtModel, self).__init__()
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

    def forward(self, src_ids, src_mask, tgt_ids, labels):
        output = self.model(input_ids=src_ids, attention_mask=src_mask, decoder_input_ids=tgt_ids, labels=labels,
                            output_hidden_states=True, return_dict=True)
        loss = output.loss
        return loss
