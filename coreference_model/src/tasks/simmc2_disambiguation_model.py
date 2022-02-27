# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20


class SIMMC2DisambiguationModel(nn.Module):
    # copy of the vqa_model with some things from the nlvr2_model

    def __init__(self):
        super().__init__()

        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        self.hid_dim = hid_dim = self.lxrt_encoder.dim

        # NLVR2 Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, 2)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        # from NLVR2 model
        # Extract feature --> Concat
        x = self.lxrt_encoder(sent, (feat, pos))
        # print("x length: {}".format(len(x)))
        # x = x.view(-1, self.hid_dim*2)
        # print("x length: {}".format(len(x)))

        # Compute logit of answers
        logit = self.logit_fc(x)

        return logit
