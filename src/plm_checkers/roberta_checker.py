# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaModel, BertPreTrainedModel, RobertaConfig
from .checker_utils import attention_mask_to_mask, ClassificationHead, soft_logic, build_pseudo_labels, \
    get_label_embeddings, temperature_annealing

import re
from math import log

from PIL import Image
import numpy as np

import random
import torchvision

from .InceptionV3 import GoogLeNet
from .encoding_img import vgg, ResNet


from transformers import AutoTokenizer, AutoModelForMaskedLM

from transformers import AutoTokenizer, AutoModelForSequenceClassification



class RobertaChecker(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config, hs=256, share_hs=128, logic_lambda=0.0, prior='rand', temperature=1):
        super().__init__(config)
        self.ynum_labels = config.num_labels  # 3
        self.znum_labels = 2
        self.hidden_size = config.hidden_size

        self.roberta =  AutoModelForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext-large")   # for wb
        self.resnet = ResNet()
        self.vgg = vgg()
        self.inv3 = GoogLeNet()
        self.myhs = hs
        self.share_hs = share_hs

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self._lambda = logic_lambda
        self.prior = prior
        self.temperature = temperature
        self._step = 0

        self.linear_1 = nn.Linear(21128, self.hidden_size) # 768
        self.linear_2 = nn.Linear(21128, self.hidden_size)  # 768
        self.linear_et = nn.Linear(self.hidden_size, self.myhs)
        self.linear_eo = nn.Linear(self.hidden_size, self.myhs)
        self.linear_ev = nn.Linear(512, self.myhs)
        self.linear_ep = nn.Linear(1000, self.myhs)

        self.linear_ez = nn.Linear(2, self.myhs // 2)

        self.linear_shared1 = nn.Linear(self.myhs, self.share_hs)
        self.linear_shared2 = nn.Linear(self.myhs, self.share_hs)
        self.bilinear = nn.Bilinear(self.myhs, self.myhs, self.myhs,
                                    bias=True)

        self.var_hidden_size = self.hidden_size // 4

        z_hid_size = self.znum_labels * 3
        self.linear_P_theta = nn.Linear(self.myhs * 2 + z_hid_size, self.var_hidden_size)

        y_hid_size = self.var_hidden_size
        self.linear_Q_phi_1 = nn.Linear(self.myhs + y_hid_size, self.var_hidden_size)
        self.linear_Q_phi_2 = nn.Linear(self.myhs + y_hid_size, self.var_hidden_size)
        self.linear_Q_phi_3 = nn.Linear(self.myhs + y_hid_size, self.var_hidden_size)


        self.linear_Q_phi_global_1 = nn.Linear(3*self.myhs + y_hid_size, self.var_hidden_size)
        self.linear_Q_phi_global_2 = nn.Linear(3*self.myhs + y_hid_size, self.var_hidden_size)
        self.linear_Q_phi_global_3 = nn.Linear(3*self.myhs + y_hid_size, self.var_hidden_size)

        # TODO: y_clf => classifier. compromise for mnli
        self.classifier = ClassificationHead(self.var_hidden_size, self.ynum_labels,
                                             config.hidden_dropout_prob)  # label embedding for y

        self.z_clf_1 = ClassificationHead(self.var_hidden_size, self.znum_labels,
                                          config.hidden_dropout_prob)
        self.z_clf_2 = ClassificationHead(self.var_hidden_size, self.znum_labels,
                                          config.hidden_dropout_prob)
        self.z_clf_3 = ClassificationHead(self.var_hidden_size, self.znum_labels,
                                          config.hidden_dropout_prob)

        # Entities memory
        self.ent_dim_in = hs
        self.ent_dim_out = hs

        self.ent_mem_hops = 1

        # mapping was not defined before
        if self.ent_mem_hops > 1:
            self.Wqent_hop = nn.Linear(self.ent_dim_out, self.ent_dim_out)
        self.W_ent_c = nn.Linear(self.ent_dim_in, self.ent_dim_out)
        self.W_ent_a = nn.Linear(self.ent_dim_in, self.ent_dim_out)
        self.bn_entMem = nn.BatchNorm1d(num_features=self.ent_dim_out)

        self.init_weights()
        self.soft = nn.Softmax(dim=-1)

    def forward(self, text_input_ids, text_attention_mask,
                img_context_input_ids_list, img_context_attention_mask_list,
                img_input,  labels=None):
        '''
        :param text_input_ids: b x L1
        :param text_attention_mask: b x L1
        :param img_context_input_ids_list: b x L2
        :param img_context_attention_mask_list: b x L2
        :param labels: (b,)
        :return: (loss, (neg_elbo, logic_loss), y, m_attn, (z_softmax, mask))
        '''
        # ====================== Representation learning =======================
        self._step += 1
        _zero = torch.tensor(0.).to(text_input_ids.device)

        x = self.roberta(text_input_ids, attention_mask=text_attention_mask)[0]

        x = torch.mean(x, dim=1, keepdim=True)

        e_t = torch.squeeze(x, dim=1)

        e_t = self.linear_1(e_t)
        e_t = self.linear_et(e_t)

     
        e_v = self.resnet(img_input)
        e_v = self.linear_ev(e_v)
        e_p = self.inv3(img_input)
        e_p = self.linear_ep(e_p)

        x = self.roberta(img_context_input_ids_list, attention_mask=img_context_attention_mask_list)[0]
        x = torch.mean(x, dim=1, keepdim=True)


        e_r = torch.squeeze(x, dim=1)
        e_r = self.linear_2(e_r)
        e_r = self.linear_eo(e_r)


        e1 = e_p
        e2 = self.bilinear(e_t, e_v)
        e3 = self.bilinear(e_t, e_r)


        e1 = torch.nn.functional.normalize(e1, p=2, dim=1)
        e2 = torch.nn.functional.normalize(e2, p=2, dim=1)
        e3 = torch.nn.functional.normalize(e3, p=2, dim=1)


        e_global = torch.cat([e_t, e_v], dim=1)  # b* 2myhs

        neg_elbo, loss, logic_loss = _zero, _zero, _zero

        if labels is not None:
            # Training
            # ======================== Q_phi ================================

            labels_onehot = F.one_hot(labels, num_classes=self.ynum_labels).to(torch.float)
            y_star_emb = get_label_embeddings(labels_onehot,
                                              self.classifier.out_proj.weight)


            z = self.Q_phi(e1, e2, e3, y_star_emb)



            z_softmax = z.softmax(-1)  # b *3 *2


            # ======================== P_theta ==============================

            z_gumbel = F.gumbel_softmax(z, tau=temperature_annealing(self.temperature, self._step),
                                        dim=-1, hard=True)
            y = self.P_theta(e_global, z_gumbel)  # b*2
            y_softmax = y.softmax(-1)

            # ======================== soft logic ===========================

            y_z = soft_logic(z_softmax)  # b x 2

            logic_loss = F.kl_div(y.log_softmax(-1), y_z)

            # ======================== ELBO =================================
            elbo_neg_p_log = F.cross_entropy(y.view(-1, self.ynum_labels), labels.view(
                -1))


            if self.prior == 'uniform':
                prior = torch.full((y_z.size(0), 3, self.znum_labels), 1 / self.znum_labels).to(y.device)

            elif self.prior == 'rand':
                prior = torch.rand([y_z.size(0), 3, self.znum_labels]).to(e1)


                for i in range(A.size(0)):

                    row_index = torch.randint(0, 3, (1,))
                    A[i, row_index] = torch.tensor([0.1, 0.9])


                    other_rows = torch.tensor([0.9, 0.1])
                    A[i, torch.arange(3) != row_index] = other_rows
                prior = A.to(e1.device)

            else:
                raise NotImplementedError(self.prior)

            elbo_kl = F.kl_div(z_softmax.log(), prior)

            neg_elbo = elbo_kl + elbo_neg_p_log


            loss = (1 - abs(self._lambda) ) * neg_elbo + abs(self._lambda) * logic_loss
        else:


            if self.prior == 'uniform':
                z = torch.full((e1.size(0), 3, self.znum_labels), 1 / self.znum_labels).to(e1.device)

            elif self.prior == 'rand':
                z = torch.rand([e1.size(0), 3, self.znum_labels]).to(e1)


                for i in range(A.size(0)):

                    row_index = torch.randint(0, 3, (1,))
                    A[i, row_index] = torch.tensor([0.1, 0.9])


                    other_rows = torch.tensor([0.9, 0.1])
                    A[i, torch.arange(3) != row_index] = other_rows
                z = A.to(e1.device)
            z_softmax = z.softmax(-1)

            for i in range(3):
                z = z_softmax.argmax(-1)
                z = F.one_hot(z, num_classes=2).to(torch.float)


                y = self.P_theta(e_global, z)
                y = y.softmax(-1)

                y_emb = get_label_embeddings(y, self.classifier.out_proj.weight)
                z = self.Q_phi(e1, e2, e3, y_emb)

                z_softmax = z.softmax(-1)


            y_softmax = y.softmax(-1)

        return (loss, (neg_elbo, logic_loss), y_softmax, z_softmax)

    def Q_phi(self, e1, e2, e3, y):
        '''
        :param e1,2,3: b x self.myhs
        :param y: b x h'
        :return: b x 3 (3个角度) x 2 (real,fake)
        '''
        z_hidden_1 = self.linear_Q_phi_1(torch.cat([y, e1], dim=-1))
        z_hidden_1 = F.tanh(z_hidden_1)
        z_1 = self.z_clf_1(z_hidden_1)  # b * 2

        z_hidden_2 = self.linear_Q_phi_2(torch.cat([y, e2], dim=-1))  #
        z_hidden_2 = F.tanh(z_hidden_2)
        z_2 = self.z_clf_2(z_hidden_2)  # 2

        z_hidden_3 = self.linear_Q_phi_3(torch.cat([y, e3], dim=-1))  #
        z_hidden_3 = F.tanh(z_hidden_3)
        z_3 = self.z_clf_3(z_hidden_3)  # 2

        z = torch.cat([z_1.unsqueeze(1), z_2.unsqueeze(1), z_3.unsqueeze(1)], dim=1)  # b*3*2
        return z




    def P_theta(self, X_global, z):  # e_global, z_gumbel
        '''
        X, z => y*
        :param X_global: b x 2myhs
        :param z: b x 3 x 2
        :return: b x 2
        '''
        b = z.size(0)
        # global classification
        _logits = torch.cat([X_global, z.reshape(b, -1)], dim=-1)  #

        _logits = self.dropout(_logits)
        _logits = self.linear_P_theta(_logits)
        _logits = torch.tanh(_logits)

        y = self.classifier(_logits)
        return y

    def generic_memory(self, query_proj, results, mem_a_weights, mem_c_weights, bn_mem, mem_hops=1,
                       query_hop_weight=None):
        u = query_proj
        for i in range(0, mem_hops):
            u = F.dropout(u, self.pdrop_mem)

            mem_a = F.relu(mem_a_weights(results))
            mem_a = F.dropout(mem_a, self.pdrop_mem)

            P = F.softmax(torch.sum(mem_a * u.unsqueeze(1), 2), dim=1)
            mem_c = F.relu(mem_c_weights(results))
            mem_c = F.dropout(mem_c, self.pdrop_mem)

            mem_out = torch.sum(P.unsqueeze(2).expand_as(mem_c) * mem_c, 1)
            mem_out = mem_out + u
            u = mem_out
        mem_out = bn_mem(mem_out)

        return mem_out







