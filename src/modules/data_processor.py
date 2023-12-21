# -*- coding: utf-8 -*-


import os
import copy
import logging
import json
import ujson as json
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset
import tensorflow as tf
import sys
from PIL import Image
from torchvision import datasets, models, transforms



logger = logging.getLogger(__name__)


data_path1 = "data/weibo/nonrumor_images/"  # wb image path . Please find the original data from the original text that proposed the dataset
data_path2 = "data/weibo/rumor_images/"  # wb

class InputExample(object):
    def __init__(self, id, text, o_text,
                 img_id, label=None):
        self.id = id
        self.text = text
        self.o_text = o_text
        self.img_id = img_id
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    def __init__(
            self,
            id,
            img_id,
            c_input_ids,
            c_attention_mask,
            c_token_type_ids,
            q_input_ids,
            q_attention_mask,
            q_token_type_ids,
            label=None,
    ):
        self.id = id
        self.c_input_ids = c_input_ids
        self.c_attention_mask = c_attention_mask
        self.c_token_type_ids = c_token_type_ids
        self.q_input_ids = q_input_ids
        self.q_attention_mask = q_attention_mask
        self.q_token_type_ids = q_token_type_ids
        self.label = label
        self.img = img_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def _create_input_ids_from_token_ids(token_ids_a, tokenizer, max_seq_length):
    # Truncate sequence
    num_special_tokens_to_add = tokenizer.num_special_tokens_to_add()
    while len(token_ids_a) > max_seq_length - num_special_tokens_to_add:
        token_ids_a = token_ids_a[:-1]

    # Add special tokens to input_ids
    input_ids = tokenizer.build_inputs_with_special_tokens(token_ids_a)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    attention_mask = [1] * len(input_ids)

    # Create token_type_ids
    token_type_ids = tokenizer.create_token_type_ids_from_sequences(token_ids_a)

    # Pad up to the sequence length
    padding_length = max_seq_length - len(input_ids)
    if tokenizer.padding_side == "right":
        input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([tokenizer.pad_token_type_id] * padding_length)
    else:
        input_ids = ([tokenizer.pad_token_id] * padding_length) + input_ids
        attention_mask = ([0] * padding_length) + attention_mask
        token_type_ids = ([tokenizer.pad_token_type_id] * padding_length) + token_type_ids

    assert len(input_ids) == max_seq_length
    assert len(attention_mask) == max_seq_length
    assert len(token_type_ids) == max_seq_length

    return input_ids, attention_mask, token_type_ids

def convert_examples_to_features(examples,tokenizer,
        max_seq1_length=32,
        max_seq2_length=32,
        verbose=True):
    features = []
    iter = tqdm(examples, desc="Converting Examples") if verbose else examples
    for (ex_index, example) in enumerate(iter):
        encoded_outputs = {"id": example.id, "img_id": example.img_id, 'label': example.label}

        # ****** for sequence 1 *******
        token_ids_a = []

        token_ids = tokenizer.encode(example.text, add_special_tokens=False)  # encode claim
        token_ids_a.extend(token_ids)

        input_ids, attention_mask, token_type_ids = _create_input_ids_from_token_ids(
            token_ids_a,
            tokenizer,
            max_seq1_length,
        )

        encoded_outputs["c_input_ids"] = input_ids
        encoded_outputs["c_attention_mask"] = attention_mask
        encoded_outputs["c_token_type_ids"] = token_type_ids

        # ****** for sequence 2 ******* #

        token_ids_b = []

        token_ids = tokenizer.encode(example.o_text, add_special_tokens=False)
        token_ids_b.extend(token_ids)

        input_ids, attention_mask, token_type_ids = _create_input_ids_from_token_ids(
            token_ids_b,
            tokenizer,
            max_seq2_length,
        )

        encoded_outputs["q_input_ids"] = input_ids
        encoded_outputs["q_attention_mask"] = attention_mask
        encoded_outputs["q_token_type_ids"] = token_type_ids

        features.append(InputFeatures(**encoded_outputs))

        if ex_index < 5 and verbose:
            logger.info("*** Example ***")
            logger.info("id: {}".format(example.id))
            logger.info("c_input_ids: {}".format(encoded_outputs["c_input_ids"]))
            logger.info('q_input_ids: {}'.format(encoded_outputs["q_input_ids"]))
            logger.info("label: {}".format(example.label))


    return features


class DataProcessor:
    def __init__(
            self,
            model_name_or_path,
            max_seq1_length,
            max_seq2_length,
            data_dir='',
            cache_dir_name='cache_check',
            overwrite_cache=False,
            mask_rate=0.
    ):
        self.model_name_or_path = model_name_or_path
        self.max_seq1_length = max_seq1_length
        self.max_seq2_length = max_seq2_length
        self.mask_rate = mask_rate

        self.data_dir = data_dir
        self.cached_data_dir = os.path.join(data_dir, cache_dir_name)
        self.overwrite_cache = overwrite_cache

        self.label2id = {"REAL": 0, "FAKE": 1}

    def _format_file(self, role):
        return os.path.join(self.data_dir, "{}.txt".format(role))

    def load_and_cache_data(self, role, tokenizer, data_tag):
        tf.io.gfile.makedirs(self.cached_data_dir)
        cached_file = os.path.join(
            self.cached_data_dir,
            "cached_features_{}_{}_{}_{}_{}".format(
                role,
                list(filter(None, self.model_name_or_path.split("/"))).pop(),
                str(self.max_seq1_length),
                str(self.max_seq2_length),
                data_tag
            ),
        )
        if os.path.exists(cached_file) and not self.overwrite_cache:
            logger.info("Loading features from cached file {}".format(cached_file))
            features = torch.load(cached_file)
        else:
            examples = []
            with tf.io.gfile.GFile(self._format_file(role)) as f:
                data = f.readlines()
                for line in tqdm(data):
                    sample = self._load_line(line)
                    examples.append(InputExample(**sample))
            features = convert_examples_to_features(examples, tokenizer, self.max_seq1_length, self.max_seq2_length,)
            if 'train' in role or 'eval' in role:
                logger.info("Saving features into cached file {}".format(cached_file))
                torch.save(features, cached_file)

        return self._create_tensor_dataset(features, tokenizer)

    def convert_inputs_to_dataset(self, inputs, tokenizer, verbose=True):
        examples = []
        for line in inputs:
            sample = self._load_line(line)
            examples.append(InputExample(**sample))
        features = convert_examples_to_features(examples, tokenizer,
                                                self.max_seq1_length, self.max_seq2_length, verbose)

        return self._create_tensor_dataset(features, tokenizer, do_predict=True)

    def _create_tensor_dataset(self, features, tokenizer, do_predict=False):
        all_c_input_ids = torch.tensor([f.c_input_ids for f in features], dtype=torch.long)
        all_c_attention_mask = torch.tensor([f.c_attention_mask for f in features], dtype=torch.long)

        all_q_input_ids = torch.tensor([f.q_input_ids for f in features], dtype=torch.long)
        all_q_attention_mask = torch.tensor([f.q_attention_mask for f in features], dtype=torch.long)



        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),  # resnet
            # transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img_list = []
        for f in features:
            if os.path.exists(os.path.join(data_path1 + f.img + ".jpg")):
                img_path = os.path.join(data_path1 + f.img + ".jpg")
            elif os.path.exists(os.path.join(data_path2 + f.img + ".jpg")):
                img_path = os.path.join(data_path2 + f.img + ".jpg")

            img = Image.open(img_path).convert('RGB')
            img = data_transforms(img)
            img_list.append(img)
        all_img = torch.stack(img_list, dim=0)  # # N* 3, 224, 224

        # if not do_predict:
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = TensorDataset(
            all_c_input_ids, all_c_attention_mask,
            all_q_input_ids, all_q_attention_mask,
            all_img, all_labels,
        )


        return dataset

    def _load_line(self, line):

        l_list = line.split(' ==sep== ')

        id = l_list[0]
        text = l_list[1]
        img_id = l_list[2]
        label = int(l_list[3])
        o_text = l_list[-2] if len(l_list[-2]) > 0 else None # image_context


        sample = {
            "id": id,
            "text": text,
            "o_text": o_text,
            "img_id": img_id,
            "label": label,
        }
        return sample
