# -*- coding: utf-8 -*-
"""
@Date ： 2020/11/19 21:08
@Author ： xyu
"""
import json
from typing import List
import torch
from numpy import linalg
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from model.bert_utils import bert_mask_wrapper, bert_sequence_output_wrapper
from pytorch_transformers import BertConfig, BertTokenizer, BertModel
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS

batch_size = 128
max_seq_len = 32

model_path = "pretrained_model/bert-base-uncased"

config = BertConfig.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
model = BertModel.from_pretrained(model_path, config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def get_dataset(sentences: List[str]):
    examples = []
    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)

        if len(tokens) == 0:
            tokens = ["-"]

        tokens = ["CLS"] + tokens + ["SEP"]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        if len(input_ids) > max_seq_len:
            input_ids = input_ids[:max_seq_len]
            # raise ValueError("sequence length is larger than max_seq_len!")

        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        if len(input_ids) < max_seq_len:
            pad_len = max_seq_len - len(input_ids)
            pad_id = tokenizer.convert_tokens_to_ids(["PAD"])[0]
            input_ids += [pad_id] * pad_len
            segment_ids += [0] * pad_len
            input_mask += [0] * pad_len

        examples.append((input_ids, input_mask, segment_ids))

    all_input_ids = torch.tensor([e[0] for e in examples], dtype=torch.long)
    all_input_mask = torch.tensor([e[1] for e in examples], dtype=torch.long)
    all_segment_ids = torch.tensor([e[2] for e in examples], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    return dataset


# 使用pooled output
def bert_embedding_1(sentences: List[str]):
    sentence_embeddings = []

    dataset = get_dataset(sentences)
    train_dataloader = DataLoader(dataset, batch_size=batch_size)

    # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2]}

        outputs = model(**inputs)
        sentence_embedding = outputs[1]
        sentence_embedding = sentence_embedding.detach().cpu().numpy().tolist()
        sentence_embeddings.extend(sentence_embedding)

    return sentence_embeddings


# 使用CLS embedding
def bert_embedding_2(sentences: List[str]):
    sentence_embeddings = []
    dataset = get_dataset(sentences)
    train_dataloader = DataLoader(dataset, batch_size=batch_size)

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2]}

        outputs = model(**inputs)
        sentence_embedding = outputs[0][:, 0]
        sentence_embedding = sentence_embedding.detach().cpu().numpy().tolist()
        sentence_embeddings.extend(sentence_embedding)

    return sentence_embeddings


# 使用sequence output average
def bert_embedding_3(sentences: List[str]):
    sentence_embeddings = []
    dataset = get_dataset(sentences)
    train_dataloader = DataLoader(dataset, batch_size=batch_size)

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2]}

        outputs = model(**inputs)
        mask = bert_mask_wrapper(batch[1])
        seq_embedding = bert_sequence_output_wrapper(outputs[0]).detach().cpu().numpy()
        b_size, sel_len = mask.shape
        mask = mask.detach().cpu().view(b_size, sel_len, 1).numpy()
        sen_embedding = np.sum(seq_embedding * mask, 1) / np.sum(mask, 1)
        sentence_embeddings.extend(sen_embedding)

    return sentence_embeddings


def cosine_similarity(a, b):
    dot = np.dot(a, b)
    product = linalg.norm(a) * linalg.norm(b)
    cos = dot / product
    sim = 0.5 + 0.5 * cos
    return sim


app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['JSON_AS_ASCII'] = False


def get_similarity(func):
    phrase1 = request.args.get('phrase1')
    phrase2 = request.args.get('phrase2')
    embeddings = func([phrase1, phrase2])
    sim = cosine_similarity(embeddings[0], embeddings[1])
    return jsonify({"sim": sim})


func2cache = {
    bert_embedding_1: {},
    bert_embedding_2: {},
    bert_embedding_3: {},
}


def post_similarity(func):
    params = json.loads(request.get_data(as_text=True))
    source_phrase = params['phrase']
    target_phrases = params['patterns']

    all_phrases = target_phrases + [source_phrase]
    all_phrases_embeddings = func(all_phrases)

    source_phrase_embed = all_phrases_embeddings[-1]
    target_phrase_embeds = all_phrases_embeddings[:-1]

    sim_list = []
    for target in target_phrase_embeds:
        sim = cosine_similarity(source_phrase_embed, target)
        sim_list.append(sim)
    return sim_list


@app.route('/bert_service/1', methods=["GET"])
def bert_embedding_api_1():
    return get_similarity(bert_embedding_1)


@app.route('/bert_service/2', methods=["GET"])
def bert_embedding_api_2():
    return get_similarity(bert_embedding_2)


@app.route('/bert_service/3', methods=["GET"])
def bert_embedding_api_3():
    return get_similarity(bert_embedding_3)


@app.route('/bert_service/all', methods=["GET"])
def bert_embedding_api_4():
    phrase1 = request.args.get('phrase1')
    phrase2 = request.args.get('phrase2')
    embeddings1 = bert_embedding_1([phrase1, phrase2])
    embeddings2 = bert_embedding_2([phrase1, phrase2])
    embeddings3 = bert_embedding_3([phrase1, phrase2])

    sim1 = cosine_similarity(embeddings1[0], embeddings1[1])
    sim2 = cosine_similarity(embeddings2[0], embeddings2[1])
    sim3 = cosine_similarity(embeddings3[0], embeddings3[1])
    return jsonify({"sim1": sim1, "sim2": sim2, "sim3": sim3})


@app.route('/bert_service/1', methods=["POST"])
def bert_embedding_post_api_1():
    return jsonify(post_similarity(bert_embedding_1))


@app.route('/bert_service/2', methods=["POST"])
def bert_embedding_post_api_2():
    return jsonify(post_similarity(bert_embedding_2))


@app.route('/bert_service/3', methods=["POST"])
def bert_embedding_post_api_3():
    return jsonify(post_similarity(bert_embedding_3))


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 5000

    app.run(host=host, port=port, threaded=False)
