import logging
import argparse
import glob
import json

import os
import random
import shutil

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data_utils.entity_aware_mhs_utils import EntityAwareMHSDataProcessor_3, EntityAwareMHSDataProcessor_4
from data_utils.mhs_utils import processors
from model.mhs import Config3BModel1, Config3BModel2Softmax
from model.semantic_role_labeling import BertSRLEncoder
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertTokenizer,
                                  RobertaConfig,
                                  XLMConfig, XLNetConfig)

from utils.spaceeval_utils import Metrics

# logging.basicConfig(format='%(asctime)s - %(pathname)s: %(lineno)d - %(levelname)s: %(message)s',
#                     level=logging.DEBUG,
#                     filemode="w",
#                     filename="log.txt",
#                     force=True)

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
                  for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'ea_transformer_config3b_1': (BertSRLEncoder, Config3BModel1, BertTokenizer, EntityAwareMHSDataProcessor_3),
    'ea_transformer_config3b_2': (BertSRLEncoder, Config3BModel2Softmax, BertTokenizer, EntityAwareMHSDataProcessor_4)
    # 'ea_transformer_config3b_2_crf'(BertSRLEncoder, )
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, processor, eval_dataset=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    args.logging_steps = t_total // args.num_train_epochs

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    # train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    #######
    best_f1 = -1
    features_num = len(processor.get_features_size())
    #######
    # for _ in train_iterator:
    for epoch in range(int(args.num_train_epochs)):
        print("Epoch：", epoch + 1)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      # 'token_type_ids': batch[2],
                      'gold_selection_matrix': batch[3],
                      'features': batch[4: 4 + features_num],
                      }
            if args.use_rel_pos:
                inputs['relative_positions'] = batch[4 + features_num]
                inputs['entity_mask'] = batch[5 + features_num]

            if len(batch) > 6 + features_num:
                inputs['gold_qs_rel_types'] = batch[6 + features_num]
                inputs['gold_o_rel_types'] = batch[7 + features_num]

            if len(batch) > 8 + features_num:
                inputs["gold_no_trigger_q"] = batch[8 + features_num]
                inputs["gold_no_trigger_o"] = batch[9 + features_num]

            # if args.use_dependency:
            #     inputs['dependency_graph'] = batch[-1]

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            epoch_iterator.set_postfix(loss=loss.item())

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer, processor, prefix=global_step, dataset=eval_dataset)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

                        # 记录最好的结果
                        if results['f1'] > best_f1:
                            best_f1 = results['f1']
                            if not os.path.exists(args.output_dir):
                                os.makedirs(args.output_dir)
                            model_to_save = model.module if hasattr(model,
                                                                    'module') else model  # Take care of distributed/parallel training
                            torch.save({'state_dict': model_to_save.state_dict()},
                                       os.path.join(args.output_dir, 'pytorch_model.bin'))
                            logger.info("Saving model checkpoint to %s", args.output_dir)

                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                #     # Save model checkpoint
                #     output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     # Take care of distributed/parallel training
                #     model_to_save = model.module if hasattr(model, 'module') else model
                #     model_to_save.save_pretrained(output_dir)
                #     torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                #     logger.info("Saving model checkpoint to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            # train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    print("best_f1:%f" % best_f1)

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, processor, prefix="", dataset=None, save_predict=True):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    features_num = len(processor.get_features_size())

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, processor, tokenizer, evaluate=True) if not dataset else dataset

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0

        pred_triples = []

        for batch in tqdm(eval_dataloader, desc="Evaluating", mininterval=5):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          # 'token_type_ids': batch[2],
                          'features': batch[4: 4 + features_num]
                          }
                if args.use_rel_pos:
                    inputs['relative_positions'] = batch[4 + features_num]
                    inputs['entity_mask'] = batch[5 + features_num]

                # if args.use_dependency:
                #     inputs['dependency_graph'] = batch[-1]

                outputs = model(**inputs)
                batch_triples = outputs[0]
                pred_triples.extend(batch_triples)

            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps

        # result = processor.evaluate(pred_triples)
        # utd_result_1 = processor.evaluate_exact(pred_triples, eval_null_roles=True, allow_null_mover=True)
        # utd_result_2 = processor.evaluate_exact(pred_triples, eval_null_roles=True, allow_null_mover=False)
        #
        # utd_result_3 = processor.evaluate_exact(pred_triples, eval_null_roles=True, allow_null_mover=True, eval_optional_roles=False)
        # utd_result_4 = processor.evaluate_exact(pred_triples, eval_null_roles=True, allow_null_mover=False,eval_optional_roles=False)
        #
        # official_result_1 = processor.evaluate_exact(pred_triples, eval_null_roles=False, allow_null_mover=True)
        # official_result_2 = processor.evaluate_exact(pred_triples, eval_null_roles=False, allow_null_mover=False)

        pred_triples = processor.post_process(pred_triples)
        if save_predict:
            processor.save_predict_result(eval_output_dir, pred_triples)
        attr_result = processor.evaluate_link_attribute(pred_triples)
        link_result = processor.evaluate_exact(pred_triples, Metrics.STRICT, eval_link_attr=False)
        strict_result = processor.evaluate_exact(pred_triples, Metrics.STRICT, eval_link_attr=True)

        all_result = {
            "result": attr_result,
            "link": link_result,
            "strict": strict_result,
        }

        results.update(strict_result["OVERALL"])

        output_eval_file = os.path.join(eval_output_dir, "eval_results_{}.txt".format(prefix))
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for eval_type, result_dict in all_result.items():
                for key, value in result_dict.items():
                    key = eval_type + " " + key
                    logger.info(" %s = %s", key, str(value))
                    writer.write("%s = %s\n" % (key, str(value)))
                logger.info("")
    return results


def load_and_cache_examples(args, processor, tokenizer, evaluate=False):
    # output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        # str(args.task_name),
        "_".join(processor.used_feature_names),
        str(args.use_rel_pos),
        str(args.max_distance)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = processor.get_dev_examples() if evaluate else processor.get_train_examples()
        features = processor.convert_examples_to_features(examples, args.max_seq_length, tokenizer,
                                                          cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                          # xlnet has a cls token at the end
                                                          cls_token=tokenizer.cls_token,
                                                          cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                          sep_token=tokenizer.sep_token,
                                                          sep_token_extra=bool(
                                                              args.model_type in ['roberta', 'roberta_crf']),
                                                          # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                          pad_on_left=bool(args.model_type in ['xlnet']),
                                                          # pad on the left for xlnet
                                                          pad_token=
                                                          tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                          pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                          max_rel_distance=args.max_distance,
                                                          use_rel_pos=args.use_rel_pos
                                                          )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_selection_matrix = torch.tensor([f.selection_matrix for f in features], dtype=torch.float32)

    all_features = (all_input_ids, all_input_mask, all_segment_ids, all_selection_matrix)

    for f_name in features[0].features.keys():
        all_features += (torch.tensor([f.features[f_name] for f in features], dtype=torch.long),)

    if args.use_rel_pos:
        all_features += (torch.tensor([f.relative_positions for f in features], dtype=torch.long),)
        all_features += (torch.tensor([f.entity_mask for f in features], dtype=torch.long),)

    if features[0].rel_types is not None:
        for name in features[0].rel_types.keys():
            all_features += (torch.tensor([f.rel_types[name] for f in features], dtype=torch.long),)

    if features[0].q_selection_matrix is not None:
        all_features += (torch.tensor([f.q_selection_matrix for f in features], dtype=torch.long),)

    if features[0].o_selection_matrix is not None:
        all_features += (torch.tensor([f.o_selection_matrix for f in features], dtype=torch.long),)

    dataset = TensorDataset(*all_features)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # My parameters
    parser.add_argument("--use_head", action='store_true', help="")

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--task_name", default="MHS", type=str, required=False,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--config_dir", default=None, type=str, required=False)

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=300,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=2000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task

    # args.task_name = args.task_name.lower()
    # if args.task_name not in processors:
    #     raise ValueError("Task not found: %s" % args.task_name)

    args.model_type = args.model_type.lower()
    encoder_class, model_class, tokenizer_class, processor_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    config_dict = json.load(open(args.config_dir))
    processor = processor_class(data_dir=args.data_dir,
                                used_feature_names=config_dict["used_feature_name"],
                                tokenizer=tokenizer, use_head=True)

    label_dict = {
        "element_label": processor.get_element_labels(),
        "qs_rel_type": processor.get_qs_rel_types(),
        "o_rel_type": processor.get_o_rel_types()
    }
    if args.model_type == "ea_transformer_config3b_2":
        label_dict["no_trigger_q_rel_type"] = processor.get_no_trigger_q_labels()
        label_dict["no_trigger_o_rel_type"] = processor.get_no_trigger_o_labels()
        label_dict["no_trigger_o_rel_type"] = processor.get_no_trigger_o_labels()

    config_dict["encoder_params"].update({'features': processor.get_features_size()})
    config_dict["model_params"].update({"labels": processor.get_labels(), "label_dict": label_dict})

    args.use_rel_pos = config_dict["use_rel_pos"]
    args.max_distance = config_dict["model_params"]["max_distance"]

    encoder = encoder_class(**config_dict["encoder_params"])
    model = model_class(encoder, **config_dict["model_params"])

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, processor, tokenizer, evaluate=False)
        eval_dataset = load_and_cache_examples(args, processor, tokenizer, evaluate=True)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, processor, eval_dataset=eval_dataset)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        # logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # model_to_save = model.module if hasattr(model,
        #                                         'module') else model  # Take care of distributed/parallel training
        # model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        json.dump(config_dict, open(os.path.join(args.output_dir, "config.json"), 'w'),
                  default=lambda x: list(x) if isinstance(x, set) else x)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        shutil.copy(args.config_dir, os.path.join(args.output_dir, "model_config.json"))
        # Load a trained model and vocabulary that you have fine-tuned
        # model = model_class.from_pretrained(args.output_dir)
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        # model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model.load_state_dict(torch.load(os.path.join(args.output_dir, 'pytorch_model.bin'))['state_dict'])
            model.to(args.device)
            result = evaluate(args, model, tokenizer, processor, prefix=global_step, save_predict=True)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
    return results


if __name__ == "__main__":
    main()
