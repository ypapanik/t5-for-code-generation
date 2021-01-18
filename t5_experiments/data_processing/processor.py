import json
import logging
import torch
from torch.utils.data import TensorDataset
from transformers import InputExample

from t5_experiments.data_processing.utils import get_encoded_code_tokens


def load_and_cache_examples(data_file, local_rank, max_seq_length, tokenizer, evaluate=False,
                            input_label="rewritten_intent", target_label='snippet'):

    if local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    nr_examples = 0
    examples = []
    labels = []
    for i, pair in enumerate(json.load(open(data_file, 'r'))):
        if (not "rewritten_intent" in pair) or not pair[input_label]:
            text_a = pair['intent']
        else:
            text_a = pair[input_label]
        label = str(pair[target_label])+''
        encoded_reconstr_code = get_encoded_code_tokens(label)
        label = ' '.join(encoded_reconstr_code)

        labels.append(label.lower())
        guid = str(i)
        ex = InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
        examples.append(ex)
        nr_examples += 1
    print('number of examples:', nr_examples)
    tokenized_inputs = tokenizer.batch_encode_plus(
        [ex.text_a for ex in examples],
        max_length=max_seq_length,
        pad_to_max_length=True,
        return_tensors="pt",
    )
    # tokenize targets
    tokenized_targets = tokenizer.batch_encode_plus(
        [ex.label for ex in examples],
        max_length=max_seq_length,
        pad_to_max_length=True,
        return_tensors="pt",
        truncation_strategy='do_not_truncate'
    )

    if local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache


    dataset = TensorDataset(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'],
                            tokenized_targets['input_ids'], tokenized_targets['attention_mask'])

    return dataset, labels

def write_t5_predictions(predictions, output_nbest_file, qa_ids):
    """Write final predictions to the json file and log-odds of null if needed."""
    logging.info("Writing predictions to: %s" % (output_nbest_file))
    nbest_json = {}
    for (qa_id, prediction) in zip(qa_ids, predictions):
            nbest_json[qa_id] = prediction
    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(nbest_json, indent=2) + "\n")