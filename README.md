# Text-to-Text Transformers for Semantic Parsing
This repository is used to finetune a T5 model on the task of semantic parsing, a.k.a. generating (Python) code out of natural language descriptions.
For more details, please refer to the relevant paper.

```
@article{papanikolaou21,
  title={Teach me how to Label: Labeling Functions from Natural Language with Text-to-text Transformers},
  author={Yannis Papanikolaou},
  journal={arXiv preprint arXiv:},
  year={2021}
}
```
Along with our code we include the relevant datasets used in the paper. The CoNaLa dataset, as well as the evaluation script are taken from the CoNaLa project website:
https://conala-corpus.github.io/

# Finetuned models
You can find a t5-small and a t5-large model, both finetuned on CoNaLa (gold+noisy data) here:
 https://huggingface.co/yannis-papanikolaou/t5-code-generation


# Results
You can reproduce results of the paper by downloading one of the above models and then run with:

`PYTHONPATH=. python t5_experiments/scripts/train_predict.py --validation-file data/conala-test.json --language-model t5-small --model-dir <finetuned-model-dir>`

You can also finetune a model on CoNaLa with:

`PYTHONPATH=. python t5_experiments/scripts/train_predict.py --training-file data/conala-train.json --validation-file data/conala-test.json --language-model t5-small --model-dir <finetuned-model-dir>`

#Notes
- To fit a larger model on a "normal" GPU, you might need to use gradient accumulation to emulate larger batches (for example a batch with 4 samples and gradient accumulaiion of 8 will emulate a batch size of 32). With this approach we could fit t5-large on a 16Gb V100.
- The code has been tested with Python 3.8.
- Do make sure that `transformers==3.1.0` when using the finetuned models, otherwise you might run into funny results, due to a major update in transformers library tokenizers.
