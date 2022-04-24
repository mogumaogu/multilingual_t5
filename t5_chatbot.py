import t5.data
from t5.data import sentencepiece_vocabulary
from t5.evaluation import metrics
from t5.data import preprocessors
from t5.data import TaskRegistry
from t5.data import TextLineTask

import functools
import tensorflow as tf
from sumeval.metrics.lang.lang_ja import LangJA
from sacrebleu import corpus_bleu, TOKENIZERS

lang_ja = LangJA()

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model"
DEFAULT_VOCAB = sentencepiece_vocabulary.SentencePieceVocabulary(
    DEFAULT_SPM_PATH)
DEFAULT_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(
        vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
    "targets": t5.data.Feature(
        vocabulary=DEFAULT_VOCAB, add_eos=True)
}

# オージス総研様と同様にBLEUを指標にしたいと思います
def bleu(targets, predictions):
  predictions = [tf.compat.as_text(x) for x in predictions]

  if isinstance(targets[0], list):
    targets = [[tf.compat.as_text(x) for x in target] for target in targets]
  else:
    targets = [tf.compat.as_text(x) for x in targets]
    targets = [targets]

  bleu_score = corpus_bleu(predictions, targets,smooth_method="exp", smooth_value=0.0,
                           force=False,lowercase=False,tokenize="ja-mecab", use_effective_order=False)
  return {"bleu": bleu_score.score}

task_name = "t5_chatbot"

tsv_path = {
    "train": "/content/train_data.tsv",
    "test": "/content/test_data.tsv",
}

TaskRegistry.add(
    task_name,
    TextLineTask,
    split_to_filepattern=tsv_path,
    text_preprocessor=[
      functools.partial(
          preprocessors.parse_tsv,
          field_names=["inputs", "targets"]),
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[bleu])
