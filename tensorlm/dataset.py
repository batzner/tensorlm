import json
import os
from collections import Counter

import math
from nltk.tokenize import RegexpTokenizer

from tensorlm.common.log import get_logger
from tensorlm.common.tokens import PAD_TOKEN, UNK_TOKEN
from tensorlm.common.util import get_chunks

LOGGER = get_logger(__name__)
VOCAB_FILE_NAME = "vocab.json"


def tokenize(sentence, level):
    if level == "char":
        # No need for tokenizing
        return list(sentence)
    elif level == "word":
        # Tokenize while keeping indentation. Glue letters and numbers to themselves but
        # keep all other chars isolated
        tokenizer = RegexpTokenizer(r'\w+|\S|\s')
        return tokenizer.tokenize(sentence)
    else:
        raise ValueError("Unknown token level: {}".format(level))


class Dataset:
    def __init__(self, path, vocab, batch_size, num_timesteps, bytes_in_memory=1000000):
        self.path = path
        self.vocab = vocab
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
        self.bytes_in_memory = bytes_in_memory

        self.text_iter = TextIterator(self.path, self.vocab.level, bytes_in_memory)
        self.next_batch_index_to_load = 0
        self.index_to_batch = {}

    def get_batch(self, batch_index):
        # Returns None if the batch_index exceeds the number of batches in the text / at the
        # end of an epoch

        if batch_index in self.index_to_batch:
            return self.index_to_batch[batch_index]

        # Check if the text_iter is already past the batch_index
        if batch_index < self.next_batch_index_to_load:
            # Reset the file iterator to 0 to start again
            self.text_iter = TextIterator(self.path, self.vocab.level, self.bytes_in_memory)
            self.next_batch_index_to_load = 0

        try:
            # Load from the text iterator until the current_batch_index equals the batch_index
            tokens = self.text_iter.__next__()
            self._update_batches(tokens)
            return self.get_batch(batch_index)
        except StopIteration:
            # We finished an epoch
            return None

    def _batch_tokens_to_ids(self, batch):
        # Translate an 2D array of tokens to ids
        batch_ids = []
        for batch_item in batch:
            ids = self.vocab.tokens_to_ids(batch_item)
            batch_ids.append(ids)
        return batch_ids

    def _update_batches(self, tokens):
        self.index_to_batch = {}

        batches = self._split_tokens_in_batches(tokens)
        for batch_inputs, batch_targets in batches:
            # Lookup the tokens to ids
            batch_input_ids = self._batch_tokens_to_ids(batch_inputs)
            batch_target_ids = self._batch_tokens_to_ids(batch_targets)

            # Each batch is a tuple with inputs and targets
            self.index_to_batch[self.next_batch_index_to_load] = (batch_input_ids, batch_target_ids)
            self.next_batch_index_to_load += 1

        LOGGER.debug("Loaded batches %d to %d", min(self.index_to_batch.keys()),
                     max(self.index_to_batch.keys()))

    def _split_tokens_in_batches(self, tokens):
        # Start the rows of batches at equidistant points in the tokens

        # The batches within a set of tokens (about 1MB of size) should follow each other exactly.
        # The sets of tokens don't need to follow each other exactly. We can neglect this
        # incorrectness but should reset the state before each set of tokens

        num_batches = math.ceil((len(tokens) - 1) / float(self.num_timesteps * self.batch_size))

        # Each batch is a tuple with inputs and targets
        batches = [([], []) for _ in range(num_batches)]

        token_index = 0
        for row_index in range(self.batch_size):
            # Take rows of batches for a specific row_index consisting of
            # num_batches * num_timesteps tokens
            rows_inputs = []
            rows_targets = []
            while (token_index < len(tokens) - 1
                   and len(rows_inputs) < num_batches * self.num_timesteps):
                rows_inputs.append(tokens[token_index])
                rows_targets.append(tokens[token_index + 1])
                token_index += 1

            # Fill up the rows_tokens if we reached the end of the tokens first
            assert len(rows_inputs) == len(rows_targets)
            if len(rows_inputs) < num_batches * self.num_timesteps:
                missing = num_batches * self.num_timesteps - len(rows_inputs)
                rows_inputs += [PAD_TOKEN for _ in range(missing)]
                rows_targets += [PAD_TOKEN for _ in range(missing)]

            # Split up the rows_tokens to distribute them to the batches
            batch_index_to_inputs_row = get_chunks(rows_inputs, self.num_timesteps)
            batch_index_to_targets_row = get_chunks(rows_targets, self.num_timesteps)

            assert len(batch_index_to_inputs_row) == num_batches

            # Append the new rows to the batches
            for batch_index, batch in enumerate(batches):
                inputs_row = batch_index_to_inputs_row[batch_index]
                targets_row = batch_index_to_targets_row[batch_index]
                batch[0].append(inputs_row)
                batch[1].append(targets_row)

        return batches

    def __iter__(self):
        self.current_iter_batch = 0
        return self

    def __next__(self):
        batch = self.get_batch(self.current_iter_batch)
        if not batch:
            raise StopIteration()
        self.current_iter_batch += 1
        return batch


class Vocabulary:
    def __init__(self, token_to_id, level):
        self.level = level
        self.token_to_id = token_to_id
        # Reverse the token to id dict
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def save_to_dir(self, dir):
        out_path = os.path.join(dir, VOCAB_FILE_NAME)
        with open(out_path, 'w') as f:
            json.dump(self.token_to_id, f)

    def tokens_to_ids(self, tokens):
        if type(tokens) != list:
            raise TypeError("tokens need to be of type list, but are {}".format(type(tokens)))

        ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_id = self.token_to_id[token]
            else:
                token_id = self.token_to_id[UNK_TOKEN]
            ids.append(token_id)

        return ids

    def ids_to_tokens(self, ids):
        return [self.id_to_token[i] for i in ids]

    def get_size(self):
        return len(self.token_to_id)

    @staticmethod
    def load_or_create(save_dir, text_path, max_vocab_size, level="char"):
        if not save_dir:
            return Vocabulary.create_from_text(text_path, max_vocab_size=max_vocab_size,
                                               level=level)

        # Try to reload
        try:
            return Vocabulary.load_from_dir(save_dir, level=level)
        except IOError:
            return Vocabulary.create_from_text(text_path, max_vocab_size=max_vocab_size,
                                               level=level)

    @staticmethod
    def load_from_dir(save_dir, level="char"):
        out_path = os.path.join(save_dir, VOCAB_FILE_NAME)
        with open(out_path) as f:
            token_to_id = json.load(f)
        return Vocabulary(token_to_id, level)

    @staticmethod
    def create_from_text(text_path, max_vocab_size, level="char"):
        LOGGER.info("Creating vocabulary from %s", text_path)

        # Get the most common tokens from the text
        token_counter = Counter()
        for tokens in TextIterator(text_path, level, bytes_in_memory=1000000):
            token_counter.update(tokens)

        # Get the id for each of the most common tokens
        token_to_id = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1
        }
        for token, _ in token_counter.most_common(max_vocab_size - len(token_to_id)):
            token_to_id[token] = len(token_to_id)

        return Vocabulary(token_to_id, level)


class TextIterator:
    def __init__(self, path, level, bytes_in_memory):
        self.path = path
        self.level = level
        self.current_byte = 0
        self.bytes_per_step = bytes_in_memory

    def __iter__(self):
        return self

    def __next__(self):
        # Load the text into memory in x MB parts

        # This will break words at the end. At char-level this is no problem. At word level
        # this introduces a little incorrectness. But with 1MB steps, it shouldn't be a problem.

        with open(self.path) as f:
            f.seek(self.current_byte)
            part = f.read(self.bytes_per_step)
            self.current_byte += self.bytes_per_step

            if part:
                # Tokenize the part based on the level
                return tokenize(part, level=self.level)
            else:
                raise StopIteration()
