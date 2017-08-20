import json
import os
from collections import Counter
from nltk.tokenize import RegexpTokenizer

from src.generating_lstm.common.tokens import PAD_TOKEN, UNK_TOKEN


class Dataset:
    def __init__(self, path, vocab, batch_size, level="char"):
        self.path = path
        self.level = level
        self.vocab = vocab
        self.batch_size = batch_size
        self.current_iter_batch = 0

        self.text_iter = TextIter(self.path, self.level)
        self.next_batch_index_to_load = 0
        self.index_to_batch = {}

    def get_batch(self, batch_index):
        if batch_index in self.index_to_batch:
            return self.index_to_batch[batch_index]

        # Check if the text_iter is already past the batch_index
        if batch_index < self.next_batch_index_to_load:
            # Reset the file iterator to 0 to start again
            self.text_iter = TextIter(self.path, self.level)
            self.next_batch_index_to_load = 0

        # Load from the text iterator until the current_batch_index equals the batch_index
        tokens = self.text_iter.__next__()
        self.update_batches(tokens)
        return self.get_batch(batch_index)

    def update_batches(self, tokens):
        self.index_to_batch = {}

        # Get all batches from the tokens
        for inputs_start in range(0, len(tokens) - 1, self.batch_size):
            inputs = tokens[inputs_start: inputs_start + self.batch_size]
            targets = tokens[inputs_start + 1: inputs_start + 1 + self.batch_size]

            # Pad the inputs and targets
            if len(inputs) < self.batch_size:
                missing = self.batch_size - len(inputs)
                inputs += [PAD_TOKEN for _ in range(missing)]

            if len(targets) < self.batch_size:
                missing = self.batch_size - len(targets)
                targets += [PAD_TOKEN for _ in range(missing)]

            # Lookup the tokens to ids
            input_ids = self.vocab.ids_for_tokens(inputs)
            target_ids = self.vocab.ids_for_tokens(targets)

            self.index_to_batch[self.next_batch_index_to_load] = (input_ids, target_ids)
            self.next_batch_index_to_load += 1

    def __iter__(self):
        return self

    def __next__(self):
        # The text iter will automatically raise the StopIteration
        batch = self.get_batch(self.current_iter_batch)
        self.current_iter_batch += 1
        return batch


class Vocabulary:
    def __init__(self, token_to_id):
        self.token_to_id = token_to_id
        # Reverse the token to id dict
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def save_to_path(self, path):
        with open(path, 'w') as f:
            json.dump(self.token_to_id, f)

    def ids_for_tokens(self, tokens):
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_id = self.token_to_id[token]
            else:
                token_id = self.token_to_id[UNK_TOKEN]
            ids.append(token_id)
        return ids

    @staticmethod
    def load_from_path(path):
        with open(path) as f:
            token_to_id = json.load(f)
        return Vocabulary(token_to_id)

    @staticmethod
    def create_from_text(text_path, max_vocab_size=60, level="char"):
        # Get the most common tokens from the text
        token_counter = Counter()
        for tokens in TextIter(text_path, level):
            token_counter.update(tokens)

        # Get the id for each of the most common tokens
        token_to_id = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1
        }
        for token, _ in token_counter.most_common(max_vocab_size - len(token_to_id)):
            token_to_id[token] = len(token_to_id)

        return Vocabulary(token_to_id)


class TextIter:
    def __init__(self, path, level, bytes_in_memory=1000000):
        self.path = path
        self.level = level
        self.current_byte = 0
        self.bytes_per_step = bytes_in_memory

    def __iter__(self):
        return self

    def __next__(self):
        # Load the text into memory in x MB parts

        # TODO: This will break words at the end. At char-level this is no problem. At word level
        # this introduces a little incorrectness. But with 1MB steps, it shouldn't be a problem.

        with open(self.path) as f:
            f.seek(self.current_byte)
            part = f.read(self.bytes_per_step)
            self.current_byte += self.bytes_per_step

            if part:
                # Tokenize the part based on the level
                if self.level == "char":
                    # No need for tokenizing
                    return list(part)
                elif self.level == "word":
                    # Tokenize while keeping indentation. Glue letters and numbers to themselves but
                    # keep all other chars isolated
                    tokenizer = RegexpTokenizer(r'\w+|\S|\s')
                    return tokenizer.tokenize(part)
                else:
                    raise ValueError("Unknown token level: {}".format(self.level))

            else:
                raise StopIteration()
