# Copyright (c) 2017 Kilian Batzner All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
"""Helper module for using a .txt file as dataset."""

import json
import os
from collections import Counter

import math
from nltk.tokenize import RegexpTokenizer

from tensorlm.common.log import get_logger
from tensorlm.common.util import get_chunks

LOGGER = get_logger(__name__)

# Special tokens to include in the vocabulary
PAD_TOKEN = "_PAD"
UNK_TOKEN = "_UNK"


def tokenize(text, level):
    """Tokenize a text into a list of strings.

    Args:
        text (str): An arbitrary string.
        level (str): Either "char" or "word". For "char", the string is split into characters. For
            "word", letters and numbers are glued to themselves and everything else is split.
            Example: "asdf df!?123 as12" -> "asdf", " ", "df", "!", "?", "123", " ", "as", "12"

    Returns:
        list[str]: The tokens

    Raises:
        ValueError: If the level is not "char" or "word"
    """

    if level == "char":
        # No need for tokenizing
        return list(text)
    elif level == "word":
        # Tokenize while keeping indentation. Glue letters and numbers to themselves but
        # keep all other chars isolated.
        tokenizer = RegexpTokenizer(r'\w+|\S|\s')
        return tokenizer.tokenize(text)
    else:
        raise ValueError("Unknown token level: {}".format(level))


class Dataset:
    """Helps iterating a .txt file in batches.

    This class can be used to load a .txt file chunk by chunk into memory (1MB chunks per default),
    tokenize the text, translate it into ids and return the ids batch by batch.

    For iterating through the whole dataset either use it as an iterator or call get_batch() with
    increasing batch indices.
    """

    def __init__(self, path, vocab, batch_size, num_timesteps, bytes_in_memory=1000000):
        """Create a new dataset for iterating over a .txt file.

        Args:
            path (str): Path to the .txt dataset file.
            vocab (Vocabulary): The vocabulary with which to translate the tokens to ids
            batch_size (int): The size of the batches returned
            num_timesteps (int): The number of timesteps / tokens in each batch row
            bytes_in_memory (int): The number of bytes that should be loaded into memory at once
        """

        self._path = path
        self._vocab = vocab
        self._batch_size = batch_size
        self._num_timesteps = num_timesteps
        self._bytes_in_memory = bytes_in_memory

        self._text_iter = TextIterator(self._path, self._vocab.level, bytes_in_memory)
        self._next_batch_index_to_load = 0
        self._index_to_batch = {} # Batches currently loaded into memory

    def get_batch(self, batch_index):
        """Return a new batch for the given index.

        If the desired batch is currently loaded into memory it is returned directly. Otherwise,
        the function iterates over the dataset file until it finds the batch.

        Args:
            batch_index (int): The index of the batch to load / return.

        Returns:
            tuple[list[list[int]]]: A tuple of the batch inputs and batch targets. Both items have
                the same type and shape. They both are a list (length = batch_size) of lists
                (length = num_timesteps) of integers (the tokens).

                Returns None if the batch_index exceeds the number of batches in the text / at the
                end of an epoch.
        """

        if batch_index in self._index_to_batch:
            return self._index_to_batch[batch_index]

        # Check if the text_iter is already past the batch_index
        if batch_index < self._next_batch_index_to_load:
            # Reset the file iterator to 0 to start again
            self._text_iter = TextIterator(self._path, self._vocab.level, self._bytes_in_memory)
            self._next_batch_index_to_load = 0

        try:
            # Load from the text iterator until the current_batch_index equals the batch_index
            tokens = self._text_iter.__next__()
            self._update_batches(tokens)
            return self.get_batch(batch_index)
        except StopIteration:
            # We finished an epoch
            return None

    def _batch_tokens_to_ids(self, batch):
        # Translate an 2D array of tokens to ids
        batch_ids = []
        for batch_item in batch:
            ids = self._vocab.tokens_to_ids(batch_item)
            batch_ids.append(ids)
        return batch_ids

    def _update_batches(self, tokens):
        # Clear the memory
        self._index_to_batch = {}

        batches = self._split_tokens_in_batches(tokens)
        for batch_inputs, batch_targets in batches:
            # Lookup the tokens to ids
            batch_input_ids = self._batch_tokens_to_ids(batch_inputs)
            batch_target_ids = self._batch_tokens_to_ids(batch_targets)

            # Each batch is a tuple with inputs and targets
            self._index_to_batch[self._next_batch_index_to_load] = (batch_input_ids, batch_target_ids)
            self._next_batch_index_to_load += 1

    def _split_tokens_in_batches(self, tokens):
        # Start the rows of batches at equidistant points in the tokens

        # The batches within a set of tokens (about 1MB of size) should follow each other exactly.
        # The sets of tokens don't need to follow each other exactly. We can neglect this
        # incorrectness but should reset the state before each set of tokens

        num_batches = math.ceil((len(tokens) - 1) / float(self._num_timesteps * self._batch_size))

        # Each batch is a tuple with inputs and targets
        batches = [([], []) for _ in range(num_batches)]

        token_index = 0
        for row_index in range(self._batch_size):
            # Take rows of batches for a specific row_index consisting of
            # num_batches * num_timesteps tokens
            rows_inputs = []
            rows_targets = []
            while (token_index < len(tokens) - 1
                   and len(rows_inputs) < num_batches * self._num_timesteps):
                rows_inputs.append(tokens[token_index])
                rows_targets.append(tokens[token_index + 1])
                token_index += 1

            # Fill up the rows_tokens if we reached the end of the tokens first
            assert len(rows_inputs) == len(rows_targets)
            if len(rows_inputs) < num_batches * self._num_timesteps:
                missing = num_batches * self._num_timesteps - len(rows_inputs)
                rows_inputs += [PAD_TOKEN for _ in range(missing)]
                rows_targets += [PAD_TOKEN for _ in range(missing)]

            # Split up the rows_tokens to distribute them to the batches
            batch_index_to_inputs_row = get_chunks(rows_inputs, self._num_timesteps)
            batch_index_to_targets_row = get_chunks(rows_targets, self._num_timesteps)

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
        """Generates batches out of the dataset one by one.

        See get_batch() for more info.

        Returns:
            tuple[list[list[int]]]: A tuple of the batch inputs and batch targets. Both items have
                the same type and shape. They both are a list (length = batch_size) of lists
                (length = num_timesteps) of integers (the tokens).
        """
        batch = self.get_batch(self.current_iter_batch)
        if not batch:
            raise StopIteration()
        self.current_iter_batch += 1
        return batch


class Vocabulary:
    """Generates a token -> id vocabulary on a .txt file.

    Only the most frequent tokens are kept in the vocabulary.

    For creation, see create_from_text().
    For loading an already generated vocabulary, see load_from_dir().
    For trying loading and falling back to creation if that failed, see load_or_create().
    """
    vocab_file_name = "vocab.json"

    def __init__(self, token_to_id, level):
        """Constructor that expects an already generated vocabulary.

        Args:
            token_to_id (dict[str:int]): A dict mapping tokens to ids.
            level (str): Either "char" or "word". The level to use for tokenizing.
        """

        self.level = level
        self.token_to_id = token_to_id
        # Reverse the token to id dict
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def save_to_dir(self, out_dir):
        """Saves the vocabulary to a vocab.json file in the given directory.

        Args:
            out_dir (str): The path to the output directory.
        """
        out_path = os.path.join(out_dir, Vocabulary.vocab_file_name)
        with open(out_path, 'w') as f:
            json.dump(self.token_to_id, f)

    def tokens_to_ids(self, tokens):
        """Translates a list of tokens to ids.

        Unkown tokens will be translated to the id of the UNK_TOKEN

        Args:
            tokens (list[str]): A list of tokens.

        Returns:
            list[int]: The ids.

        Raises:
            TypeError: If the tokens are not a list.
        """

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
        """Translates a list of ids to tokens.

        Args:
            ids (list[int]): A list of ids.

        Returns:
            list[str]: The tokens.
        """
        return [self.id_to_token[i] for i in ids]

    def get_size(self):
        """Returns the size of the vocabulary.

        Returns:
            int: The size.
        """
        return len(self.token_to_id)

    @staticmethod
    def load_or_create(save_dir, text_path, max_vocab_size, level="char"):
        """Try to load a vocabulary from the file system and create one if that failed.

        Args:
            save_dir (str): The path to the directory containing the vocabulary to load. This
                parameter may be None or a nonsense string.
            text_path (str): The path to the dataset file to read if the vocabulary loading failed.
            max_vocab_size (int): Maximum size of the vocabulary.
            level (str): Level for tokenizing - either "char" or "word".

        Returns:
            Vocabulary: The loaded / created vocabulary.
        """

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
        """Loads a vocabulary from the file system.

        Args:
            save_dir (str): The path to the directory containing the vocab.json file.
            level (str): Level for tokenizing - either "char" or "word".

        Returns:
            Vocabulary: The loaded vocabulary.

        Raises:
            IOError: If the vocabulary could not be loaded from the file system.
        """

        out_path = os.path.join(save_dir, Vocabulary.vocab_file_name)
        with open(out_path) as f:
            token_to_id = json.load(f)
        return Vocabulary(token_to_id, level)

    @staticmethod
    def create_from_text(text_path, max_vocab_size, level="char"):
        """Creates a new vocabulary.

        Loads the text file in 1MB chunks, tokenizes the text and counts the tokens. Then, it builds
        a token -> id map for the most frequent tokens.

        Args:
            text_path (str): The path to the dataset file to read if the vocabulary loading failed.
            max_vocab_size (int): Maximum size of the vocabulary.
            level (str): Level for tokenizing - either "char" or "word".

        Returns:
            Vocabulary: The created vocabulary.
        """

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
    """Iterates over potentially large .txt files and returns the content in chunks."""

    def __init__(self, path, level, bytes_in_memory):
        """Creates a new iterator.

        Args:
            path (str): Path to the file to load.
            level (str): Level for tokenizing the loaded parts.
            bytes_in_memory (int): Number of bytes to load into memory at once.
        """
        self.path = path
        self.level = level
        self.current_byte = 0
        self.bytes_per_step = bytes_in_memory

    def __iter__(self):
        return self

    def __next__(self):
        """Loads the next part from the file and tokenizes it

        Returns:
            list[str]: The tokens loaded.
        """

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
