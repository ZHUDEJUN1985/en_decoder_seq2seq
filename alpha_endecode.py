# coding=utf-8

import os
import sys
import codecs
import re
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import numpy as np
from tensorflow.contrib import legacy_seq2seq
from tensorflow.contrib import rnn
from tensorflow.contrib.deprecated import scalar_summary, merge_all_summaries


class HParam():
    def __init__(self):
        pass

    epoch = 100
    batch_size = 128
    max_sequence_length = 12
    rnn_size = 100
    num_layers = 3
    encoding_embedding_size = 12
    decoding_embedding_size = 12
    lr = 0.001
    grad_clip = 5
    log_dir = './logs'
    metadata = 'metadata.tsv'


class DataGenerator():
    def __init__(self, filename1, filename2, args):
        self.batch_size = args.batch_size
        self.data = ''
        f1 = codecs.open(filename1, 'r', encoding='utf-8')
        self.source_data = f1.read()
        f1.close()
        f2 = codecs.open(filename2, 'r', encoding='utf-8')
        self.target_data = f2.read()
        f2.close()

        special_char = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
        words_list_source = list(set([character for line in self.source_data.split('\n') for character in line]))
        words_list_target = list(set([character for line in self.target_data.split('\n') for character in line]))
        self.source_vocab_size = len(words_list_source)
        self.target_vocab_size = len(words_list_target)

        self.source_id_to_char = {idx: word for idx, word in enumerate(special_char + words_list_source)}
        self.source_char_to_id = {word: idx for idx, word in enumerate(special_char + words_list_source)}
        self.target_id_to_char = {idx: word for idx, word in enumerate(special_char + words_list_target)}
        self.target_char_to_id = {word: idx for idx, word in enumerate(special_char + words_list_target)}

        self.pointer = 0
        self.source_vocab = [word for word in self.source_data.split('\n')]
        self.target_vocab = [word for word in self.target_data.split('\n')]
        self.source_pad_id = self.source_char_to_id['<PAD>']
        self.source_go_id = self.source_char_to_id['<GO>']
        self.target_pad_id = self.target_char_to_id['<PAD>']
        self.target_go_id = self.target_char_to_id['<GO>']

        self.source_id = [[self.source_char_to_id.get(letter, self.source_char_to_id['<UNK>']) for letter in line]
                          for line in self.source_data.split('\n')]
        self.target_id = [[self.target_char_to_id.get(letter, self.target_char_to_id['<UNK>']) for letter in line]
                          for line in self.target_data.split('\n')]

    def save_metadata(self, filename):
        with open(filename) as f:
            f.write('id\tchar\n')
            for i in xrange(self.source_vocab_size):
                c = self.source_id_to_char[i]
                f.write('{}\t{}\n'.format(i, c))

    def pad_sentence_batch(self, sentence_batch, pad_int):
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + str(pad_int) * (max_sentence - len(sentence)) for sentence in sentence_batch]

    def next_batch(self):
        sources_batch = self.source_vocab[self.pointer: self.pointer + self.batch_size]
        targets_batch = self.target_vocab[self.pointer: self.pointer + self.batch_size]
        self.pointer += self.batch_size
        if self.pointer > self.source_vocab_size:
            self.pointer = 0

        id_sources_batch = []
        for item in sources_batch:
            temp_item = []
            for ch in item:
                if ch == '0':
                    temp_item.append('0')
                    continue
                ch_to_id = self.source_char_to_id[ch]
                temp_item.append(str(ch_to_id))
            new_item = ''.join(temp_item)
            id_sources_batch.append(new_item)

        id_targets_batch = []
        for item in targets_batch:
            temp_item = []
            for ch in item:
                if ch == '0':
                    temp_item.append('0')
                    continue
                ch_to_id = self.source_char_to_id[ch]
                temp_item.append(str(ch_to_id))
            new_item = ''.join(temp_item)
            id_targets_batch.append(new_item)

        pad_sources_batch = self.pad_sentence_batch(id_sources_batch, self.source_pad_id)
        pad_targets_batch = self.pad_sentence_batch(id_targets_batch, self.target_pad_id)

        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_sources_lengths = []
        for source in pad_sources_batch:
            pad_sources_lengths.append(len(source))

        new_pad_sources_batch = []
        for item_word in pad_sources_batch:
            temp_word = []
            for i in item_word:
                temp_word.append(int(i))
            new_pad_sources_batch.append(temp_word)

        new_pad_targets_batch = []
        for item_word in pad_targets_batch:
            temp_word = []
            for i in item_word:
                temp_word.append(int(i))
            new_pad_targets_batch.append(temp_word)

        return new_pad_sources_batch, new_pad_targets_batch, pad_sources_lengths, pad_targets_lengths


class Model():
    def __init__(self, args, data):
        with tf.name_scope("inputs"):
            # print("#####%d" % args.max_sequence_length)
            self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.max_sequence_length])
            self.target_data = tf.placeholder(tf.int32, [args.batch_size, args.max_sequence_length])
            self.input_data_lengths = tf.placeholder(tf.int32, [args.batch_size])
            self.target_data_lengths = tf.placeholder(tf.int32, [args.batch_size])

        with tf.name_scope("encoder"):
            def get_lstm_cell(rnn_size):
                lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                                    initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                return lstm_cell

            self.cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(args.rnn_size) for _ in range(args.num_layers)])
            self.initial_state = self.cell.zero_state(args.batch_size, tf.float32)

            encoder_embed_input = tf.contrib.layers.embed_sequence(self.input_data, data.source_vocab_size,
                                                                   args.encoding_embedding_size)

            encoder_output, encoder_state = tf.nn.dynamic_rnn(self.cell, encoder_embed_input,
                                                              initial_state=self.initial_state)

        with tf.name_scope("decoder"):
            def process_decoder_input(target_data, begin_id, batch_size):
                ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
                decoder_input = tf.concat([tf.fill([batch_size, 1], begin_id), ending], 1)
                decoder_input_length = []
                for i in range(batch_size):
                    x = decoder_input[i].shape[0]
                    x = int(x)
                    decoder_input_length.append(x)
                return decoder_input, decoder_input_length

            decoder_embeddings = tf.Variable(tf.random_uniform([data.target_vocab_size, args.decoding_embedding_size]))
            new_decoder_input, new_decoder_input_length = process_decoder_input(self.target_data, data.target_go_id,
                                                                                args.batch_size)
            decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, new_decoder_input)

            def get_decoder_cell(rnn_size):
                decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                                       initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                return decoder_cell

            cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(args.rnn_size) for _ in range(args.num_layers)])
            output_layer = layers_core.Dense(data.target_vocab_size,
                                             kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        with tf.variable_scope("decode"):
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                                sequence_length=new_decoder_input_length,
                                                                time_major=False)

            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell, training_helper, encoder_state, output_layer)

            training_decoder_output, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder, impute_finished=True,
                                                                           maximum_iterations=args.max_sequence_length)

        with tf.variable_scope("decode", reuse=True):
            start_tokens = tf.tile(tf.constant([data.target_char_to_id['<GO>']], dtype=tf.int32), [args.batch_size],
                                   name='start_tokens')

            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings, start_tokens,
                                                                         data.target_char_to_id['<EOS>'])
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell, predicting_helper, encoder_state, output_layer)
            predicting_decoder_output, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder, impute_finished=True,
                                                                             maximum_iterations=args.max_sequence_length)

        with tf.name_scope("loss"):
            training_logits = tf.identity(training_decoder_output.rnn_output, name='logits')
            predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')

            # print(self.target_data_lengths)
            # print(self.target_data_lengths.shape)
            # print(type(self.target_data_lengths))
            masks = tf.sequence_mask(self.target_data_lengths, args.max_sequence_length, dtype=tf.float32, name='masks')

            # print(training_logits)
            # print(training_logits.shape)
            # print(self.target_data)
            # print(self.target_data.shape)
            # print(masks)
            # print(masks.shape)
            loss = tf.contrib.seq2seq.sequence_loss(training_logits, self.target_data, masks)
            self.cost = tf.reduce_sum(loss) / args.batch_size

        with tf.name_scope('optimize'):
            self.lr = tf.placeholder(tf.float32, [])
            optimizer = tf.train.AdamOptimizer(self.lr)
            gradients = optimizer.compute_gradients(self.cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            self.train_op = optimizer.apply_gradients(capped_gradients)


def train(data, model, args):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # writer = tf.summary.FileWriter(args.log_dir, sess.graph)

        for i in range(500):
            print("i=%d" % i)
            x_batch, y_batch, x_batch_lengths, y_batch_lengths = data.next_batch()
            # print(x_batch)
            # print(x_batch.shape)
            feed_dict = {model.input_data: x_batch, model.target_data: y_batch,
                         model.target_data_lengths: y_batch_lengths, model.lr: args.lr}
            # print(feed_dict)
            # print(x_batch)
            # print(x_batch.shape)
            # print(model.input_data)
            train_loss, _ = sess.run([model.cost, model.train_op], feed_dict)

            if i % 10 == 0:
                print('Step:{}/{}, training_loss:{:4f}'.format(i, args.epoch, train_loss))
            if i % 100 == 0 or (i + 1) == iter:
                saver.save(sess, os.path.join(args.log_dir, 'alpha_endecode.ckpt'), global_step=i)


def main():
    args = HParam()
    data = DataGenerator('sources.txt', 'targets.txt', args)
    model = Model(args, data)
    train(data, model, args)


if __name__ == '__main__':
    main()
