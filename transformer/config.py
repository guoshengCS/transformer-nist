class TrainTaskConfig(object):
    use_gpu = True
    # the epoch number to train.
    pass_num = 50
    # the number of sequences contained in a mini-batch.
    batch_size = 56
    # the hyper parameters for Adam optimizer.
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.98
    eps = 1e-9
    # the parameters for learning rate scheduling.
    warmup_steps = 15000
    # the flag indicating to use average loss or sum loss when training.
    use_avg_cost = True  #False
    # the weight used to mix up the ground-truth distribution and the fixed
    # uniform distribution in label smoothing when training.
    # Set this as zero if label smoothing is not wanted.
    label_smooth_eps = 0.1
    # the directory for saving trained models.
    model_dir = "trained_models"


class InferTaskConfig(object):
    use_gpu = True
    # the number of examples in one run for sequence generation.
    batch_size = 1
    # the parameters for beam search.
    beam_size = 5
    max_length = 100
    # the number of decoded sentences to output.
    n_best = 1
    # the flags indicating whether to output the special tokens.
    output_bos = False
    output_eos = False
    output_unk = False
    # the directory for loading the trained model.
    model_path = 'trained_models_sum/pass_29.infer.model'


class ModelHyperParams(object):
    # This model directly uses paddle.dataset.wmt16 in which <bos>, <eos> and
    # <unk> token has alreay been added. As for the <pad> token, any token
    # included in dict can be used to pad, since the paddings' loss will be
    # masked out and make no effect on parameter gradients.
    # size of source word dictionary.
    src_vocab_size = 30001
    # size of target word dictionay
    trg_vocab_size = 30001
    # index for <bos> token
    bos_idx = 0
    # index for <eos> token
    eos_idx = 1
    # index for <unk> token
    unk_idx = 2
    # max length of sequences.
    # The size of position encoding table should at least plus 1, since the
    # sinusoid position encoding starts from 1 and 0 can be used as the padding
    # token for position encoding.
    max_length = 150
    # the dimension for word embeddings, which is also the last dimension of
    # the input and output of multi-head attention, position-wise feed-forward
    # networks, encoder and decoder.
    d_model = 512
    # size of the hidden layer in position-wise feed-forward networks.
    d_inner_hid = 2048
    # the dimension that keys are projected to for dot-product attention.
    d_key = 64
    # the dimension that values are projected to for dot-product attention.
    d_value = 64
    # number of head used in multi-head attention.
    n_head = 8
    # number of sub-layers to be stacked in the encoder and decoder.
    n_layer = 6
    # dropout rate used by all dropout layers.
    dropout = 0.1


# Names of position encoding table which will be initialized externally.
pos_enc_param_names = (
    "src_pos_enc_table",
    "trg_pos_enc_table", )
# Names of all data layers in encoder listed in order.
encoder_input_data_names = (
    "src_word",
    "src_pos",
    "src_slf_attn_bias",
    "src_data_shape",
    "src_slf_attn_pre_softmax_shape",
    "src_slf_attn_post_softmax_shape", )
# Names of all data layers in decoder listed in order.
decoder_input_data_names = (
    "trg_word",
    "trg_pos",
    "trg_slf_attn_bias",
    "trg_src_attn_bias",
    "trg_data_shape",
    "trg_slf_attn_pre_softmax_shape",
    "trg_slf_attn_post_softmax_shape",
    "trg_src_attn_pre_softmax_shape",
    "trg_src_attn_post_softmax_shape",
    "enc_output", )
# Names of label related data layers listed in order.
label_data_names = (
    "lbl_word",
    "lbl_weight", )
encoder_data_input_fields = (
    "src_word",
    "src_pos",
    "src_slf_attn_bias", )
encoder_util_input_fields = (
    "src_data_shape",
    "src_slf_attn_pre_softmax_shape",
    "src_slf_attn_post_softmax_shape", )
decoder_data_input_fields = (
    "trg_word",
    "trg_pos",
    "trg_slf_attn_bias",
    "trg_src_attn_bias",
    "enc_output", )
decoder_util_input_fields = (
    "trg_data_shape",
    "trg_slf_attn_pre_softmax_shape",
    "trg_slf_attn_post_softmax_shape",
    "trg_src_attn_pre_softmax_shape",
    "trg_src_attn_post_softmax_shape", )
input_descs = {
    "src_word": [(1 * (ModelHyperParams.max_length + 1), 1L), "int64"],
    "src_pos": [(1 * (ModelHyperParams.max_length + 1), 1L), "int64"],
    "src_slf_attn_bias":
    [(1, ModelHyperParams.n_head, (ModelHyperParams.max_length + 1),
      (ModelHyperParams.max_length + 1)), "float32"],
    "src_data_shape": [(3L, ), "int32"],
    "src_slf_attn_pre_softmax_shape": [(2L, ), "int32"],
    "src_slf_attn_post_softmax_shape": [(4L, ), "int32"],
    "trg_word": [(1 * (ModelHyperParams.max_length + 1), 1L), "int64"],
    "trg_pos": [(1 * (ModelHyperParams.max_length + 1), 1L), "int64"],
    "trg_slf_attn_bias": [(1, ModelHyperParams.n_head,
                           (ModelHyperParams.max_length + 1),
                           (ModelHyperParams.max_length + 1)), "float32"],
    "trg_src_attn_bias": [(1, ModelHyperParams.n_head,
                           (ModelHyperParams.max_length + 1),
                           (ModelHyperParams.max_length + 1)), "float32"],
    "trg_data_shape": [(3L, ), "int32"],
    "trg_slf_attn_pre_softmax_shape": [(2L, ), "int32"],
    "trg_slf_attn_post_softmax_shape": [(4L, ), "int32"],
    "trg_src_attn_pre_softmax_shape": [(2L, ), "int32"],
    "trg_src_attn_post_softmax_shape": [(4L, ), "int32"],
    "enc_output": [(1, (ModelHyperParams.max_length + 1),
                    ModelHyperParams.d_model), "float32"],
    "lbl_word": [(1 * (ModelHyperParams.max_length + 1), 1L), "int64"],
    "lbl_weight": [(1 * (ModelHyperParams.max_length + 1), 1L), "float32"],
}
