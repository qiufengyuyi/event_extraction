# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops
import re
import tensorflow as tf


# def create_optimizer(loss, init_lr, num_train_steps,num_warmup_steps, clip_norm):
#     """Creates an optimizer training op."""
#     global_step = tf.train.get_or_create_global_step()

#     learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

#     # Implements linear decay of the learning rate.
#     # learning_rate = tf.train.polynomial_decay(
#     #     learning_rate,
#     #     global_step,
#     #     num_train_steps,
#     #     end_learning_rate=0.0,
#     #     power=1.0,
#     #     cycle=False)

#     # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
#     # learning rate will be `global_step/num_warmup_steps * init_lr`.
#     # if num_warmup_steps:
#     #   global_steps_int = tf.cast(global_step, tf.int32)
#     #   warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

#     #   global_steps_float = tf.cast(global_steps_int, tf.float32)
#     #   warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

#     #   warmup_percent_done = global_steps_float / warmup_steps_float
#     #   warmup_learning_rate = init_lr * warmup_percent_done

#     #   is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
#     #   learning_rate = (
#     #       (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

#     # It is recommended that you use this optimizer for fine tuning, since this
#     # is how the model was trained (note that the Adam m/v variables are NOT
#     # loaded from init_checkpoint.)
#     # optimizer = tf.train.AdamOptimizer(
#     #     learning_rate=learning_rate)
#     different_layer_vars = []

#     optimzer_list = []
#     tvars = tf.trainable_variables()
#     print(len(tvars))
#     # print(embedding_vars)
#     one_layer_vars_len = 0
#     var_name_list = []
#     list_24 = []
#     list_18 = []
#     list_12 = []
#     list_6 = []
#     for layer in range(24):
#         if layer <24 and layer >=18:
#             cur_layer_lr = learning_rate
#             layer_pattern = r"bert.*layer_{}\/".format(layer)
#             cur_var_list = [var for var in tvars if re.search(r""+layer_pattern,var.name)]
#             one_layer_vars_len = len(cur_var_list)
#             list_24.extend(cur_var_list)
#             var_name_list.extend([var.name for var in cur_var_list])
#         elif layer < 18 and layer >= 12:
#             cur_layer_lr = learning_rate/2.0
#             layer_pattern = r"bert.*layer_{}\/".format(layer)
#             cur_var_list = [var for var in tvars if re.search(r""+layer_pattern,var.name)]
#             one_layer_vars_len = len(cur_var_list)
#             list_18.extend(cur_var_list)
#             var_name_list.extend([var.name for var in cur_var_list])
#         elif layer < 12 and layer <=6:
#             cur_layer_lr = learning_rate / 4.0
#             layer_pattern = r"bert.*layer_{}\/".format(layer)
#             cur_var_list = [var for var in tvars if re.search(r""+layer_pattern,var.name)]
#             one_layer_vars_len = len(cur_var_list)
#             list_12.extend(cur_var_list)
#             var_name_list.extend([var.name for var in cur_var_list])
#         else:
#             cur_layer_lr = learning_rate / 8.0
#             layer_pattern = r"bert.*layer_{}\/".format(layer)
#             cur_var_list = [var for var in tvars if re.search(r""+layer_pattern,var.name)]
#             one_layer_vars_len = len(cur_var_list)
#             list_6.extend(cur_var_list)
#             var_name_list.extend([var.name for var in cur_var_list])
#         if layer == 18 or layer == 12 or layer == 6 or layer == 0:
#             optimzer_list.append(AdamWeightDecayOptimizer(learning_rate=cur_layer_lr,weight_decay_rate=0.01,beta_1=0.9,beta_2=0.999,epsilon=1e-6,exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]))
#     other_vars = [var for var in tvars if var.name not in var_name_list]
#     # print(len(different_layer_vars))
#     print(other_vars)
#     # print(other_vars)
#     other_optimizer = AdamWeightDecayOptimizer(
#       learning_rate=learning_rate,
#       weight_decay_rate=0.01,
#       beta_1=0.9,
#       beta_2=0.999,
#       epsilon=1e-6,
#       exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
#     grads = tf.gradients(loss, list_24+list_18+list_12+list_6+other_vars)

#     # This is how the model was pre-trained.
#     (grads, _) = tf.clip_by_global_norm(grads, clip_norm=clip_norm)
#     train_op_list = []
#     for i in range(4):
#         if i == 0:
#             grads_layer_cur_layer = grads[0:len(list_24)]
#             cur_train_op = optimzer_list[i].apply_gradients(zip(grads_layer_cur_layer, list_24), global_step=global_step)
#             train_op_list.append(cur_train_op)
#         elif i == 1:
#             grads_layer_cur_layer = grads[len(list_24):len(list_24)+len(list_18)]
#             cur_train_op = optimzer_list[i].apply_gradients(zip(grads_layer_cur_layer, list_18), global_step=global_step)
#             train_op_list.append(cur_train_op)
#         elif i == 2:
#             grads_layer_cur_layer = grads[len(list_24)+len(list_18):len(list_24)+len(list_18)+len(list_12)]
#             cur_train_op = optimzer_list[i].apply_gradients(zip(grads_layer_cur_layer, list_12), global_step=global_step)
#             train_op_list.append(cur_train_op)
#         else:
#             grads_layer_cur_layer = grads[len(list_24)+len(list_18)+len(list_12):len(list_24)+len(list_18)+len(list_12)+len(list_6)]
#             cur_train_op = optimzer_list[i].apply_gradients(zip(grads_layer_cur_layer, list_6), global_step=global_step)
#             train_op_list.append(cur_train_op)
#     # for i in range(24):
#     #         grads_layer_cur_layer = grads[one_layer_vars_len*i:one_layer_vars_len*(i+1)]
#     #         cur_train_op = optimzer_list[i].apply_gradients(zip(grads_layer_cur_layer, different_layer_vars[one_layer_vars_len*i:one_layer_vars_len*(i+1)]), global_step=global_step)
#     #         train_op_list.append(cur_train_op)

#     grads_layer_others = grads[len(list_24)+len(list_18)+len(list_12)+len(list_6):]
#     # print(grads_layer_others)
#     other_train_op = other_optimizer.apply_gradients(zip(grads_layer_others, other_vars), global_step=global_step)

#     train_op_list.append(other_train_op)
#     # grads_nocrf = grads[:len(tvars_no_crf)]
#     # grads_crf = grads[len(tvars_no_crf):]
#     # train_op_1 = optimizer.apply_gradients(
#     #     zip(grads_nocrf, tvars_no_crf), global_step=global_step)
#     # train_op_2 = optimizer_2.apply_gradients(
#     #     zip(grads_crf, tvars_crf), global_step=global_step)

#     # train_op = optimizer.apply_gradients(
#     #     zip(grads, tvars), global_step=global_step)

#     new_global_step = global_step + 1
#     train_op = tf.group(train_op_list, [global_step.assign(new_global_step)])
#     return train_op

def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):
    """Creates an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    # # Implements linear decay of the learning rate.
    # learning_rate = tf.train.polynomial_decay(
    #     learning_rate,
    #     global_step,
    #     num_train_steps,
    #     end_learning_rate=0.0,
    #     power=1.0,
    #     cycle=False)

    # # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # # learning rate will be `global_step/num_warmup_steps * init_lr`.
    # if num_warmup_steps:
    #     global_steps_int = tf.cast(global_step, tf.int32)
    #     warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    #     global_steps_float = tf.cast(global_steps_int, tf.float32)
    #     warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    #     warmup_percent_done = global_steps_float / warmup_steps_float
    #     warmup_learning_rate = init_lr * warmup_percent_done

    #     is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    #     learning_rate = (
    #         (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=5.0)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op


def create_optimizer_for_bert_theseus(loss, init_lr, num_train_steps, clip_norm, finetune_suc=False):
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    # learning_rate = tf.train.polynomial_decay(
    #     learning_rate,
    #     global_step,
    #     num_train_steps,
    #     end_learning_rate=0.0,
    #     power=1.0,
    #     cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    # if num_warmup_steps:
    #   global_steps_int = tf.cast(global_step, tf.int32)
    #   warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
    #
    #   global_steps_float = tf.cast(global_steps_int, tf.float32)
    #   warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
    #
    #   warmup_percent_done = global_steps_float / warmup_steps_float
    #   warmup_learning_rate = init_lr * warmup_percent_done
    #
    #   is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    #   learning_rate = (
    #       (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    tvars = tf.trainable_variables()

    if not finetune_suc:
        # only compute suc_layer
        tvars_update = [var for var in tvars if re.search(
            r"suc_layer_\d+", var.name)]
    else:
        tvars_update = tvars
    grads = tf.gradients(loss, tvars_update)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=clip_norm)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars_update), global_step=global_step)

    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op


def create_optimizer_for_bert_distilling(loss, init_lr, num_train_steps, clip_norm, grad_update_var_prefix="ta_bert"):
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    # learning_rate = tf.train.polynomial_decay(
    #     learning_rate,
    #     global_step,
    #     num_train_steps,
    #     end_learning_rate=0.0,
    #     power=1.0,
    #     cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    # if num_warmup_steps:
    #   global_steps_int = tf.cast(global_step, tf.int32)
    #   warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
    #
    #   global_steps_float = tf.cast(global_steps_int, tf.float32)
    #   warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
    #
    #   warmup_percent_done = global_steps_float / warmup_steps_float
    #   warmup_learning_rate = init_lr * warmup_percent_done
    #
    #   is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    #   learning_rate = (
    #       (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    tvars = tf.trainable_variables()

    tvars_update = [var for var in tvars if re.search(
        r""+grad_update_var_prefix, var.name)]
    # print(tvars_update)
    grads = tf.gradients(loss, tvars_update)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=clip_norm)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars_update), global_step=global_step)

    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op


def create_optimizer_for_bertcrf(loss, init_lr, clip_norm):
    """Creates an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    optimizer_2 = AdamWeightDecayOptimizer(
        learning_rate=learning_rate * 100,
        weight_decay_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    tvars = tf.trainable_variables()
    # grads = tf.gradients(loss, tvars)
    tvars_no_crf = [var for var in tvars if not re.search(r"crf", var.name)]
    tvars_crf = [var for var in tvars if re.search(r"crf", var.name)]
#     print([var.name for var in tvars_crf])
    grads = tf.gradients(loss, tvars_no_crf + tvars_crf)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=clip_norm)
    grads_nocrf = grads[:len(tvars_no_crf)]
    grads_crf = grads[len(tvars_no_crf):]
    train_op_1 = optimizer.apply_gradients(
        zip(grads_nocrf, tvars_no_crf), global_step=global_step)
    train_op_2 = optimizer_2.apply_gradients(
        zip(grads_crf, tvars_crf), global_step=global_step)

    new_global_step = global_step + 1
    train_op = tf.group(train_op_1, train_op_2, [
                        global_step.assign(new_global_step)])
    return train_op


class AdamWeightDecayOptimizer(optimizer.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def _prepare(self):
        self.learning_rate_t = ops.convert_to_tensor(
            self.learning_rate, name='learning_rate')
        self.weight_decay_rate_t = ops.convert_to_tensor(
            self.weight_decay_rate, name='weight_decay_rate')
        self.beta_1_t = ops.convert_to_tensor(self.beta_1, name='beta_1')
        self.beta_2_t = ops.convert_to_tensor(self.beta_2, name='beta_2')
        self.epsilon_t = ops.convert_to_tensor(self.epsilon, name='epsilon')

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, 'm', self._name)
            self._zeros_slot(v, 'v', self._name)

    def _apply_dense(self, grad, var):
        learning_rate_t = math_ops.cast(
            self.learning_rate_t, var.dtype.base_dtype)
        beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
        beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
        weight_decay_rate_t = math_ops.cast(
            self.weight_decay_rate_t, var.dtype.base_dtype)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        # Standard Adam update.
        next_m = (
            tf.multiply(beta_1_t, m) +
            tf.multiply(1.0 - beta_1_t, grad))
        next_v = (
            tf.multiply(beta_2_t, v) + tf.multiply(1.0 - beta_2_t,
                                                   tf.square(grad)))

        update = next_m / (tf.sqrt(next_v) + epsilon_t)

        if self._do_use_weight_decay(var.name):
            update += weight_decay_rate_t * var

        update_with_lr = learning_rate_t * update

        next_param = var - update_with_lr

        return control_flow_ops.group(*[var.assign(next_param),
                                        m.assign(next_m),
                                        v.assign(next_v)])

    def _resource_apply_dense(self, grad, var):
        learning_rate_t = math_ops.cast(
            self.learning_rate_t, var.dtype.base_dtype)
        beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
        beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
        weight_decay_rate_t = math_ops.cast(
            self.weight_decay_rate_t, var.dtype.base_dtype)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        # Standard Adam update.
        next_m = (
            tf.multiply(beta_1_t, m) +
            tf.multiply(1.0 - beta_1_t, grad))
        next_v = (
            tf.multiply(beta_2_t, v) + tf.multiply(1.0 - beta_2_t,
                                                   tf.square(grad)))

        update = next_m / (tf.sqrt(next_v) + epsilon_t)

        if self._do_use_weight_decay(var.name):
            update += weight_decay_rate_t * var

        update_with_lr = learning_rate_t * update

        next_param = var - update_with_lr

        return control_flow_ops.group(*[var.assign(next_param),
                                        m.assign(next_m),
                                        v.assign(next_v)])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        learning_rate_t = math_ops.cast(
            self.learning_rate_t, var.dtype.base_dtype)
        beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
        beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
        weight_decay_rate_t = math_ops.cast(
            self.weight_decay_rate_t, var.dtype.base_dtype)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        m_t = state_ops.assign(m, m * beta_1_t,
                               use_locking=self._use_locking)

        m_scaled_g_values = grad * (1 - beta_1_t)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)

        v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
        v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)

        update = m_t / (math_ops.sqrt(v_t) + epsilon_t)

        if self._do_use_weight_decay(var.name):
            update += weight_decay_rate_t * var

        update_with_lr = learning_rate_t * update

        var_update = state_ops.assign_sub(var,
                                          update_with_lr,
                                          use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
                x, i, v, use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
                [resource_variable_ops.resource_scatter_add(
                    x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(
            grad, var, indices, self._resource_scatter_add)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True
