mindspore.mint
===============

mindspore.mint提供了大量的functional、nn、优化器接口，API用法及功能等与业界主流用法一致，方便用户参考使用。
mint接口当前是实验性接口，在图编译模式为O0和PyNative模式下性能比ops更优。当前暂不支持图下沉模式及CPU、GPU后端，后续会逐步完善。

模块导入方法如下：

.. code-block::

    from mindspore import mint

MindSpore中 `mindspore.mint` 接口与上一版本相比，新增、删除和支持平台的变化信息请参考 `mindspore.mint API接口变更 <https://gitee.com/mindspore/docs/blob/master/resource/api_updates/mint_api_updates_cn.md>`_ 。

Tensor
---------------

创建运算
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.arange
    mindspore.mint.bernoulli
    mindspore.mint.bincount
    mindspore.mint.clone
    mindspore.mint.eye
    mindspore.mint.einsum
    mindspore.mint.empty
    mindspore.mint.empty_like
    mindspore.mint.full
    mindspore.mint.full_like
    mindspore.mint.linspace
    mindspore.mint.ones
    mindspore.mint.ones_like
    mindspore.mint.randint
    mindspore.mint.randint_like
    mindspore.mint.randn
    mindspore.mint.randn_like
    mindspore.mint.randperm
    mindspore.mint.zeros
    mindspore.mint.zeros_like

索引、切分、连接、突变运算
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.cat
    mindspore.mint.chunk
    mindspore.mint.concat
    mindspore.mint.count_nonzero
    mindspore.mint.gather
    mindspore.mint.index_select
    mindspore.mint.masked_select
    mindspore.mint.permute
    mindspore.mint.reshape
    mindspore.mint.scatter
    mindspore.mint.scatter_add
    mindspore.mint.split
    mindspore.mint.narrow
    mindspore.mint.nonzero
    mindspore.mint.tile
    mindspore.mint.tril
    mindspore.mint.select
    mindspore.mint.squeeze
    mindspore.mint.stack
    mindspore.mint.swapaxes
    mindspore.mint.transpose
    mindspore.mint.triu
    mindspore.mint.unbind
    mindspore.mint.unique_consecutive
    mindspore.mint.unsqueeze
    mindspore.mint.where

随机采样
------------

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.multinomial
    mindspore.mint.normal
    mindspore.mint.rand_like
    mindspore.mint.rand

数学运算
------------------

逐元素运算
^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.abs
    mindspore.mint.add
    mindspore.mint.addmv
    mindspore.mint.acos
    mindspore.mint.acosh
    mindspore.mint.arccos
    mindspore.mint.arccosh
    mindspore.mint.arcsin
    mindspore.mint.arcsinh
    mindspore.mint.arctan
    mindspore.mint.arctan2
    mindspore.mint.arctanh
    mindspore.mint.asin
    mindspore.mint.asinh
    mindspore.mint.atan
    mindspore.mint.atan2
    mindspore.mint.atanh
    mindspore.mint.bitwise_and
    mindspore.mint.bitwise_or
    mindspore.mint.bitwise_xor
    mindspore.mint.ceil
    mindspore.mint.clamp
    mindspore.mint.cos
    mindspore.mint.cosh
    mindspore.mint.cross
    mindspore.mint.diff
    mindspore.mint.div
    mindspore.mint.divide
    mindspore.mint.erf
    mindspore.mint.erfc
    mindspore.mint.erfinv
    mindspore.mint.exp
    mindspore.mint.exp2
    mindspore.mint.expm1
    mindspore.mint.fix
    mindspore.mint.float_power
    mindspore.mint.floor
    mindspore.mint.fmod
    mindspore.mint.frac
    mindspore.mint.lerp
    mindspore.mint.log
    mindspore.mint.log1p
    mindspore.mint.log2
    mindspore.mint.log10
    mindspore.mint.logaddexp
    mindspore.mint.logical_and
    mindspore.mint.logical_not
    mindspore.mint.logical_or
    mindspore.mint.logical_xor
    mindspore.mint.mul
    mindspore.mint.mv
    mindspore.mint.nansum
    mindspore.mint.nan_to_num
    mindspore.mint.neg
    mindspore.mint.negative
    mindspore.mint.pow
    mindspore.mint.polar
    mindspore.mint.ravel
    mindspore.mint.reciprocal
    mindspore.mint.remainder
    mindspore.mint.roll
    mindspore.mint.round
    mindspore.mint.rsqrt
    mindspore.mint.sigmoid
    mindspore.mint.sign
    mindspore.mint.sin
    mindspore.mint.sinc
    mindspore.mint.sinh
    mindspore.mint.softmax
    mindspore.mint.sqrt
    mindspore.mint.square
    mindspore.mint.sub
    mindspore.mint.t
    mindspore.mint.tan
    mindspore.mint.tanh
    mindspore.mint.trunc
    mindspore.mint.xlogy

Reduction运算
^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.amax
    mindspore.mint.amin
    mindspore.mint.argmax
    mindspore.mint.argmin
    mindspore.mint.all
    mindspore.mint.any
    mindspore.mint.cumprod
    mindspore.mint.histc
    mindspore.mint.logsumexp
    mindspore.mint.max
    mindspore.mint.mean
    mindspore.mint.median
    mindspore.mint.min
    mindspore.mint.norm
    mindspore.mint.prod
    mindspore.mint.sum
    mindspore.mint.std
    mindspore.mint.std_mean
    mindspore.mint.unique
    mindspore.mint.var
    mindspore.mint.var_mean

比较运算
^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.allclose
    mindspore.mint.argsort
    mindspore.mint.eq
    mindspore.mint.equal
    mindspore.mint.greater
    mindspore.mint.greater_equal
    mindspore.mint.gt
    mindspore.mint.isclose
    mindspore.mint.isfinite
    mindspore.mint.isinf
    mindspore.mint.isneginf
    mindspore.mint.le
    mindspore.mint.less
    mindspore.mint.less_equal
    mindspore.mint.lt
    mindspore.mint.maximum
    mindspore.mint.minimum
    mindspore.mint.ne
    mindspore.mint.not_equal
    mindspore.mint.topk
    mindspore.mint.sort

BLAS和LAPACK运算
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.addbmm
    mindspore.mint.addmm
    mindspore.mint.baddbmm
    mindspore.mint.bmm
    mindspore.mint.dot
    mindspore.mint.inverse
    mindspore.mint.matmul
    mindspore.mint.meshgrid
    mindspore.mint.mm
    mindspore.mint.outer
    mindspore.mint.trace

其他运算
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.broadcast_to
    mindspore.mint.cdist
    mindspore.mint.cummax
    mindspore.mint.cummin
    mindspore.mint.cumsum
    mindspore.mint.flatten
    mindspore.mint.flip
    mindspore.mint.repeat_interleave
    mindspore.mint.searchsorted
    mindspore.mint.tril

mindspore.mint.nn
------------------

损失函数
^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.L1Loss

卷积层
^^^^^^^^^^^^^^^^^^
.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Conv2d
    mindspore.mint.nn.Conv3d
    mindspore.mint.nn.ConvTranspose2d
    mindspore.mint.nn.Fold
    mindspore.mint.nn.Unfold

归一化层
^^^^^^^^^^^^^^^^^^
.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.BatchNorm1d
    mindspore.mint.nn.BatchNorm2d
    mindspore.mint.nn.BatchNorm3d
    mindspore.mint.nn.GroupNorm
    mindspore.mint.nn.LayerNorm
    mindspore.mint.nn.SyncBatchNorm

非线性激活层 (加权和，非线性)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.ELU
    mindspore.mint.nn.GELU
    mindspore.mint.nn.Hardshrink
    mindspore.mint.nn.Hardsigmoid
    mindspore.mint.nn.Hardswish
    mindspore.mint.nn.LogSigmoid
    mindspore.mint.nn.LogSoftmax
    mindspore.mint.nn.Mish
    mindspore.mint.nn.PReLU
    mindspore.mint.nn.ReLU
    mindspore.mint.nn.ReLU6
    mindspore.mint.nn.SELU
    mindspore.mint.nn.SiLU
    mindspore.mint.nn.Softmax
    mindspore.mint.nn.Softshrink
    mindspore.mint.nn.Tanh

嵌入层
^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Embedding

线性层
^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Linear

Dropout层
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Dropout
    mindspore.mint.nn.Dropout2d

池化层
^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.AdaptiveAvgPool1d
    mindspore.mint.nn.AdaptiveAvgPool2d
    mindspore.mint.nn.AdaptiveAvgPool3d
    mindspore.mint.nn.AvgPool2d
    mindspore.mint.nn.MaxUnpool2d

填充层
^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.ConstantPad1d
    mindspore.mint.nn.ConstantPad2d
    mindspore.mint.nn.ConstantPad3d
    mindspore.mint.nn.ReflectionPad1d
    mindspore.mint.nn.ReflectionPad2d
    mindspore.mint.nn.ReflectionPad3d
    mindspore.mint.nn.ReplicationPad1d
    mindspore.mint.nn.ReplicationPad2d
    mindspore.mint.nn.ReplicationPad3d
    mindspore.mint.nn.ZeroPad1d
    mindspore.mint.nn.ZeroPad2d
    mindspore.mint.nn.ZeroPad3d

损失函数
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.BCELoss
    mindspore.mint.nn.BCEWithLogitsLoss
    mindspore.mint.nn.CrossEntropyLoss
    mindspore.mint.nn.MSELoss
    mindspore.mint.nn.NLLLoss
    mindspore.mint.nn.SmoothL1Loss

图像处理层
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Upsample

工具
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Identity

mindspore.mint.nn.functional
-----------------------------

卷积函数
^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.conv2d
    mindspore.mint.nn.functional.conv3d
    mindspore.mint.nn.functional.conv_transpose2d
    mindspore.mint.nn.functional.fold
    mindspore.mint.nn.functional.unfold

池化函数
^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.adaptive_avg_pool1d
    mindspore.mint.nn.functional.adaptive_avg_pool2d
    mindspore.mint.nn.functional.avg_pool1d
    mindspore.mint.nn.functional.avg_pool2d
    mindspore.mint.nn.functional.max_pool2d
    mindspore.mint.nn.functional.max_unpool2d

非线性激活函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.batch_norm
    mindspore.mint.nn.functional.elu
    mindspore.mint.nn.functional.gelu
    mindspore.mint.nn.functional.group_norm
    mindspore.mint.nn.functional.hardshrink
    mindspore.mint.nn.functional.hardsigmoid
    mindspore.mint.nn.functional.hardswish
    mindspore.mint.nn.functional.layer_norm
    mindspore.mint.nn.functional.leaky_relu
    mindspore.mint.nn.functional.log_softmax
    mindspore.mint.nn.functional.logsigmoid
    mindspore.mint.nn.functional.mish
    mindspore.mint.nn.functional.prelu
    mindspore.mint.nn.functional.relu
    mindspore.mint.nn.functional.relu6
    mindspore.mint.nn.functional.relu_
    mindspore.mint.nn.functional.selu
    mindspore.mint.nn.functional.sigmoid
    mindspore.mint.nn.functional.silu
    mindspore.mint.nn.functional.softmax
    mindspore.mint.nn.functional.softplus
    mindspore.mint.nn.functional.softshrink
    mindspore.mint.nn.functional.tanh

归一化函数
^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.normalize

线性函数
^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.linear

Dropout函数
^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.dropout
    mindspore.mint.nn.functional.dropout2d

稀疏函数
^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.embedding
    mindspore.mint.nn.functional.one_hot

损失函数
^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.binary_cross_entropy
    mindspore.mint.nn.functional.binary_cross_entropy_with_logits
    mindspore.mint.nn.functional.l1_loss
    mindspore.mint.nn.functional.mse_loss
    mindspore.mint.nn.functional.nll_loss
    mindspore.mint.nn.functional.smooth_l1_loss

Vision函数
^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.interpolate
    mindspore.mint.nn.functional.grid_sample
    mindspore.mint.nn.functional.pad

mindspore.mint.optim
---------------------

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.optim.Adam
    mindspore.mint.optim.AdamW

mindspore.mint.linalg
----------------------

逆数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.linalg.inv
    mindspore.mint.linalg.matrix_norm
    mindspore.mint.linalg.norm
    mindspore.mint.linalg.vector_norm

mindspore.mint.special
----------------------

逐元素运算
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.special.erfc
    mindspore.mint.special.exp2
    mindspore.mint.special.expm1
    mindspore.mint.special.log1p
    mindspore.mint.special.log_softmax
    mindspore.mint.special.round
    mindspore.mint.special.sinc

mindspore.mint.distributed
--------------------------------

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.distributed.all_gather
    mindspore.mint.distributed.all_gather_into_tensor
    mindspore.mint.distributed.all_gather_object
    mindspore.mint.distributed.all_reduce
    mindspore.mint.distributed.all_to_all
    mindspore.mint.distributed.all_to_all_single
    mindspore.mint.distributed.barrier
    mindspore.mint.distributed.batch_isend_irecv
    mindspore.mint.distributed.broadcast
    mindspore.mint.distributed.broadcast_object_list
    mindspore.mint.distributed.destroy_process_group
    mindspore.mint.distributed.gather
    mindspore.mint.distributed.gather_object
    mindspore.mint.distributed.get_backend
    mindspore.mint.distributed.get_global_rank
    mindspore.mint.distributed.get_group_rank
    mindspore.mint.distributed.get_process_group_ranks
    mindspore.mint.distributed.get_rank
    mindspore.mint.distributed.get_world_size
    mindspore.mint.distributed.init_process_group
    mindspore.mint.distributed.irecv
    mindspore.mint.distributed.isend
    mindspore.mint.distributed.new_group
    mindspore.mint.distributed.P2POp
    mindspore.mint.distributed.recv
    mindspore.mint.distributed.reduce
    mindspore.mint.distributed.reduce_scatter
    mindspore.mint.distributed.reduce_scatter_tensor
    mindspore.mint.distributed.scatter
    mindspore.mint.distributed.scatter_object_list
    mindspore.mint.distributed.send
