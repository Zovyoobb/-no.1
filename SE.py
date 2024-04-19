import tensorflow as tf
from tensorflow.keras import layers, Input, Model
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# se注意力机制
def se_block(inputs, ratio=4):  # ratio代表第一个全连接层下降通道数的系数

    # 获取输入特征图的通道数
    in_channel = inputs.shape[-1]

    # 全局平均池化[h,w,c]==>[None,c]
    x = layers.GlobalAveragePooling2D()(inputs)

    # [None,c]==>[1,1,c]
    x = layers.Reshape(target_shape=(1, 1, in_channel))(x)

    # [1,1,c]==>[1,1,c/4]
    x = layers.Dense(in_channel // ratio)(x)  # 全连接下降通道数

    # relu激活
    x = tf.nn.relu(x)

    # [1,1,c/4]==>[1,1,c]
    x = layers.Dense(in_channel)(x)  # 全连接上升通道数

    # sigmoid激活，权重归一化
    x = tf.nn.sigmoid(x)

    # [h,w,c]*[1,1,c]==>[h,w,c]
    outputs = layers.multiply([inputs, x])  # 归一化权重和原输入特征图逐通道相乘

    return outputs


# 测试SE注意力机制
# if __name__ == '__main__':
#     # 构建输入
#     inputs = Input([56, 56, 24])
#
#     x = se_block(inputs)  # 接收SE返回值
#
#     model = Model(inputs, x)  # 构建网络模型
#
#     print(x.shape)  # (None, 56, 56, 24)
#     model.summary()  # 输出SE模块的结构
