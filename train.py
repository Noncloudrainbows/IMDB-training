import tensorflow as tf
import numpy as np

# 加载 IMDB 数据集
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.imdb.load_data()

# 对输入序列进行填充，使其具有相同的长度
train_x = tf.keras.preprocessing.sequence.pad_sequences(train_x, maxlen=500)
test_x = tf.keras.preprocessing.sequence.pad_sequences(test_x, maxlen=500)

# 计算数据集中的最大单词索引值
max_index = np.max([np.max(x) for x in train_x])
print("Max index:", max_index)

# 构建模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=max_index+1, output_dim=16, input_length=500),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(train_x, train_y, epochs=10, batch_size=40, validation_split=0.2)

# 在测试集上评估模型
loss, accuracy = model.evaluate(test_x, test_y)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)