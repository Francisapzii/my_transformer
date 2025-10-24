# my_transformer
1. 首先对于transformer，其网络架构为decoder加上encoder；
2. encoder的架构是embedding + self attention + layer normalization with residual+ ffn + layer normalization with residual，
3. decoder的架构是embedding + cross attention + layer normalization with residual + ffn + layer normalization with residual；
4. ffn是线性神经网络网络；layer normalization是将每个token内部的特征上，取均值和方差，使数据处于【0，1】之间。
5. embedding中包括：token embedding + positional embedding
6. self attention采用多头机制，使每个token都能够查询到与其他token之间的关系分值，并将每个输入token与序列中其他token的关系，转化为输出token；
7. cross attention中，有两个attention，结构完全相同，第一attention的输入是目标token序列，例如在训练时，输入整个目标序列；在推理时，初始输入start，每次输出的token，追加到目标输入序列中。
8. cross attention的第二个attention的输入：Q是此一个attention的输出，K和V是encoder的输出。
9. cross attention的第一个attention，其输入是start加上已经输出的序列；在训练阶段，将要输出的目标序列是已知的，每次输入一个token的效率太低，一般采用输入整个序列，然后采用look ahead mask，使这一层的输出的每个token，只跟输入中对应token自己以及之前的token计算相关关系。同时，批量的目标token需要对齐成同意长度，短序列通过不上padding token补齐，这时需要padding mask，避免计算真实token与padding token之间的关系，所以， target mask是look ahead mask与padding mask的通过“或”计算的结果。
10. cross attention的第二个attention，其mask是输入的source mask，因为，K和V来自于encoder的输出。


