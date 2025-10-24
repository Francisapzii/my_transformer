# my_transformer
1. 首先对于transformer，其网络架构为decoder加上encoder；
2. encoder的架构是embedding + self attention + layer normalization with residual+ ffn + layer normalization with residual，
3. decoder的架构是embedding + cross attention + layer normalization with residual + ffn + layer normalization with residual；
4. ffn是线性神经网络网络；layer normalization是将每个token内部的特征上，取均值和方差，使数据处于【0，1】之间。
5. embedding中包括：token embedding + positional embedding
6. self attention采用多头机制，使每个token都能够查询到与其他token之间的关系分值，并将每个输入token与序列中其他token的关系，转化为输出token；
7. cross attention中，有两个attention，结构完全相同，第一attention的输入是目标

