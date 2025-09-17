\begin{itemize}[itemsep=0pt, topsep=0pt, label=\textbullet]
  \item 以 SASRec 模型作为baseline，初始评分 \textbf{0.014}。
  \item 将 交叉熵损失替换为 Triplet Loss，拉近用户与正样本距离，推远负样本，显式建模排序关系，评分提升至 \textbf{0.032}；使用 InfoNCE 对比学习损失，利用噪声对比学习，提升 表示判别性 和泛化能力，进一步提升至 \textbf{0.052}。
  \item 在推理阶段，将预测方式由“下一行为预测”改为“使用完整用户序列预测”，评分提升至 \textbf{0.065}。
  \item 引入 时间特征和点击特征，模型更好捕捉用户行为语义和时间差异，评分提升至 \textbf{0.078}。
  \item 通过 调整温度系数与扩展模型规模，评分提升至 \textbf{0.086}。
  \item User和Item embedding采用zero-init初始化，模型输入包含多模态特征、类别特征，这些特征比ID更稳定、更能泛化，把ID embedding置零防止模型过早依赖ID的regularization作用，让模型先学会用强信号，在后期微调ID embedding作为补充，评分达到\textbf{0.089}。
  \item 采用 HSTU 模型结构 替换原有 SASRec，HSTU 会把长序列划分为局部片段和 全局表示，这样可以同时捕捉 短期兴趣和长期偏好。评分达到 \textbf{0.096}，相较baseline提升近 7 倍。
\end{itemize}
