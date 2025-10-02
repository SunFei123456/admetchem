from argparse import Namespace

import torch.nn as nn

from model.AModel import AModel
from model.HModel import HModel



class MoleculeModel(nn.Module):


    def __init__(self, classification: bool, multiclass: bool, pretrain: bool):
        """
        Initializes the MoleculeModel.

        :param classification: 模型是否为二分类模型.
        :param multiclass: 模型是否为多分类模型.
        :param pretrain: 模型是否使用预训练模型.

        """
        super(MoleculeModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)
        self.pretrain = pretrain

    # 这里的encoder指的是 pharmhgt 或者cmpnn
    def create_encoder(self, args, encoder_name):
        """
        创建Encoder

        :param args: 参数
        """
        self.encoder_name = encoder_name
        if encoder_name == 'AModel':
            self.encoder = AModel(args)
        elif encoder_name == 'HModel':
            self.encoder = HModel(args)


    def create_ffn(self, args):
        """
        为模型创建前馈网络。

        :param args: 参数
        """

        #1、是否是多分类
        self.multiclass = args["dataset_type"] == 'multiclass'
        if self.multiclass:
            #类别的数量
            self.num_classes = args["num_task"]

        #2、embedding的维度
        first_linear_dim = args["hid_dim"] * 1

        from model.util import get_func
        dropout = nn.Dropout(args["dropout"])
        activation = get_func(args['act'])

        #3、创建FFN层
        if args["ffn_num_layers"] == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args["output_size"])
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args["ffn_hidden_size"])
            ]
            for _ in range(args["ffn_num_layers"] - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args["ffn_hidden_size"], args["ffn_hidden_size"]),
                ])

            ffn.extend([
                activation,
                dropout,
                nn.Linear(args["ffn_hidden_size"], args["output_size"]),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)
    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: 输入.
        :return: 输出.
        """
        if not self.pretrain:
            # 非预训练阶段

            encoder_output = self.encoder(*input)
            output = self.ffn(encoder_output)



            # 在训练过程中不要应用 Sigmoid，因为使用了 BCEWithLogitsLoss，BCELogitsLoss内部已经用过了sigmoid了
            # 所以 如果是not training的时候 就加上sigmoid

            if self.classification and not self.training:
                output = self.sigmoid(output)
            if self.multiclass:
                output = output.reshape(
                    (output.size(0), -1, self.num_classes))
                if not self.training:
                    output = self.multiclass_softmax(
                        output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss
            return output,encoder_output
        else:
            # 预训练阶段这个就是投影头
            output = self.ffn(self.encoder(*input))
            embedding = None
            if self.encoder_name == 'AModel':
                embedding = self.encoder.get_atom_embeddings(*input)
                return output, embedding
            else:
                embedding = self.encoder.get_frag_embeddings(*input)
                index = self.encoder.get_atomToFragment_index(*input)
                return output, embedding,index


