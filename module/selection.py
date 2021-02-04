import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import relu, gelu, tanh
from torch.nn.parameter import Parameter
import math


class OriMHSLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels):
        super(OriMHSLayer, self).__init__()
        self.selection_u = nn.Linear(input_size, hidden_size)
        self.selection_w = nn.Linear(input_size, hidden_size)
        self.v = nn.Linear(hidden_size, num_labels)
        self.bias = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

    @staticmethod
    def broadcasting(left, right):
        x2 = left.permute(1, 0, 2).unsqueeze(-1)
        y2 = right.permute(0, 2, 1).unsqueeze(0)
        return (x2 + y2).permute(1, 0, 3, 2)

    def forward(self, input_tensor):
        left = self.selection_u(input_tensor)
        right = self.selection_w(input_tensor)
        outer_sum = self.broadcasting(left, right)
        outer_sum_bias = outer_sum + self.bias
        output = self.v(outer_sum_bias).permute(0, 1, 3, 2)
        return output


class BaseMHSLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, activation="relu"):
        super(BaseMHSLayer, self).__init__()
        self.selection_u = nn.Linear(input_size, hidden_size)
        self.selection_v = nn.Linear(input_size, hidden_size)
        self.selection_uv = nn.Linear(2 * hidden_size, hidden_size)
        self.relation_embedding = nn.Embedding(num_embeddings=num_labels, embedding_dim=hidden_size)
        self.activation = ACT2FN[activation]

    def forward(self, input_tensor):
        B, L, H = input_tensor.size()
        u = self.activation(self.selection_u(input_tensor)).unsqueeze(2).expand(B, L, L, -1)
        v = self.activation(self.selection_v(input_tensor)).unsqueeze(1).expand(B, L, L, -1)
        uv = self.activation(self.selection_uv(torch.cat((u, v), dim=-1)))
        output = torch.einsum('bijh,rh->birj', uv, self.relation_embedding.weight)
        return output


# class LinearLayer(nn.Module):
#     def __init__(self, input_size, hidden_size, num_labels, activation="relu"):
#         super(LinearLayer, self).__init__()
#         self.s_fc = nn.Linear(input_size, hidden_size)
#         self.t_fc = nn.Linear(input_size, hidden_size)
#         self.fc = nn.Linear(2 * hidden_size, hidden_size)
#         # self.relation_embedding = nn.Embedding(num_embeddings=num_labels, embedding_dim=hidden_size)
#         self.weight = nn.Parameter(torch.Tensor(num_labels, hidden_size))
#         init.normal_(self.weight)
#         self.activation = ACT2FN[activation]
#
#     def forward(self, input_tensor):
#         B, L, H = input_tensor.size()
#         s = self.activation(self.s_fc(input_tensor)).unsqueeze(2).expand(B, L, L, -1)
#         t = self.activation(self.t_fc(input_tensor)).unsqueeze(1).expand(B, L, L, -1)
#         o = self.activation(self.fc(torch.cat((s, t), dim=-1)))
#         output = torch.einsum('bijh,rh->birj', o, self.weight)
#         return output

# class LinearLayer(nn.Module):
#     def __init__(self, input_size, hidden_size, num_labels, activation="relu"):
#         super(LinearLayer, self).__init__()
#         self.s_fc = nn.Linear(input_size, hidden_size)
#         self.t_fc = nn.Linear(input_size, hidden_size)
#         self.fc = nn.Linear(hidden_size * 2, num_labels)
#         self.activation = ACT2FN[activation]
#         # self.fc1 = nn.Linear(input_size, hidden_size)
#         # self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
#         # self.fc3 = nn.Linear(hidden_size, num_labels)
#         self.init_weights()
#
#     def init_weights(self):
#         def _init_weights(module):
#             """ Initialize the weights """
#             if isinstance(module, (nn.Linear, nn.Embedding)):
#                 module.weight.data.normal_()
#             if isinstance(module, nn.Linear) and module.bias is not None:
#                 module.bias.data.zero_()
#
#         self.apply(_init_weights)
#
#     def forward(self, input_tensor):
#         # input_tensor = self.fc1(input_tensor)
#         B, L, H = input_tensor.size()
#         # s = input_tensor.unsqueeze(2).expand(B, L, L, -1)
#         # t = input_tensor.unsqueeze(1).expand(B, L, L, -1)
#         # s = self.s_fc(input_tensor).unsqueeze(2).expand(B, L, L, -1)
#         # t = self.t_fc(input_tensor).unsqueeze(1).expand(B, L, L, -1)
#         s = self.activation(self.s_fc(input_tensor)).unsqueeze(2).expand(B, L, L, -1)
#         t = self.activation(self.t_fc(input_tensor)).unsqueeze(1).expand(B, L, L, -1)
#         output = self.fc(torch.cat((s, t), dim=-1))
#         output = output.permute(0, 1, 3, 2)
#         return output


class LinearLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, activation="relu"):
        super(LinearLayer, self).__init__()
        self.s_fc = nn.Linear(input_size, hidden_size)
        self.t_fc = nn.Linear(input_size, hidden_size)
        # self.fc = nn.Linear(hidden_size * 2, num_labels)
        self.activation = ACT2FN[activation]
        # self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_labels, bias=False)
        self.init_weights()

    def init_weights(self):
        self.fc3.weight.data.normal_()
        # def _init_weights(module):
        #     """ Initialize the weights """
        #     if isinstance(module, (nn.Linear, nn.Embedding)):
        #         module.weight.data.normal_()
        #     if isinstance(module, nn.Linear) and module.bias is not None:
        #         module.bias.data.zero_()
        # self.apply(_init_weights)

    def forward(self, input_tensor):
        # input_tensor = self.fc1(input_tensor)
        B, L, H = input_tensor.size()
        # s = input_tensor.unsqueeze(2).expand(B, L, L, -1)
        # t = input_tensor.unsqueeze(1).expand(B, L, L, -1)
        # s = self.s_fc(input_tensor).unsqueeze(2).expand(B, L, L, -1)
        # t = self.t_fc(input_tensor).unsqueeze(1).expand(B, L, L, -1)
        s = self.activation(self.s_fc(input_tensor)).unsqueeze(2).expand(B, L, L, -1)
        t = self.activation(self.t_fc(input_tensor)).unsqueeze(1).expand(B, L, L, -1)
        output = self.activation(self.fc2(torch.cat((s, t), dim=-1)))
        output = self.fc3(output)
        output = output.permute(0, 1, 3, 2)
        return output


class LinearLayer2(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, activation="relu"):
        super(LinearLayer2, self).__init__()
        self.s_fc = nn.Linear(input_size, hidden_size)
        self.t_fc = nn.Linear(input_size, hidden_size)
        self.activation = ACT2FN[activation]
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_labels)
        self.init_weights()

    def init_weights(self):
        self.fc3.weight.data.normal_()

    def forward(self, input_tensor):
        B, L, H = input_tensor.size()
        s = self.activation(self.s_fc(input_tensor)).unsqueeze(2).expand(B, L, L, -1)
        t = self.activation(self.t_fc(input_tensor)).unsqueeze(1).expand(B, L, L, -1)
        output = self.activation(self.fc2(torch.cat((s, t), dim=-1)))
        output = self.fc3(output)
        output = output.permute(0, 1, 3, 2)
        return output

# 消融实验，没有FNN
class LinearLayer3(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, activation="relu"):
        super(LinearLayer3, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = ACT2FN[activation]
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_labels)
        self.init_weights()

    def init_weights(self):
        self.fc3.weight.data.normal_()

    def forward(self, input_tensor):
        B, L, H = input_tensor.size()
        input_tensor = self.fc1(input_tensor)

        s = input_tensor.unsqueeze(2).expand(B, L, L, -1)
        t = input_tensor.unsqueeze(1).expand(B, L, L, -1)
        output = self.activation(self.fc2(torch.cat((s, t), dim=-1)))
        output = self.fc3(output)
        output = output.permute(0, 1, 3, 2)
        return output



class SelfAttnMHSLayer(nn.Module):
    def __init__(self, input_size, num_labels):
        super(SelfAttnMHSLayer, self).__init__()
        self.dense = nn.Linear(input_size, num_labels)

    def forward(self, attention_probs, value_attentions, relative_positions=None):
        B, H, S, E = value_attentions.size()
        value_attentions = value_attentions.unsqueeze(2).expand(B, H, S, S, E)

        attention_probs = attention_probs.unsqueeze(-1)
        output = torch.mul(value_attentions, attention_probs)

        if relative_positions is not None:
            relative_positions = relative_positions.view(B, S, S, H, E).permute(0, 3, 1, 2, 4)
            output = output + torch.mul(relative_positions, attention_probs)

        output = output.permute(0, 3, 2, 1, 4).contiguous().view(B, S, S, E * H)

        output = self.dense(output)
        output = output.permute(0, 1, 3, 2)

        return output


class BiLinearMHSLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, add_bias=True):
        super(BiLinearMHSLayer, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.bi_linear = nn.Linear(hidden_size, hidden_size * num_labels, bias=False)

        self.add_bias = add_bias
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

    def forward(self, input_tensor):
        input_t = self.fc(input_tensor)
        B, S, E = input_t.size()
        value_t = input_t.unsqueeze(1).expand(B, S, S, E).permute(0, 1, 3, 2)
        bi_lin = self.bi_linear(input_t)
        bi_lin = bi_lin.view(B, S, -1, E)
        if self.add_bias:
            bi_lin += self.bias
        output = torch.matmul(bi_lin, value_t)
        return output


class BiLinearMHSLayer2(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, add_bias=True):
        super(BiLinearMHSLayer2, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.bi_linear = nn.Linear(hidden_size, hidden_size * num_labels, bias=False)

        self.add_bias = add_bias
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(num_labels), requires_grad=True)

        self.init_weights()

    def init_weights(self):
        self.bi_linear.weight.data.normal_()
        # def _init_weights(module):
        #     """ Initialize the weights """
        #     if isinstance(module, (nn.Linear, nn.Embedding)):
        #         module.weight.data.normal_()
        #     if isinstance(module, nn.Linear) and module.bias is not None:
        #         module.bias.data.zero_()
        # self.apply(_init_weights)

    def forward(self, input_tensor):
        input_tensor = self.fc(input_tensor)
        B, S, E = input_tensor.size()
        key = self.bi_linear(input_tensor)
        key = key.view(B, S, E, -1).permute(0, 3, 1, 2)
        value = input_tensor.unsqueeze(1).permute(0, 1, 3, 2)

        output = torch.matmul(key, value)
        if self.add_bias:
            output = output.permute(0, 2, 3, 1)
            output += self.bias
            output = output.permute(0, 1, 3, 2)
        else:
            output = output.permute(0, 2, 1, 3)

        return output


class BiLinearMHSLayer3(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, add_bias=True):
        super(BiLinearMHSLayer3, self).__init__()
        self.bi_linear = nn.Linear(input_size, input_size * num_labels, bias=False)

        self.add_bias = add_bias
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)

    def forward(self, input_tensor):
        # input_t = self.fc(input_tensor)
        B, S, E = input_tensor.size()
        value_t = input_tensor.unsqueeze(1).permute(0, 1, 3, 2)
        bi_lin = self.bi_linear(input_tensor)
        bi_lin = bi_lin.view(B, S, -1, E)
        if self.add_bias:
            bi_lin += self.bias
        output = torch.matmul(bi_lin, value_t)
        return output


class BiLinearMHSLayer4(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, add_bias=True):
        super(BiLinearMHSLayer4, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.bi_linear = nn.Linear(hidden_size, hidden_size * num_labels, bias=False)

        self.add_bias = add_bias
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(num_labels), requires_grad=True)

    def forward(self, input_tensor):
        input_tensor = self.fc(input_tensor)
        B, S, E = input_tensor.size()
        key = self.bi_linear(input_tensor)
        key = key.view(B, S, E, -1).permute(0, 3, 1, 2)
        value = input_tensor.unsqueeze(1).permute(0, 1, 3, 2)

        output = torch.matmul(key, value)
        if self.add_bias:
            output = output.permute(0, 2, 3, 1)
            output += self.bias
            output = output.permute(0, 1, 3, 2)
        else:
            output = output.permute(0, 2, 1, 3)

        return output


class BiLinearMHSLayer5(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, add_bias=True, activation="relu"):
        super(BiLinearMHSLayer5, self).__init__()
        self.bi_linear = nn.Linear(hidden_size, hidden_size * num_labels, bias=False)

        self.s_fc = nn.Linear(input_size, hidden_size)
        self.t_fc = nn.Linear(input_size, hidden_size)

        self.activation = ACT2FN[activation]
        self.add_bias = add_bias
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(num_labels), requires_grad=True)

    def forward(self, input_tensor):
        s = self.activation(self.s_fc(input_tensor))
        t = self.activation(self.t_fc(input_tensor))

        B, S, E = s.size()
        key = self.bi_linear(s)
        key = key.view(B, S, E, -1).permute(0, 3, 1, 2)
        value = t.unsqueeze(1).permute(0, 1, 3, 2)

        output = torch.matmul(key, value)
        if self.add_bias:
            output = output.permute(0, 2, 3, 1)
            output += self.bias
            output = output.permute(0, 1, 3, 2)
        else:
            output = output.permute(0, 2, 1, 3)

        return output


class BiAffineLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, activation="relu"):
        super().__init__()
        self.bi_linear = nn.Linear(hidden_size, hidden_size * num_labels, bias=False)

        self.s_fc = nn.Linear(input_size, hidden_size)
        self.t_fc = nn.Linear(input_size, hidden_size)

        self.fc = nn.Linear(2 * hidden_size, num_labels)

        self.activation = ACT2FN[activation]
        # self.bias = nn.Parameter(torch.zeros(num_labels), requires_grad=True)

    def forward(self, input_tensor):
        s = self.activation(self.s_fc(input_tensor))
        t = self.activation(self.t_fc(input_tensor))

        B, S, E = s.size()
        key = self.bi_linear(s)
        key = key.view(B, S, E, -1).permute(0, 3, 1, 2)
        value = t.unsqueeze(1).permute(0, 1, 3, 2)

        output_1 = torch.matmul(key, value)
        output_1 = output_1.permute(0, 2, 1, 3)

        u = s.unsqueeze(2).expand(B, S, S, -1)
        v = t.unsqueeze(1).expand(B, S, S, -1)
        uv = torch.cat((u, v), dim=-1)
        output_2 = self.fc(uv)
        output_2 = output_2.permute(0, 1, 3, 2)

        output = output_1 + output_2

        return output


class BiAffineLayer2(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, activation="relu"):
        super().__init__()
        self.bi_linear = nn.Linear(hidden_size, hidden_size * num_labels, bias=False)

        self.s_fc = nn.Linear(input_size, hidden_size)
        self.t_fc = nn.Linear(input_size, hidden_size)

        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_labels)

        self.weight = nn.Parameter(torch.Tensor(num_labels, hidden_size), requires_grad=True)
        init.normal_(self.weight)
        self.activation = ACT2FN[activation]
        # self.bias = nn.Parameter(torch.zeros(num_labels), requires_grad=True)

    def forward(self, input_tensor):
        s = self.activation(self.s_fc(input_tensor))
        t = self.activation(self.t_fc(input_tensor))

        B, S, E = s.size()
        key = self.bi_linear(s)
        key = key.view(B, S, E, -1).permute(0, 3, 1, 2)
        value = t.unsqueeze(1).permute(0, 1, 3, 2)

        output_1 = torch.matmul(key, value)
        output_1 = output_1.permute(0, 2, 1, 3)

        u = s.unsqueeze(2).expand(B, S, S, -1)
        v = t.unsqueeze(1).expand(B, S, S, -1)
        uv = self.activation(self.fc1(torch.cat((u, v), dim=-1)))
        output_2 = self.fc2(uv)
        output_2 = output_2.permute(0, 1, 3, 2)

        output = output_1 + output_2

        return output


class BiAffineLayer3(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, activation="relu"):
        super(BiAffineLayer3, self).__init__()
        self.s_fc = nn.Linear(input_size, hidden_size)
        self.t_fc = nn.Linear(input_size, hidden_size)
        self.activation = ACT2FN[activation]
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_labels)

        self.bi_linear = nn.Linear(hidden_size, hidden_size * num_labels, bias=False)

        self.init_weights()

    def init_weights(self):
        self.fc3.weight.data.normal_()
        self.bi_linear.weight.data.normal_()

    def forward(self, input_tensor):
        B, L, H = input_tensor.size()
        s = self.activation(self.s_fc(input_tensor))
        t = self.activation(self.t_fc(input_tensor))

        B, S, E = s.size()
        key = self.bi_linear(s)
        key = key.view(B, S, E, -1).permute(0, 3, 1, 2)
        value = t.unsqueeze(1).permute(0, 1, 3, 2)
        output1 = torch.matmul(key, value)
        output1 = output1.permute(0, 2, 1, 3)

        u = s.unsqueeze(2).expand(B, L, L, -1)
        v = t.unsqueeze(1).expand(B, L, L, -1)
        output2 = self.activation(self.fc2(torch.cat((u, v), dim=-1)))
        output2 = self.fc3(output2)
        output2 = output2.permute(0, 1, 3, 2)
        output = output1 + output2
        return output




# class LinearLayer(nn.Module):
#     def __init__(self, input_size, hidden_size, num_labels):
#         super(LinearLayer, self).__init__()
#         self.s_fc = nn.Linear(input_size, hidden_size)
#         self.t_fc = nn.Linear(input_size, hidden_size)
#         self.fc = nn.Linear(hidden_size * 2, num_labels)
#
#     def forward(self, input_tensor):
#         B, L, H = input_tensor.size()
#         s = self.s_fc(input_tensor).unsqueeze(2).expand(B, L, L, -1)
#         t = self.t_fc(input_tensor).unsqueeze(1).expand(B, L, L, -1)
#         uv = torch.cat((s, t), dim=-1)
#         output = self.fc(uv)
#         output = output.permute(0, 1, 3, 2)
#         return output


class DependencyLinearLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, num_dep_labels):
        super(DependencyLinearLayer, self).__init__()
        self.s_fc = nn.Linear(input_size, hidden_size)
        self.t_fc = nn.Linear(input_size, hidden_size)
        self.dep_embedding = nn.Embedding(num_dep_labels, hidden_size)
        self.classifier = nn.Linear(hidden_size * 3, num_labels)

    def forward(self, input_tensor, dependency_graph):
        B, L, H = input_tensor.size()
        s = self.s_fc(input_tensor).unsqueeze(2).expand(B, L, L, -1)
        t = self.t_fc(input_tensor).unsqueeze(1).expand(B, L, L, -1)
        dep_embed = self.dep_embedding(dependency_graph)

        features = torch.cat((s, t, dep_embed), -1)
        output = self.classifier(features)

        return output.permute(0, 1, 3, 2)


class DependencyBaseMHSLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, num_dep_labels, activation="relu"):
        super().__init__()
        self.selection_u = nn.Linear(input_size, hidden_size)
        self.selection_v = nn.Linear(input_size, hidden_size)
        self.dep_embedding = nn.Embedding(num_dep_labels, hidden_size)

        self.selection_uv = nn.Linear(3 * hidden_size, hidden_size)
        self.relation_embedding = nn.Embedding(num_embeddings=num_labels, embedding_dim=hidden_size)
        self.activation = ACT2FN[activation]

    def forward(self, input_tensor, dependency_graph):
        B, L, H = input_tensor.size()
        u = self.activation(self.selection_u(input_tensor)).unsqueeze(2).expand(B, L, L, -1)
        v = self.activation(self.selection_v(input_tensor)).unsqueeze(1).expand(B, L, L, -1)
        d = self.activation(self.dep_embedding(dependency_graph))
        uv = self.activation(self.selection_uv(torch.cat((u, v, d), dim=-1)))
        output = torch.einsum('bijh,rh->birj', uv, self.relation_embedding.weight)
        return output


class DepBiAffineLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, num_dep_labels, activation="relu"):
        super().__init__()
        self.bi_linear = nn.Linear(hidden_size, hidden_size * num_labels, bias=False)

        self.s_fc = nn.Linear(input_size, hidden_size)
        self.t_fc = nn.Linear(input_size, hidden_size)

        self.fc = nn.Linear(3 * hidden_size, num_labels)

        self.dep_embedding = nn.Embedding(num_dep_labels, hidden_size)

        self.activation = ACT2FN[activation]
        # self.bias = nn.Parameter(torch.zeros(num_labels), requires_grad=True)

    def forward(self, input_tensor, dependency_graph):
        s = self.activation(self.s_fc(input_tensor))
        t = self.activation(self.t_fc(input_tensor))

        B, S, E = s.size()
        key = self.bi_linear(s)
        key = key.view(B, S, E, -1).permute(0, 3, 1, 2)
        value = t.unsqueeze(1).permute(0, 1, 3, 2)

        output_1 = torch.matmul(key, value)
        output_1 = output_1.permute(0, 2, 1, 3)

        u = s.unsqueeze(2).expand(B, S, S, -1)
        v = t.unsqueeze(1).expand(B, S, S, -1)
        d = self.activation(self.dep_embedding(dependency_graph))
        uv = torch.cat((u, v, d), dim=-1)
        output_2 = self.fc(uv)
        output_2 = output_2.permute(0, 1, 3, 2)

        output = output_1 + output_2

        return output


class DepBiAffineLayer2(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, num_dep_labels, activation="relu"):
        super().__init__()
        self.bi_linear = nn.Linear(hidden_size, hidden_size * num_labels, bias=False)

        self.s_fc = nn.Linear(input_size, hidden_size)
        self.t_fc = nn.Linear(input_size, hidden_size)

        self.fc1 = nn.Linear(3 * hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_labels)

        self.dep_embedding = nn.Embedding(num_dep_labels, hidden_size)

        self.weight = nn.Parameter(torch.Tensor(num_labels, hidden_size), requires_grad=True)
        init.normal_(self.weight)
        self.activation = ACT2FN[activation]
        # self.bias = nn.Parameter(torch.zeros(num_labels), requires_grad=True)

    def forward(self, input_tensor, dependency_graph):
        s = self.activation(self.s_fc(input_tensor))
        t = self.activation(self.t_fc(input_tensor))

        B, S, E = s.size()
        key = self.bi_linear(s)
        key = key.view(B, S, E, -1).permute(0, 3, 1, 2)
        value = t.unsqueeze(1).permute(0, 1, 3, 2)

        output_1 = torch.matmul(key, value)
        output_1 = output_1.permute(0, 2, 1, 3)

        u = s.unsqueeze(2).expand(B, S, S, -1)
        v = t.unsqueeze(1).expand(B, S, S, -1)
        d = self.activation(self.dep_embedding(dependency_graph))

        uv = self.activation(self.fc1(torch.cat((u, v, d), dim=-1)))
        output_2 = self.fc2(uv)
        output_2 = output_2.permute(0, 1, 3, 2)

        output = output_1 + output_2

        return output


ACT2FN = {"gelu": gelu, "relu": relu, "tanh": tanh}

INFERENCE_CLASS = {
    "OriMHSLayer": OriMHSLayer,
    "BaseMHSLayer": BaseMHSLayer,
    "LinearLayer": LinearLayer,
    "LinearLayer2": LinearLayer2,
    "LinearLayer3": LinearLayer3,
    "SelfAttnMHSLayer": SelfAttnMHSLayer,
    "BiLinearMHSLayer": BiLinearMHSLayer,
    "BiLinearMHSLayer2": BiLinearMHSLayer2,
    "BiLinearMHSLayer3": BiLinearMHSLayer3,
    "BiLinearMHSLayer4": BiLinearMHSLayer4,
    "BiLinearMHSLayer5": BiLinearMHSLayer5,
    "BiAffineLayer": BiAffineLayer,
    "BiAffineLayer2": BiAffineLayer2,
    "BiAffineLayer3": BiAffineLayer3,
    "DepBiAffineLayer": DepBiAffineLayer,
    "DepBiAffineLayer2": DepBiAffineLayer2,
    "DepBaseLayer": DependencyBaseMHSLayer
}
